"""計測中のマシンの状態を記録する (bench.py が使う)。

  GPU    nvidia-smi -lms 500: グラフィックス / メモリのクロック、温度、電力、使用率、クロックが下がった理由
  CPU    Windows のカウンター (typeperf, 1 秒ごと): Processor Frequency x % Processor Performance = 実効クロック
         CPU の温度は管理者権限がないと読めない (MSAcpi_ThermalZoneTemperature) ので記録しない
  他の処理 psutil: 計測の間に各プロセスが使った CPU 時間 (エンジンと計測自身を除く) の上位
  VRAM   エンジンの専用 / 共有 GPU メモリ (Windows の GPU Process Memory カウンター)。共有が増えていれば、
         VRAM からシステムメモリへはみ出している
"""
import csv
import io
import subprocess
import threading
import time

import psutil

# nvidia-smi の clocks_event_reasons のビット
REASONS = {0x1: "idle", 0x2: "app_clocks", 0x4: "sw_power_cap", 0x8: "hw_slowdown", 0x20: "sw_thermal",
           0x40: "hw_thermal", 0x80: "hw_power_brake", 0x100: "display_clocks"}


class Telemetry:
    def __init__(self, engine_pid):
        self.engine_pid = engine_pid
        self.gpu, self.cpu = [], []
        self._procs = []

    def _read_gpu(self):
        for line in self._smi.stdout:
            p = [x.strip() for x in line.split(",")]
            if len(p) < 6:
                continue
            try:
                self.gpu.append({"t": time.time(), "clk": float(p[0]), "mclk": float(p[1]), "temp": float(p[2]),
                                 "power": float(p[3]), "util": float(p[4]), "reasons": int(p[5], 16)})
            except ValueError:
                pass

    def _read_cpu(self):
        for row in csv.reader(io.TextIOWrapper(self._tp.stdout, encoding="cp932", errors="replace")):
            if len(row) < 3:
                continue
            try:
                freq, perf = float(row[1]), float(row[2])
            except ValueError:
                continue
            self.cpu.append({"t": time.time(), "mhz": freq * perf / 100.0})

    def _cpu_times(self):
        out = {}
        for p in psutil.process_iter(["pid", "name", "cpu_times"]):
            ct = p.info.get("cpu_times")
            if ct is not None:
                out[p.info["pid"]] = (p.info["name"], ct.user + ct.system)
        return out

    def start(self):
        self._t0 = time.time()
        self._smi = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=clocks.gr,clocks.mem,temperature.gpu,power.draw,utilization.gpu,"
             "clocks_event_reasons.active", "--format=csv,noheader,nounits", "-lms", "500"],
            stdout=subprocess.PIPE, text=True)
        threading.Thread(target=self._read_gpu, daemon=True).start()
        self._tp = subprocess.Popen(["typeperf", r"\Processor Information(_Total)\Processor Frequency",
                                     r"\Processor Information(_Total)\% Processor Performance", "-si", "1"],
                                    stdout=subprocess.PIPE)
        threading.Thread(target=self._read_cpu, daemon=True).start()
        self._before = self._cpu_times()
        return self

    def stop(self):
        t1 = time.time()
        after = self._cpu_times()
        for p in (self._smi, self._tp):
            p.terminate()
        me = {psutil.Process().pid, self.engine_pid}
        try:
            me |= {c.pid for c in psutil.Process().children(recursive=True)}
            me |= {psutil.Process(self.engine_pid).ppid()}   # bench/server.py
        except psutil.Error:
            pass
        used = {}
        for pid, (name, t) in after.items():
            if pid in me or pid == 0 or name in ("System Idle Process",):
                continue
            dt = t - self._before[pid][1] if pid in self._before else t   # 計測中に起動したものは全部
            used[name] = used.get(name, 0.0) + max(0.0, dt)
        wall = max(1e-9, t1 - self._t0)
        eng = after.get(self.engine_pid, (None, 0.0))[1] - self._before.get(self.engine_pid, (None, 0.0))[1]
        top = sorted(used.items(), key=lambda x: -x[1])[:3]
        g = [s for s in self.gpu if s["t"] >= self._t0]
        busy = [s for s in g if s["util"] >= 50] or g   # 計算している時の値 (待ちの合間を除く)
        reasons = 0
        for s in busy:
            reasons |= s["reasons"]
        mean = lambda xs: round(sum(xs) / len(xs), 1) if xs else ""
        return {
            "gpu_clk_mhz": mean([s["clk"] for s in busy]),
            "gpu_clk_min_mhz": min([s["clk"] for s in busy], default=""),
            "gpu_mem_clk_mhz": mean([s["mclk"] for s in busy]),
            "gpu_temp_max_c": max([s["temp"] for s in g], default=""),
            "gpu_power_w": mean([s["power"] for s in busy]),
            "gpu_util_pct": mean([s["util"] for s in g]),
            "gpu_slowdown": "+".join(v for k, v in REASONS.items() if reasons & k and k != 0x1) or "none",
            "cpu_mhz": mean([s["mhz"] for s in self.cpu if s["t"] >= self._t0]),
            "cpu_temp_c": "",   # 管理者権限なしでは読めない
            "other_cpu_pct": round(100.0 * sum(used.values()) / wall / psutil.cpu_count(), 1),
            "other_top": " ".join(f"{n}:{100.0 * v / wall:.0f}%" for n, v in top),
            "engine_cpu_pct": round(100.0 * eng / wall / psutil.cpu_count(), 1),   # of all the CPU's threads
            "engine_cpu_cores": round(eng / wall, 2),
            **gpu_memory(self.engine_pid),
        }


def gpu_memory(pid):
    """エンジンの専用 / 共有 GPU メモリ (MiB)。"""
    ps = (f"(Get-Counter '\\GPU Process Memory(pid_{pid}_*)\\Dedicated Usage','\\GPU Process Memory(pid_{pid}_*)"
          f"\\Shared Usage' -ErrorAction SilentlyContinue).CounterSamples | % {{ $_.Path + '=' + $_.CookedValue }}")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                         encoding="cp932", errors="replace", timeout=60).stdout
    ded = sh = 0.0
    for line in out.splitlines():
        if "=" not in line:
            continue
        path, v = line.rsplit("=", 1)
        try:
            v = float(v)
        except ValueError:
            continue
        if "dedicated" in path.lower():
            ded += v
        elif "shared" in path.lower():
            sh += v
    return {"engine_gpu_dedicated_mib": round(ded / 2**20), "engine_gpu_shared_mib": round(sh / 2**20)}
