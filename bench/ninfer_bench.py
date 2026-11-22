"""NInfer (iamwavecut/ninfer-all) か llama.cpp の llama-server を、自前エンジンと同じ入力で測る (計測のみ)。

ninfer-serve を 127.0.0.1:PORT (既定 18471) で起動し、/completion にトークン列 (bench/data/*.ids) を送って、応答の
timings (prompt_per_second、predicted_per_second) を読む。
  8K    生成: gen.ids から 216 トークン (素・温度 1.0 と greedy)。読み込み: prefill_8k.ids (cache_prompt なし)
  128K  long_code_base + long_code_tail (131072 トークン) を読み、続けて同じプロンプトで生成 (前の状態を再利用)
温度 1.0 は temperature=1.0 top_p=1 min_p=0 seed=1234。NInfer の top_k は 20 が上限 (0 以下でも 20) なので、自前エンジンの
top_k=0 (語彙全体) とは候補の数が違う。

使い方: python -I bench/ninfer_bench.py [--long] [--reps 2] [--label メモ] [-- ninfer-serve の追加の設定]
        python -I bench/ninfer_bench.py --llamacpp [-- llama-server の追加の設定]   (ref/llama.cpp のビルド)
結果は bench/ninfer_results.csv に追記。サーバーのログは bench/logs/ninfer.log
"""
import argparse
import csv
import datetime
import json
import pathlib
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bench  # noqa: E402
import telemetry  # noqa: E402

ROOT = bench.ROOT
NINFER = ROOT / "ref" / "ninfer" / "bin"
MODEL = ROOT / "models" / "ninfer" / "Inkling-2-Mini-UD-IQ4_XS-ninfer-v3.ninfer"
OUT = ROOT / "bench" / "ninfer_results.csv"
LLAMA = ROOT / "ref" / "llama.cpp" / "build" / "bin"
LLAMA_MODEL = ROOT / "models" / "Mainline_Support" / "Inkling-2-Mini-UD-IQ4_XS.gguf"   # llama.cpp の inkling2_loop 形式
T1 = {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "seed": 1234}
GREEDY = {"temperature": -1.0}


def post(port, body, timeout=7200):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def up(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def find_exe(name):
    hits = sorted(NINFER.rglob(name + ".exe"))
    if not hits:
        raise SystemExit(f"{name}.exe が {NINFER} にない")
    return hits[0]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", action="store_true")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--label", default="")
    ap.add_argument("--port", type=int, default=18471)
    ap.add_argument("--max-context", type=int, default=0, help="既定: 8K は 16384、--long は 132096")
    ap.add_argument("--greedy", action="store_true", help="greedy の生成も測る")
    ap.add_argument("--llamacpp", action="store_true", help="NInfer の代わりに llama.cpp の llama-server")
    ap.add_argument("--attach", default="", help="起動済みのサーバーを測る: そのエンジンのプロセス名 (例 strata-inkling)")
    ap.add_argument("serve_args", nargs="*")
    a = ap.parse_args()
    if a.port in (8080, 3000, 8765, 8081, 18457):
        raise SystemExit("そのポートは使わない")
    if not a.attach:
        bench.stop_server()
    ctx = a.max_context or (132096 if a.long else 16384)
    if a.llamacpp:
        exe = LLAMA / "llama-server.exe"
        cmd = [str(exe), "-m", str(LLAMA_MODEL), "--host", "127.0.0.1", "--port", str(a.port), "-c", str(ctx),
               "-fa", "on", "-t", str(bench.THREADS), "--no-webui"] + a.serve_args
    else:
        exe = find_exe("ninfer-serve")
        cmd = [str(exe), str(MODEL), "--model-id", "inkling-2-mini", "--host", "127.0.0.1",
               "--port", str(a.port), "--max-context", str(ctx), "--spec", "none"] + a.serve_args
    log = open(ROOT / "bench" / "logs" / ("llamacpp.log" if a.llamacpp else "ninfer.log"), "w", encoding="utf-8",
               errors="replace")
    print("起動: " + " ".join(cmd[1:]), flush=True)
    t0 = time.time()
    if a.attach:   # 起動済み: 何も起動せず、終わっても止めない
        import psutil

        class Attached:
            pid = next(q.pid for q in psutil.process_iter(["name"]) if (q.info["name"] or "").startswith(a.attach))
            returncode = None

            def poll(self):
                return None

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0
        p = Attached()
    else:
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(exe.parent))
    try:
        while not up(a.port):
            if p.poll() is not None:
                raise SystemExit(f"サーバーが終了した (コード {p.returncode})。bench/logs/ のログを見る")
            if time.time() - t0 > 900:
                raise SystemExit("ninfer-serve が 15 分で起動しない")
            time.sleep(2)
        print(f"起動まで {time.time() - t0:.0f} 秒", flush=True)
        gen = bench.ids("gen.ids")
        n_new = bench.GEN_SKIP + bench.GEN_COUNT
        # ウォームアップ (数えない)
        post(a.port, {"prompt": gen, "n_predict": 32, "ignore_eos": True, "cache_prompt": False, **GREEDY})
        tele = telemetry.Telemetry(p.pid).start()
        res = {"t1": [], "greedy": [], "prefill": []}
        if not a.long:
            pre = bench.ids("prefill_8k.ids")
            for r in range(a.reps):
                j = post(a.port, {"prompt": gen, "n_predict": n_new, "ignore_eos": True, "cache_prompt": False, **T1})
                res["t1"].append(j["timings"]["predicted_per_second"])
                if a.greedy:
                    j = post(a.port, {"prompt": gen, "n_predict": n_new, "ignore_eos": True, "cache_prompt": False,
                                      **GREEDY})
                    res["greedy"].append(j["timings"]["predicted_per_second"])
                j = post(a.port, {"prompt": [1000 + r] + pre[1:], "n_predict": 1, "cache_prompt": False, **GREEDY})
                res["prefill"].append(j["timings"]["prompt_per_second"])
        else:
            prompt = bench.ids("long_code_base.ids") + bench.ids("long_code_tail.ids")
            prompt = prompt[:ctx - n_new - 8]
            j = post(a.port, {"prompt": prompt, "n_predict": 1, "cache_prompt": True, **GREEDY})
            res["prefill"].append(j["timings"]["prompt_per_second"])
            res["prefill_n"] = j["timings"]["prompt_n"]
            for r in range(a.reps):
                j = post(a.port, {"prompt": prompt, "n_predict": n_new, "ignore_eos": True, "cache_prompt": True, **T1})
                res["t1"].append(j["timings"]["predicted_per_second"])
                res.setdefault("cache_n", []).append(j["timings"].get("cache_n"))
                if a.greedy:
                    j = post(a.port, {"prompt": prompt, "n_predict": n_new, "ignore_eos": True, "cache_prompt": True,
                                      **GREEDY})
                    res["greedy"].append(j["timings"]["predicted_per_second"])
        machine = tele.stop()
    finally:
        p.terminate()
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            p.kill()
        log.close()
    mean = lambda xs: round(sum(xs) / len(xs), 2) if xs else ""
    row = {"time": datetime.datetime.now().isoformat(timespec="seconds"), "label": a.label,
           "mode": "long" if a.long else "short", "max_context": ctx, "serve_args": " ".join(a.serve_args),
           "gen_tps_t1": mean(res["t1"]), "gen_tps_t1_runs": " ".join(f"{x:.2f}" for x in res["t1"]),
           "gen_tps_greedy": mean(res["greedy"]), "prefill_tps": mean(res["prefill"]),
           "prefill_tokens": res.get("prefill_n", 8192), "cache_n": str(res.get("cache_n", "")), **machine}
    out = OUT.with_name("llamacpp_results.csv" if a.llamacpp else "strata_upstream_results.csv" if a.attach else OUT.name)
    new = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"[{'llama.cpp' if a.llamacpp else 'NInfer'} {row['mode']}] 温度 1.0 の生成 {row['gen_tps_t1']} tok/s ({row['gen_tps_t1_runs']})"
          + (f"  greedy {row['gen_tps_greedy']} tok/s" if res["greedy"] else "")
          + f"  読み込み {row['prefill_tps']} tok/s ({row['prefill_tokens']} トークン)  GPU 使用率 {machine['gpu_util_pct']}%  "
          f"エンジンの CPU {machine['engine_cpu_cores']} コア分  専用 VRAM {machine['engine_gpu_dedicated_mib']} MiB  "
          f"共有 {machine['engine_gpu_shared_mib']} MiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
