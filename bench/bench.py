"""Inkling 2 Mini の速度を測り、コミット ID 付きで bench/results.csv に 1 行追記する。

  python bench/bench.py                 普段の計測: 生成 (約 200 トークン) と 読み込み (8K トークン)。1 分以内
  python bench/bench.py --long          128K の計測 (指示があったときだけ): 120K まで読んだ状態を復元し、残り 8K の
                                        読み込みと、その後の生成を測る。初回だけ 120K を読んで保存する (数分)
  python bench/bench.py --restart       エンジンを読み込み直してから測る (ビルドし直したときは自動で読み込み直す)
  python bench/bench.py --stop          常駐サーバーを止める
  オプション: --label 文字列 (CSV のメモ欄)、--reps N (各計測の回数、既定 2)

常駐サーバー (bench/server.py, 127.0.0.1:18457) がエンジンを 1 回だけ読み込み、使い回す。
数値はエンジン内部の時刻:
  生成    GEN の times=1 で出る TT 行 (トークンを出した時刻、エンジンの時計) から、最初の 16 トークンを除いた
          200 トークンの tok/s。greedy (temperature=0)、EOS で止めない (ignore_eos=1)
  読み込み DONE 行の prompt_ms と読んだトークン数 (エンジンが計った読み込み時間) から tok/s。毎回先頭のトークンを
          変えて、前回の会話の再利用が起きないようにする (DONE の reused が 0 であることを確かめる)
"""
import argparse
import csv
import datetime
import json
import os
import pathlib
import re
import socket
import statistics
import subprocess
import sys
import time

import psutil

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))   # -I でも同じフォルダーの telemetry を読む
import telemetry  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "bench" / "data"
RESULTS = ROOT / "bench" / "results.csv"
STRATA = ROOT / "strata"
# 開発用の配置 (strata/build-inkling2) か、公開用の配置 (build/bin、tools/build.bat が作る) のどちらか
BUILD = STRATA / "build-inkling2" if (STRATA / "build-inkling2").exists() else ROOT / "build" / "bin"
if os.environ.get("INKLING2_BUILD"):   # 別のビルドを測る (例: 公開用の配布ビルド)
    BUILD = pathlib.Path(os.environ["INKLING2_BUILD"])
EXE = BUILD / "strata-inkling2.exe"
MODEL = ROOT / "models" / "Inkling-2-Mini-UD-IQ4_XS.gguf"
HOST, PORT = "127.0.0.1", 18457

MAX_CONTEXT = 132096        # 128K + 生成ぶん
THREADS = 12                # CPU スレッドの上限 (12)
# lookup (会話内の繰り返しによる推測、結果は変わらない) は窓 8 で既定で有効。素の速度はリクエストの lookup=0 で測る
BASE_FLAGS = ["--max-context", str(MAX_CONTEXT), "--threads", str(THREADS), "--window", "8", "--lookup", "3",
              "--prompt-batch", "2048",   # 2048: 128K の残り 8K の読み込みが 1149 -> 1300 tok/s、生成は約 -7%
              "--vram-reserve-mib", "400",   # VRAM の空き約 400 MB: エキスパート 1078 -> 1166 個、128K の生成 +2.5%
              "--ram-experts-mib", "9216"]   # VRAM の外のエキスパートを全部 RAM に置く (約 9 GB)。既定 (空き RAM - 4 GiB)
                                             # だと他のアプリの RAM 使用で変わって生成が 5% ほど揺れ、5120 では一部が
                                             # ファイルキャッシュ経由になり 128K の生成が 35 -> 39 tok/s と遅かった
ENGINE_FLAGS = list(BASE_FLAGS)   # main() で --kv を足す
ENGINE_LOG = ROOT / "bench" / "logs" / "engine.log"

GEN_SKIP = 16               # 計測から除く最初のトークン数
GEN_COUNT = 200             # 計測するトークン数
REQUEST_TIMEOUT = 120.0     # 1 リクエストの上限 (秒)
LONG_TIMEOUT = 1800.0       # 120K を読む初回の上限 (秒)
RESTORE_TIMEOUT = 900.0     # 数 GB の保存ファイルを復元する上限 (秒)


def session_path(kv, text="docs"):
    return DATA / f"long_120k_{kv}{'' if text == 'docs' else '_' + text}.session"


def load_info():
    """今のエンジンの読み込み時の報告: 空いていた RAM と、VRAM の外のエキスパートに取った RAM (MiB)。"""
    text = ENGINE_LOG.read_text(encoding="utf-8", errors="replace")
    text = text[text.rfind(chr(10) + "==== "):]
    free = re.search(r"([0-9.]+) GiB of RAM free for", text)
    held = re.search(r"(\d+) MiB of RAM for the [0-9.]+ GiB of experts outside VRAM", text)
    return {"ram_free_at_load_mib": round(float(free.group(1)) * 1024) if free else "",
            "ram_for_experts_mib": int(held.group(1)) if held else ""}


def vram_free():
    """エンジンの最後の報告での VRAM の空き (MiB)。"""
    text = ENGINE_LOG.read_text(encoding="utf-8", errors="replace")
    m = re.findall(r"(\d+) MiB VRAM free", text[text.rfind(chr(10) + "==== "):])
    return int(m[-1]) if m else ""


def cached_experts():
    """エンジンのログの最後の報告から、VRAM に載っているエキスパートの数 (と全体の数) を読む。"""
    text = ENGINE_LOG.read_text(encoding="utf-8", errors="replace")
    text = text[text.rfind("\n==== "):]   # 今のエンジンの分だけ
    last = None
    for m in re.finditer(r"(\d+) of (\d+) routed experts in VRAM|expert cache (\d+) experts", text):
        last = (int(m.group(1)), int(m.group(2))) if m.group(1) else (int(m.group(3)), 3072)
    return last or (0, 3072)


def ids(name):
    return [int(x) for x in (DATA / name).read_text().split(",")]


class Conn:
    def __init__(self):
        self.s = socket.create_connection((HOST, PORT), timeout=5)
        self.f = self.s.makefile("rwb", buffering=0)

    def send(self, line):
        self.f.write((line + "\n").encode())

    def read_until(self, prefixes, timeout):
        """prefixes のどれかで始まる行が来るまで読み、それまでの行をすべて返す。"""
        self.s.settimeout(timeout)
        got = []
        deadline = time.time() + timeout
        while True:
            if time.time() > deadline:
                raise TimeoutError(f"{timeout:.0f} 秒で応答が終わらない")
            raw = self.f.readline()
            if not raw:
                raise ConnectionError("サーバーが切断した")
            line = raw.decode().rstrip("\r\n")
            got.append(line)
            if line.startswith(("ERR", "FATAL", "SERR")):
                raise RuntimeError(line)
            if line.startswith(prefixes):
                return got

    def close(self):
        self.f.close()   # makefile() の参照が残るとソケットが閉じず、常駐サーバーが次の接続を受けない
        self.s.close()


def server_up():
    try:
        socket.create_connection((HOST, PORT), timeout=0.5).close()
        return True
    except OSError:
        return False


def server_info():
    c = Conn()
    c.send("@INFO")
    info = json.loads(c.read_until(("{",), 10)[-1])
    c.close()
    return info


def stop_server():
    if not server_up():
        return
    c = Conn()
    c.send("@QUIT")
    try:
        c.read_until(("@BYE",), 60)
    except (ConnectionError, TimeoutError, OSError):
        pass
    c.close()
    for _ in range(60):
        if not server_up():
            return
        time.sleep(0.5)


def start_server():
    args = [str(EXE), "--model", str(MODEL), "--serve"] + ENGINE_FLAGS
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS if sys.platform == "win32" else 0
    subprocess.Popen([sys.executable, "-I", str(ROOT / "bench" / "server.py"), "--"] + args,
                     creationflags=flags, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=open(ROOT / "bench" / "logs" / "server.log", "ab"))
    t0 = time.time()
    while not server_up():
        if time.time() - t0 > 180:
            raise SystemExit("サーバーが 180 秒で起動しない (bench/logs/engine.log を見る)")
        time.sleep(0.5)
    print(f"エンジンを読み込んだ ({time.time() - t0:.1f} 秒)")


def ensure_server(restart):
    (ROOT / "bench" / "logs").mkdir(parents=True, exist_ok=True)
    if server_up():
        info = server_info()
        want = [str(EXE), "--model", str(MODEL), "--serve"] + ENGINE_FLAGS
        stale = info["args"] != want or abs(info["exe_mtime"] - EXE.stat().st_mtime) > 1e-3
        if restart or stale:
            print("エンジンを読み込み直す" + (" (ビルドか設定が変わった)" if stale and not restart else ""))
            stop_server()
        else:
            return info
    start_server()
    return server_info()


SAMPLING = "temperature=0"   # main() が --temp で変える (1.0 なら推奨設定: top_k=0 top_p=1 min_p=0、seed 固定)


T1 = "temperature=1.0 top_k=0 top_p=1 min_p=0 seed=1234"   # the model card's sampling


def gen_line(max_new, prompt, extra="", sampling=None):
    return (f"GEN {max_new} {sampling or SAMPLING} ignore_eos=1 times=1 {extra}"
            f"{','.join(map(str, prompt))}").replace("  ", " ")


def parse_done(line):
    p = line.split()
    return {"produced": int(p[1]), "prompt": int(p[2]), "prompt_ms": float(p[3]), "decode_ms": float(p[4]),
            "reused": int(p[8]), "experts_vram": int(p[9]), "experts": int(p[10]), "read": int(p[14]),
            "drafts_right": int(p[6]), "drafts": int(p[7])}


def run_gen(c, prompt, timeout=REQUEST_TIMEOUT, sampling=None, lookup=False):
    c.send(gen_line(GEN_SKIP + GEN_COUNT, prompt, extra="" if lookup else "lookup=0 ", sampling=sampling))
    got = c.read_until(("DONE",), timeout)
    stamps = [float(l.split()[2]) for l in got if l.startswith("TT ")]
    done = parse_done(got[-1])
    if len(stamps) < GEN_SKIP + GEN_COUNT:
        raise RuntimeError(f"生成が {len(stamps)} トークンで終わった")
    # トークン GEN_SKIP-1 を出した時刻から GEN_SKIP+GEN_COUNT-1 までの GEN_COUNT トークン
    dt = (stamps[GEN_SKIP + GEN_COUNT - 1] - stamps[GEN_SKIP - 1]) / 1e6
    return GEN_COUNT / dt, done


def run_prefill(c, prompt, nonce, timeout=REQUEST_TIMEOUT):
    # 先頭に毎回違う普通のトークンを置き、会話の再利用を防ぐ
    c.send(f"GEN 1 temperature=0 {','.join(map(str, [1000 + nonce % 50000] + prompt))}")
    got = c.read_until(("DONE",), timeout)
    done = parse_done(got[-1])
    if done["reused"] != 0:
        raise RuntimeError(f"読み込みの計測で {done['reused']} トークンが再利用された")
    return 1000.0 * done["read"] / done["prompt_ms"], done


def memory(pid):
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.split(",")
    p = psutil.Process(pid)
    m = p.memory_full_info() if sys.platform == "win32" else p.memory_info()
    return {"vram_used_mib": int(out[0]), "vram_total_mib": int(out[1]),
            "ram_ws_mib": round(m.rss / 2**20), "ram_private_mib": round(getattr(m, "private", m.rss) / 2**20)}


def commit():
    head = subprocess.run(["git", "-C", str(STRATA), "rev-parse", "--short=9", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(STRATA), "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    return head + ("-dirty" if dirty else "")


FIELDS = ["time", "commit", "mode", "kv", "label", "gen_tps", "gen_tps_runs", "prefill_tps", "prefill_tps_runs",
          "prefill_tokens", "context_at", "vram_used_mib", "ram_ws_mib", "ram_private_mib", "experts_vram_pct_gen",
          "experts_cached", "engine_flags", "decode", "gpu_clk_mhz", "gpu_clk_min_mhz", "gpu_mem_clk_mhz",
          "gpu_temp_max_c", "gpu_power_w", "gpu_util_pct", "gpu_slowdown", "cpu_mhz", "cpu_temp_c", "other_cpu_pct",
          "other_top", "engine_gpu_dedicated_mib", "engine_gpu_shared_mib", "ram_free_at_load_mib",
          "ram_for_experts_mib", "vram_free_mib", "temperature", "drafts_accepted", "drafts_offered", "gen_tps_t1",
          "gen_tps_t1_runs", "drafts_accepted_t1", "drafts_offered_t1", "prefill_full_tps", "gen_lookup_tps",
          "gen_lookup_tps_t1", "lookup_accepted", "lookup_offered", "bench_text", "engine_cpu_pct", "engine_cpu_cores"]


def append(row):
    new = not RESULTS.exists()
    if not new:
        with open(RESULTS, newline="", encoding="utf-8") as f:
            old = list(csv.DictReader(f))
        if old and list(old[0].keys()) != FIELDS:   # 列が増えた: 古い行を新しい列にそろえて書き直す
            with open(RESULTS, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                w.writeheader()
                for r in old:
                    r.setdefault("kv", "fp16")
                    if not r.get("decode"):
                        r["decode"] = "plain"
                    w.writerow({k: r.get(k, "") for k in FIELDS})
    with open(RESULTS, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", action="store_true")
    ap.add_argument("--restart", action="store_true")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--kv", choices=["fp16", "q8_0"], default="q8_0")
    ap.add_argument("--extra", default="", help='エンジンに足す設定 (例: "--vram-pool")。CSV の engine_flags に残る')
    ap.add_argument("--env", action="append", default=[], help="エンジンに渡す環境変数 K=V (読み込み直す。CSV に残る)")
    ap.add_argument("--lookup", action="store_true",
                    help="会話内の繰り返しによる推測 (結果は変わらない) をエンジンの既定 (--window 8 --lookup 3) で使う。"
                         "CSV の decode 欄が lookup になり、素の速度 (plain) とは別に記録される")
    ap.add_argument("--mtp", action="store_true",
                    help="DSpark ドラフター (INKLING2_DSPARK, models/official/mtp.safetensors) で推測デコード。--window 8。"
                         "CSV の decode 欄が mtp (lookup と併用なら mtp+lookup)。128K は保存した状態を使わず全部読む")
    ap.add_argument("--temp", type=float, default=0.0, help="生成の温度 (0: greedy)")
    ap.add_argument("--text", choices=["docs", "code"], default="docs",
                    help="128K の文脈: docs (llama.cpp の文書) か code (このフォークの engine/src とコードを書く指示)")
    ap.add_argument("--no-full", action="store_true", help="--long で 128K の全読み (prefill_full) を省く")
    ap.add_argument("--quick", action="store_true",
                    help="生成の比較だけ: 素・温度 1.0 を --reps 回 (greedy / lookup / 読み込みの計測 / 128K の全読みなし。"
                         "--long は保存した状態を 1 回復元して後ろの 8K を読み、続けて生成する)")
    ap.add_argument("--window", type=int, default=0, help="推測の窓 (下書き長 + 1)。--mtp / --lookup と使う")
    a = ap.parse_args()
    global SAMPLING
    if a.temp > 0:
        SAMPLING = f"temperature={a.temp} top_k=0 top_p=1 min_p=0 seed=1234"
    if a.mtp:
        a.env.append("INKLING2_DSPARK=" + str((ROOT / "models" / "official" / "mtp.safetensors").resolve()))
        ENGINE_FLAGS[ENGINE_FLAGS.index("--window") + 1] = "8"
    if a.lookup:
        ENGINE_FLAGS[ENGINE_FLAGS.index("--window") + 1] = "8"
        ENGINE_FLAGS[ENGINE_FLAGS.index("--lookup") + 1] = "3"
    if a.window > 0:
        ENGINE_FLAGS[ENGINE_FLAGS.index("--window") + 1] = str(a.window)
    for kv in a.env:
        k, _, v = kv.partition("=")
        os.environ[k] = v
    if a.env:
        a.restart = True   # 環境変数はエンジンの起動時にしか効かない
    if a.kv != "fp16":
        ENGINE_FLAGS.extend(["--kv", a.kv])
    ENGINE_FLAGS.extend(a.extra.split())
    session_120k = session_path(a.kv, a.text)
    if a.stop:
        stop_server()
        print("止めた")
        return 0

    t_start = time.time()
    info = ensure_server(a.restart)
    c = Conn()
    gen_prompt = ids("gen.ids")
    nonce = int(time.time())
    if not info["warmed"]:
        # 読み込み直後の 1 回目は遅い (エキスパートのキャッシュが温まっていない) ので数えない
        run_gen(c, gen_prompt)
        run_prefill(c, ids("prefill_8k.ids"), nonce)
        nonce += 1
        c.send("@WARMED")
        c.read_until(("@OK",), 10)
        print(f"ウォームアップ済み ({time.time() - t_start:.1f} 秒)")

    gens, pres, done_gen, drafts = [], [], None, []
    gens_t1, drafts_t1, full = [], [], []
    lk, lk_t1 = [], []   # lookup: (tok/s, accepted, offered)
    tele = telemetry.Telemetry(info["pid"]).start()
    if a.quick:
        if a.long:
            pre = "long_code" if a.text == "code" else "long"
            base, tail = ids(f"{pre}_base.ids"), ids(f"{pre}_tail.ids")
            if not session_120k.exists():
                raise RuntimeError(f"{session_120k} がない: --quick なしで 1 回作る")
            mode, context_at, n_pre, prompt = "long", len(base), len(tail), base + tail
            t0 = time.time()
            c.send(f"RESTORE {session_120k}")
            print("  " + c.read_until(("RESTORED",), RESTORE_TIMEOUT)[-1] + f"  ({time.time() - t0:.1f} 秒)")
            c.send(f"GEN 1 temperature=0 {','.join(map(str, prompt))}")
            done = parse_done(c.read_until(("DONE",), REQUEST_TIMEOUT)[-1])
            if done["reused"] < len(base) or done["read"] != len(tail):
                raise RuntimeError(f"復元が効いていない: reused={done['reused']} read={done['read']}")
            pres.append(1000.0 * done["read"] / done["prompt_ms"])
        else:
            mode, context_at, n_pre, prompt = "short", 0, 0, gen_prompt
        for _ in range(a.reps):
            tps1, d1 = run_gen(c, prompt, sampling=T1)
            gens_t1.append(tps1)
            drafts_t1.append((d1["drafts_right"], d1["drafts"]))
            done_gen = d1
    elif not a.long:
        mode, context_at, n_pre = "short", 0, 8192
        for _ in range(a.reps):
            tps, done_gen = run_gen(c, gen_prompt)
            gens.append(tps)
            drafts.append((done_gen["drafts_right"], done_gen["drafts"]))
            tps1, d1 = run_gen(c, gen_prompt, sampling=T1)
            gens_t1.append(tps1)
            drafts_t1.append((d1["drafts_right"], d1["drafts"]))
            if not a.mtp:
                for smp, out in ((None, lk), (T1, lk_t1)):
                    t_, d_ = run_gen(c, gen_prompt, sampling=smp, lookup=True)
                    out.append((t_, d_["drafts_right"], d_["drafts"]))
            tps, _ = run_prefill(c, ids("prefill_8k.ids"), nonce)
            nonce += 1
            pres.append(tps)
    else:
        pre = "long_code" if a.text == "code" else "long"
        mode, base, tail = "long", ids(f"{pre}_base.ids"), ids(f"{pre}_tail.ids")
        context_at, n_pre = len(base), len(tail)
        if a.mtp:   # the drafter's context needs the taps of every position: the whole 128K is read
            for _ in range(a.reps):
                c.send(f"GEN 1 temperature=0 {','.join(map(str, [1000 + nonce % 50000] + base[1:] + tail))}")
                nonce += 1
                done = parse_done(c.read_until(("DONE",), LONG_TIMEOUT)[-1])
                pres.append(1000.0 * done["read"] / done["prompt_ms"])
                full.append(pres[-1])
                tps, done_gen = run_gen(c, [1000 + (nonce - 1) % 50000] + base[1:] + tail)
                gens.append(tps)
                drafts.append((done_gen["drafts_right"], done_gen["drafts"]))
                tps1, d1 = run_gen(c, [1000 + (nonce - 1) % 50000] + base[1:] + tail, sampling=T1)
                gens_t1.append(tps1)
                drafts_t1.append((d1["drafts_right"], d1["drafts"]))
            n_pre = len(base) + len(tail)
        elif not session_120k.exists():
            print(f"{len(base)} トークンを読んで保存する (初回だけ)")
            t0 = time.time()
            c.send(f"GEN 1 temperature=0 {','.join(map(str, base))}")
            d = parse_done(c.read_until(("DONE",), LONG_TIMEOUT)[-1])
            print(f"  読み込み {d['read']} トークン {d['prompt_ms'] / 1000:.1f} 秒 "
                  f"({1000.0 * d['read'] / d['prompt_ms']:.1f} tok/s)")
            c.send(f"SAVE {session_120k}")
            print("  " + c.read_until(("SAVED",), LONG_TIMEOUT)[-1] +
                  f"  ファイル {session_120k.stat().st_size / 2**30:.2f} GiB  計 {time.time() - t0:.0f} 秒")
        for _ in range(0 if a.mtp else a.reps):
            t0 = time.time()
            c.send(f"RESTORE {session_120k}")
            print("  " + c.read_until(("RESTORED",), RESTORE_TIMEOUT)[-1] + f"  ({time.time() - t0:.1f} 秒)")
            c.send(f"GEN 1 temperature=0 {','.join(map(str, base + tail))}")
            done = parse_done(c.read_until(("DONE",), REQUEST_TIMEOUT)[-1])
            if done["reused"] < len(base) or done["read"] != len(tail):
                raise RuntimeError(f"復元が効いていない: reused={done['reused']} read={done['read']}")
            pres.append(1000.0 * done["read"] / done["prompt_ms"])
            tps, done_gen = run_gen(c, base + tail)
            gens.append(tps)
            drafts.append((done_gen["drafts_right"], done_gen["drafts"]))
            tps1, d1 = run_gen(c, base + tail, sampling=T1)
            gens_t1.append(tps1)
            drafts_t1.append((d1["drafts_right"], d1["drafts"]))
            for smp, out in ((None, lk), (T1, lk_t1)):
                t_, d_ = run_gen(c, base + tail, sampling=smp, lookup=True)
                out.append((t_, d_["drafts_right"], d_["drafts"]))
        if not a.mtp and not a.no_full:   # the whole 128K from the start, once (no reuse: its first token is new)
            c.send(f"GEN 1 temperature=0 {','.join(map(str, [1000 + nonce % 50000] + base[1:] + tail))}")
            nonce += 1
            d = parse_done(c.read_until(("DONE",), LONG_TIMEOUT)[-1])
            full.append(1000.0 * d["read"] / d["prompt_ms"])
    machine = tele.stop()
    mem = memory(info["pid"])
    c.close()
    n_cached, n_all = cached_experts()

    row = {
        "time": datetime.datetime.now().isoformat(timespec="seconds"), "commit": commit(), "mode": mode,
        "label": a.label + (" quick" if a.quick else ""), "gen_tps": round(statistics.mean(gens), 2) if gens else "",
        "gen_tps_runs": " ".join(f"{x:.2f}" for x in gens), "prefill_tps": round(statistics.mean(pres), 1) if pres else "",
        "prefill_tps_runs": " ".join(f"{x:.1f}" for x in pres), "prefill_tokens": n_pre, "context_at": context_at,
        "vram_used_mib": mem["vram_used_mib"], "ram_ws_mib": mem["ram_ws_mib"],
        "ram_private_mib": mem["ram_private_mib"],
        "experts_vram_pct_gen": round(100.0 * done_gen["experts_vram"] / max(1, done_gen["experts"]), 1),
        "experts_cached": n_cached, "kv": a.kv, "engine_flags": " ".join(ENGINE_FLAGS + [f"env:{e}" for e in a.env]),
        "decode": ("mtp+lookup" if a.lookup else "mtp") if a.mtp else "lookup" if a.lookup else "plain",
        "temperature": a.temp,
        "gen_tps_t1": round(statistics.mean(gens_t1), 2) if gens_t1 else "",
        "gen_tps_t1_runs": " ".join(f"{x:.2f}" for x in gens_t1),
        "drafts_accepted_t1": sum(r for r, _ in drafts_t1), "drafts_offered_t1": sum(d for _, d in drafts_t1),
        "prefill_full_tps": round(statistics.mean(full), 1) if full else "",
        "bench_text": a.text if a.long else "chat",
        "gen_lookup_tps": round(statistics.mean(x[0] for x in lk), 2) if lk else "",
        "gen_lookup_tps_t1": round(statistics.mean(x[0] for x in lk_t1), 2) if lk_t1 else "",
        "lookup_accepted": sum(x[1] for x in lk + lk_t1), "lookup_offered": sum(x[2] for x in lk + lk_t1),
        "drafts_accepted": sum(r for r, _ in drafts), "drafts_offered": sum(d for _, d in drafts),
    }
    row.update(machine)
    row.update(load_info())
    row["vram_free_mib"] = vram_free()
    append(row)
    print(f"[{mode} kv={a.kv} {row['decode']}] 生成 {row['gen_tps']} tok/s ({row['gen_tps_runs']})  読み込み {row['prefill_tps']} "
          f"tok/s ({row['prefill_tps_runs']}, {n_pre} トークン, {context_at} トークン目から)  "
          f"VRAM {mem['vram_used_mib']}/{mem['vram_total_mib']} MiB  RAM {mem['ram_ws_mib']} MiB "
          f"(private {mem['ram_private_mib']})  VRAM のエキスパート {n_cached}/{n_all}  生成時の命中率 "
          f"{row['experts_vram_pct_gen']}%  [{row['commit']}]  {time.time() - t_start:.1f} 秒")
    print(f"  GPU {row['gpu_clk_mhz']} MHz (最低 {row['gpu_clk_min_mhz']}, メモリ {row['gpu_mem_clk_mhz']})  "
          f"{row['gpu_temp_max_c']}℃  {row['gpu_power_w']} W  使用率 {row['gpu_util_pct']}%  低下理由 "
          f"{row['gpu_slowdown']}  CPU {row['cpu_mhz']} MHz  他の処理 {row['other_cpu_pct']}% ({row['other_top']})  "
          f"エンジンの CPU {row['engine_cpu_cores']} コア分 ({row['engine_cpu_pct']}%)  "
          f"共有GPUメモリ {row['engine_gpu_shared_mib']} MiB  VRAM の空き {row['vram_free_mib']} MiB  "
          f"読み込み時の空きRAM {row['ram_free_at_load_mib']} MiB (エキスパート用 {row['ram_for_experts_mib']})")
    print(f"  温度 1.0 の生成 {row['gen_tps_t1']} tok/s ({row['gen_tps_t1_runs']})" +
          (f"  128K を最初から読む {row['prefill_full_tps']} tok/s" if full else ""))
    if lk:
        print(f"  lookup: 生成 {row['gen_lookup_tps']} tok/s、温度 1.0 {row['gen_lookup_tps_t1']} tok/s "
              f"(受け入れ {row['lookup_accepted']} / {row['lookup_offered']})")
    if row["drafts_offered_t1"]:
        print(f"  推測 (温度 1.0): {row['drafts_accepted_t1']} / {row['drafts_offered_t1']} を受け入れ")
    if row["drafts_offered"]:
        print(f"  推測: {row['drafts_accepted']} / {row['drafts_offered']} を受け入れ "
              f"({100.0 * row['drafts_accepted'] / row['drafts_offered']:.0f}%)  温度 {a.temp}")
    print(f"記録: {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
