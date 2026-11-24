"""KV の形式ごとの長い文脈の速さ (開発用、--kv k8v4 の比較)。

1 つの形式と文脈の長さにつき、エンジンを起動して次を 1 回ずつ測る。
  1. 最初から全部読む (保存した状態は使わない)
     (前半を読んでから全体を送り、後ろ 8K だけを読む: その 8K の速さが「後ろ 8K の読み込み」)
  2. 続けて生成する (温度 1.0、17〜216 トークン目の速さ)
エキスパートの数とヒット率は、生成のあとにエンジンが出す 1 行から読む。
結果は bench/logs/kv4_bench.jsonl に 1 行ずつ足す。

使い方: python -I bench/kv4_bench.py --kv q8_0 k8v4 --ctx 128k 256k [--env K=V ...]
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench as B           # noqa: E402
import upstream_bench as U  # noqa: E402

CTX = {"128k": (132096, "long_code"), "256k": (263168, "long_code256")}
LOG = "kv4_engine.log"
OUT = B.ROOT / "bench" / "logs" / "kv4_bench.jsonl"


def vram_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.split()[0])


def last_cache_line(since):
    text = (B.ROOT / "bench" / "logs" / LOG).read_text(encoding="utf-8", errors="replace")[since:]
    m = None
    for m in re.finditer(r"expert cache (\d+) experts \((\d+) when full.*?at (\d+) tokens.*?hit rate ([\d.]+)%", text):
        pass
    kv = re.findall(r"(k8v4 KV cache.*|q8_0 KV cache.*)", text)
    return (m.groups() if m else None), (kv[-1] if kv else "")


def run(kv, ctx, env):
    max_ctx, pre = CTX[ctx]
    base, tail = U.ids(f"{pre}_base.ids"), U.ids(f"{pre}_tail.ids")
    flags = list(B.BASE_FLAGS)
    flags[flags.index("--max-context") + 1] = str(max_ctx)
    log_path = B.ROOT / "bench" / "logs" / LOG
    since = log_path.stat().st_size if log_path.exists() else 0
    env = dict(env, INKLING2_REPORT_CACHE="1")   # the engine's line a request: the expert cache and its hit rate
    for k, v in env.items():
        os.environ[k] = v
    e = U.Engine([str(B.EXE), "--model", str(B.MODEL), "--serve"] + flags + ["--kv", kv], log=LOG)
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "kv": kv, "ctx": ctx, "env": env, "load_s": round(e.load_s, 1)}
    try:
        U_gen = U.ids("gen.ids")
        e.gen(32, U.T1, U_gen)   # the expert cache's first use (not counted)
        first = [1234] + base[1:]
        _, d1 = e.gen(1, "temperature=0", first)   # the first part from the start
        _, d2 = e.gen(1, "temperature=0", first + tail)   # the rest: the last 8K, the first part reused
        read1, read2 = d1["prompt"] - d1["reused"], d2["prompt"] - d2["reused"]
        row["full_tokens"] = read1 + read2
        row["full_tps"] = round(1000.0 * (read1 + read2) / (d1["prompt_ms"] + d2["prompt_ms"]), 1)
        row["tail_tokens"], row["tail_reused"] = read2, d2["reused"]
        row["tail_tps"] = round(1000.0 * read2 / d2["prompt_ms"], 1)
        print(f"[{kv} {ctx}] 全読み {row['full_tokens']} トークン {(d1['prompt_ms'] + d2['prompt_ms']) / 1000:.0f} 秒 "
              f"({row['full_tps']} tok/s)、うち後ろ {read2} トークン ({d2['reused']} 使い回し) {row['tail_tps']} tok/s",
              flush=True)
        stamps, d = e.gen(U.SKIP + U.COUNT, B.T1, first + tail)
        row["gen_tps"] = round(U.write_rate(stamps)[0], 2)
        row["gen_reused"] = d["reused"]
        row["vram_used_mib"] = vram_used_mib()
    finally:
        e.quit()
        for k in env:
            os.environ.pop(k, None)
    cache, kvline = last_cache_line(since)
    if cache:
        row["experts_vram"], row["experts_full"], row["at"], row["hit_rate"] = int(cache[0]), int(cache[1]),             int(cache[2]), float(cache[3])
    row["kv_line"] = kvline
    print(f"[{kv} {ctx}] 生成 {row.get('gen_tps')} tok/s  エキスパート {row.get('experts_vram')} (位置 {row.get('at')}) "
          f"命中 {row.get('hit_rate')}%  VRAM {row.get('vram_used_mib')} MiB  {kvline}", flush=True)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv", nargs="+", default=["q8_0", "k8v4"])
    ap.add_argument("--ctx", nargs="+", default=["128k", "256k"])
    ap.add_argument("--env", action="append", default=[])
    a = ap.parse_args()
    env = dict(x.split("=", 1) for x in a.env)
    B.stop_server()   # the bench server's engine, if one is up, holds the card
    for ctx in a.ctx:
        for kv in a.kv:
            run(kv, ctx, env)


if __name__ == "__main__":
    main()
