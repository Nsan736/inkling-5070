"""文脈の長さごとの「最初から読む」読み込みの速さ (開発用)。

長さごとにエンジンを起動し直し (--max-context は長さ + 2048)、gen.ids で 32 トークン書いてから (数えない)、
long_code の先頭から N トークンを 1 回読む。--gen を付けると、続けて温度 1.0 で書く速さ (17〜216 トークン目) も測る (数えない生成を 1 回してから)。
結果は bench/logs/full_reads.jsonl に 1 行ずつ足す。

使い方: python -I bench/full_reads.py [--len 4k 32k 64k 128k] [--gen 128k] [--env K=V ...] [--label 名前]
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench as B           # noqa: E402
import upstream_bench as U  # noqa: E402

LEN = {"4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768, "64k": 65536, "128k": 131072, "256k": 253000}
LOG = "full_reads_engine.log"
OUT = B.ROOT / "bench" / "logs" / "full_reads.jsonl"


def run(name, gen, env, label, extra, reps):
    n = LEN[name]
    src = "long_code256" if n > 131072 else "long_code"
    text = U.ids(f"{src}_base.ids") + U.ids(f"{src}_tail.ids")
    flags = list(B.BASE_FLAGS)
    flags[flags.index("--max-context") + 1] = str(n + 2048)
    for k, v in env.items():
        os.environ[k] = v
    e = U.Engine([str(B.EXE), "--model", str(B.MODEL), "--serve"] + flags + ["--kv", "q8_0"] + extra, log=LOG)
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "len": name, "label": label, "env": env, "extra": extra,
           "load_s": round(e.load_s, 1)}
    try:
        e.gen(32, U.T1, U.ids("gen.ids"))   # the expert cache's first use (not counted)
        log_path = B.ROOT / "bench" / "logs" / LOG
        for r in range(reps):
            since = log_path.stat().st_size
            prompt = [1234 + r] + text[1:n]   # another first token: nothing reused
            _, d = e.gen(1, "temperature=0", prompt)
            row.setdefault("read_tps_all", []).append(round(1000.0 * (d["prompt"] - d["reused"]) / d["prompt_ms"], 1))
            time.sleep(0.5)
            phases = re.findall(r"prompt phases \((.*)", log_path.read_text(encoding="utf-8", errors="replace")[since:])
            if phases:
                row.setdefault("phases", []).append(phases[-1][:400])
        row["tokens"] = d["prompt"] - d["reused"]
        row["read_s"] = round(d["prompt_ms"] / 1000, 2)
        row["read_tps"] = row["read_tps_all"][-1]
        if gen:
            e.gen(U.SKIP + U.COUNT, B.T1, prompt)   # one answer first: the expert cache settles (not counted)
            stamps, _ = e.gen(U.SKIP + U.COUNT, B.T1, prompt)
            row["gen_tps"] = round(U.write_rate(stamps)[0], 2)
    finally:
        e.quit()
        for k in env:
            os.environ.pop(k, None)
    print(f"[{label} {name}] 読み込み {row['tokens']} トークン {row['read_s']} 秒 ({row['read_tps_all']} tok/s)"
          + (f"  生成 {row['gen_tps']} tok/s" if gen else ""), flush=True)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--len", nargs="+", default=["4k", "32k", "64k", "128k"])
    ap.add_argument("--gen", nargs="*", default=[], help="生成も測る長さ")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--extra", default="", help="エンジンに足す設定")
    ap.add_argument("--label", default="")
    ap.add_argument("--reps", type=int, default=1, help="同じ長さを読む回数 (先頭を変えて使い回させない)")
    a = ap.parse_args()
    env = dict(x.split("=", 1) for x in a.env)
    B.stop_server()
    for name in a.len:
        run(name, name in a.gen, env, a.label, a.extra.split(), a.reps)


if __name__ == "__main__":
    main()
