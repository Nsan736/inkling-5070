"""中くらいの読み込み (600 / 2,000 トークン) を続けて繰り返したときの読み込み時間と、その後の生成の速さ。
エンジンを bench.py と同じ引数で起動し (ウォームアップ 1 回)、600 と 2,000 トークンを交互に各 3 回 (先頭を毎回変えて
再利用させない)、最後に gen.ids で 120 トークンを生成する。

  python -I bench/medium_reads.py [--env K=V ...] [--label 名前]
"""
import argparse
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bench as B  # noqa: E402
import upstream_bench as U  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    for kv in a.env:
        k, _, v = kv.partition("=")
        os.environ[k] = v
    ids = U.ids("prefill_8k.ids")
    e = U.Engine([str(B.EXE), "--model", str(B.MODEL), "--serve"] + B.BASE_FLAGS + ["--kv", "q8_0"], log="cold_engine.log")
    e.gen(20, "temperature=0", U.ids("gen.ids"))
    reads = {600: [], 2000: []}
    for k in range(3):
        for n in (600, 2000):
            _, d = e.gen(1, "temperature=0", [1000 + k * 7 + n] + ids[:n])
            reads[n].append(round(d["prompt_ms"]))
    st, _ = e.gen(120, "temperature=1.0 top_k=0 top_p=1 min_p=0 seed=1234 ignore_eos=1 lookup=0", U.ids("gen.ids"))
    e.quit()
    gen = round(119 / (st[-1] - st[0]), 1)
    print(f"[{a.label}] 600: {reads[600]} ms  2000: {reads[2000]} ms  生成 {gen} tok/s", flush=True)
    with open(B.ROOT / "bench" / "logs" / "medium_reads.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "label": a.label, "env": a.env,
                            "reads_ms": reads, "gen": gen}, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
