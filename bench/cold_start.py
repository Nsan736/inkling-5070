"""起動直後 (エキスパートの使用回数が 0、ウォームアップなし) の最初の生成の速さ。エンジンを bench.py と同じ引数で
起動し、gen.ids に続けて N トークン (温度 1.0) を生成させ、区間ごとの tok/s を出す。

  python -I bench/cold_start.py [--tokens 400] [--reps 2] [--env K=V ...] [--label 名前]

結果は bench/logs/cold_start.jsonl に 1 行ずつ。エンジンの stderr は bench/logs/cold_engine.log。
"""
import argparse
import json
import os
import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bench as B  # noqa: E402
import upstream_bench as U  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    for kv in a.env:
        k, _, v = kv.partition("=")
        os.environ[k] = v
    args = [str(B.EXE), "--model", str(B.MODEL), "--serve"] + B.BASE_FLAGS + ["--kv", "q8_0"]
    edges = [0, 100, 200, a.tokens]
    runs = []
    for _ in range(a.reps):
        e = U.Engine(args, log="cold_engine.log")
        st, d = e.gen(a.tokens, "temperature=1.0 top_k=0 top_p=1 min_p=0 seed=1234 ignore_eos=1 lookup=0",
                      U.ids("gen.ids"))
        e.quit()
        part = {}
        for lo, hi in zip(edges, edges[1:]):
            if len(st) >= hi:
                part[f"{lo + 1}-{hi}"] = round((hi - max(lo, 1)) / (st[hi - 1] - st[max(lo, 1) - 1]), 2)
        runs.append({"parts": part, "tokens": len(st), "load_s": round(e.load_s, 1)})
        print(f"[{a.label}] " + "  ".join(f"{k}: {v} tok/s" for k, v in part.items()) + f"  (load {e.load_s:.0f} s)",
              flush=True)
        time.sleep(2)
    with open(B.ROOT / "bench" / "logs" / "cold_start.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "label": a.label, "env": a.env, "runs": runs,
                            "commit": B.commit()}, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
