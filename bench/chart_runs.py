"""README のグラフ用の計測 (開発用)。版ごと・文脈ごとにエンジンを 1 回起動し、文脈を 1 回読んで (読み込みの速さ)、
数えない答え 1 回、lookup ありの答え、lookup なしの答えを出させる (温度 1.0、bench.py の T1)。

  v034  今の既定 (INKLING2_GPU_TOKEN / SLOT_ORDER / FAST_PICK がオン、窓 32)
  v032  0.3.2 と同じデコードの計算: その 3 つを =0、窓 8
  v010  v0.1.0 のビルド (INKLING2_BUILD で指す)、当時の setup の設定

依頼は dspark_probe.py の code (前置きの公開ソースコード + 新しいコードを書く依頼)。結果は bench/logs/chart_runs.jsonl。

使い方: python -I bench/chart_runs.py [--ctx 8k 64k 100k 128k] [--versions v032 v034]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench as B            # noqa: E402
import upstream_bench as U   # noqa: E402
import copy_window as C      # noqa: E402
import dspark_probe as D     # noqa: E402

OUT = B.ROOT / "bench" / "logs" / "chart_runs.jsonl"
VERSIONS = {
    "v034": ({}, "32"),
    "v032": ({"INKLING2_GPU_TOKEN": "0", "INKLING2_SLOT_ORDER": "0", "INKLING2_FAST_PICK": "0"}, "8"),
    # v0.1.0's own build (INKLING2_BUILD), with what its setup wrote for 12 GB / 32 GB: the window 8 and a max context
    # of 132,096 whatever the prompt (the others: the prompt's length plus 1,024)
    "v010": ({}, "8"),
}
FIXED_CONTEXT = {"v010": 132096}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", nargs="+", default=["8k", "64k", "100k", "128k"])
    ap.add_argument("--versions", nargs="+", default=["v032", "v034"])
    a = ap.parse_args()
    tok = C.strata_tokenizer.Tokenizer.from_dir(C.PUB / "data" / "tokenizer")
    from serve.frontend import ChatTemplate   # noqa: E402
    tmpl = ChatTemplate(C.PUB / "data" / "tokenizer" / "chat_template.jinja")
    B.stop_server()
    for ctx in a.ctx:
        prompt = D.prompt_ids(tok, tmpl, "code", ctx)
        for v in a.versions:
            env, window = VERSIONS[v]
            for k, val in env.items():
                os.environ[k] = val
            flags = list(B.BASE_FLAGS)
            flags[flags.index("--max-context") + 1] = str(FIXED_CONTEXT.get(v, D.CTX[ctx] + 1024))
            flags[flags.index("--window") + 1] = window
            e = U.Engine([str(B.EXE), "--model", str(B.MODEL), "--serve"] + flags + ["--kv", "q8_0"], log="chart_runs.log")
            try:
                _, d = e.gen(1, "temperature=0", prompt)   # the prompt read from the start
                read = d["prompt"] / (d["prompt_ms"] / 1000.0)
                e.gen(U.SKIP + U.COUNT, B.T1, prompt)        # warm-up answer (not counted)
                on, _ = e.gen(U.SKIP + U.COUNT, B.T1, prompt)
                off, _ = e.gen(U.SKIP + U.COUNT, B.T1 + " lookup=0", prompt)
                row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "version": v, "ctx": ctx, "prompt": d["prompt"],
                       "read_tps": round(read, 1), "lookup_tps": round(U.write_rate(on)[0], 2),
                       "no_lookup_tps": round(U.write_rate(off)[0], 2)}
                print(json.dumps(row), flush=True)
                with open(OUT, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
            finally:
                e.quit()
                for k in env:
                    os.environ.pop(k, None)


if __name__ == "__main__":
    main()
