"""会話の続きを読む速さ (開発用)。エージェントの会話のように、前の会話の後ろに新しい発言を足して読む場面。

組 (前の会話のトークン数:足すトークン数) ごとに、普段使いの設定 (local/try8078.json) の引数と環境変数で開発版の
エンジンを起動し直す。gen.ids を 1 回 (数えない)、前の会話を読み (数えない)、その後ろに足したものを読む。
足した分の読み込みの速さ (使い回した分を除く) を出す。--times で INKLING2_PREFILL_TIMES=1 の内訳の行も出す
(段ごとに待つので遅くなる)。結果は bench/logs/turn_reads.jsonl。
使い方: python -I bench/turn_reads.py [--pairs 60000:4000 30000:4000 ...] [--env K=V ...] [--times] [--label ...]
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

OUT = B.ROOT / "bench" / "logs" / "turn_reads.jsonl"
LOG = "turn_reads.log"
CONFIG = B.ROOT / "local" / "try8078.json"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=["60000:4000"])
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--times", action="store_true")
    ap.add_argument("--answer", type=int, default=0, help="前の会話のあとに書かせるトークン数 (温度 1.0、既定 0 = 1 トークン)")
    ap.add_argument("--arg", action="append", default=[], help="起動の引数の上書き (例: stage-ahead=8192)")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    tok = C.strata_tokenizer.Tokenizer.from_dir(C.PUB / "data" / "tokenizer")
    from serve.frontend import ChatTemplate   # noqa: E402
    tmpl = ChatTemplate(C.PUB / "data" / "tokenizer" / "chat_template.jinja")
    long = D.prompt_ids(tok, tmpl, "code", "128k")
    base = U.ids("long_code_base.ids") + U.ids("long_code_tail.ids")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for x in a.arg:
        k, v = x.split("=", 1)
        if "--" + k in cfg["args"]:
            cfg["args"][cfg["args"].index("--" + k) + 1] = v
        else:
            cfg["args"] += ["--" + k, v]
    env = dict(cfg.get("env") or {})
    env.update(dict(x.split("=", 1) for x in a.env))
    if a.times:
        env["INKLING2_PREFILL_TIMES"] = "1"
        env["INKLING2_REPORT_CACHE"] = "1"
    for k in [k for k in os.environ if k.startswith(("INKLING2_", "STRATA_"))]:
        os.environ.pop(k)
    os.environ.update(env)
    B.stop_server()
    log = B.ROOT / "bench" / "logs" / LOG
    for pair in a.pairs:
        prev, add = map(int, pair.split(":"))
        first = long[:prev]
        at = prev + 5000 if prev + 5000 + add <= len(base) else 1000
        second = first + base[at:at + add] + first[-6:]
        start = log.stat().st_size if log.exists() else 0
        e = U.Engine([str(B.EXE)] + list(cfg["args"]) + ["--serve"], log=LOG)
        try:
            e.gen(1, "temperature=0", U.ids("gen.ids"))
            e.gen(1, "temperature=0", first)
            if a.answer > 0:   # an answer as long as a turn's, before the next message (its tokens are not in second)
                e.gen(a.answer, "temperature=1.0 seed=1", first)
            t0 = time.time()
            _, d = e.gen(1, "temperature=0", second)
            n = d["prompt"] - d["reused"]
            r = n / d["prompt_ms"] * 1000
        finally:
            e.quit()
        row = {"label": a.label, "prev": prev, "add": add, "read": n, "reused": d["reused"], "tok_s": round(r, 1),
               "prompt_ms": round(d["prompt_ms"], 1), "env": a.env, "answer": a.answer}
        print(f"[{a.label} {prev}:{add}] {n} read after {d['reused']} reused: {r:.0f} tok/s "
              f"({d['prompt_ms']:.0f} ms)", flush=True)
        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if a.times:
            text = log.read_text(encoding="utf-8", errors="replace")[start:]
            lines = [l for l in text.splitlines() if "prompt phases" in l]
            if lines:
                print("   ", lines[-1].split("strata-")[-1][:600], flush=True)


if __name__ == "__main__":
    main()
