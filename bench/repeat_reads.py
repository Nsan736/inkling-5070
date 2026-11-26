"""中くらいの読み込みを続けて繰り返したときの速さ (開発用)。会話で新しい発言を読むたびの形に近い。

1 つのエンジンで、毎回ちがう N トークンの依頼 (前置きの資料の別の部分) を読み、32 トークン書かせる。
読み込みの時間 (prompt_ms) と、その直後の生成の速さ (17 トークン目から) を回ごとに出す。

使い方: python -I bench/repeat_reads.py [--tokens 600 2000] [--times 5] [--env K=V ...] [--label ...]
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

OUT = B.ROOT / "bench" / "logs" / "repeat_reads.jsonl"
CONFIG = B.ROOT / "local" / "try8078.json"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", nargs="+", type=int, default=[600, 2000])
    ap.add_argument("--times", type=int, default=5)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    tok = C.strata_tokenizer.Tokenizer.from_dir(C.PUB / "data" / "tokenizer")
    from serve.frontend import ChatTemplate   # noqa: E402
    tmpl = ChatTemplate(C.PUB / "data" / "tokenizer" / "chat_template.jinja")
    base = U.ids("long_code_base.ids")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    env = dict(cfg.get("env") or {})
    env.update(dict(x.split("=", 1) for x in a.env))
    os.environ.update(env)
    B.stop_server()
    e = U.Engine([str(B.EXE)] + list(cfg["args"]) + ["--serve"], log="repeat_reads.log")
    rows = []
    try:
        at = 0
        for n in a.tokens:
            for i in range(a.times):
                text = tok.decode(base[at:at + n])
                at += n
                prompt = tok.encode(tmpl.render([{"role": "user", "content": "Summarize this code:\n\n" + text}],
                                                reasoning_effort="none"), parse_special=True)
                st, d = e.gen(48, B.T1 + " lookup=0", prompt)
                rate = U.write_rate(st)[0]
                rows.append({"tokens": n, "i": i, "prompt": d["prompt"], "reused": d["reused"],
                             "read_ms": round(d["prompt_ms"], 1), "write_tps": round(rate, 2) if rate else None})
                print(f"[{a.label} {n}] #{i}: read {d['prompt'] - d['reused']} tokens in {d['prompt_ms']:.0f} ms, "
                      f"then {rows[-1]['write_tps']} tok/s", flush=True)
    finally:
        e.quit()
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": a.label, "env": a.env,
                            "rows": rows}) + "\n")


if __name__ == "__main__":
    main()
