"""2 つの設定 (環境変数) の、読み込みの速さとその後の生成の速さを交互に比べる (開発用)。

普段使いの設定 (local/try8078.json) の引数と環境変数で、開発版のエンジンを起動する。測るものごとにエンジンを
起動し直す (前の読み込みを使い回させない)。測るもの:
  8k / 32k / 128k  コードの依頼を最初から読む (dspark_probe.py の code)
  turn             60K の会話を読み (数えない)、その後ろに約 4,000 トークンを足して読む (会話の続き)
  gen8k / gen100k  最初から読み、続けて温度 1.0 で書く速さ (bench.py の T1、17〜216 トークン目)
設定の順番は回ごとに入れ替える (A B、B A、...)。結果は bench/logs/read_ab.jsonl。
使い方: python -I bench/read_ab.py --b K=V [--a K=V ...] [--items 8k 32k 128k turn gen8k gen100k] [--rounds 2]
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

OUT = B.ROOT / "bench" / "logs" / "read_ab.jsonl"
CONFIG = B.ROOT / "local" / "try8078.json"
CTX = {"8k": "8k", "32k": "32k", "128k": "128k", "gen8k": "8k", "gen100k": "100k"}


def engine(env):
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    full = dict(cfg.get("env") or {})
    full.update(dict(x.split("=", 1) for x in env))
    for k in [k for k in os.environ if k.startswith(("INKLING2_", "STRATA_"))]:
        os.environ.pop(k)
    os.environ.update(full)
    return U.Engine([str(B.EXE)] + list(cfg["args"]) + ["--serve"], log="read_ab.log")


def rate(d):
    return d["prompt"] - d["reused"], (d["prompt"] - d["reused"]) / d["prompt_ms"] * 1000


def measure(env, item, prompts):
    e = engine(env)
    try:
        e.gen(1, "temperature=0", U.ids("gen.ids"))   # the first call after the load (not counted)
        if item == "turn":
            e.gen(1, "temperature=0", prompts["turn60"])
            _, d = e.gen(1, "temperature=0", prompts["turn64"])
            n, r = rate(d)
            return {"read": n, "tok_s": round(r, 1), "reused": d["reused"]}
        _, d = e.gen(1, "temperature=0", prompts[CTX[item]])
        n, r = rate(d)
        res = {"read": n, "tok_s": round(r, 1)}
        if item.startswith("gen"):
            st, _ = e.gen(U.SKIP + U.COUNT, B.T1, prompts[CTX[item]])
            res["gen_tok_s"] = round(U.write_rate(st)[0], 2)
        return res
    finally:
        e.quit()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", action="append", default=[], help="K=V of setting A (none: the config's)")
    ap.add_argument("--b", action="append", default=[], help="K=V of setting B")
    ap.add_argument("--items", nargs="+", default=["8k", "32k", "128k", "turn", "gen8k", "gen100k"])
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    tok = C.strata_tokenizer.Tokenizer.from_dir(C.PUB / "data" / "tokenizer")
    from serve.frontend import ChatTemplate   # noqa: E402
    tmpl = ChatTemplate(C.PUB / "data" / "tokenizer" / "chat_template.jinja")
    prompts = {c: D.prompt_ids(tok, tmpl, "code", c) for c in set(CTX[i] for i in a.items if i in CTX)}
    if "turn" in a.items:
        base = U.ids("long_code_base.ids") + U.ids("long_code_tail.ids")
        first = D.prompt_ids(tok, tmpl, "code", "64k")[:60000]
        more = base[70000:74000]
        prompts["turn60"] = first
        prompts["turn64"] = first + more + first[-6:]
    B.stop_server()
    t0 = time.time()
    for rnd in range(a.rounds):
        order = [("A", a.a), ("B", a.b)] if rnd % 2 == 0 else [("B", a.b), ("A", a.a)]
        for name, env in order:
            for item in a.items:
                r = measure(env, item, prompts)
                row = {"label": a.label, "round": rnd, "set": name, "env": env, "item": item, **r}
                print(f"[{a.label} round {rnd} {name} {item}] {r}  ({time.time() - t0:.0f} s)", flush=True)
                with OUT.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
