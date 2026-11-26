"""2 つの設定 (環境変数) の、貪欲な生成の一致と生成の速さを比べる (開発用)。

普段使いの設定 (local/try8078.json) の引数と環境変数で、開発版のエンジンを起動する。設定ごとに 1 つのエンジンで、
文脈ごとに: コードの依頼を読み、数えない答え 1 回、貪欲な 216 トークン (lookup なし、トークン列を比べる)、
温度 1.0 の 216 トークン (lookup なし、bench.py の T1、17〜216 トークン目の tok/s)。
設定の順番は回ごとに入れ替える (A B、B A、...)。結果は bench/logs/fuse_ab.jsonl。

使い方: python -I bench/fuse_ab.py --b INKLING2_FUSE=2 [--a K=V ...] [--ctx 8k 100k] [--rounds 2]
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

OUT = B.ROOT / "bench" / "logs" / "fuse_ab.jsonl"
CONFIG = B.ROOT / "local" / "try8078.json"


def greedy(e, n, prompt):
    e.p.stdin.write(f"GEN {n} temperature=0 lookup=0 {','.join(map(str, prompt))}\n".encode())
    e.p.stdin.flush()
    toks = []
    while True:
        _, line = e.get(3600)
        if line.startswith("T "):
            toks.append(int(line.split()[1]))
        elif line.startswith("DONE"):
            return toks
        elif line.startswith("ERR"):
            raise RuntimeError(line)


def run(env, prompts, label):
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    full = dict(cfg.get("env") or {})
    full.update(dict(x.split("=", 1) for x in env))
    old = {k: os.environ.get(k) for k in full}
    os.environ.update(full)
    out = {}
    try:
        e = U.Engine([str(B.EXE)] + list(cfg["args"]) + ["--serve"], log="fuse_ab.log")
        try:
            for ctx, prompt in prompts.items():
                e.gen(U.SKIP + U.COUNT, B.T1 + " lookup=0", prompt)   # warm-up answer (not counted)
                toks = greedy(e, U.SKIP + U.COUNT, prompt)
                st, _ = e.gen(U.SKIP + U.COUNT, B.T1 + " lookup=0", prompt)
                out[ctx] = (toks, round(U.write_rate(st)[0], 2))
                print(f"  [{label} {ctx}] {out[ctx][1]} tok/s", flush=True)
        finally:
            e.quit()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", action="append", default=[], help="K=V of setting A (none: the defaults)")
    ap.add_argument("--b", action="append", default=[], help="K=V of setting B")
    ap.add_argument("--ctx", nargs="+", default=["8k", "100k"])
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    tok = C.strata_tokenizer.Tokenizer.from_dir(C.PUB / "data" / "tokenizer")
    from serve.frontend import ChatTemplate   # noqa: E402
    tmpl = ChatTemplate(C.PUB / "data" / "tokenizer" / "chat_template.jinja")
    prompts = {c: D.prompt_ids(tok, tmpl, "code", c) for c in a.ctx}
    B.stop_server()
    got = {"A": [], "B": []}
    for r in range(a.rounds):
        order = ["A", "B"] if r % 2 == 0 else ["B", "A"]
        for name in order:
            got[name].append(run(a.a if name == "A" else a.b, prompts, name))
    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": a.label, "a": a.a, "b": a.b}
    def diff(ta, tb):
        return next((i for i, (p, q) in enumerate(zip(ta, tb)) if p != q), None if len(ta) == len(tb) else 0)

    for ctx in a.ctx:
        runs = [("A", i, x[ctx][0]) for i, x in enumerate(got["A"])] + [("B", i, x[ctx][0]) for i, x in enumerate(got["B"])]
        pairs = {f"{n1}{i1}-{n2}{i2}": diff(t1, t2) for k, (n1, i1, t1) in enumerate(runs)
                 for (n2, i2, t2) in runs[k + 1:]}
        same = all(v is None for v in pairs.values())
        ra = [x[ctx][1] for x in got["A"]]
        rb = [x[ctx][1] for x in got["B"]]
        row[ctx] = {"greedy_same": same, "first_diff": pairs, "a_tps": ra, "b_tps": rb,
                    "gain_pct": round(100.0 * (sum(rb) / len(rb) / (sum(ra) / len(ra)) - 1.0), 2),
                    "tokens": {f"{n}{i}": t for n, i, t in runs}}
        shown = {k: v for k, v in pairs.items() if v is not None}
        print(f"{ctx}: greedy {'same in every run' if same else f'first differences {shown}'}; A {ra} B {rb} "
              f"({row[ctx]['gain_pct']:+.2f}%)", flush=True)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
