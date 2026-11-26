"""普段使いの設定ファイル (local/try*.json) のエンジン・引数・環境変数で測る (開発用)。
文脈を最初から 1 回読み (読み込みの速さ)、数えない答え 1 回、lookup ありの答え、lookup なしの答え (温度 1.0、bench.py の T1)。
依頼は dspark_probe.py の code。結果は bench/logs/config_runs.jsonl。

使い方: python -I bench/config_runs.py CONFIG --ctx 128k [--kv q8_0] [--label ...]
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

OUT = B.ROOT / "bench" / "logs" / "config_runs.jsonl"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--ctx", required=True)
    ap.add_argument("--kv", default="", help="replace the config's --kv")
    ap.add_argument("--set", action="append", default=[], metavar="FLAG=VALUE",
                    help="replace the config's --FLAG's value (e.g. --set prompt-batch=2048)")
    ap.add_argument("--label", default="")
    ap.add_argument("--exe", default="", help="replace the config's exe (dev: the build's)")
    ap.add_argument("--env", action="append", default=[], help="K=V on top of the config's env")
    ap.add_argument("--times", type=int, default=1, help="lookup / no-lookup pairs, alternated")
    ap.add_argument("--no-read", action="store_true", help="read the prompt with the warm-up answer, no read rate")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    args = list(cfg["args"])
    if a.kv:
        args[args.index("--kv") + 1] = a.kv
    for kv in a.set:
        flag, value = kv.split("=", 1)
        args[args.index("--" + flag) + 1] = value
    tok = C.strata_tokenizer.Tokenizer.from_dir(C.PUB / "data" / "tokenizer")
    from serve.frontend import ChatTemplate   # noqa: E402
    tmpl = ChatTemplate(C.PUB / "data" / "tokenizer" / "chat_template.jinja")
    prompt = D.prompt_ids(tok, tmpl, "code", a.ctx)
    B.stop_server()
    env = dict(cfg.get("env") or {})
    env.update(dict(x.split("=", 1) for x in a.env))
    os.environ.update(env)
    log = B.ROOT / "bench" / "logs" / "config_runs.log"
    at = log.stat().st_size if log.exists() else 0
    e = U.Engine([a.exe or cfg["exe"]] + args + ["--serve"], log="config_runs.log")
    try:
        read, n_prompt = None, len(prompt)
        if not a.no_read:
            _, d = e.gen(1, "temperature=0", prompt)
            read, n_prompt = round(d["prompt"] / (d["prompt_ms"] / 1000.0), 1), d["prompt"]
        e.gen(U.SKIP + U.COUNT, B.T1, prompt)   # warm-up answer (not counted)
        got = {"on": [], "off": []}
        for i in range(a.times):
            pair = [("on", ""), ("off", " lookup=0")]
            for name, extra in (pair if i % 2 == 0 else pair[::-1]):
                st, _ = e.gen(U.SKIP + U.COUNT, B.T1 + extra, prompt)
                got[name].append(round(U.write_rate(st)[0], 2))
                print(f"  {name}: {got[name][-1]} tok/s", flush=True)
        row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "config": Path(a.config).name, "label": a.label,
               "env": a.env, "kv": args[args.index("--kv") + 1], "ctx": a.ctx, "prompt": n_prompt, "read_tps": read,
               "lookup_tps": got["on"], "no_lookup_tps": got["off"]}
        print(json.dumps(row), flush=True)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    finally:
        e.quit()
    text = log.read_text(encoding="utf-8", errors="replace")[at:]
    for line in text.splitlines():
        if "answer steps" in line or " steps: " in line or "lookup policy" in line:
            print("   ", line.split("strata-")[-1][:300])


if __name__ == "__main__":
    main()
