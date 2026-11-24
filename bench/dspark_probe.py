"""DSpark の下書き・検証の内訳 (開発用、計測のみ)。

文脈 8K / 64K で、素の生成 (lookup のみ) と DSpark (INKLING2_DSPARK) を比べる。依頼は 2 つ (温度 1.0、256 トークン)。
  text  前置きの資料 + 普通の文章を書く依頼
  code  前置きの資料 + 新しいコードを書く依頼 (写しではない)
前置きは公開のソースコード (bench/data の long_code をトークナイザーで戻したもの)。
生成は、同じ依頼で数えない生成を 1 回してから測る (--cold で起動直後の 1 回目)。
エンジンの報告 (INKLING2_REPORT_CACHE) から、書いたトークンあたりの CPU のエキスパート、命中率、ドラフターの時間を読む。
結果は bench/logs/dspark_probe.jsonl に 1 行ずつ足す。

使い方: python -I bench/dspark_probe.py [--ctx 8k 64k] [--modes base dspark] [--env K=V ...]
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
import copy_window as C     # noqa: E402

LOG = "dspark_probe_engine.log"
OUT = B.ROOT / "bench" / "logs" / "dspark_probe.jsonl"
DRAFTER = B.ROOT / "models" / "official" / "mtp.safetensors"
CTX = {"8k": 8192, "32k": 32768, "64k": 65536, "128k": 131072, "256k": 253000}
ASK = {
    "text": "Write a long, plain essay (no headings, no lists) about how small harbor towns changed when railways "
            "arrived: trade, work, food, daily life and what was lost. Write at least 800 words.",
    "code": "Write a complete Python module (standard library only) for a small task queue: a Task dataclass, a "
            "priority queue with retries and backoff, a worker pool with threads, persistence to a JSON file, and "
            "unit tests with unittest. Write only the code in one ```python block.",
}


def prompt_ids(tok, tmpl, task, ctx):
    room = CTX[ctx] - 900
    src = ("long_code256" if CTX[ctx] > 131072 else "long_code")
    filler = tok.decode((U.ids(f"{src}_base.ids") + U.ids(f"{src}_tail.ids"))[:room])
    content = "Here is some reference source code, for context only:\n\n" + filler + "\n\n----\n\n" + ASK[task]
    return tok.encode(tmpl.render([{"role": "user", "content": content}], reasoning_effort="none"), parse_special=True)


def report(text):
    out = {}
    m = re.findall(r"per written token: ([\d.]+) ms, CPU experts ([\d.]+) ms, router wait ([\d.]+) ms, ([\d.]+) CPU experts"
                   r".*?hit rate ([\d.]+)%", text)
    if m:
        out.update(zip(["ms_tok", "cpu_ms_tok", "router_ms_tok", "cpu_experts_tok", "hit"], map(float, m[-1])))
    m = re.findall(r"expert cache (\d+) experts", text)
    if m:
        out["experts_vram"] = int(m[-1])
    m = re.findall(r"DSpark this answer: (\d+) blocks, ([\d.]+) ms in the drafter \(([\d.]+) ms a block\), (\d+) guesses, (\d+) right", text)
    if m:
        out["blocks"], out["drafter_ms"], out["drafter_ms_block"] = int(m[-1][0]), float(m[-1][1]), float(m[-1][2])
        out["guesses"], out["right"] = int(m[-1][3]), int(m[-1][4])
    m = re.findall(r"DSpark drafter .*?: (\d+) MiB of VRAM", text)
    if m:
        out["drafter_mib"] = int(m[-1])
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", nargs="+", default=["8k", "64k"])
    ap.add_argument("--modes", nargs="+", default=["base", "dspark"])
    ap.add_argument("--tasks", nargs="+", default=["text", "code"])
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--label", default="")
    ap.add_argument("--window", type=int, default=0, help="--window (0: bench.py's 8)")
    ap.add_argument("--cold", action="store_true", help="measure the first answer (no warm-up answer)")
    ap.add_argument("--no-lookup", action="store_true", help="lookup=0 for the answers")
    ap.add_argument("--threads", type=int, default=0, help="--threads (0: bench.py's)")
    ap.add_argument("--reserve", type=int, default=0, help="--vram-reserve-mib (0: bench.py's)")
    a = ap.parse_args()
    extra_env = dict(x.split("=", 1) for x in a.env)
    tok = C.strata_tokenizer.Tokenizer.from_dir(C.PUB / "data" / "tokenizer")
    from serve.frontend import ChatTemplate   # noqa: E402
    tmpl = ChatTemplate(C.PUB / "data" / "tokenizer" / "chat_template.jinja")
    B.stop_server()
    log_path = B.ROOT / "bench" / "logs" / LOG
    for ctx in a.ctx:
        prompts = {t: prompt_ids(tok, tmpl, t, ctx) for t in a.tasks}
        for mode in a.modes:
            env = dict(extra_env, INKLING2_REPORT_CACHE="1")
            if mode == "dspark":
                env["INKLING2_DSPARK"] = str(DRAFTER)
            for k, v in env.items():
                os.environ[k] = v
            flags = list(B.BASE_FLAGS)
            flags[flags.index("--max-context") + 1] = str(CTX[ctx] + 1024)
            if a.window:
                flags[flags.index("--window") + 1] = str(a.window)
            if a.threads:
                flags[flags.index("--threads") + 1] = str(a.threads)
            if a.reserve:
                flags[flags.index("--vram-reserve-mib") + 1] = str(a.reserve)
            since = log_path.stat().st_size if log_path.exists() else 0
            e = U.Engine([str(B.EXE), "--model", str(B.MODEL), "--serve"] + flags + ["--kv", "q8_0"], log=LOG)
            try:
                e.gen(16, "temperature=0", U.ids("gen.ids"))   # warm-up (not counted)
                for t in a.tasks:
                    e.gen(1, "temperature=0", prompts[t])         # the prompt, read once (not counted)
                    if not a.cold:   # one answer first, so that the expert cache settles (not counted)
                        e.gen(U.SKIP + U.COUNT, B.T1 + (" lookup=0" if a.no_lookup else ""), prompts[t])
                    at = log_path.stat().st_size
                    stamps, d = e.gen(U.SKIP + U.COUNT, B.T1 + (" lookup=0" if a.no_lookup else ""), prompts[t])
                    time.sleep(0.5)
                    rep = report(log_path.read_text(encoding="utf-8", errors="replace")[at:])
                    if "drafter_mib" not in rep:
                        rep.update({k: v for k, v in report(log_path.read_text(encoding="utf-8", errors="replace")[since:]).items()
                                    if k == "drafter_mib"})
                    row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "label": a.label, "ctx": ctx, "mode": mode,
                           "task": t, "tps": round(U.write_rate(stamps)[0], 2), "produced": d["produced"]}
                    row.update(rep)
                    print(f"[{ctx} {mode} {t}] {row['tps']} tok/s  " + "  ".join(f"{k}={v}" for k, v in rep.items()),
                          flush=True)
                    with open(OUT, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row) + "\n")
            finally:
                e.quit()
                for k in env:
                    os.environ.pop(k, None)


if __name__ == "__main__":
    main()
