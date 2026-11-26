"""--prompt-batch ごとの読み込みの速さ (開発用)。文脈ごとに新しいエンジンで、コードの依頼を最初から 1 回読む。
--times で INKLING2_PREFILL_TIMES=1 (段ごとに待つので遅くなる) の内訳の行も出す。

使い方: python -I bench/prefill_batch.py [--ctx 8k 32k] [--batches 2048 4096] [--times]
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench as B            # noqa: E402
import upstream_bench as U   # noqa: E402
import copy_window as C      # noqa: E402
import dspark_probe as D     # noqa: E402

LOG = "prefill_batch.log"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", nargs="+", default=["8k", "32k"])
    ap.add_argument("--batches", nargs="+", default=["2048", "4096"])
    ap.add_argument("--times", action="store_true")
    ap.add_argument("--env", action="append", default=[], help="K=V for the engine (several: --env A=1 --env B=2)")
    ap.add_argument("--warm", action="store_true", help="read another prompt first (gen.ids, not counted)")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    tok = C.strata_tokenizer.Tokenizer.from_dir(C.PUB / "data" / "tokenizer")
    from serve.frontend import ChatTemplate   # noqa: E402
    tmpl = ChatTemplate(C.PUB / "data" / "tokenizer" / "chat_template.jinja")
    B.stop_server()
    log = B.ROOT / "bench" / "logs" / LOG
    if a.times:
        os.environ["INKLING2_PREFILL_TIMES"] = "1"
        os.environ["INKLING2_REPORT_CACHE"] = "1"
    for ctx in a.ctx:
        prompt = D.prompt_ids(tok, tmpl, "code", ctx)
        for b in a.batches:
            flags = list(B.BASE_FLAGS)
            flags[flags.index("--max-context") + 1] = str(D.CTX[ctx] + 1024)
            flags[flags.index("--prompt-batch") + 1] = b
            flags[flags.index("--window") + 1] = "32"
            at = log.stat().st_size if log.exists() else 0
            extra = dict(x.split("=", 1) for x in a.env)
            os.environ.update(extra)
            e = U.Engine([str(B.EXE), "--model", str(B.MODEL), "--serve"] + flags + ["--kv", "q8_0"], log=LOG)
            try:
                if a.warm:
                    e.gen(1, "temperature=0", U.ids("gen.ids"))
                _, d = e.gen(1, "temperature=0", prompt)
                import psutil
                free = psutil.virtual_memory().available / 2 ** 30
                print(f"[{ctx} batch {b} {a.label}] {d['prompt']} tokens read at "
                      f"{d['prompt'] / d['prompt_ms'] * 1000:.0f} tok/s; RAM available {free:.1f} GiB", flush=True)
            finally:
                e.quit()
                for k in extra:
                    os.environ.pop(k, None)
            text = log.read_text(encoding="utf-8", errors="replace")[at:]
            for line in text.splitlines():
                if "routed experts in VRAM" in line or "prompt phases" in line or "RAM pool of" in line:
                    print("   ", line.split("strata-")[-1][:400])


if __name__ == "__main__":
    main()
