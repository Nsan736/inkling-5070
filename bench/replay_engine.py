"""サーバーがエンジンに送った命令 (GEN の行) を、エンジン単体に順に送り直す (開発用、不具合の再現)。

命令の行は、エンジンの前に置いた中継 (標準入力を書き写す) で取る。STOP の行は飛ばし、GEN には seed を足して
毎回同じ答えにする。設定ファイルの args と env でエンジンを起動する (--exe、--env で上書き)。

使い方: python -I bench/replay_engine.py 命令のファイル 設定.json [--exe ...] [--env K=V ...] [--prefix ...]
  --prefix  エンジンの前に付けるコマンド (例: compute-sanitizer の実行ファイルと引数、空白で区切る)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import upstream_bench as U   # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("commands")
    ap.add_argument("config")
    ap.add_argument("--exe", default="")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--prefix", default="")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--log", default="replay_engine.log")
    ap.add_argument("--stops", action="store_true", help="STOP の行も送る (前の GEN の DONE のすぐ後)")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    env = dict(cfg.get("env") or {})
    env.update(dict(x.split("=", 1) for x in a.env))
    os.environ.update(env)
    lines = [l.rstrip("\r\n") for l in open(a.commands, encoding="utf-8") if l.startswith("GEN") or (a.stops and l.startswith("STOP"))]
    args = a.prefix.split() + [a.exe or cfg["exe"], "--serve"] + list(cfg["args"])
    print(f"{len(lines)} GEN lines; env {sorted(k for k in env if k.startswith('INKLING2_'))}", flush=True)
    t0 = time.time()
    e = U.Engine(args, log=a.log)
    print(f"ready in {time.time() - t0:.0f} s", flush=True)
    try:
        for i, line in enumerate(lines):
            if line.startswith("STOP"):
                e.p.stdin.write(b"STOP\n")
                e.p.stdin.flush()
                continue
            head, ids = line.rsplit(" ", 1)
            if a.seed and " seed=" not in head:
                head += f" seed={a.seed}"
            e.p.stdin.write(f"{head} {ids}\n".encode())
            e.p.stdin.flush()
            n = 0
            t = time.time()
            while True:
                _, out = e.get(3600)
                if out.startswith("T "):
                    n += 1
                elif out.startswith("DONE"):
                    print(f"#{i}: {len(ids.split(','))} prompt tokens, {n} written in {time.time() - t:.1f} s", flush=True)
                    break
                elif out.startswith("ERR"):
                    print(f"#{i}: {out}", flush=True)
                    return 1
    except RuntimeError as x:
        print(f"the engine ended: {x}", flush=True)
        return 1
    finally:
        e.quit()
    print("all done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
