"""エンジン単体に、長い読み込みと短い生成を高速で繰り返させる (開発用、まれに落ちる不具合を早く出す)。

記録した GEN の行 (bench/replay_engine.py と同じ形) から、システムの部分 (sys_at まで) を取り出す。
毎回、その後ろに長さを変えた利用者の発言 (システムの部分から切り出したトークン) を付けて送る。
サーバーに近いタイミングを作るために、STOP を読み込みや生成の途中のランダムな時点で送る。
前の命令の DONE を待たずに次の GEN を送ることもできる。

使い方: python -I bench/stress_engine.py 命令のファイル 設定.json [--env K=V ...] [--arg 名前=値 ...]
  --read A,B      読み込むトークン数の範囲 (既定 1500,6500)
  --write A,B     書かせるトークン数の上限の範囲 (既定 8,120)
  --stop P        STOP を途中で送る割合 (既定 0.5)。読み込み中と生成中を半々にする
  --overlap P     DONE を待たずに STOP と次の GEN を続けて送る割合 (既定 0)
  --seconds N     ここまでで終える (既定 540)
  --arg           起動の引数を上書きする (例: --arg vram-reserve-mib=100)
"""
import argparse
import json
import os
import queue
import random
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import upstream_bench as U   # noqa: E402

SAMPLING = "temperature=0.7 top_p=0.8 top_k=20 penalty_present=1.5 penalty_last_n=64"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("commands")
    ap.add_argument("config")
    ap.add_argument("--exe", default="")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--arg", action="append", default=[])
    ap.add_argument("--read", default="1500,6500")
    ap.add_argument("--write", default="8,120")
    ap.add_argument("--stop", type=float, default=0.5)
    ap.add_argument("--overlap", type=float, default=0.0)
    ap.add_argument("--seconds", type=int, default=540)
    ap.add_argument("--rng", type=int, default=0)
    ap.add_argument("--log", default="stress_engine.log")
    a = ap.parse_args()
    rng = random.Random(a.rng or None)
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    env = dict(cfg.get("env") or {})
    env.update(dict(x.split("=", 1) for x in a.env))
    os.environ.update(env)
    args = list(cfg["args"])
    for x in a.arg:
        k, v = x.split("=", 1)
        if "--" + k in args:
            args[args.index("--" + k) + 1] = v
        else:
            args += ["--" + k, v]
    line = next(l for l in open(a.commands, encoding="utf-8") if l.startswith("GEN"))
    head, ids = line.rstrip("\r\n").rsplit(" ", 1)
    sys_at = int(next(x for x in head.split() if x.startswith("sys_at=")).split("=")[1])
    system = [int(x) for x in ids.split(",")[:sys_at]]
    words = [t for t in system[16:] if t < 200000]
    r0, r1 = map(int, a.read.split(","))
    w0, w1 = map(int, a.write.split(","))
    print(f"system {sys_at} tokens; env {sorted(k[9:] for k in env if k.startswith('INKLING2_'))}; args {args[2:]}",
          flush=True)
    e = U.Engine([a.exe or cfg["exe"], "--serve"] + args, log=a.log)
    print(f"ready in {e.load_s:.0f} s", flush=True)

    def send(text):
        e.p.stdin.write((text + "\n").encode())
        e.p.stdin.flush()

    def gen_line():
        n = rng.randint(r0, r1)
        at = rng.randint(0, len(words) - n - 1) if n < len(words) else 0
        prompt = system + [200000, 200004] + words[at:at + n] + [200010, 200001, 200004]
        return (f"GEN {rng.randint(w0, w1)} {SAMPLING} turn_at={len(prompt) - 2} sys_at={sys_at} "
                + ",".join(map(str, prompt))), n

    t0, it, stops, overlaps, waiting = time.time(), 0, 0, 0, 0
    try:
        text, n = gen_line()
        send(text)
        waiting = 1
        while waiting:
            plan = rng.random()
            stop_at = None   # seconds after the send (in the read), or a count of written tokens
            if plan < a.stop / 2:
                stop_at = ("time", rng.uniform(0.0, 2.5))
            elif plan < a.stop:
                stop_at = ("token", rng.randint(1, 30))
            sent_at, written, stopped = time.time(), 0, False
            while True:
                if stop_at and not stopped and stop_at[0] == "time" and time.time() - sent_at >= stop_at[1]:
                    send("STOP")
                    stopped = True
                    stops += 1
                try:
                    _, out = e.get(0.02 if stop_at and not stopped and stop_at[0] == "time" else 600)
                except queue.Empty:
                    continue
                if out.startswith("T "):
                    written += 1
                    if stop_at and not stopped and stop_at[0] == "token" and written >= stop_at[1]:
                        send("STOP")
                        stopped = True
                        stops += 1
                        if rng.random() < a.overlap and time.time() - t0 < a.seconds:
                            text, n = gen_line()   # the next one before this one's DONE
                            send(text)
                            waiting += 1
                            overlaps += 1
                elif out.startswith("DONE"):
                    break
                elif out.startswith("ERR"):
                    print(f"#{it}: {out}  (read {n}, {written} written, after {time.time() - t0:.0f} s, "
                          f"{stops} stops, {overlaps} overlaps)", flush=True)
                    return 1
            it += 1
            waiting -= 1
            if it % 25 == 0:
                print(f"{it} requests in {time.time() - t0:.0f} s ({stops} stops, {overlaps} overlaps)", flush=True)
            if waiting == 0 and time.time() - t0 < a.seconds:
                text, n = gen_line()
                send(text)
                waiting = 1
    except RuntimeError as x:
        print(f"#{it}: the engine ended: {x}  (after {time.time() - t0:.0f} s)", flush=True)
        return 1
    finally:
        e.quit()
    print(f"no failure: {it} requests in {time.time() - t0:.0f} s ({stops} stops, {overlaps} overlaps)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
