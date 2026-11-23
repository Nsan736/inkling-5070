"""run.bat's work: start the OpenAI-compatible server (server/serve/server.py) with config/inkling.json.

It listens on 127.0.0.1 only.  The port is the config's "port" (setup.py --port N changes it) or --port N here.
While it runs, the engine's log (config "log") is followed and a short English line is printed when the engine is
ready (the experts it put in VRAM) and after each request (speed, experts in VRAM, how many of the experts a written
token used were in VRAM).  --quiet leaves those lines out.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LOADED = re.compile(r"loaded in ([\d.]+) s; (\d+) of (\d+) routed experts in VRAM \(([\d.]+) GiB\)")
REQUEST = re.compile(r"strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in [\d.]+ ms \(([\d.]+) tok/s\), "
                     r"(\d+) generated in [\d.]+ ms \(([\d.]+) tok/s\)")
CACHE = re.compile(r"expert cache (\d+) experts \((\d+) when full, ([\d.]+) GiB\) at (\d+) tokens.*?"
                   r"([\d.]+) CPU experts .*?hit rate ([\d.]+)%")


def follow(log: Path, stop: threading.Event):
    """Print the engine's state from its log, from the end it has now (lines of earlier runs are not repeated)."""
    total = 0
    req = None
    while not stop.is_set() and not log.exists():
        time.sleep(0.5)
    if stop.is_set():
        return
    with open(log, "r", encoding="utf-8", errors="replace") as f:
        f.seek(0, os.SEEK_END)
        buf = ""
        while not stop.is_set():
            chunk = f.read()
            if not chunk:
                time.sleep(0.5)
                continue
            buf += chunk
            *lines, buf = buf.split("\n")
            for line in lines:
                m = LOADED.search(line)
                if m:
                    n, total = int(m.group(2)), int(m.group(3))
                    print(f"[engine] ready in {float(m.group(1)):.0f} s: {n} of {total} experts in VRAM "
                          f"({100.0 * n / total:.0f}%, {float(m.group(4)):.1f} GiB); the others run on the CPU",
                          flush=True)
                    continue
                m = REQUEST.search(line)
                if m:
                    req = m
                    continue
                m = CACHE.search(line)
                if m and req:
                    n, full = int(m.group(1)), int(m.group(2))
                    share = f" ({100.0 * n / total:.0f}%)" if total else ""
                    read = int(req.group(3))
                    reading = f"read {read} tokens at {float(req.group(4)):.0f} tok/s" if read > 1 else "nothing new to read"
                    print(f"[engine] request: {reading}, wrote {req.group(5)} tokens at {float(req.group(6)):.1f} tok/s; "
                          f"context {int(m.group(4))} tokens; experts in VRAM {n}{share}"
                          + (f", {full} when the context is short" if full != n else "")
                          + f"; hit rate {float(m.group(6)):.1f}% ({float(m.group(5)):.1f} experts a token on the CPU)",
                          flush=True)
                    req = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "inkling.json"))
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--api-key", default=None, help="require this key (Authorization: Bearer ...)")
    ap.add_argument("--quiet", action="store_true", help="no lines about the engine's experts and speed")
    a, rest = ap.parse_known_args()
    cfg_path = Path(a.config)
    if not cfg_path.exists():
        print(f"ERROR: {cfg_path} is missing; run setup.bat first", flush=True)
        return 1
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    for key in ("exe", "tokenizer"):
        if not Path(cfg[key]).exists():
            print(f"ERROR: {cfg[key]} is missing; run setup.bat again", flush=True)
            return 1
    port = a.port or int(cfg.get("port") or 8080)
    cmd = [sys.executable, "-I", str(ROOT / "server" / "serve" / "server.py"), "--engine", "strata", "--config",
           str(cfg_path), "--host", "127.0.0.1", "--port", str(port)]
    if a.api_key:
        cmd += ["--api-key", a.api_key]
    env = dict(os.environ)
    stop = threading.Event()
    if not a.quiet and cfg.get("log"):
        env["INKLING2_REPORT_CACHE"] = "1"   # the engine's line after each answer (its expert cache, its hit rate)
        threading.Thread(target=follow, args=(Path(cfg["log"]), stop), daemon=True).start()
    print(f"Starting Inkling 2 Mini on http://127.0.0.1:{port}/v1 (Ctrl+C or close this window to stop)", flush=True)
    try:
        return subprocess.call(cmd + rest, cwd=str(ROOT), env=env)
    except KeyboardInterrupt:
        return 0
    finally:
        stop.set()


if __name__ == "__main__":
    sys.exit(main())
