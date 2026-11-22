"""run.bat's work: start the OpenAI-compatible server (server/serve/server.py) with config/inkling.json.

It listens on 127.0.0.1 only.  The port is the config's "port" (setup.py --port N changes it) or --port N here.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config" / "inkling.json"))
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--api-key", default=None, help="require this key (Authorization: Bearer ...)")
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
    print(f"Starting Inkling 2 Mini on http://127.0.0.1:{port}/v1 (Ctrl+C or close this window to stop)", flush=True)
    try:
        return subprocess.call(cmd + rest, cwd=str(ROOT))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
