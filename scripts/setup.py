"""setup.bat's work: check the PC, fetch the model, extract its tokenizer, choose the settings, write the config.

  python scripts/setup.py                      everything (the model download is about 15.7 GB)
  python scripts/setup.py --configure-only     only choose the settings again (the model and tokenizer are there)
  python scripts/setup.py --limit-ram-gb 16 --limit-vram-gb 8   settings for a smaller PC (also to try them here)

The config is config/inkling.json: the server's run config (the engine, its arguments, the tokenizer, the port).
Run setup again after moving the folder or changing the PC's RAM or GPU.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = "Inkling-2-Mini-UD-IQ4_XS.gguf"
MODEL_REPO = "unsloth/Inkling-2-Mini-GGUF"
MODEL_REV = "45fd08764f2b0bf16b03a5483a80e5c7aaec8772"   # the revision this was tested with
MODEL_SIZE = 15733755232
MODEL_SHA256 = "a861d21e7c6d7bc69d8cdcdfb9eb195fd37ef1cd2ab31499db61cd7d30d9b3a1"
MODEL_URL = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REV}/{MODEL_FILE}"
MIN_DRIVER = 580          # CUDA 13 runtime (the release is built with CUDA 13.3)
MIN_CC = (7, 5)           # Turing and newer; the release has code for 7.5, 8.6, 8.9 and 12.0
DEFAULT_PORT = 8080


def say(msg=""):
    print(msg, flush=True)


def fail(msg):
    say(f"ERROR: {msg}")
    sys.exit(1)


def ram_gib():
    class MS(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                    ("avail", ctypes.c_ulonglong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                    ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ae", ctypes.c_ulonglong)]
    m = MS()
    m.len = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.total / 2**30, m.avail / 2**30


def gpus():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,compute_cap",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    res = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 5:
            cc = tuple(int(x) for x in p[4].split("."))
            res.append({"index": int(p[0]), "name": p[1], "driver": p[2], "vram_mib": int(float(p[3])), "cc": cc})
    return res


def physical_cores():
    try:
        import psutil
        return psutil.cpu_count(logical=False) or os.cpu_count() or 4
    except ImportError:
        return max(1, (os.cpu_count() or 8) // 2)


def engine_exe():
    for p in (ROOT / "bin" / "strata-inkling2.exe", ROOT / "build" / "bin" / "strata-inkling2.exe"):
        if p.exists():
            return p
    return None


def sha256(path):
    h = hashlib.sha256()
    done, t0, last = 0, time.time(), 0.0
    with open(path, "rb") as f:
        while True:
            b = f.read(16 << 20)
            if not b:
                break
            h.update(b)
            done += len(b)
            if time.time() - last > 5:
                last = time.time()
                say(f"  checking {100 * done / MODEL_SIZE:5.1f}%")
    return h.hexdigest()


def download(dst: Path):
    part = dst.with_suffix(dst.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    say(f"Downloading {MODEL_FILE} ({MODEL_SIZE / 1e9:.1f} GB) from huggingface.co/{MODEL_REPO}")
    req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "inkling2-mini-setup"})
    if have:
        req.add_header("Range", f"bytes={have}-")
        say(f"  resuming at {have / 1e9:.1f} GB")
    t0, last = time.time(), 0.0
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have else "wb") as f:
        if have and r.status != 206:
            f.truncate(0)
            have = 0
        done = have
        while True:
            b = r.read(8 << 20)
            if not b:
                break
            f.write(b)
            done += len(b)
            if time.time() - last > 5:
                last = time.time()
                rate = (done - have) / max(1e-9, time.time() - t0) / 1e6
                say(f"  {100 * done / MODEL_SIZE:5.1f}%  {done / 1e9:.1f} / {MODEL_SIZE / 1e9:.1f} GB  {rate:.0f} MB/s")
    if part.stat().st_size != MODEL_SIZE:
        fail(f"the download stopped at {part.stat().st_size} of {MODEL_SIZE} bytes; run setup.bat again to resume")
    part.replace(dst)


def choose(ram_total, vram_mib, vram_limit_mib, cores):
    """Engine arguments for this much RAM and VRAM (GiB / MiB).  Measured on one PC only (RTX 5070 12 GB, 32 GB):
    the smaller tiers are cautious guesses that leave room, not tuned values."""
    v = vram_limit_mib / 1024
    if v >= 11.5:
        ctx, batch, reserve = 131072, 2048, 400
    elif v >= 9.5:
        ctx, batch, reserve = 65536, 1024, 500
    elif v >= 7.5:
        ctx, batch, reserve = 32768, 1024, 600
    else:
        ctx, batch, reserve = 16384, 512, 600
    # a smaller budget than the card: the rest is left free as well
    reserve += max(0, vram_mib - vram_limit_mib)
    # experts outside VRAM kept in RAM (the rest is read through the file cache): the OS, other programs and the
    # engine's own ~2.5 GB get 14 GB first
    ram_experts = int(min(9216, max(1024, (ram_total - 14) * 1024)))
    threads = max(2, min(12, cores))
    args = ["--max-context", str(ctx + 1024), "--threads", str(threads), "--window", "8", "--lookup", "3",
            "--prompt-batch", str(batch), "--vram-reserve-mib", str(reserve), "--ram-experts-mib", str(ram_experts),
            "--kv", "q8_0"]
    return args, {"context": ctx, "prompt_batch": batch, "vram_reserve_mib": reserve, "ram_experts_mib": ram_experts,
                  "threads": threads}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--configure-only", action="store_true", help="choose the settings again; download nothing")
    ap.add_argument("--limit-ram-gb", type=float, help="settings as if the PC had this much RAM")
    ap.add_argument("--limit-vram-gb", type=float, help="settings for this much VRAM (the rest is left free)")
    ap.add_argument("--port", type=int, default=0, help=f"the server's port (default {DEFAULT_PORT})")
    ap.add_argument("--context", type=int, default=0, help="the context length instead of the chosen one")
    ap.add_argument("--gpu", type=int, default=0, help="which NVIDIA GPU (nvidia-smi's index)")
    ap.add_argument("--config", default=str(ROOT / "config" / "inkling.json"))
    a = ap.parse_args()
    say("== Checking this PC")
    if sys.platform != "win32":
        fail("this setup is for Windows")
    if sys.version_info < (3, 10):
        fail("Python 3.10 or newer is needed")
    gs = gpus()
    if not gs:
        fail("no NVIDIA GPU found (nvidia-smi did not answer). Install the NVIDIA driver first.")
    g = next((x for x in gs if x["index"] == a.gpu), None) or fail(f"no GPU with index {a.gpu}")
    major = int(g["driver"].split(".")[0])
    say(f"  GPU {g['index']}: {g['name']}, {g['vram_mib']} MiB, compute capability {g['cc'][0]}.{g['cc'][1]}, "
        f"driver {g['driver']}")
    if major < MIN_DRIVER:
        fail(f"driver {g['driver']} is too old: version {MIN_DRIVER} or newer is needed (CUDA 13)")
    if g["cc"] < MIN_CC:
        fail("this GPU is older than Turing (compute capability 7.5); it is not supported")
    if g["cc"] not in ((7, 5), (8, 6), (8, 9), (12, 0)):
        say("  note: this GPU generation was not built for directly; it may not run (built for 7.5, 8.6, 8.9, 12.0)")
    total, avail = ram_gib()
    say(f"  RAM {total:.1f} GiB ({avail:.1f} GiB free now)")
    if total < 15:
        say("  WARNING: less than 16 GB of RAM; the model's experts will be read from disk and run slowly")
    cores = physical_cores()
    say(f"  CPU: {cores} physical cores")
    models = Path(a.models_dir)
    models.mkdir(parents=True, exist_ok=True)
    model = models / MODEL_FILE
    free = shutil.disk_usage(models).free / 1e9
    say(f"  free disk space at {models}: {free:.1f} GB")
    exe = engine_exe()
    if not exe:
        fail("the engine is missing: bin\\strata-inkling2.exe (release package) or build\\bin (tools\\build.bat)")

    if not a.configure_only:
        say("== The model")
        if model.exists() and model.stat().st_size == MODEL_SIZE and (models / (MODEL_FILE + ".ok")).exists():
            say(f"  {model} is there and was checked before")
        else:
            if not (model.exists() and model.stat().st_size == MODEL_SIZE):
                need = (MODEL_SIZE - (model.with_suffix(".gguf.part").stat().st_size
                                      if model.with_suffix(".gguf.part").exists() else 0)) / 1e9 + 1
                if free < need:
                    fail(f"{need:.1f} GB of free disk space is needed at {models}")
                download(model)
            say(f"  checking the SHA-256 of {model.name} (a minute or two)")
            got = sha256(model)
            if got != MODEL_SHA256:
                fail(f"SHA-256 mismatch: {got}, expected {MODEL_SHA256}. Delete {model} and run setup again.")
            (models / (MODEL_FILE + ".ok")).write_text(MODEL_SHA256)
            say("  SHA-256 OK")
        say("== The tokenizer")
        tok = ROOT / "data" / "tokenizer"
        if not (tok / "vocab.json").exists():
            (ROOT / "data").mkdir(exist_ok=True)
            subprocess.run([sys.executable, "-I", str(ROOT / "server" / "tools" / "strata_tokenizer.py"), "--gguf",
                            str(model), "--out", str(ROOT / "data"), "--check"], check=True)
        say(f"  {tok}")
    elif not model.exists():
        fail(f"{model} is missing; run setup.bat without --configure-only first")

    say("== Settings")
    vram_limit = int(a.limit_vram_gb * 1024) if a.limit_vram_gb else g["vram_mib"]
    ram_total = min(total, a.limit_ram_gb) if a.limit_ram_gb else total
    args, chosen = choose(ram_total, g["vram_mib"], min(vram_limit, g["vram_mib"]), cores)
    if a.context:
        i = args.index("--max-context")
        args[i + 1] = str(a.context + 1024)
        chosen["context"] = a.context
    for k, v in chosen.items():
        say(f"  {k}: {v}")
    cfg_path = Path(a.config)
    old = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    port = a.port or old.get("port") or DEFAULT_PORT
    cfg = {"exe": str(exe), "cwd": str(ROOT), "tokenizer": str(ROOT / "data" / "tokenizer"),
           "model_name": "inkling-2-mini", "display_name": "Inkling 2 Mini UD-IQ4_XS",
           "args": ["--model", str(model)] + args,
           "host": "127.0.0.1", "port": port, "log": str(ROOT / "logs" / "engine.log"),
           "sampling": {"temperature": 1.0}, "chosen": chosen,
           "limits": {"ram_gb": a.limit_ram_gb, "vram_gb": a.limit_vram_gb}}
    if a.gpu:
        cfg["gpu"], cfg["gpus_asked"] = a.gpu, True   # the server tells the engine this card
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    say(f"  wrote {cfg_path}")
    say("")
    say(f"Done. Start the server with run.bat; it listens on http://127.0.0.1:{port}/v1 (OpenAI-compatible).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
