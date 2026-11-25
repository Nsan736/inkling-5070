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
# --model: the default IQ4_XS GGUF, or one of unsloth's smaller GGUFs of the same revision (below)
MODELS = {
    "iq4_xs": {"file": MODEL_FILE, "url": MODEL_URL, "size": MODEL_SIZE, "sha256": MODEL_SHA256,
               "repo": MODEL_REPO, "display": "Inkling 2 Mini UD-IQ4_XS"},
}
# --model: unsloth's smaller GGUFs of the same revision, for 8 GB and 6 GB cards or long contexts (lower quality)
for _key, _q, _size, _sha in (
        ("q3_k_xl", "UD-Q3_K_XL", 13_631_778_016, "f72f396d1b479b14d76346df80baa449c8b1f399653bff38fa4764b359a40048"),
        ("q2_k_xl", "UD-Q2_K_XL", 9_138_715_328, "20292e8142b69535ca88a5d04439d0fdeb4dc698d44c002dca706dcf92901441"),
        ("iq2_xxs", "UD-IQ2_XXS", 8_486_236_352, "4fe958f73a667d8082f911a4347e52bc34696a7816cf3d5b9f7c8a81d5c5ceaa")):
    MODELS[_key] = {"file": f"Inkling-2-Mini-{_q}.gguf", "size": _size, "sha256": _sha, "repo": MODEL_REPO,
                    "url": f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REV}/Inkling-2-Mini-{_q}.gguf",
                    "display": f"Inkling 2 Mini {_q}"}
HEADER_BYTES = 32 << 20   # the IQ4_XS GGUF's metadata (the tokenizer) lies in its first 32 MiB
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
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,compute_cap,memory.free",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    res = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 5:
            cc = tuple(int(x) for x in p[4].split("."))
            free = int(float(p[5])) if len(p) >= 6 and p[5].replace(".", "").isdigit() else int(float(p[3]))
            res.append({"index": int(p[0]), "name": p[1], "driver": p[2], "vram_mib": int(float(p[3])), "cc": cc,
                        "free_mib": free})
    return res


def physical_cores():
    try:
        import psutil
        return psutil.cpu_count(logical=False) or os.cpu_count() or 4
    except ImportError:
        return max(1, (os.cpu_count() or 8) // 2)


# --vision cpu: the image encoder (PolyStrata's strata-vision on llama.cpp's mtmd, built without CUDA: no VRAM) and
# ggml-org's encoder weights, the file PolyStrata checks Inkling 2's pictures with
VISION_MMPROJ = {"file": "mmproj-Inkling-2-Mini-Q8_0.gguf", "size": 345_910_944,
                 "sha256": "233287272ec5ac39b4168677aba54a4c024f202410f5629fac1830e1059f8abb",
                 "repo": "ggml-org/Inkling-2-Mini-GGUF",
                 "url": "https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/resolve/"
                        "db61c5b492a6a034d986328d28274af0a439aa6c/mmproj-Inkling-2-Mini-Q8_0.gguf"}


def vision_exe():
    for p in (ROOT / "bin" / "strata-vision.exe", ROOT / "build-vision" / "bin" / "strata-vision.exe"):
        if p.exists():
            return p
    return None


def engine_exe():
    for p in (ROOT / "bin" / "strata-inkling2.exe", ROOT / "build" / "bin" / "strata-inkling2.exe"):
        if p.exists():
            return p
    return None


def fetch_header(dst: Path):
    """The IQ4_XS GGUF's first HEADER_BYTES: its metadata, the tokenizer of every build."""
    req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "inkling-5070-setup",
                                                     "Range": f"bytes=0-{HEADER_BYTES - 1}"})
    say(f"  fetching the tokenizer: the first {HEADER_BYTES >> 20} MiB of {MODEL_FILE}")
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    if len(data) != HEADER_BYTES:
        fail(f"the tokenizer's download stopped at {len(data)} of {HEADER_BYTES} bytes; run setup again")
    dst.write_bytes(data)


def sha256(path, size=None):
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
                say(f"  checking {100 * done / (size or MODEL_SIZE):5.1f}%")
    return h.hexdigest()


def download(dst: Path, m=None):
    m = m or MODELS["iq4_xs"]
    size = m["size"]
    part = dst.with_suffix(dst.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    say(f"Downloading {m['file']} ({size / 1e9:.1f} GB) from huggingface.co/{m['repo']}")
    req = urllib.request.Request(m["url"], headers={"User-Agent": "inkling-5070-setup"})
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
                say(f"  {100 * done / size:5.1f}%  {done / 1e9:.1f} / {size / 1e9:.1f} GB  {rate:.0f} MB/s")
    if part.stat().st_size != size:
        fail(f"the download stopped at {part.stat().st_size} of {size} bytes; run setup.bat again to resume")
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
    # pinned RAM for a long prompt's rows (--read-mib): fewer passes over the prompt, so the experts outside VRAM
    # are copied fewer times (32K to 128K read 4-6% faster with 2-4 GiB on the PC above); only while reading
    read_mib = 2048 if ram_total >= 30 else 1024 if ram_total >= 15 else 512
    args = ["--max-context", str(ctx + 1024), "--threads", str(threads), "--window", "32", "--lookup", "3",
            "--prompt-batch", str(batch), "--vram-reserve-mib", str(reserve), "--ram-experts-mib", str(ram_experts),
            "--read-mib", str(read_mib), "--kv", "q8_0"]
    return args, {"context": ctx, "prompt_batch": batch, "vram_reserve_mib": reserve, "ram_experts_mib": ram_experts,
                  "read_mib": read_mib, "threads": threads}


# What setup expects of a choice: the GGUFs' sizes (experts and the rest, MiB; 3072 routed experts: 24 layers of
# 128), and what was measured on one PC (RTX 5070 12 GB with the VRAM limited by --vram-reserve-mib to act as an 8 GB
# or 6 GB card, Core Ultra 7 265K, DDR5-5600 32 GB, q8_0 KV, prompt batches of 2048, 2026-11-23): tok/s of reading
# an 8K prompt / of writing at temperature 1.0 after it, and at 64K / 128K / 256K (the last 8K read after the rest).
# kld: 2K's mean KL divergence from unsloth's Q8_K_XL (llama.cpp's logits).  256K (12 GB): with LONG_ARGS, the
# last 8K read after 245,758 tokens; reading all 253,950 from the start: 1355 tok/s (IQ2_XXS).
# IQ4_XS's writing at 12 GB is 0.3.2's (2026-11-24), measured after one warm-up answer on the same prompt (the first
# answer after a start is about 10% slower); its reading and the other rows are 0.3.1's.
SIZES = {"iq4_xs": (13385, 1138), "q3_k_xl": (11633, 1051), "q2_k_xl": (7522, 946), "iq2_xxs": (6915, 931)}
MEASURED = {
    "iq4_xs": {"kld": 0.0026, 12: {8: (2162, 70.7), 64: (1725, 54.7), 128: (1327, 44.3), 256: (848, 36.6)},
               8: {8: (1902, 51.4), 64: (1541, 34.3), 128: (1214, 25.3)}, 6: {8: (1795, 35.0), 64: (1446, 19.4)}},
    "q3_k_xl": {"kld": 0.0090, 12: {8: (2186, 69.0), 64: (1741, 55.4), 128: (1340, 45.1)},
                8: {8: (2003, 57.6), 64: (1650, 38.8), 128: (1253, 26.1)}, 6: {8: (1949, 39.0), 64: (1532, 22.2)}},
    "q2_k_xl": {"kld": 0.0535, 12: {8: (2604, 74.0), 64: (1962, 53.7), 128: (1432, 45.4), 256: (942, 38.1)},
                8: {8: (2325, 67.0), 64: (1811, 43.5), 128: (1351, 27.0)}, 6: {8: (2272, 43.9), 64: (1710, 23.9)}},
    "iq2_xxs": {"kld": 0.0687, 12: {8: (2870, 79.9), 64: (1966, 63.0), 128: (1509, 51.6), 256: (956, 36.9)},
                8: {8: (2362, 66.7), 64: (1841, 45.7), 128: (1397, 26.3)}, 6: {8: (2388, 44.5), 64: (1755, 21.5)}},
}
N_EXPERTS = 3072
KV_TOKEN = {"q8_0": 17408, "fp16": 32768}   # bytes a token (the engine's rows: 16 of 1088 bytes at q8_0)
# MiB the engine takes beside the experts and the trunk, mostly a prompt batch's scratch: 1180 + 0.777 a token of
# the batch (measured: 2730-2800 at 2048, 1975 at 1024, 1577 at 512)


def engine_fixed(batch):
    return 1180 + 0.777 * batch

LONG_SLACK = 250      # MiB less of it once the experts gave way to the KV cache (64K fit a 6 GB budget, 128K not)


def plan(key, free_mib, reserve, ctx, batch=2048, kv="q8_0"):
    """(experts in VRAM at a short context, at ctx, MiB the context lacks) for a model of SIZES."""
    experts, trunk = SIZES[key]
    each = experts / N_EXPERTS
    room = free_mib - reserve - trunk - engine_fixed(batch)
    kv_mib = ctx * KV_TOKEN[kv] / 2**20
    short = max(0, min(N_EXPERTS, int(room / each)))
    full = max(0, min(N_EXPERTS, int((room - kv_mib) / each)))
    return short, full, max(0.0, kv_mib - room - LONG_SLACK)


def measured(key, vram_gb, ctx_k):
    """The nearest measurement at or below this much VRAM: (GB it was measured at, (read, write) tok/s) or None."""
    m = MEASURED.get(key, {})
    for v in (12, 8, 6):
        if vram_gb + 0.25 >= v and ctx_k in m.get(v, {}):
            return v, m[v][ctx_k]
    return None


def context_arg(v: str) -> int:
    """--context: a number of tokens, or 64k / 128k / 256k (1024 tokens a k)."""
    v = v.strip().lower()
    return int(v[:-1]) * 1024 if v.endswith("k") else int(v)


def quality_note(key):
    """The warning for a model whose 2K KLD is ten times IQ4_XS's or more ("" for the others)."""
    if key not in MEASURED or key == "iq4_xs":
        return ""
    r = MEASURED[key]["kld"] / MEASURED["iq4_xs"]["kld"]
    return f"fast, but its measured quality drop is large (2K KLD about {r:.0f}x IQ4_XS's)" if r >= 10 else ""


def expect(key, g, budget_mib, chosen):
    """setup's estimate for the choice: VRAM, KV, RAM, experts in VRAM, speed; then other contexts and models."""
    say("== What to expect (estimates, not measurements of this PC)")
    if key not in SIZES:
        say("  no estimate for Strata's SQ builds")
        return
    if quality_note(key):
        say(f"  NOTE: {MODELS[key]['display']} is {quality_note(key)}; the default iq4_xs is the closest to Q8")
    ctx, reserve = chosen["context"], chosen["vram_reserve_mib"]
    # what other programs hold now stays theirs; a smaller budget than the card (--limit-vram-gb) takes the rest
    # off the free VRAM, and off the reserve choose() added for it
    cut = max(0, g["vram_mib"] - budget_mib)
    free = g.get("free_mib", g["vram_mib"]) - cut
    reserve_own = reserve - cut
    usable = free - reserve_own
    short, full, lack = plan(key, free, reserve_own, ctx, chosen["prompt_batch"])
    experts, trunk = SIZES[key]
    each = experts / N_EXPERTS
    kv_gib = ctx * KV_TOKEN["q8_0"] / 2**30
    say(f"  usable VRAM: {usable / 1024:.1f} GiB ({free} MiB free now, {reserve_own} MiB left free)")
    held = g["vram_mib"] - g.get("free_mib", g["vram_mib"])
    if held > 1536:
        say(f"  note: other programs hold {held / 1024:.1f} GiB of VRAM now (the engine still running, a game, ...); "
            "these estimates assume they keep it. Close them and run setup --configure-only again for the full card.")
    say(f"  KV cache at {ctx // 1024}K tokens: {kv_gib:.1f} GiB (q8_0)")
    say(f"  experts in VRAM: about {short} of {N_EXPERTS} ({100 * short / N_EXPERTS:.0f}%) with a short context, "
        f"{full} ({100 * full / N_EXPERTS:.0f}%) at {ctx // 1024}K")
    outside = (N_EXPERTS - short) * each
    ram = min(chosen["ram_experts_mib"], outside)
    say(f"  experts in RAM: {ram / 1024:.1f} GiB of the {outside / 1024:.1f} GiB outside VRAM"
        + ("" if ram >= outside else "; the rest is read from the file cache or the disk"))
    if lack > 0:
        say(f"  WARNING: the KV cache at {ctx // 1024}K needs {lack / 1024:.1f} GiB more VRAM than is left: "
            f"choose a shorter --context or a smaller model")
    gb = budget_mib / 1024
    for c in (8, ctx // 1024):
        m = measured(key, gb, c)
        if m:
            how = "prompt batches of 1024, 200 MiB left free" if c == 256 else "prompt batches of 2048"
            say(f"  speed at {c}K (measured on an RTX 5070 limited to {m[0]} GB, {how}): reading {m[1][0]} tok/s, "
                f"writing {m[1][1]} tok/s")
    rows = [("context", "KV cache", "experts in VRAM at full", "write tok/s (measured)")]
    for c in (32, 64, 128, 256):
        s_, f_, l_ = plan(key, free, reserve_own, c * 1024, chosen["prompt_batch"])
        m = measured(key, gb, c)
        rows.append((f"{c}K", f"{c * 1024 * KV_TOKEN['q8_0'] / 2**30:.1f} GiB",
                     "does not fit" if l_ > 0 else f"{f_} ({100 * f_ / N_EXPERTS:.0f}%)",
                     f"{m[1][1]} ({m[0]} GB)" if m else "-"))
    m256 = measured(key, gb, 256)
    if m256 and ctx < 256 * 1024 and gb >= 11.5 and key != "iq4_xs":
        say(f"  256K fits this card with this model: setup --context 256k (measured: writing {m256[1][1]} tok/s, "
            f"reading the last 8K {m256[1][0]} tok/s; prompt batches of 1024, 200 MiB left free)")
    say(f"  other contexts for this model (setup --context N):")
    table(rows)
    rows = [("model", "file", "quality: KLD vs Q8 at 2K", f"experts in VRAM at {ctx // 1024}K",
             f"write tok/s at 8K / {ctx // 1024}K (measured)", "note")]
    for k in SIZES:
        s_, f_, l_ = plan(k, free, reserve_own, ctx, chosen["prompt_batch"])
        m8, mc = measured(k, gb, 8), measured(k, gb, ctx // 1024)
        rows.append((k + (" (this)" if k == key else ""), f"{MODELS[k]['size'] / 1e9:.1f} GB", f"{MEASURED[k]['kld']}",
                     "does not fit" if l_ > 0 else f"{f_} ({100 * f_ / N_EXPERTS:.0f}%)",
                     f"{m8[1][1] if m8 else '-'} / {mc[1][1] if mc else '-'}", quality_note(k)))
    say(f"  other models at this context (setup --model NAME; lower KLD is closer to Q8):")
    table(rows)
    say("  Speeds were measured on one PC (Core Ultra 7 265K, DDR5-5600); a slower CPU or RAM writes more slowly")
    say("  when many experts are outside VRAM. run.bat prints what the engine actually holds and its hit rate.")


def table(rows):
    w = [max(len(str(r[i])) for r in rows) for i in range(len(rows[0]))]
    for j, r in enumerate(rows):
        say("    " + "  ".join(str(x).ljust(w[i]) for i, x in enumerate(r)))
        if j == 0:
            say("    " + "  ".join("-" * w[i] for i in range(len(w))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=str(ROOT / "models"))
    ap.add_argument("--configure-only", action="store_true", help="choose the settings again; download nothing")
    ap.add_argument("--limit-ram-gb", type=float, help="settings as if the PC had this much RAM")
    ap.add_argument("--limit-vram-gb", type=float, help="settings for this much VRAM (the rest is left free)")
    ap.add_argument("--port", type=int, default=0, help=f"the server's port (default {DEFAULT_PORT})")
    ap.add_argument("--vision", choices=("off", "cpu"), default=None,
                    help="pictures in requests: cpu (default) runs the image encoder on the CPU - no VRAM, about "
                         "0.5 GB of RAM while pictures come; about 0.7 s for a 640 x 400 picture, 5.8 s for 1920 x "
                         "1200 on a Core Ultra 7 265K; off: text only")
    ap.add_argument("--vision-max-tokens", type=int, default=1024,
                    help="the most tokens a picture takes (a larger one is scaled down first); default 1024")
    ap.add_argument("--thinking", choices=("low", "medium", "high"), default="medium",
                    help="the thinking level for requests that do not send one (reasoning_effort); default medium. "
                         "With high the model often drafts whole files in its thinking and an agent's reply reaches "
                         "max_tokens; a request's own reasoning_effort still wins")
    ap.add_argument("--context", type=context_arg, default=0,
                    help="the context length instead of the chosen one: tokens, or 64k / 128k / 256k")
    ap.add_argument("--gpu", type=int, default=0, help="which NVIDIA GPU (nvidia-smi's index)")
    ap.add_argument("--config", default=str(ROOT / "config" / "inkling.json"))
    ap.add_argument("--model", choices=sorted(MODELS), default="iq4_xs",
                    help="iq4_xs (default: unsloth's UD-IQ4_XS GGUF), or unsloth's smaller q3_k_xl / q2_k_xl / "
                         "iq2_xxs (faster on 8 GB and 6 GB cards and at 256K, lower quality; checked at 2K only)")
    a = ap.parse_args()
    m = MODELS[a.model]
    # --vision: cpu unless asked otherwise; when not asked for, a missing encoder (a build without it) is text only
    vision_asked = a.vision is not None
    a.vision = a.vision or "cpu"
    if a.vision == "cpu" and not vision_exe():
        if vision_asked:
            fail("the image encoder is missing: bin\\strata-vision.exe (release package) or build-vision\\bin "
                 "(tools\\build_vision.bat)")
        say("note: no image encoder (bin\\strata-vision.exe): pictures stay off (tools\\build_vision.bat builds it)")
        a.vision = "off"
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
    model = models / m["file"]
    free = shutil.disk_usage(models).free / 1e9
    say(f"  free disk space at {models}: {free:.1f} GB")
    exe = engine_exe()
    if not exe:
        fail("the engine is missing: bin\\strata-inkling2.exe (release package) or build\\bin (tools\\build.bat)")

    if not a.configure_only:
        say("== The model")
        size, want = m["size"], m["sha256"]
        part_path = model.with_suffix(model.suffix + ".part")
        if model.exists() and model.stat().st_size == size and (models / (m["file"] + ".ok")).exists():
            say(f"  {model} is there and was checked before")
        else:
            if not (model.exists() and model.stat().st_size == size):
                need = (size - (part_path.stat().st_size if part_path.exists() else 0)) / 1e9 + 1
                if free < need:
                    fail(f"{need:.1f} GB of free disk space is needed at {models}")
                download(model, m)
            say(f"  checking the SHA-256 of {model.name} (a minute or two)")
            got = sha256(model, size)
            if got != want:
                fail(f"SHA-256 mismatch: {got}, expected {want}. Delete {model} and run setup again.")
            (models / (m["file"] + ".ok")).write_text(want)
            say("  SHA-256 OK")
        say("== The tokenizer")
        tok = ROOT / "data" / "tokenizer"
        if not (tok / "vocab.json").exists():
            (ROOT / "data").mkdir(exist_ok=True)
            src = model if model.suffix == ".gguf" else models / MODEL_FILE   # every GGUF carries the tokenizer
            if not (src.exists() and src.stat().st_size in (MODEL_SIZE, m["size"])):   # a model without a GGUF: the IQ4_XS GGUF's metadata alone
                src = models / (MODEL_FILE + ".header")
                if not (src.exists() and src.stat().st_size == HEADER_BYTES):
                    fetch_header(src)
            subprocess.run([sys.executable, "-I", str(ROOT / "server" / "tools" / "strata_tokenizer.py"), "--gguf",
                            str(src), "--out", str(ROOT / "data"), "--check"], check=True)
        say(f"  {tok}")
        if a.vision == "cpu":
            say("== The image encoder")
            mp = models / VISION_MMPROJ["file"]
            if not (mp.exists() and mp.stat().st_size == VISION_MMPROJ["size"] and (models / (mp.name + ".ok")).exists()):
                if not (mp.exists() and mp.stat().st_size == VISION_MMPROJ["size"]):
                    download(mp, VISION_MMPROJ)
                if sha256(mp, VISION_MMPROJ["size"]) != VISION_MMPROJ["sha256"]:
                    fail(f"SHA-256 mismatch for {mp.name}. Delete it and run setup again.")
                (models / (mp.name + ".ok")).write_text(VISION_MMPROJ["sha256"])
            say(f"  {mp}")
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
        if a.context > 131072 and min(vram_limit, g["vram_mib"]) / 1024 >= 11.5:
            # 256K on 12 GB (IQ2_XXS, measured): prompt batches of 1024 and 200 MiB left free put 1903 experts in
            # VRAM in place of 1485 (2048 / 400): writing 31.9 -> 36.9 tok/s, the last 8K read 1008 -> 956 tok/s
            for flag, key, val in (("--prompt-batch", "prompt_batch", 1024), ("--vram-reserve-mib", "vram_reserve_mib", 200)):
                args[args.index(flag) + 1] = str(val)
                chosen[key] = val
    for k, v in chosen.items():
        say(f"  {k}: {v}")
    say(f"  thinking: {a.thinking} (for requests that send no reasoning_effort)")
    expect(a.model, g, min(vram_limit, g["vram_mib"]), chosen)
    cfg_path = Path(a.config)
    old = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    port = a.port or old.get("port") or DEFAULT_PORT
    cfg = {"exe": str(exe), "cwd": str(ROOT), "tokenizer": str(ROOT / "data" / "tokenizer"),
           "model_name": "inkling-2-mini", "display_name": m["display"], "model_choice": a.model,
           "args": ["--model", str(model)] + args,
           "host": "127.0.0.1", "port": port, "log": str(ROOT / "logs" / "engine.log"),
           "sampling": {"temperature": 1.0}, "reasoning_effort": a.thinking, "chosen": chosen,
           "limits": {"ram_gb": a.limit_ram_gb, "vram_gb": a.limit_vram_gb}}
    if a.gpu:
        cfg["gpu"], cfg["gpus_asked"] = a.gpu, True   # the server tells the engine this card
    if a.vision == "cpu":
        vexe, mp = vision_exe(), models / VISION_MMPROJ["file"]
        if not mp.exists():
            if vision_asked:
                fail(f"{mp} is missing; run setup.bat --vision cpu without --configure-only first")
            say(f"  note: {mp.name} is missing (setup without --configure-only fetches it): pictures stay off")
            vexe = None
        # on the CPU: it starts with the first picture and stops after two idle minutes (the server's default for a
        # CPU encoder), so its RAM is taken only while pictures come
        if vexe:
            cfg["vision"] = {"exe": str(vexe), "mmproj": str(mp), "model": str(model), "threads": chosen["threads"],
                             "max_tokens": a.vision_max_tokens}
            say(f"  vision: on the CPU ({chosen['threads']} threads, up to {a.vision_max_tokens} tokens a picture; "
                "setup --vision off for text only)")
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    say(f"  wrote {cfg_path}")
    say("")
    say(f"Done. Start the server with run.bat; it listens on http://127.0.0.1:{port}/v1 (OpenAI-compatible).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
