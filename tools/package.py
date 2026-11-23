"""Make the Windows release package (a .zip for GitHub Releases) from a release build (tools\\build.bat).

The package holds bin\\ (strata-inkling2.exe, inkling2_logits_dump.exe, strata-vision.exe, the CUDA and C++ runtime DLLs they need),
the server, the scripts, the docs and the licenses.  The model is not in it: setup.bat downloads it.

  python tools/package.py [--version 0.3.1] [--build build]

Redistributed runtime files:
  cublas64_13.dll, cublasLt64_13.dll   NVIDIA CUDA Toolkit, redistributable under the CUDA EULA's Attachment A
  msvcp140*.dll, vcruntime140*.dll     Microsoft Visual C++ runtime, redistributable (VC\\Redist\\MSVC\\*\\x64)
The CUDA runtime itself (cudart) is linked statically.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXES = ("strata-inkling2.exe", "inkling2_logits_dump.exe")
CUDA_DLLS = ("cublas64_13.dll", "cublasLt64_13.dll")
CRT_DLLS = ("msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "msvcp140_atomic_wait.dll", "vcruntime140.dll",
            "vcruntime140_1.dll")
TREE = ("server", "scripts", "docs", "setup.bat", "run.bat", "requirements.txt", "README.md", "README.en.md", "LICENSE",
        "NOTICE", "UPSTREAM.md")


def cuda_bin():
    base = os.environ.get("CUDA_PATH") or ""
    for d in (Path(base) / "bin" / "x64", Path(base) / "bin"):
        if (d / CUDA_DLLS[0]).exists():
            return d
    sys.exit("CUDA_PATH's bin has no cublas64_13.dll: install the CUDA Toolkit 13.x")


def crt_dir():
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio" / \
        "Installer" / "vswhere.exe"
    vs = subprocess.run([str(vswhere), "-latest", "-products", "*", "-property", "installationPath"], capture_output=True,
                        text=True).stdout.strip()
    hits = sorted(glob.glob(os.path.join(vs, "VC", "Redist", "MSVC", "*", "x64", "Microsoft.VC*.CRT")))
    if not hits:
        sys.exit("Visual Studio's VC\\Redist\\MSVC\\*\\x64\\Microsoft.VC*.CRT was not found")
    return Path(hits[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="0.3.1")
    ap.add_argument("--build", default=str(ROOT / "build"))
    a = ap.parse_args()
    bin_src = Path(a.build) / "bin"
    for e in EXES:
        if not (bin_src / e).exists():
            sys.exit(f"{bin_src / e} is missing: run tools\\build.bat first")
    name = f"inkling-5070-{a.version}-windows-x64-cuda13"
    stage = ROOT / "dist" / name
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "bin").mkdir(parents=True)
    for e in EXES:
        shutil.copy2(bin_src / e, stage / "bin" / e)
    # the image encoder (toolsuild_vision.bat): setup's --vision cpu, the default
    vision = ROOT / "build-vision" / "bin" / "strata-vision.exe"
    if not vision.exists():
        sys.exit(f"{vision} is missing: run tools\build_vision.bat first (LLAMA_DIR: a llama.cpp checkout)")
    shutil.copy2(vision, stage / "bin" / vision.name)
    cb = cuda_bin()
    for d in CUDA_DLLS:
        shutil.copy2(cb / d, stage / "bin" / d)
    crt = crt_dir()
    for d in CRT_DLLS:
        if (crt / d).exists():
            shutil.copy2(crt / d, stage / "bin" / d)
    for t in TREE:
        src = ROOT / t
        if src.is_dir():
            shutil.copytree(src, stage / t, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        elif src.exists():
            shutil.copy2(src, stage / t)
    eula = Path(os.environ.get("CUDA_PATH", "")) / "EULA.txt"
    lic = stage / "licenses"
    lic.mkdir()
    if eula.exists():
        shutil.copy2(eula, lic / "CUDA-EULA.txt")
    for f in ("LICENSE",):
        shutil.copy2(ROOT / "third_party" / "llama.cpp" / f, lic / "llama.cpp-LICENSE.txt")
    shutil.copy2(ROOT / "engine" / "LICENSE.PolyStrata", lic / "PolyStrata-LICENSE.txt")
    shutil.copy2(ROOT / "third_party" / "zstd-dec" / "LICENSE", lic / "zstd-LICENSE.txt")   # its decoder is linked
    zpath = ROOT / "dist" / f"{name}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(stage.rglob("*")):
            if p.is_file():
                z.write(p, Path(name) / p.relative_to(stage))
    print(f"{zpath} ({zpath.stat().st_size / 2**20:.0f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
