"""Before a Ninja build with MSVC: make sure Ninja will record the headers each .cpp / .c file includes.

Ninja finds a source's headers in cl's /showIncludes lines by the prefix CMake saw when it configured the build
(rules.ninja's msvc_deps_prefix, "Note: including file:" in English).  cl writes that line in the language of
Visual Studio and in the console's code page: in Japanese "メモ: インクルード ファイル:" as cp932 under code page 932
and as UTF-8 under 65001.  When the build runs under another code page than the configure did, no line matches,
every object is recorded with no headers, and a change to a header rebuilds nothing that includes it.

So this script compares the prefix cl writes now with the one in rules.ninja.  If they differ, it removes CMake's
record of the C and C++ compilers (the next configure detects the prefix again) and every C / C++ object whose
recorded headers are none (Ninja builds them again, this time with their headers).  The build scripts set the
code page to UTF-8 (65001) first, so that the configure and every build see the same one.

  python tools/ninja_deps_check.py BUILD_DIR      (cl.exe on the PATH: run it after vcvars64.bat)
Exit 0: nothing to do or fixed; the caller then runs cmake (configure) and the build.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def cl_prefix() -> bytes | None:
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "probe_header.h").write_text("#define PROBE 0\n")
        (Path(d) / "probe.cpp").write_text('#include "probe_header.h"\nint probe() { return PROBE; }\n')
        out = subprocess.run(["cl", "/nologo", "/showIncludes", "/Zs", "probe.cpp"], cwd=d, capture_output=True).stdout
    for line in out.splitlines():
        i = line.find(b"probe_header.h")
        if i > 0:
            # the prefix is the line up to the header's path (a drive letter, or the path cl was given)
            m = re.match(rb"^(.*?:)\s+[A-Za-z]:\\", line)
            return (m.group(1) if m else line[:i]).rstrip()
    return None


def main() -> int:
    build = Path(sys.argv[1])
    rules = build / "CMakeFiles" / "rules.ninja"
    if not rules.exists():
        return 0   # not configured yet: the configure detects the prefix under this code page
    have = None
    for line in rules.read_bytes().splitlines():
        if line.startswith(b"msvc_deps_prefix = "):
            have = line[len(b"msvc_deps_prefix = "):].rstrip()
    want = cl_prefix()
    if want is None or have is None:
        print("ninja_deps_check: could not read cl's or Ninja's /showIncludes prefix; nothing changed")
        return 0
    if want == have:
        return 0
    print(f"ninja_deps_check: cl writes {want!r}, the build expects {have!r}: CMake detects the compilers again, "
          "and the objects recorded without headers are built again")
    for f in (build / "CMakeFiles").glob("*/CMake*Compiler.cmake"):
        if f.name in ("CMakeCCompiler.cmake", "CMakeCXXCompiler.cmake"):
            f.unlink()
    # the C / C++ objects whose recorded headers are none
    out = subprocess.run(["ninja", "-C", str(build), "-t", "deps"], capture_output=True, text=True,
                         errors="replace").stdout
    removed = 0
    for m in re.finditer(r"^(\S+\.(?:c|cpp|cc|cxx)\.obj): #deps (\d+)", out, re.M):
        if m.group(2) == "0":
            p = build / m.group(1)
            if p.exists():
                p.unlink()
                removed += 1
    print(f"ninja_deps_check: {removed} objects without headers removed")
    if shutil.which("cmake"):
        subprocess.run(["cmake", str(build)], check=True, capture_output=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
