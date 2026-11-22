"""Nsight Systems で生成中の GPU を記録する (計測のみ)。

nsys は Windows で標準入力をエンジンに渡さないので、常駐サーバー (bench/server.py) ごと nsys の下で起動し、
TCP でコマンドを送る (エンジンは server.py の子プロセスとして記録される)。

使い方: python -I bench/profile_decode.py {8k|128k}   -> bench/nsys/g{8k|128k}.nsys-rep
"""
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bench  # noqa: E402

NSYS = r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64\nsys.exe"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    which = sys.argv[1]
    bench.stop_server()
    out = bench.ROOT / "bench" / "nsys" / f"g{which}"
    bench.ENGINE_FLAGS.extend(["--kv", "q8_0"])
    args = [str(bench.EXE), "--model", str(bench.MODEL), "--serve"] + bench.ENGINE_FLAGS
    prof = subprocess.Popen([NSYS, "profile", "-t", "cuda", "--cuda-graph-trace=node", "-s", "none", "--cpuctxsw=none",
                             "-f", "true", "-o", str(out), sys.executable, "-I", str(bench.ROOT / "bench" / "server.py"),
                             "--"] + args, stdout=open(str(out) + ".log", "w"), stderr=subprocess.STDOUT)
    t0 = time.time()
    while not bench.server_up():
        if prof.poll() is not None or time.time() - t0 > 300:
            raise SystemExit("サーバーが起動しない")
        time.sleep(1)
    c = bench.Conn()
    lines = (bench.ROOT / "bench" / "nsys" / f"cmd_{which}.txt").read_text().splitlines()
    for line in lines:
        if line == "QUIT":
            break
        c.send(line)
        want = ("RESTORED",) if line.startswith("RESTORE") else ("DONE",)
        got = c.read_until(want, 900)
        print(got[-1][:120], flush=True)
    c.close()
    bench.stop_server()
    prof.wait(timeout=900)
    print(f"記録: {out}.nsys-rep ({time.time() - t0:.0f} 秒)")


if __name__ == "__main__":
    main()
