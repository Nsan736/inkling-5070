"""--kv k8v4 のセッションの保存と復元の確認 (開発用)。

エンジンで約 3,000 トークンを読んで状態を保存し、同じ文章を送って 48 トークンを greedy で書く。別のエンジンで復元し、
同じ文章を送って、同じ 48 トークンが出るか (と、読み直さずに使い回したか) を見る。どちらも最後の 1 トークンを
読み直す同じ経路になる。

使い方: python -I bench/kv4_session_check.py [--kv k8v4]
"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench as B           # noqa: E402
import upstream_bench as U  # noqa: E402

LOG = "kv4_session.log"


def gen_tokens(e, n, prompt):
    e.p.stdin.write(f"GEN {n} temperature=0 lookup=0 {','.join(map(str, prompt))}\n".encode())
    e.p.stdin.flush()
    toks = []
    while True:
        _, line = e.get(600)
        if line.startswith("T "):
            toks.append(int(line.split()[1]))
        elif line.startswith("DONE"):
            p = line.split()
            return toks, int(p[8]) if len(p) > 8 else 0
        elif line.startswith("ERR"):
            raise RuntimeError(line)


def command(e, cmd, want):
    e.p.stdin.write((cmd + "\n").encode())
    e.p.stdin.flush()
    while True:
        _, line = e.get(600)
        if line.startswith(want):
            return line
        if line.startswith("ERR"):
            raise RuntimeError(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv", default="k8v4")
    a = ap.parse_args()
    flags = list(B.BASE_FLAGS)
    flags[flags.index("--max-context") + 1] = "16384"
    args = [str(B.EXE), "--model", str(B.MODEL), "--serve"] + flags + ["--kv", a.kv]
    prompt = U.ids("long_code_base.ids")[:3000]
    path = Path(tempfile.gettempdir()) / f"kv4_check_{a.kv}.session"
    e = U.Engine(args, log=LOG)
    try:
        gen_tokens(e, 1, prompt)
        print(command(e, f"SAVE {path}", "SAVED"))
        first, _ = gen_tokens(e, 48, prompt)   # after the save: the same path as after the restore
        print("保存したエンジンの 48 トークン:", first[:12], "...")
    finally:
        e.quit()
    e = U.Engine(args, log=LOG)
    try:
        print(command(e, f"RESTORE {path}", "RESTORED"))
        path2 = path.with_suffix(".again")
        print(command(e, f"SAVE {path2}", "SAVED"))   # restored then saved: the same bytes as the first save
        same_file = path.read_bytes() == path2.read_bytes()
        print(f"復元してすぐ保存したファイル: {'同じ' if same_file else '違う'} ({path2.stat().st_size} バイト)")
        path2.unlink()
        again, reused = gen_tokens(e, 48, prompt)
        print(f"復元後の 48 トークン (使い回し {reused}):", again[:12], "...")
    finally:
        e.quit()
    path.unlink(missing_ok=True)
    same = first == again
    print(f"保存前と復元後: {'同じ' if same else '違う'} (一致 {sum(x == y for x, y in zip(first, again))}/48)、"
          f"使い回し {reused} / {len(prompt)}")
    return 0 if same_file else 1


if __name__ == "__main__":
    sys.exit(main())
