"""strata-inkling2 --serve を常駐させ、127.0.0.1:18457 で行プロトコルを中継する小さなデーモン。

モデルの読み込みは起動時の 1 回だけ。bench.py は毎回ここにつなぎ、エンジンの stdin に行を書き、stdout の行を受け取る。
同時に使えるクライアントは 1 つ。エンジンの stderr は bench/logs/engine.log に追記する。

デーモン自身への命令 (エンジンには渡さない):
  @INFO   起動時の情報を 1 行の JSON で返す (pid, args, exe の更新時刻, warm 済みか)
  @WARMED warm 済みの印を付ける
  @QUIT   エンジンとデーモンを止める

起動は bench.py が行う (python bench/server.py -- エンジンの引数...)。
"""
import json
import os
import pathlib
import queue
import socket
import subprocess
import sys
import threading
import time

HOST, PORT = "127.0.0.1", 18457
ROOT = pathlib.Path(__file__).resolve().parent.parent
LOGS = ROOT / "bench" / "logs"


def main():
    args = sys.argv[sys.argv.index("--") + 1:]
    LOGS.mkdir(parents=True, exist_ok=True)
    err = open(LOGS / "engine.log", "ab")
    err.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(args)}\n".encode())
    err.flush()
    env = dict(os.environ, INKLING2_REPORT_CACHE="1")   # 回答ごとにエキスパートのキャッシュの大きさを stderr に出す
    eng = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, bufsize=0, env=env)
    lines = queue.Queue()

    def pump():
        for raw in eng.stdout:
            lines.put(raw.decode("utf-8", "replace").rstrip("\r\n"))
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()
    info = {"pid": eng.pid, "args": args, "exe_mtime": os.path.getmtime(args[0]), "warmed": False,
            "started": time.time(), "ready": None}
    # READY が来るまで待つ (来なければ終了)
    while True:
        line = lines.get()
        if line is None:
            sys.exit("engine exited before READY")
        if line.startswith("READY"):
            info["ready"] = line
            break
        if line.startswith("INFO"):
            info["engine_info"] = line

    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, PORT))
    srv.listen(16)   # 1 だと、前の接続を閉じている間に来た次の接続が断られる
    while eng.poll() is None:
        conn, _ = srv.accept()
        f = conn.makefile("rwb", buffering=0)
        stop = threading.Event()

        def relay():
            while not stop.is_set():
                try:
                    line = lines.get(timeout=0.2)
                except queue.Empty:
                    continue
                if line is None:
                    stop.set()
                    break
                try:
                    f.write((line + "\n").encode())
                except OSError:
                    stop.set()
                    lines.put(line)   # 次のクライアントに渡す
                    break

        t = threading.Thread(target=relay, daemon=True)
        t.start()
        try:
            for raw in f:
                cmd = raw.decode().rstrip("\r\n")
                if cmd == "@INFO":
                    f.write((json.dumps(info) + "\n").encode())
                elif cmd == "@WARMED":
                    info["warmed"] = True
                    f.write(b"@OK\n")
                elif cmd == "@QUIT":
                    eng.stdin.write(b"QUIT\n")
                    eng.stdin.flush()
                    try:
                        eng.wait(30)
                    except subprocess.TimeoutExpired:
                        eng.kill()
                    f.write(b"@BYE\n")
                    conn.close()
                    return
                else:
                    eng.stdin.write((cmd + "\n").encode())
                    eng.stdin.flush()
        except OSError:
            pass
        stop.set()
        t.join()
        conn.close()


if __name__ == "__main__":
    main()
