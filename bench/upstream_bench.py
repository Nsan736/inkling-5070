"""本家 Strata の strata-inkling.exe (--serve の行プロトコル) を、bench.py と同じトークン列・同じ数え方で測る。

  python -I bench/upstream_bench.py --exe ref/strata-v0220-build/strata-inkling.exe --model models/X.gguf
      [--long] [--full128k] [--dspark PATH] [--reps 2] [--label 名前] [-- 本家への追加の引数]

- 生成: GEN 216 を送り、T の行が届いた時刻で 17〜216 トークン目の 200 トークンを数える (bench.py と同じ区間)。
  本家の GEN には ignore_eos がないので、EOS で早く終わったときはそのトークン数を記録する
- 温度 1.0 は temperature=1.0 top_k=128 top_p=1 min_p=0 seed=1234 (本家の top_k の上限は 128。0 は 1 になる)
- 8K 読み込み: 先頭に毎回違うトークンを置いた prefill_8k.ids を GEN 1 で読む
- --long: long_code_base.ids (120K) を読み、続けて tail (8K) だけを読む速さ、その後 128K での生成
- --full128k: 128K (base + tail) を最初から読む速さ
結果は 1 行の JSON で bench/logs/upstream_results.jsonl に追記。本家の stderr は bench/logs/upstream_engine.log。
"""
import argparse
import json
import pathlib
import queue
import subprocess
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "bench" / "data"
LOGS = ROOT / "bench" / "logs"
SKIP, COUNT = 16, 200
GREEDY = "temperature=0"
T1 = "temperature=1.0 top_k=128 top_p=1 min_p=0 seed=1234"


def ids(name):
    return [int(x) for x in (DATA / name).read_text().replace(",", " ").split()]


class Engine:
    def __init__(self, args, log="upstream_engine.log"):
        LOGS.mkdir(exist_ok=True)
        self.err = open(LOGS / log, "ab")
        self.err.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(args)}\n".encode())
        self.err.flush()
        self.p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, bufsize=0)
        self.q = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        t0 = time.time()
        while True:
            t, line = self.get(1800)
            if line.startswith("READY"):
                self.ready = line
                self.load_s = time.time() - t0
                break

    def _pump(self):
        for raw in self.p.stdout:
            self.q.put((time.perf_counter(), raw.decode("utf-8", "replace").rstrip("\r\n")))
        self.q.put((time.perf_counter(), None))

    def get(self, timeout):
        t, line = self.q.get(timeout=timeout)
        if line is None:
            raise RuntimeError("the engine ended (bench/logs/upstream_engine.log)")
        return t, line

    def gen(self, max_new, sampling, prompt, timeout=3600):
        self.p.stdin.write(f"GEN {max_new} {sampling} {','.join(map(str, prompt))}\n".encode())
        self.p.stdin.flush()
        stamps = []
        while True:
            t, line = self.get(timeout)
            if line.startswith("T "):
                stamps.append(t)
            elif line.startswith("DONE"):
                p = line.split()
                done = {"produced": int(p[1]), "prompt": int(p[2]), "prompt_ms": float(p[3]),
                        "decode_ms": float(p[4]), "finish": p[5], "drafts_right": int(p[6]), "drafts": int(p[7]),
                        "reused": int(p[8]) if len(p) > 8 else 0}
                return stamps, done
            elif line.startswith("ERR"):
                raise RuntimeError(line)

    def quit(self):
        try:
            self.p.stdin.write(b"QUIT\n")
            self.p.stdin.flush()
            self.p.wait(30)
        except Exception:
            self.p.kill()


def write_rate(stamps):
    """tok/s over tokens 17..216 (or what came, from token 17)."""
    if len(stamps) > SKIP + 1:
        last = min(len(stamps), SKIP + COUNT) - 1
        return (last - (SKIP - 1)) / (stamps[last] - stamps[SKIP - 1]), len(stamps)
    return None, len(stamps)


def read_rate(done):
    n = done["prompt"] - done["reused"]
    return 1000.0 * n / done["prompt_ms"] if done["prompt_ms"] > 0 else None, n


def nvsmi():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return int(out or 0)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--long", action="store_true")
    ap.add_argument("--full128k", action="store_true")
    ap.add_argument("--dspark", default=None)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--label", default="")
    ap.add_argument("--t1-only", action="store_true", help="生成は温度 1.0 だけ")
    a = ap.parse_args(argv)
    a.exe, a.model = str(pathlib.Path(a.exe).resolve()), str(pathlib.Path(a.model).resolve())
    if a.dspark:
        a.dspark = str(pathlib.Path(a.dspark).resolve())
    args = [a.exe, "--serve", "--native", a.model] + (["--dspark", a.dspark] if a.dspark else []) + extra
    res = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "label": a.label, "args": args[1:]}
    e = Engine(args)
    res["ready"], res["load_s"] = e.ready, round(e.load_s, 1)
    print(f"[{a.label}] {e.ready} ({e.load_s:.0f} s)", flush=True)
    nonce = int(time.time())
    try:
        if not a.long:
            e.gen(SKIP + COUNT, GREEDY, ids("gen.ids"))   # warm-up
            pre = []
            for _ in range(a.reps):
                _, d = e.gen(1, GREEDY, [1000 + nonce % 50000] + ids("prefill_8k.ids"))
                nonce += 1
                pre.append(read_rate(d)[0])
            res["prefill8k"] = pre
            prompt = ids("gen.ids")
        else:
            base, tail = ids("long_code_base.ids"), ids("long_code_tail.ids")
            _, d = e.gen(1, GREEDY, base)
            res["read120k"] = read_rate(d)
            _, d = e.gen(1, GREEDY, base + tail)
            res["read_tail"] = read_rate(d)
            prompt = base + tail
        res["vram_used_mib"] = nvsmi()
        for name, smp in ((("greedy", GREEDY),) if not a.t1_only else ()) + (("t1", T1),):
            runs = []
            for _ in range(a.reps):
                st, d = e.gen(SKIP + COUNT, smp, prompt)
                r, n = write_rate(st)
                runs.append({"tps": r, "tokens": n, "finish": d["finish"], "drafts": [d["drafts_right"], d["drafts"]],
                             "reused": d["reused"]})
            res[name] = runs
            print(f"  {name}: " + ", ".join(f"{x['tps']:.2f} tok/s ({x['tokens']} tok, {x['finish']}, drafts "
                                            f"{x['drafts'][0]}/{x['drafts'][1]})" if x["tps"] else f"{x['tokens']} tok"
                                            for x in runs), flush=True)
        if a.full128k:
            base, tail = ids("long_code_base.ids"), ids("long_code_tail.ids")
            _, d = e.gen(1, GREEDY, [1000 + nonce % 50000] + base[1:] + tail)
            res["read128k_full"] = read_rate(d)
        for k in ("prefill8k", "read120k", "read_tail", "read128k_full"):
            if k in res:
                print(f"  {k}: {res[k]}", flush=True)
    finally:
        e.quit()
    with open(LOGS / "upstream_results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
