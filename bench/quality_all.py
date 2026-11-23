"""品質確認タスク (無人で回す): 暫定採用の構成ごとに 2K と 128K の品質確認を順に実行し、1 つのレポートにまとめる。

  2K    bench/quality.py と同じ照合 (llama.cpp の参照ロジット bench/data/ref_logits.bin、後半 1023 位置)
  128K  120K を超える文脈の最後の 1023 位置を、llama.cpp (690b87d、KV fp16) の参照ロジットと照合する。参照は
        ref/llama.cpp の llama-perplexity に LLAMA_PPL_TAIL (tools/llamacpp_ppl_tail.patch、このフォークのローカル変更:
        チャンクの最後の N 位置だけを保存) を足したもので、初回に 1 回だけ作り bench/data/ref128k_logits.bin に置く。
        エキスパートは全部 CPU で、--no-op-offload (llama.cpp 690b87d は CPU に置いたエキスパートの重みを読み込み中に GPU へ
        写して計算する経路で、層 14 以降を CPU に置くと NaN になった。2K の参照 PPL 3.5162 に対しこの設定で 3.5187)

構成は bench/data/quality_configs.json (名前、kv、環境変数、エンジンの設定、model: 別のモデルのファイル、group: 順番)。
q8_0 (既定の IQ4_XS) は比較のため必ず入れる (KV q4 などが不合格だったときに、同じ確認の q8_0 と並べて見られるように)。

2K は 2 つの参照と照合する: llama.cpp の IQ4_XS (KV fp16、bench/data/ref_logits.bin) と、unsloth の Q8_K_XL
(models/Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf がある場合。同じ文章・同じトークンで初回に作り
bench/data/ref_q8_logits.bin に置く)。Q8 の方は量子化そのものの差も入るので、IQ4_XS / SQ-10 / SQ-15 を同じ物差しで並べる用。

順番 (時間が足りないときに後ろが飛ぶ): 128K の参照、既定の 128K、Q8 の 2K 参照、各構成の 2K (group 順)、
近似オプション (group "option") の 128K、SQ (group "sq") の 128K。--deadline HH:MM を付けると、目安の所要時間で
その時刻に間に合わない手順は飛ばして記録する。

判定
  2K    bench/data/quality_baseline.json から KLD 平均 2 倍以内、上位1一致の低下 0.5 ポイント以内
  128K  同じ実行の q8_0 の 128K の値から同じ基準 (q8_0 自身は参照との差を記録するだけ)

途中で止まっても、同じコマンドをもう一度実行すれば終わった手順を飛ばして続きから再開する (--fresh で最初から)。
進み具合と結果は bench/quality_runs/<名前>/run.log に時刻付きで出る。最後の行が「完了」か「失敗」。
レポートは bench/quality_runs/<名前>/report.md。所要時間の目安は開始時と手順ごとにログに出る。

使い方:
  python -I bench/quality_all.py                 全部 (初回は 128K の参照作りを含む)
  python -I bench/quality_all.py --estimate      所要時間の目安だけ出す
  python -I bench/quality_all.py --smoke         仕組みの確認: 4K 文脈の小さな参照で 128K の手順を通す (数分)
"""
import argparse
import datetime
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import bench  # noqa: E402

ROOT = bench.ROOT
DATA = ROOT / "bench" / "data"
RUNS = ROOT / "bench" / "quality_runs"
DUMP = bench.BUILD / "inkling2_logits_dump.exe"
PPL = ROOT / "ref" / "llama.cpp" / "build" / "bin" / "llama-perplexity.exe"
REF_MODEL = ROOT / "models" / "Mainline_Support" / "Inkling-2-Mini-UD-IQ4_XS.gguf"
Q8_MODEL = ROOT / "models" / "Mainline_Support" / "Inkling-2-Mini-UD-Q8_K_XL.gguf"
REF_Q8 = DATA / "ref_q8_logits.bin"
CONFIGS = DATA / "quality_configs.json"
TIMES = DATA / "quality_times.json"   # 前回までに測った手順ごとの所要時間 (目安に使う)
TAIL = 1023                            # 照合する位置の数 (2K と同じ)
VOCAB = 200058
DEFAULT_TIMES = {"ref128k": 7200, "q2k": 120, "q128k": 420, "ref_smoke": 120, "q_smoke": 90, "ref2k_q8": 600}

DEFAULT_CONFIGS = [
    {"name": "q8_0", "kv": "q8_0", "env": {}, "extra": [],
     "note": "現在の既定 (KV q8_0、スピン 200000)。比較の基準として必ず入れる"},
]


class Run:
    def __init__(self, name, fresh):
        self.dir = RUNS / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        if fresh and self.state_path.exists():
            self.state_path.unlink()
        self.state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {}
        self.log_path = self.dir / "run.log"

    def log(self, msg):
        line = f"{datetime.datetime.now().isoformat(timespec='seconds')}  {msg}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def done(self, key):
        return key in self.state

    def put(self, key, value):
        self.state[key] = value
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.state_path)


def load_times():
    t = dict(DEFAULT_TIMES)
    if TIMES.exists():
        t.update(json.loads(TIMES.read_text(encoding="utf-8")))
    return t


def save_time(kind, seconds):
    t = json.loads(TIMES.read_text(encoding="utf-8")) if TIMES.exists() else {}
    t[kind] = round(seconds)
    TIMES.write_text(json.dumps(t, ensure_ascii=False, indent=1), encoding="utf-8")


def configs():
    if not CONFIGS.exists():
        CONFIGS.write_text(json.dumps(DEFAULT_CONFIGS, ensure_ascii=False, indent=1), encoding="utf-8")
    cs = json.loads(CONFIGS.read_text(encoding="utf-8"))
    if not any(c["kv"] == "q8_0" and not c.get("env") and not c.get("extra") for c in cs):
        cs.insert(0, DEFAULT_CONFIGS[0])
    return cs


def group(c):
    return c.get("group", "base")


def steps(cs, smoke):
    """(key, kind, config or None) in order: the default's long check first, then every 2K, the options' long
    checks, the SQ models' long checks (the later ones are the first to go when time runs out)."""
    if smoke:
        out = [("ref_smoke", "ref_smoke", None)]
        return out + [(f"q_smoke:{c['name']}", "q_smoke", c) for c in cs]
    out = [("ref128k", "ref128k", None)]
    base = [c for c in cs if group(c) == "base"]
    out += [(f"q128k:{c['name']}", "q128k", c) for c in base[:1]]
    if Q8_MODEL.exists():
        out.append(("ref2k_q8", "ref2k_q8", None))
    order = {"base": 0, "sq": 1, "option": 2}
    out += [(f"q2k:{c['name']}", "q2k", c) for c in sorted(cs, key=lambda c: order.get(group(c), 3))]
    out += [(f"q128k:{c['name']}", "q128k", c) for c in base[1:]]
    out += [(f"q128k:{c['name']}", "q128k", c) for c in cs if group(c) == "option"]
    out += [(f"q128k:{c['name']}", "q128k", c) for c in cs if group(c) == "sq"]
    return out


def fmt_s(s):
    s = int(s)
    return f"{s // 3600} 時間 {s % 3600 // 60} 分" if s >= 3600 else f"{s // 60} 分 {s % 60} 秒"


def ref_text(min_chars):
    """llama.cpp の文書 (MIT License) をつないだ文章。足りなければ繰り返す。"""
    docs = ROOT / "ref" / "llama.cpp"
    files = sorted(docs.glob("docs/*.md")) + sorted(docs.glob("docs/*/*.md")) + sorted(docs.glob("tools/*/README.md"))
    parts = [f.read_text(encoding="utf-8", errors="replace") for f in files]
    text = "\n\n".join(parts)
    while len(text) < min_chars:
        text += "\n\n" + "\n\n".join(parts)
    return text


def make_ref(run, smoke):
    """llama.cpp の参照: n_ctx トークンの 1 チャンク、最後の TAIL 位置のロジット。"""
    n_ctx = 4096 if smoke else 124928   # 128K 文脈 (エンジンの --max-context 132096 に収まる)
    out = DATA / ("ref_smoke_logits.bin" if smoke else "ref128k_logits.bin")
    if out.exists() and out.stat().st_size > 1000:
        run.log(f"参照は作成済み: {out}")
        return {"path": str(out), "n_ctx": n_ctx}
    src = DATA / ("ref_smoke_src.txt" if smoke else "ref128k_src.txt")
    src.write_text(ref_text(n_ctx * 6), encoding="utf-8")   # 1 トークン 3〜4 文字。余るぶんは使われない
    tmp = out.with_suffix(".part")
    env = dict(os.environ, LLAMA_PPL_TAIL=str(TAIL))
    # -ub 128: llama.cpp (690b87d) のこのモデルのアテンションは KQ を (ubatch x 文脈 x ヘッド) の行列で持つので、128K で
    # 既定の 512 は 12 GB の計算バッファになり VRAM に入らない
    cmd = [str(PPL), "-m", str(REF_MODEL), "-f", str(src), "-c", str(n_ctx), "--chunks", "1", "-b", "2048", "-ub", "128",
           "-ngl", "99", "--n-cpu-moe", "24", "--no-op-offload", "-t", str(bench.THREADS), "-fa", "on", "--kl-divergence-base", str(tmp)]
    run.log("llama.cpp で参照を作る: " + " ".join(cmd[1:]))
    with open(run.dir / "ref.log", "w", encoding="utf-8") as f:
        subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=4 * 3600)
    tmp.replace(out)
    return {"path": str(out), "n_ctx": n_ctx}


def make_ref_q8(run):
    """Q8_K_XL の 2K 参照: ref_logits.bin と同じ文章 (ref_text_src.txt) の 2048 トークン 1 チャンク。トークンが
    ref_logits.bin と同じことを確かめてから使う。"""
    if REF_Q8.exists() and REF_Q8.stat().st_size > 1000:
        run.log(f"Q8 の参照は作成済み: {REF_Q8}")
    else:
        tmp = REF_Q8.with_suffix(".part")
        cmd = [str(PPL), "-m", str(Q8_MODEL), "-f", str(DATA / "ref_text_src.txt"), "-c", "2048", "--chunks", "1",
               "-b", "2048", "-ngl", "99", "--n-cpu-moe", "24", "--no-op-offload", "-t", str(bench.THREADS), "-fa", "on",
               "--kl-divergence-base", str(tmp)]
        run.log("llama.cpp で Q8 の参照を作る: " + " ".join(cmd[1:]))
        with open(run.dir / "ref_q8.log", "w", encoding="utf-8") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=2 * 3600)
        tmp.replace(REF_Q8)
    toks = []
    for ref in (DATA / "ref_logits.bin", REF_Q8):
        out = run.dir / (ref.stem + ".tokens.txt")
        subprocess.run([sys.executable, "-I", str(ROOT / "bench/compare_logits.py"), "--ref", str(ref),
                        "--tokens-out", str(out)], check=True, capture_output=True)
        toks.append(out.read_text().strip())
    if toks[0] != toks[1]:
        raise RuntimeError("Q8 の参照のトークンが IQ4_XS の参照と違う")
    return {"path": str(REF_Q8), "n_ctx": 2048}


def ref_tokens(ref):
    tok = pathlib.Path(ref["path"]).with_suffix(".tokens.txt")
    if not tok.exists():
        subprocess.run([sys.executable, "-I", str(ROOT / "bench/compare_logits.py"), "--ref", ref["path"],
                        "--ref-tail", str(TAIL), "--tokens-out", str(tok)], check=True, capture_output=True)
    return [int(x) for x in tok.read_text().split(",")]


def model_of(c):
    return pathlib.Path(c["model"]) if c.get("model") else bench.MODEL


def dump(run, c, prompt_file, decode, out_dir, max_context, timeout):
    env = dict(os.environ)
    env.update({k: str(v) for k, v in c.get("env", {}).items()})
    cmd = [str(DUMP), str(model_of(c)), "--tokens-file", str(prompt_file), "--decode", decode, "--vocab", str(VOCAB),
           "--out", str(out_dir), "--threads", str(bench.THREADS), "--kv", c["kv"], "--prompt-batch", "2048",
           "--ram-experts-mib", "9216", "--vram-reserve-mib", "400", "--max-context", str(max_context)] + c.get("extra", [])
    with open(out_dir / "dump.log", "w", encoding="utf-8") as f:
        subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True, timeout=timeout)


def compare(ref_path, engine, first, tail):
    args = [sys.executable, "-I", str(ROOT / "bench/compare_logits.py"), "--ref", str(ref_path), "--engine", str(engine),
            "--engine-vocab", str(VOCAB), "--engine-first", str(first)]
    if tail:
        args += ["--ref-tail", str(tail)]
    res = subprocess.run(args, check=True, capture_output=True, text=True, encoding="utf-8").stdout
    ppl = re.search(r"PPL 参照 ([0-9.]+)  エンジン ([0-9.]+)", res)
    return {"kld": float(re.search(r"KLD 平均 ([0-9.]+)", res).group(1)),
            "kld99": float(re.search(r"99% ([0-9.]+)", res).group(1)),
            "top1": float(re.search(r"上位1一致 ([0-9.]+)", res).group(1)),
            "ppl_ref": float(ppl.group(1)), "ppl_engine": float(ppl.group(2))}


def q2k(run, c):
    out = run.dir / f"work_2k_{safe(c['name'])}"
    out.mkdir(exist_ok=True)
    decode = (DATA / "ref_decode.txt").read_text().strip()
    dump(run, c, DATA / "ref_prompt1024.txt", decode, out, 4096, 1800)
    r = compare(DATA / "ref_logits.bin", out / "decode_logits.f32", 1024, 0)
    if run.done("ref2k_q8"):   # the same logits against Q8_K_XL's
        r["q8"] = compare(REF_Q8, out / "decode_logits.f32", 1024, 0)
    (out / "decode_logits.f32").unlink()
    return r


def q128k(run, c, ref, smoke):
    toks = ref_tokens(ref)
    n_ctx = ref["n_ctx"]
    first = n_ctx - 1 - TAIL
    out = run.dir / f"work_{'smoke' if smoke else '128k'}_{safe(c['name'])}"
    out.mkdir(exist_ok=True)
    prompt = out / "prompt.txt"
    prompt.write_text(",".join(map(str, toks[:first])))
    decode = ",".join(map(str, toks[first:first + TAIL]))
    dump(run, c, prompt, decode, out, 8192 if smoke else 132096, run.timeout or 3 * 3600)
    r = compare(ref["path"], out / "decode_logits.f32", first, TAIL)
    (out / "decode_logits.f32").unlink()
    return r


def safe(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def verdict(r, base):
    if not base:
        return ""
    ok = r["kld"] <= 2.0 * base["kld"] and r["top1"] >= base["top1"] - 0.5
    return (f"{'合格' if ok else '不合格'} (KLD {r['kld'] / base['kld']:.2f} 倍、上位1一致 {r['top1'] - base['top1']:+.2f})")


def report(run, cs, smoke):
    b2 = DATA / "quality_baseline.json"
    base2 = None
    if b2.exists():
        j = json.loads(b2.read_text(encoding="utf-8"))
        base2 = {"kld": j["kld_mean"], "top1": j["top1_pct"]}
    long_key = "q_smoke" if smoke else "q128k"
    q8 = next((c for c in cs if c["kv"] == "q8_0" and not c.get("env") and not c.get("extra")), None)
    base_long = run.state.get(f"{long_key}:{q8['name']}") if q8 else None
    lines = [f"# 品質確認レポート ({run.dir.name})", "",
             f"- 作成 {datetime.datetime.now().isoformat(timespec='seconds')}、エンジン {bench.commit()}",
             "- 2K: llama.cpp 参照 (KV fp16) との照合、判定は bench/data/quality_baseline.json "
             f"(KLD {base2['kld'] if base2 else '?'}、上位1一致 {base2['top1'] if base2 else '?'}%) から KLD 2 倍以内・上位1一致 -0.5 以内",
             f"- {'4K (仕組みの確認)' if smoke else '128K'}: 文脈 {run.state.get('ref_smoke' if smoke else 'ref128k', {}).get('n_ctx', '?')} "
             f"トークンの最後の {TAIL} 位置を llama.cpp 参照 (KV fp16) と照合、判定は同じ実行の q8_0 の値から同じ基準", "",
             "| 構成 | 2K KLD | 2K 上位1 | 2K PPL (参照/エンジン) | 2K 判定 | 長文 KLD | 長文 KLD 99% | 長文 上位1 | 長文 PPL (参照/エンジン) | 長文 判定 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for c in cs:
        r2 = run.state.get(f"q2k:{c['name']}")
        rl = run.state.get(f"{long_key}:{c['name']}")
        cell = lambda r, k, f="{}": f.format(r[k]) if r else "未実行"
        lines.append(" | ".join([
            f"| {c['name']}",
            cell(r2, "kld", "{:.5f}"), cell(r2, "top1", "{:.2f}%"),
            f"{r2['ppl_ref']:.4f} / {r2['ppl_engine']:.4f}" if r2 else "未実行", verdict(r2, base2) if r2 else "",
            cell(rl, "kld", "{:.5f}"), cell(rl, "kld99", "{:.5f}"), cell(rl, "top1", "{:.2f}%"),
            f"{rl['ppl_ref']:.4f} / {rl['ppl_engine']:.4f}" if rl else "未実行",
            ("基準" if c is q8 else verdict(rl, base_long)) if rl else ""]) + " |")
    if any((run.state.get(f"q2k:{c['name']}") or {}).get("q8") for c in cs):
        lines += ["", "## 2K、unsloth Q8_K_XL の参照との照合 (同じトークン、llama.cpp、KV fp16)", "",
                  "| 構成 | KLD | KLD 99% | 上位1一致 | PPL (参照/エンジン) |", "|---|---|---|---|---|"]
        for c in cs:
            r = (run.state.get(f"q2k:{c['name']}") or {}).get("q8")
            lines.append(f"| {c['name']} | " + (f"{r['kld']:.5f} | {r['kld99']:.5f} | {r['top1']:.2f}% | "
                                                 f"{r['ppl_ref']:.4f} / {r['ppl_engine']:.4f} |" if r else "未実行 | | | |"))
    skipped = run.state.get("_skipped", [])
    if skipped:
        lines += ["", "飛ばした手順 (締め切りに間に合わない見込み): " + "、".join(skipped)]
    lines += ["", "構成の中身:", ""] + [f"- {c['name']}: モデル {model_of(c).name}、kv {c['kv']}、環境変数 {c.get('env') or 'なし'}、"
                                     f"設定 {' '.join(c.get('extra', [])) or 'なし'}。{c.get('note', '')}" for c in cs]
    (run.dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run.dir / "report.md"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="実行の名前 (既定: main、--smoke なら smoke)。同じ名前で再開する")
    ap.add_argument("--fresh", action="store_true", help="終わった手順も最初からやり直す")
    ap.add_argument("--smoke", action="store_true", help="4K 文脈の小さな参照で 128K の手順の仕組みだけ確かめる")
    ap.add_argument("--estimate", action="store_true", help="所要時間の目安だけ出す")
    ap.add_argument("--deadline", default=None, help="HH:MM (この PC の時刻)。間に合わない見込みの手順は飛ばす")
    a = ap.parse_args()
    run = Run(a.run or ("smoke" if a.smoke else "main"), a.fresh)
    run.timeout = None
    deadline = None
    if a.deadline:
        hh, mm = map(int, a.deadline.split(":"))
        now = datetime.datetime.now()
        deadline = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if deadline < now:
            deadline += datetime.timedelta(days=1)
    cs = configs()
    todo = steps(cs, a.smoke)
    times = load_times()
    for key, k, c in todo:   # a config's own measured time, where there is one
        if c and group(c) != "base" and f"{k}:{c['name']}" in times:
            times[key] = times[f"{k}:{c['name']}"]
    left = sum(times.get(key, times[k]) for key, k, _ in todo if not run.done(key))
    if a.estimate:
        print(f"手順 {len(todo)} 個 (残り {sum(1 for key, _, _ in todo if not run.done(key))} 個)、目安 {fmt_s(left)}")
        for key, k, _ in todo:
            print(f"  {'済み' if run.done(key) else fmt_s(times.get(key, times[k])):>12}  {key}")
        return 0
    run.log(f"開始: 構成 {', '.join(c['name'] for c in cs)}。手順 {len(todo)} 個、残りの目安 {fmt_s(left)}")
    bench.stop_server()   # 常駐サーバーの VRAM を空ける
    ref = run.state.get("ref_smoke" if a.smoke else "ref128k")
    try:
        for key, kind, c in todo:
            if run.done(key):
                continue
            if deadline is not None:
                room = (deadline - datetime.datetime.now()).total_seconds()
                est = times.get(key, times[kind])
                if est > room:
                    run.log(f"手順 {key} を飛ばす: 目安 {fmt_s(est)}、締め切り {a.deadline} まで {fmt_s(max(0, room))}")
                    run.put("_skipped", run.state.get("_skipped", []) + [key])
                    left -= est
                    continue
                run.timeout = max(60, int(room))
            run.log(f"手順 {key} を始める (目安 {fmt_s(times.get(key, times[kind]))}、残りの目安 {fmt_s(max(0, left))}、"
                    f"終わりの見込み {(datetime.datetime.now() + datetime.timedelta(seconds=max(0, left))).strftime('%H:%M')})")
            t0 = time.time()
            if kind == "ref2k_q8":
                r = make_ref_q8(run)
            elif kind in ("ref128k", "ref_smoke"):
                ref = make_ref(run, a.smoke)
                r = ref
            elif kind == "q2k":
                r = q2k(run, c)
            else:
                r = q128k(run, c, ref, a.smoke)
            dt = time.time() - t0
            # the SQ models' and the options' times are kept apart from the default's (their speeds differ)
            tkind = kind if not c or group(c) == "base" else f"{kind}:{c['name']}"
            save_time(tkind, dt)
            run.put(key, r)
            left -= times.get(key, times[kind])
            run.log(f"手順 {key} が終わった ({fmt_s(dt)}): {json.dumps(r, ensure_ascii=False)}。残りの目安 {fmt_s(max(0, left))}")
            report(run, cs, a.smoke)
    except BaseException as e:
        run.log(f"失敗: {type(e).__name__}: {e}。同じコマンドで続きから再開できる")
        with open(run.log_path, "a", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        report(run, cs, a.smoke)
        return 1
    path = report(run, cs, a.smoke)
    run.log(f"完了: レポート {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
