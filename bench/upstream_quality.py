"""本家 Strata の strata-inkling.exe の 2K 品質を、自前エンジンと同じ参照・同じ位置で測る。

1. 本家の --dump-ref で、参照と同じ 2,048 トークン (bench/data/ref_tokens.txt) を teacher-forced で読ませ、
   全位置のロジットを書き出す (形式: int32 n, int32 V, int32 ids[n], float32 logits[n*V])
2. その後半 (位置 1024〜2046) を compare_logits.py で llama.cpp の参照 (IQ4_XS: ref_logits.bin、Q8_K_XL:
   ref_q8_logits.bin) と照合する (自前エンジンの quality_2k.py と同じ照合)
3. 確認として本家の --kld-eval も使う: llama.cpp の参照は後半の位置しか持たないので、前半を本家自身のロジット
   (KLD 0、上位1一致) で埋めた参照を作り、--kld-eval の全体の値から後半だけの値を逆算する

  python -I bench/upstream_quality.py --exe ref/strata-v0220-build/strata-inkling.exe --model models/X.gguf
      --name 名前 [-- 本家への追加の引数]
結果は bench/quality_runs/upstream/<名前>.json。作業ファイル (約 1.6 GB ずつ) は終わったら消す。
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
import compare_logits as cl  # noqa: E402

DATA = ROOT / "bench" / "data"
REFS = {"iq4_xs": DATA / "ref_logits.bin", "q8": DATA / "ref_q8_logits.bin"}


def compare(ref, engine_raw, vocab):
    out = subprocess.run([sys.executable, "-I", str(ROOT / "bench/compare_logits.py"), "--ref", str(ref),
                          "--engine", str(engine_raw), "--engine-vocab", str(vocab), "--engine-first", "0"],
                         check=True, capture_output=True, text=True, encoding="utf-8").stdout
    ppl = re.search(r"PPL 参照 ([0-9.]+)  エンジン ([0-9.]+)", out)
    return {"kld": float(re.search(r"KLD 平均 ([0-9.]+)", out).group(1)),
            "kld99": float(re.search(r"99% ([0-9.]+)", out).group(1)),
            "top1": float(re.search(r"上位1一致 ([0-9.]+)", out).group(1)),
            "ppl_ref": float(ppl.group(1)), "ppl_engine": float(ppl.group(2))}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args(argv)
    exe, model = str(pathlib.Path(a.exe).resolve()), str(pathlib.Path(a.model).resolve())
    out_dir = ROOT / "bench" / "quality_runs" / "upstream"
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / f"work_{a.name}"
    work.mkdir(exist_ok=True)
    toks = (DATA / "ref_tokens.txt").read_text().strip()
    dump = work / "dump.bin"
    log = open(work / "dump.log", "w", encoding="utf-8")
    subprocess.run([exe, "--dump-ref", str(dump), "--native", model, "--prompt-ids", toks, "--max-context", "4096"]
                   + extra, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
    raw = np.memmap(dump, dtype=np.int32, mode="r")
    n, V = int(raw[0]), int(raw[1])
    logits = np.memmap(dump, dtype=np.float32, mode="r", offset=8 + 4 * n, shape=(n, V))
    eng = work / "logits.f32"   # the positions from 0, as compare_logits.py reads an engine's
    logits.tofile(eng)
    res = {"name": a.name, "model": pathlib.Path(model).name, "args": extra, "n": n, "vocab": V}
    for k, ref in REFS.items():
        res[k] = compare(ref, eng, V)
        print(f"{a.name} vs {k}: KLD {res[k]['kld']:.5f}  上位1 {res[k]['top1']:.2f}%  PPL {res[k]['ppl_engine']:.4f}",
              flush=True)
    # --kld-eval: the second half from the reference, the first half the engine's own (contributing 0 and a match)
    for k, ref in REFS.items():
        n_ctx, n_vocab, n_chunk, tokens, nv, first, n_tok, rr = cl.read_ref(ref)
        ids = np.array([int(x) for x in toks.split(",")], dtype=np.int32)
        assert len(ids) == n and (tokens[:n] == ids).all(), "the reference's tokens differ"
        hyb = work / f"kld_{k}.bin"
        with open(hyb, "wb") as f:
            np.array([n, n_vocab], dtype=np.int32).tofile(f)
            ids.tofile(f)
            for p in range(n):
                if first <= p < first + n_tok:
                    cl.ref_logprobs(rr, nv, n_vocab, p - first).astype(np.float32).tofile(f)
                else:
                    np.asarray(logits[p, :n_vocab], dtype=np.float32).tofile(f)
        o = subprocess.run([exe, "--kld-eval", str(hyb), "--native", model, "--max-context", "4096"] + extra,
                           capture_output=True, text=True, timeout=3600)
        m = re.search(r"n=(\d+) \| Mean KLD=([0-9.]+) \| Top-1=([0-9.]+)%", o.stdout)
        if m:
            ne, kl, t1 = int(m.group(1)), float(m.group(2)), float(m.group(3))
            res[f"kld_eval_{k}"] = {"raw": m.group(0), "kld_second_half": kl * ne / n_tok,
                                    "top1_second_half": (t1 / 100 * ne - (ne - n_tok)) / n_tok * 100}
            print(f"  --kld-eval {k}: {m.group(0)} -> 後半 {n_tok} 位置: KLD "
                  f"{res[f'kld_eval_{k}']['kld_second_half']:.5f}  上位1 {res[f'kld_eval_{k}']['top1_second_half']:.2f}%",
                  flush=True)
        else:
            res[f"kld_eval_{k}"] = {"error": (o.stdout + o.stderr)[-400:]}
            print(f"  --kld-eval {k}: 失敗 {(o.stdout + o.stderr)[-300:]}", flush=True)
        hyb.unlink()
    del logits, raw
    eng.unlink()
    dump.unlink()
    (out_dir / f"{a.name}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
