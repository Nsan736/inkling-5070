"""参照ロジット (llama-perplexity --kl-divergence-base の出力) と、エンジンが出したロジットを照合する。

参照ファイルの形式 (llama.cpp tools/perplexity/perplexity.cpp):
  "_logits_" | int32 n_ctx | int32 n_vocab | int32 n_chunk | int32 tokens[n_ctx*n_chunk]
  以降、チャンクごとに位置 n_ctx/2 .. n_ctx-2 の各トークンについて
  uint16[nv] (nv = 2*((n_vocab+1)/2)+4)。先頭 4 要素は float32 の scale と min_log_prob。
  log_prob[i] = min_log_prob + scale * q[i]  (q=0 は min_log_prob 以下を表す)

エンジン側のロジット: float32 の生ファイル [n_pos, n_vocab_engine] (位置 0 から)。
  n_vocab_engine が参照と違う場合は短い方に合わせる (padding 分の語彙は無視)。

使い方:
  python bench/compare_logits.py --ref bench/data/ref_logits.bin --tokens-out bench/data/ref_tokens.txt
      参照のトークン列だけ書き出す (エンジンに流す入力)
  python bench/compare_logits.py --ref bench/data/ref_logits.bin --engine out.f32 --engine-vocab 201024
      照合して KLD / 上位一致率 を表示する
"""
import argparse
import struct
import sys

import numpy as np


def read_ref(path, tail=0):
    with open(path, "rb") as f:
        if f.read(8) != b"_logits_":
            raise SystemExit("参照ファイルの形式が違う")
        n_ctx, n_vocab, n_chunk = struct.unpack("<iii", f.read(12))
        tokens = np.frombuffer(f.read(4 * n_ctx * n_chunk), dtype=np.int32)
        nv = 2 * ((n_vocab + 1) // 2) + 4
        first = n_ctx - 1 - tail if tail > 0 else n_ctx // 2   # LLAMA_PPL_TAIL (ref/llama.cpp, local)
        n_tok = n_ctx - 1 - first
        raw = np.frombuffer(f.read(), dtype=np.uint16)
    return n_ctx, n_vocab, n_chunk, tokens, nv, first, n_tok, raw


def ref_logprobs(raw, nv, n_vocab, idx):
    row = raw[idx * nv:(idx + 1) * nv]
    scale, min_lp = np.frombuffer(row[:4].tobytes(), dtype=np.float32)
    q = row[4:4 + n_vocab].astype(np.float32)
    return min_lp + scale * q


def log_softmax(x):
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--tokens-out")
    ap.add_argument("--engine")
    ap.add_argument("--engine-vocab", type=int)
    ap.add_argument("--engine-first", type=int, default=0, help="エンジンのロジット 1 行目が表す位置")
    ap.add_argument("--ref-tail", type=int, default=0,
                    help="参照が LLAMA_PPL_TAIL=N で作ったもの (チャンクの最後の N 位置だけ) のとき N")
    a = ap.parse_args()

    n_ctx, n_vocab, n_chunk, tokens, nv, first, n_tok, raw = read_ref(a.ref, a.ref_tail)
    if n_chunk != 1:
        raise SystemExit("チャンク 1 個の参照だけ扱う")
    print(f"参照: n_ctx={n_ctx} n_vocab={n_vocab} 照合する位置 {first}..{first + n_tok - 1} ({n_tok} 個)")

    if a.tokens_out:
        with open(a.tokens_out, "w") as f:
            f.write(",".join(str(int(t)) for t in tokens))
        print(f"トークン列 {len(tokens)} 個を書き出した: {a.tokens_out}")

    if not a.engine:
        return 0
    ev = a.engine_vocab or n_vocab
    eng = np.fromfile(a.engine, dtype=np.float32)
    n_pos = eng.size // ev
    eng = eng[:n_pos * ev].reshape(n_pos, ev)
    base = a.engine_first
    if base > first or base + n_pos < first + n_tok:
        raise SystemExit(f"エンジンのロジットが範囲を覆っていない: 位置 {base}..{base + n_pos - 1}")
    nvc = min(n_vocab, ev)

    klds, top1, nll_ref, nll_eng = [], 0, 0.0, 0.0
    for i in range(n_tok):
        lp_ref = ref_logprobs(raw, nv, n_vocab, i)[:nvc]
        lp_eng = log_softmax(eng[first - base + i, :nvc].astype(np.float64))
        # llama-perplexity --kl-divergence と同じ計算: log 確率が -16 以下の語彙は除き、正規化し直さない
        lp64 = lp_ref.astype(np.float64)
        keep = lp64 > -16.0
        klds.append(float((np.exp(lp64[keep]) * (lp64[keep] - lp_eng[keep])).sum()))
        top1 += int(lp_ref.argmax() == lp_eng.argmax())
        nxt = int(tokens[first + i + 1])
        if nxt < nvc:
            nll_ref -= float(lp_ref[nxt])
            nll_eng -= float(lp_eng[nxt])
    k = np.array(klds)
    print(f"KLD 平均 {k.mean():.5f}  中央 {np.median(k):.5f}  99% {np.percentile(k, 99):.5f}  最大 {k.max():.5f}")
    print(f"上位1一致 {100.0 * top1 / n_tok:.2f}%")
    print(f"PPL 参照 {np.exp(nll_ref / n_tok):.4f}  エンジン {np.exp(nll_eng / n_tok):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
