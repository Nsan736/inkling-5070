"""Nsight Systems の記録 (sqlite に書き出したもの) から、生成中の GPU の使い方を出す (計測のみ)。

生成の区間: 最後のプロンプト読み込み (attn_mma_kernel = テンソルコアのプロンプト用アテンション) が終わった後から、
最後のカーネルまで。その区間で
  - カーネルの実行時間の合計 (同時に走ったものは重ねて数えない: 時間の和集合)
  - GPU が何もしていない時間 (区間 - カーネルとコピーの和集合)
  - カーネルの種類ごとの合計 (上位)
を 1 トークンあたりで出す。

使い方: python -I bench/nsys_decode.py bench/nsys/g8k.sqlite [生成トークン数 216]
"""
import collections
import sqlite3
import sys


def union(iv):
    iv.sort()
    tot, cs, ce = 0, None, None
    for s, e in iv:
        if cs is None or s > ce:
            if cs is not None:
                tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if cs is not None:
        tot += ce - cs
    return tot


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    db = sqlite3.connect(sys.argv[1])
    n_tok = int(sys.argv[2]) if len(sys.argv) > 2 else 216
    names = dict(db.execute("select id, value from StringIds"))
    k = [(s, e, names.get(n, "?")) for s, e, n in
         db.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL")]
    try:
        m = [(s, e) for s, e in db.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMCPY")]
    except sqlite3.Error:
        m = []
    prompt_end = max((e for s, e, n in k if "attn_mma_kernel" in n), default=0)
    dec = [(s, e, n) for s, e, n in k if s > prompt_end]
    if not dec:
        raise SystemExit("生成のカーネルが見つからない")
    t0, t1 = min(s for s, _, _ in dec), max(e for _, e, _ in dec)
    mem = [(s, e) for s, e in m if s >= t0 and e <= t1]
    span = t1 - t0
    kern = union([(s, e) for s, e, _ in dec])
    busy = union([(s, e) for s, e, _ in dec] + mem)
    by = collections.Counter()
    cnt = collections.Counter()
    for s, e, n in dec:
        short = n[-60:]
        by[short] += e - s
        cnt[short] += 1
    per = 1e-6 / n_tok   # ns -> ms per token
    print(f"生成の区間 {span * 1e-9:.2f} s ({n_tok} トークン): 1 トークン {span * per:.2f} ms")
    print(f"  カーネル (和集合) {kern * per:.2f} ms、コピーを含む GPU 使用 {busy * per:.2f} ms、"
          f"GPU が空いている時間 {(span - busy) * per:.2f} ms ({100.0 * (span - busy) / span:.0f}%)")
    print(f"  カーネル数 {len(dec) / n_tok:.0f} 個/トークン、コピー {len(mem) / n_tok:.0f} 個/トークン")
    print("  カーネルの種類別 (1 トークンあたり ms, 個数):")
    for name, t in by.most_common(14):
        print(f"    {t * per:6.3f} ms  {cnt[name] / n_tok:6.1f} 個  {name}")


if __name__ == "__main__":
    main()
