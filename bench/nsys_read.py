"""Nsight Systems の記録 (sqlite) から、最後の読み込みの GPU の時間をカーネルごとに出す (計測のみ)。

区間: カーネルの間が 0.3 秒以上空いた所で切り、attn_mma_kernel (読み込みのアテンション) を含む最後の区間。
使い方: python -I bench/nsys_read.py 記録.sqlite [上位の数 40] [間の ms]
  間の ms を付けると、アテンションの間が それより空いた最後の所から後ろだけを数える (会話の続きの読み込み)。
"""
import collections
import sqlite3
import sys


def union(iv):
    iv = sorted(iv)
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
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    names = dict(db.execute("select id, value from StringIds"))
    k = sorted((s, e, names.get(n, "?")) for s, e, n in
               db.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL"))
    parts, cur = [], [k[0]]
    for x in k[1:]:
        if x[0] - cur[-1][1] > 300_000_000:
            parts.append(cur)
            cur = []
        cur.append(x)
    parts.append(cur)
    reads = [p for p in parts if any("attn_mma" in x[2] for x in p)]
    p = reads[-1]
    first = next(i for i, x in enumerate(p) if "attn_mma" in x[2])
    last = max(i for i, x in enumerate(p) if "attn_mma" in x[2] or "mul_mat_q" in x[2])
    if len(sys.argv) > 3:   # the last read alone: from after the attention before the last gap of N ms between attentions
        gap = float(sys.argv[3]) * 1e6
        at = [x[0] for x in p if "attn_mma" in x[2]]
        cut = max(i for i in range(1, len(at)) if at[i] - at[i - 1] > gap)
        lo = next(i for i, x in enumerate(p) if x[0] > at[cut - 1])
        p = p[lo:last + 40]
    else:
        p = p[max(0, first - 40):last + 40]
    by = collections.defaultdict(lambda: [0, 0])
    for s, e, n in p:
        by[n][0] += e - s
        by[n][1] += 1
    total = sum(v[0] for v in by.values())
    w0, w1 = p[0][0], p[-1][1]
    ku = union([(s, e) for s, e, _ in p])
    try:
        mc = [(max(s, w0), min(e, w1), b, k) for s, e, b, k in
              db.execute("select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY") if e > w0 and s < w1]
    except sqlite3.OperationalError:
        mc = []
    h2d = [(s, e) for s, e, b, k in mc if k == 1]
    both = union([(s, e) for s, e, _ in p] + [(s, e) for s, e, _, _ in mc])
    print(f"read window {(w1 - w0) / 1e6:.0f} ms wall, kernels {total / 1e6:.0f} ms ({ku / 1e6:.0f} ms of the window "
          f"busy), {len(p)} launches; host to card {sum(b for _, _, b, k in mc if k == 1) / 1e9:.1f} GB in "
          f"{union(h2d) / 1e6:.0f} ms; nothing on the card {(w1 - w0 - both) / 1e6:.0f} ms")
    for n, (t, c) in sorted(by.items(), key=lambda x: -x[1][0])[:top]:
        print(f"{t / 1e6:8.1f} ms {c:6d}x {t / c / 1e3:8.1f} us  {n[:90]}")


if __name__ == "__main__":
    main()
