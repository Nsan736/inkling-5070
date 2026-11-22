"""Nsight Systems の記録 (sqlite) から、生成中に GPU が空いた隙間を、前後のカーネル名で分けて集計する (計測のみ)。

使い方: python -I bench/nsys_gaps.py bench/nsys/g8k.sqlite [生成トークン数 216]
"""
import collections
import sqlite3
import sys


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    db = sqlite3.connect(sys.argv[1])
    n_tok = int(sys.argv[2]) if len(sys.argv) > 2 else 216
    names = dict(db.execute("select id, value from StringIds"))
    ev = [(s, e, names.get(n, "?")) for s, e, n in
          db.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL")]
    try:
        ev += [(s, e, "[memcpy]") for s, e in db.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMCPY")]
    except sqlite3.Error:
        pass
    prompt_end = max((e for s, e, n in ev if "attn_mma_kernel" in n), default=0)
    ev = sorted((x for x in ev if x[0] > prompt_end), key=lambda x: x[0])
    gaps = collections.Counter()
    count = collections.Counter()
    sizes = collections.Counter()
    end, prev = ev[0][1], ev[0][2]
    for s, e, n in ev[1:]:
        if s > end:
            g = s - end
            key = f"{prev[:34]:34s} -> {n[:34]}"
            gaps[key] += g
            count[key] += 1
            b = "<5us" if g < 5000 else "5-20us" if g < 20000 else "20-100us" if g < 100000 else ">100us"
            sizes[b] += g
        if e > end:
            end, prev = e, n
    total = sum(gaps.values())
    per = 1e-6 / n_tok
    print(f"隙間の合計 {total * per:.2f} ms/トークン")
    print("  大きさ別:", "  ".join(f"{k} {v * per:.2f} ms" for k, v in sorted(sizes.items())))
    print("  前 -> 後 のカーネル別 (上位):")
    for k, v in gaps.most_common(12):
        print(f"    {v * per:6.3f} ms  {count[k] / n_tok:5.1f} 回  平均 {v / count[k] / 1000:6.1f} us  {k}")


if __name__ == "__main__":
    main()
