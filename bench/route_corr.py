"""INKLING2_ROUTE_LOG=FILE の記録から、エキスパートの選ばれ方の相関を測る (開発用、計測のみ)。

1 行が 1 訪問: 訪問 層 (エキスパートの層) 番号:VRAM にあるか x8。書いたトークンの訪問だけ (48 個で 1 トークン)。
見るもの (VRAM にない = CPU で計算した選択について):
  - 訪問の区分ごとの、1 訪問あたりの CPU の選択の数
  - ループの 2・3 回目: 同じトークンの前の回 (同じ層) でも選ばれていた割合 (前の回の後に VRAM へ送れば当たる分)
  - 直前のトークンの同じ訪問でも選ばれていた割合、直前 4 トークンのどれかで選ばれていた割合
  - 層 L と L+1 (同じ回の続く訪問): L の選択から L+1 の選択を当てる上限の目安として、前半のトークンで数えた
    共起の上位 8 個が後半でどれだけ当たるか
使い方: python -I bench/route_corr.py FILE [--visits 48]
"""
import argparse
import collections

ap = argparse.ArgumentParser()
ap.add_argument("file")
ap.add_argument("--visits", type=int, default=48)
a = ap.parse_args()
rows = []
for line in open(a.file, encoding="utf-8"):
    p = line.split()
    if len(p) < 3:
        continue
    picks = [tuple(int(x) for x in q.split(":")) for q in p[2:]]
    rows.append((int(p[0]), int(p[1]), picks))
# tokens: runs of visits 0..47 in order
toks, cur = [], []
for r in rows:
    if r[0] == 0 and cur:
        toks.append(cur)
        cur = []
    cur.append(r)
if cur:
    toks.append(cur)
toks = [t for t in toks if len(t) == a.visits]
print(f"rows {len(rows)}, whole tokens {len(toks)}")


def cls(v):
    return "prelude 0-5" if v < 6 else "pass 1" if v < 18 else "pass 2" if v < 30 else "pass 3" if v < 42 else "epilogue 18-23"


cpu = collections.Counter()
allp = collections.Counter()
for t in toks:
    for v, il, picks in t:
        allp[cls(v)] += len(picks)
        cpu[cls(v)] += sum(1 for _, c in picks if not c)
n = len(toks)
print("CPU picks a token, by part (of 8 picks a visit):")
for k in ("prelude 0-5", "pass 1", "pass 2", "pass 3", "epilogue 18-23"):
    print(f"  {k:15s} {cpu[k] / n:6.2f} of {allp[k] / n:5.0f}  ({100 * cpu[k] / max(1, allp[k]):.1f}%)")

# later passes: picked in an earlier pass of the same token at the same layer
hit_prev_pass = tot_late = 0
for t in toks:
    by = {v: {e for e, _ in picks} for v, il, picks in t}
    for v, il, picks in t:
        if 18 <= v < 42:
            earlier = set()
            q = v - 12
            while q >= 6:
                earlier |= by[q]
                q -= 12
            for e, c in picks:
                if not c:
                    tot_late += 1
                    hit_prev_pass += e in earlier
print(f"passes 2-3: CPU picks also picked by an earlier pass of the token: {hit_prev_pass} of {tot_late} "
      f"({100 * hit_prev_pass / max(1, tot_late):.1f}%)")

# the token before, and the 4 before, at the same visit
for back in (1, 4):
    hit = tot = 0
    for i in range(back, n):
        for v in range(a.visits):
            seen = set()
            for b in range(1, back + 1):
                seen |= {e for e, _ in toks[i - b][v][2]}
            for e, c in toks[i][v][2]:
                if not c:
                    tot += 1
                    hit += e in seen
    print(f"CPU picks also picked at the same visit by one of the {back} token(s) before: {hit} of {tot} "
          f"({100 * hit / max(1, tot):.1f}%)")

# the visit before (same token): co-occurrence from the first half, top 8 predicted, scored on the second half
half = n // 2
co = collections.defaultdict(collections.Counter)
for t in toks[:half]:
    for v in range(1, a.visits):
        for e1, _ in t[v - 1][2]:
            for e2, _ in t[v][2]:
                co[(v, e1)][e2] += 1
hit = tot = 0
for t in toks[half:]:
    for v in range(1, a.visits):
        score = collections.Counter()
        for e1, _ in t[v - 1][2]:
            score.update(co[(v, e1)])
        guess = {e for e, _ in score.most_common(8)}
        for e, c in t[v][2]:
            if not c:
                tot += 1
                hit += e in guess
print(f"CPU picks among the top 8 guessed from the visit before (co-occurrence of the first half): {hit} of {tot} "
      f"({100 * hit / max(1, tot):.1f}%)")
