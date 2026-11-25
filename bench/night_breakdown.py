"""INKLING2_VISIT_TRACE の記録から、1 トークンの時間を区間ごとに分ける (開発用、計測のみ)。

lookup なし (1 トークンずつ) の答えの記録を読む。訪問 48 個で 1 トークン。訪問ごとに:
  head     前の訪問の終わり (g5、なければ g4) -> ルーターの結果 (g1): 前の訪問の tail、この訪問の混ぜ合わせ、アテンション、ルーター
  route    g1 -> キャッシュのエキスパートの起動 (g3): ルーターの結果がホストに届き、起動するまで (転送・同期)
  gpu      g3 -> g4: VRAM のエキスパート
  cpu_wait g4 -> g5: CPU のエキスパートの終わりを待つ (CPU の分がない訪問は 0)
トークンの最初の訪問の head には、前のトークンの出力 (lm_head、サンプリング) が入るので「その他」に分ける。
使い方: python -I bench/night_breakdown.py FILE [--visits 48]
"""
import argparse
import statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("file")
ap.add_argument("--visits", type=int, default=48)
a = ap.parse_args()
lines = open(a.file, encoding="utf-8").read().split("\n")
start = max(i for i, l in enumerate(lines) if l.startswith("cal "))
off = int(lines[start].split()[1])
ev = []
for l in lines[start + 1:]:
    p = l.split()
    if p and p[0] in ("g", "h"):
        ev.append((int(p[2]) + (off if p[0] == "g" else 0), p[0] + p[1]))
ev.sort()
visits, cur = [], None
last = {}
for t, k in ev:
    if k in ("g1", "g2"):
        last[k] = t
    if k == "h10":
        if cur:
            visits.append(cur)
        cur = {"g1": last.get("g1"), "h10": t}
        continue
    if cur is not None and k not in cur and k in ("g3", "g4", "g5", "h13", "h14", "h15", "h16"):
        cur[k] = t
if cur:
    visits.append(cur)
V = a.visits
n_tok = len(visits) // V
visits = visits[len(visits) - n_tok * V:]
print(f"visits {len(visits)}, tokens {n_tok}")


def phys(k):
    return k if k < 6 else 6 + (k - 6) % 12 if k < 42 else k - 24


per = {k: {"head": [], "route": [], "gpu": [], "cpu_wait": []} for k in range(V)}
tok_ms = []
for i in range(1, len(visits)):
    v, p = visits[i], visits[i - 1]
    k = i % V
    end = p.get("g5") or p.get("g4")
    if not all([end, v.get("g1"), v.get("g3"), v.get("g4")]):
        continue
    per[k]["head"].append(v["g1"] - end)
    per[k]["route"].append(v["g3"] - v["g1"])
    per[k]["gpu"].append(v["g4"] - v["g3"])
    per[k]["cpu_wait"].append((v["g5"] - v["g4"]) if v.get("g5") else 0)
for i in range(1, n_tok):
    a0, b0 = visits[(i - 1) * V].get("g1"), visits[i * V].get("g1")
    if a0 and b0:
        tok_ms.append((b0 - a0) / 1e6)
med = {k: {x: (st.median(y) / 1e3 if y else 0) for x, y in d.items()} for k, d in per.items()}
print(f"token: median {st.median(tok_ms):.2f} ms ({1000 / st.median(tok_ms):.1f} tok/s)")
print("visit phys   head  route    gpu  cpu_wait (us, median)")
for k in range(V):
    m = med[k]
    print(f"{k:5d} {phys(k):4d} {m['head']:6.0f} {m['route']:6.0f} {m['gpu']:6.0f} {m['cpu_wait']:8.0f}")
heads = [med[k]["head"] for k in range(1, V)]
base = st.median(heads)
glob = [k for k in range(1, V) if med[k]["head"] > 1.8 * base]
tot = {x: sum(med[k][x] for k in range(V)) / 1e3 for x in ("head", "route", "gpu", "cpu_wait")}
other = med[0]["head"] / 1e3
g_extra = sum(med[k]["head"] - base for k in glob) / 1e3
print(f"global-looking visits (head > 1.8x the median {base:.0f} us): {glob} -> phys {[phys(k) for k in glob]}")
print(f"sum of medians: head {tot['head'] - other:.2f} ms (of it the global visits' attention above a window "
      f"visit's {g_extra:.2f} ms), route {tot['route']:.2f}, gpu {tot['gpu']:.2f}, cpu_wait {tot['cpu_wait']:.2f}, "
      f"other (output, sampling: visit 0's head) {other:.2f}")
