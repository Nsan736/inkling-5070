"""INKLING2_VISIT_TRACE=FILE の記録を、生成の訪問ごとに並べて区間の中央値を出す (開発用、計測のみ)。

使い方: python -I bench/visit_trace.py FILE   (最後の依頼の分を読む)
"""
import collections
import statistics as st
import sys

lines = open(sys.argv[1], encoding="utf-8").read().split("\n")
# the last section (the last "cal" line and what follows)
start = max(i for i, l in enumerate(lines) if l.startswith("cal "))
off = int(lines[start].split()[1])
g, h = [], []
for l in lines[start + 1:]:
    p = l.split()
    if not p:
        continue
    if p[0] == "g":
        g.append((int(p[2]) + off, int(p[1])))
    elif p[0] == "h":
        h.append((int(p[2]), int(p[1])))
g.sort()
h.sort()
print("counts g:", collections.Counter(t for _, t in g), " h:", collections.Counter(t for _, t in h))

# visits: each host 10 (router noticed); the card's 2 just before it; then until the next 10
ev = sorted([(t, "g", k) for t, k in g] + [(t, "h", k) for t, k in h])
visits = []
cur = None
last_g2 = None
last_g1 = None
for t, w, k in ev:
    if w == "g" and k == 1:
        last_g1 = t
    if w == "g" and k == 2:
        last_g2 = t
    if w == "h" and k == 10:
        if cur:
            visits.append(cur)
        cur = {"g1": last_g1, "g2": last_g2, "h10": t}
        continue
    if cur is None:
        continue
    key = ("g" if w == "g" else "h") + str(k)
    cur.setdefault(key, t)
if cur:
    visits.append(cur)
# also: the card's stamp 1 of the NEXT visit after this one's 16 (how long the card took to come back to a router)
print("visits:", len(visits))


def med(xs):
    xs = [x for x in xs if x is not None]
    return (st.median(xs) / 1000, st.mean(xs) / 1000, len(xs)) if xs else None


def gap(v, a, b):
    return v[b] - v[a] if a in v and b in v and v[a] is not None and v[b] is not None else None


rows = [
    ("card: router results in host memory (g1) -> router event (g2)", "g1", "g2"),
    ("card router event (g2) -> host notices (h10)", "g2", "h10"),
    ("host notices (h10) -> CPU experts called (h13)", "h10", "h13"),
    ("CPU experts called (h13) -> first product (h14)", "h13", "h14"),
    ("host notices (h10) -> launches begin on their thread (h11)", "h10", "h11"),
    ("launches begin (h11) -> card starts them (g3)", "h11", "g3"),
    ("card cached experts (g3 -> g4)", "g3", "g4"),
    ("CPU first product (h14) -> CPU done (h15)", "h14", "h15"),
    ("card cached done (g4) -> CPU done (h15)", "g4", "h15"),
    ("CPU done (h15) -> moe returns (h16)", "h15", "h16"),
    ("moe returns (h16) -> card adds the CPU sum (g5)", "h16", "g5"),
    ("card cached done (g4) -> card adds the CPU sum (g5)", "g4", "g5"),
]
cpu_visits = [v for v in visits if "h14" in v]
print(f"visits with CPU experts: {len(cpu_visits)} of {len(visits)}")
for name, a, b in rows:
    m = med([gap(v, a, b) for v in cpu_visits])
    if m:
        print(f"{name:62s} median {m[0]:8.1f} us  mean {m[1]:8.1f} us  (n={m[2]})")
# per visit: the card's idle from g4 to g5 split into: CPU still computing (to h15), host after (h15 -> g5)
both = [v for v in cpu_visits if all(k in v for k in ("g3", "g4", "h14", "h15", "g5"))]
lag = [v["h14"] - v["g3"] for v in both]
print("CPU first product - card cached start (h14 - g3): median %.1f us, mean %.1f us" % (st.median(lag) / 1e3, st.mean(lag) / 1e3))

g1n = [v["h10"] - v["g1"] for v in visits if v["g1"]]
print("router results in host memory (g1) -> host notices (h10): median %.1f us, mean %.1f us" % (st.median(g1n) / 1e3, st.mean(g1n) / 1e3))
cpu_ms = sum(v["h15"] - v["h14"] for v in cpu_visits) / 1e6
print("CPU first product -> done, total %.0f ms over %d visits; %.1f MB a visit if 2.9 experts of 4.48 MB" % (cpu_ms, len(cpu_visits), 2.9 * 4.48))
# the whole step: the card's gaps between visits (g5 or g4 of visit v -> g1 of visit v+1)
nxt = [visits[i + 1]["g1"] - visits[i].get("g5", visits[i].get("g4", visits[i]["h16"])) for i in range(len(visits) - 1)
       if visits[i + 1]["g1"]]
print("card: this visit's last stamp -> next visit's router results: median %.1f us" % (st.median(nxt) / 1e3))
tot = (visits[-1]["h16"] - visits[0]["h10"]) / 1e6
print(f"span {tot:.0f} ms for {len(visits)} visits: {1e3 * tot / len(visits):.1f} us a visit")
