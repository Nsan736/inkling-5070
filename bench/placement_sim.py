"""エキスパートの置き場と外側の層の先読みの見積もり (開発用、GPU は使わない)。INKLING2_ROUTE_LOG の記録を読む。

時間の目安 (この PC の計測から):
  CPU の 1 個   大きさ / 56 GB/s (NVFP4 4.48 MB で約 80 µs)、その訪問に CPU の分があれば呼び出しの固定 20 µs
  GPU の 1 個   7.7 µs (VRAM のエキスパート 8 個で 62 µs、試作の計測)
  GPU が止まる時間 = max(0, その訪問の CPU の時間 - GPU の VRAM の分の時間)

置き方 (同じバイト数の VRAM):
  log     記録の中の実際の置き場 (そのときのエンジンの入れ替えを含む)
  count   使われた回数 / 大きさ の順 (エンジンの初めの詰め方と同じ考え方)
  stall   GPU が止まる時間を減らす順 (回数 x 止まる時間の差で、まとめて選び直す)
count と stall は前半のトークンで決め、後半で比べる。

使い方: python -I bench/placement_sim.py ROUTE_LOG [VRAM の GB、既定 5.5]
"""
import sys
from collections import defaultdict

import numpy as np

NE = 128
BYTES = {20: 5.474, 21: 5.474, 22: 4.811, 23: 4.313}   # MB an expert, else 4.479
GPU_US, CPU_FIXED_US, CPU_GBPS = 7.7, 20.0, 56.0


def phys(home):
    return home if home < 24 else home - 24


def mb(home):
    return BYTES.get(phys(home), 4.479)


def load(path):
    toks, cur = [], None
    for line in open(path):
        p = line.split()
        if len(p) < 3:
            continue
        v, home = int(p[0]), int(p[1])
        if v == 0:
            cur = []
            toks.append(cur)
        if cur is None:
            continue
        cur.append((v, home, [(int(a), int(b)) for a, b in (x.split(":") for x in p[2:])]))
    return [t for t in toks if len(t) == 48]


def stall_of(toks, invram):
    """GPU が止まる時間 (µs) の 1 トークンあたり、全体と外側の層 (18〜23)。invram(home, e, flag) -> bool"""
    tot, outer = 0.0, 0.0
    for t in toks:
        for v, home, picks in t:
            cpu = gpu = 0.0
            for e, f in picks:
                if invram(home, e, f):
                    gpu += GPU_US
                else:
                    cpu += mb(home) * 1000.0 / CPU_GBPS
            if cpu > 0:
                cpu += CPU_FIXED_US
            s = max(0.0, cpu - gpu)
            tot += s
            if v >= 42:
                outer += s
    return tot / len(toks), outer / len(toks)


def cpu_picks(toks, invram):
    return sum(1 for t in toks for v, h, ps in t for e, f in ps if not invram(h, e, f)) / len(toks)


def by_count(train, budget):
    use = defaultdict(int)
    for t in train:
        for v, h, ps in t:
            for e, _ in ps:
                use[(h, e)] += 1
    order = sorted(use, key=lambda k: -use[k] / mb(k[0]))
    chosen, room = set(), budget
    for k in order:
        if mb(k[0]) <= room:
            chosen.add(k)
            room -= mb(k[0])
    return chosen


def by_stall(train, budget, step=24):
    """GPU が止まる時間をいちばん減らすものから、まとめて step 個ずつ選ぶ (選ぶたびに計算し直す)。"""
    occ_t, occ_v, occ_k = [], [], []
    keys = {}
    for ti, t in enumerate(train):
        for v, h, ps in t:
            for e, _ in ps:
                k = keys.setdefault((h, e), len(keys))
                occ_t.append(ti), occ_v.append(v), occ_k.append(k)
    occ_t, occ_v, occ_k = map(np.array, (occ_t, occ_v, occ_k))
    kinv = {i: k for k, i in keys.items()}
    t_cpu = np.array([mb(kinv[i][0]) * 1000.0 / CPU_GBPS for i in range(len(keys))])
    size = np.array([mb(kinv[i][0]) for i in range(len(keys))])
    inv = np.zeros(len(keys), bool)
    room = budget
    slot = occ_t * 48 + occ_v
    while True:
        on = inv[occ_k]
        cpu = np.bincount(slot, weights=np.where(on, 0.0, t_cpu[occ_k]), minlength=len(train) * 48)
        ncpu = np.bincount(slot, weights=(~on).astype(float), minlength=len(train) * 48)
        gpu = np.bincount(slot, weights=np.where(on, GPU_US, 0.0), minlength=len(train) * 48)
        c_full = cpu + np.where(ncpu > 0, CPU_FIXED_US, 0.0)
        before = np.maximum(0.0, c_full - gpu)
        c2 = cpu[slot] - t_cpu[occ_k] + np.where(ncpu[slot] - 1 > 0, CPU_FIXED_US, 0.0)
        after = np.maximum(0.0, c2 - (gpu[slot] + GPU_US))
        gain = np.where(on, 0.0, before[slot] - after)
        per = np.bincount(occ_k, weights=gain, minlength=len(keys)) / size
        per[inv] = -1
        picked = 0
        for i in np.argsort(-per):
            if per[i] <= 0 or picked >= step:
                break
            if size[i] <= room:
                inv[i] = True
                room -= size[i]
                picked += 1
        if picked == 0:
            break
    # the room left goes by use, as the cache would fill it
    rest = by_count(train, room + sum(size[inv]))
    chosen = {kinv[i] for i in range(len(keys)) if inv[i]}
    for k in sorted(rest, key=lambda k: 0):
        if k not in chosen and mb(k[0]) <= room:
            chosen.add(k)
            room -= mb(k[0])
    return chosen


def prefetch(toks, invram, back, cap):
    """外側の層 (訪問 42〜47) の CPU の分のうち、同じ訪問で back トークン前までに使われたものを、cap 個まで先に VRAM
    へ送れたとき: (当たった割合, 1 トークンの外側の層の止まる時間 µs)。"""
    hit = tot = 0
    stall = 0.0
    for ti, t in enumerate(toks):
        want = []
        for b in range(1, back + 1):
            if ti - b < 0:
                continue
            for v, h, ps in toks[ti - b]:
                if v >= 42:
                    want += [(v, e) for e, f in ps if not invram(h, e, f)]
        sent = []
        for w in want:
            if w not in sent and len(sent) < cap:
                sent.append(w)
        sent = set(sent)
        for v, h, ps in t:
            if v < 42:
                continue
            cpu = gpu = 0.0
            for e, f in ps:
                if invram(h, e, f):
                    gpu += GPU_US
                    continue
                tot += 1
                if (v, e) in sent:
                    hit += 1
                    gpu += GPU_US
                else:
                    cpu += mb(h) * 1000.0 / CPU_GBPS
            if cpu > 0:
                cpu += CPU_FIXED_US
            stall += max(0.0, cpu - gpu)
    return hit / max(tot, 1), stall / len(toks)


def main():
    toks = load(sys.argv[1])
    half = len(toks) // 2
    train, test = toks[:half], toks[half:]
    # the VRAM the log's cache held: the bytes of the experts it shows in VRAM (seen), the most of any token
    seen = set()
    for t in toks:
        for v, h, ps in t:
            for e, f in ps:
                if f:
                    seen.add((h, e))
    # the cache's size at a time: 1,225 experts at 100K (the engine's report, 0.3.4), about 5.5 GB; the experts seen in
    # VRAM over the whole log are more (the cache trades them while it writes)
    budget = float(sys.argv[2]) * 1024 if len(sys.argv) > 2 else 5.5 * 1024
    log_f = lambda h, e, f: bool(f)
    cnt = by_count(train, budget)
    stl = by_stall(train, budget)
    print(f"tokens {len(toks)} (train {len(train)}, test {len(test)}); VRAM for experts {budget / 1024:.2f} GB "
          f"(the log shows {len(seen)} experts in VRAM over its tokens)")
    for name, f in (("log", log_f), ("count", lambda h, e, _: (h, e) in cnt), ("stall", lambda h, e, _: (h, e) in stl)):
        s, o = stall_of(test, f)
        print(f"  {name:6s} on the test half: GPU stalled {s / 1000:.2f} ms a token (outer layers {o / 1000:.2f}), "
              f"CPU experts {cpu_picks(test, f):.1f}")
    for pname, f in (("the log's", log_f), ("the count", lambda h, e, _: (h, e) in cnt)):
        print(f"prefetch of the outer layers (visits 42-47), on {pname} placement, test half:")
        _, base = stall_of(test, f)
        print(f"  none: outer stall {base / 1000:.2f} ms")
        for back in (1, 2):
            for cap in (16, 32, 64):
                h, st = prefetch(test, f, back, cap)
                print(f"  from {back} token(s) back, at most {cap} sent: {100 * h:.0f}% of their CPU picks hit, "
                      f"outer stall {st / 1000:.2f} ms")


if __name__ == "__main__":
    main()
