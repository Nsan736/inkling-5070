"""INKLING2_GPU_TOKEN=1 の 1 トークンの内訳 (開発用、計測のみ)。INKLING2_VISIT_TRACE の記録 (最後の回答) を読む。

グラフの中の刻み (engine.cpp): 100 + 訪問 その訪問の始まり、41 層の頭の終わり、42 VRAM のエキスパートの終わり、
43 CPU の和を足した、44 層の終わり。45 / 46 はアテンションの始まりと終わり (inkling.cpp)。
訪問 v の物理層: 0〜5 はそのまま、6〜41 は 6 + (v - 6) % 12、42〜47 は v - 24。グローバル層は物理 5 / 11 / 17 / 23。

使い方: python -I bench/gt_breakdown.py 記録 [--step-ms 19.31]
"""
import argparse
import statistics as st


def phys(v):
    return v if v < 6 else 6 + (v - 6) % 12 if v < 42 else v - 24


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--step-ms", type=float, default=0.0, help="the host's time a step (the report's answer steps)")
    a = ap.parse_args()
    blocks, cur = [], None
    for line in open(a.trace, encoding="utf-8"):
        p = line.split()
        if p[0] == "cal":
            cur = []
            blocks.append(cur)
        elif p[0] == "g" and cur is not None:
            cur.append((int(p[1]), int(p[2])))
    g = blocks[-1]
    # tokens: from a 100 (visit 0) to the next
    toks, t = [], None
    for tag, ns in g:
        if tag == 100:
            t = {"visits": {}, "start": ns}
            toks.append(t)
            v = 0
        if t is None:
            continue
        if tag >= 100:
            v = tag - 100
            t["visits"][v] = {"start": ns}
        elif tag in (41, 42, 43, 44, 45, 46) and v in t["visits"]:
            t["visits"][v].setdefault(tag, ns)
    toks = [t for t in toks if len(t["visits"]) == 48 and all(44 in x for x in t["visits"].values())]
    toks = toks[1:]   # the first after a dump may hold its start
    per = {k: [] for k in ("swa_attn", "glob_attn", "head_rest", "experts", "wait_inner", "wait_outer", "tail",
                           "gaps", "graph")}
    for t in toks:
        vs = t["visits"]
        acc = dict.fromkeys(per, 0.0)
        for v in range(48):
            x = vs[v]
            attn = (x.get(46, 0) - x.get(45, 0)) / 1e6 if 45 in x and 46 in x else 0.0
            head = (x[41] - x["start"]) / 1e6
            acc["glob_attn" if phys(v) in (5, 11, 17, 23) else "swa_attn"] += attn
            acc["head_rest"] += head - attn
            acc["experts"] += (x[42] - x[41]) / 1e6
            acc["wait_inner" if 6 <= v < 42 else "wait_outer"] += (x[43] - x[42]) / 1e6
            acc["tail"] += (x[44] - x[43]) / 1e6
            if v + 1 < 48:
                acc["gaps"] += (vs[v + 1]["start"] - x[44]) / 1e6
        acc["graph"] = (vs[47][44] - vs[0]["start"]) / 1e6
        for k in per:
            per[k].append(acc[k])
    m = {k: st.median(v) for k, v in per.items()}
    print(f"tokens {len(toks)} (medians, ms a token)")
    names = {"swa_attn": "attention, window layers (40)", "glob_attn": "attention, global layers (8)",
             "head_rest": "head without attention (norms, mixes, QKV, router)",
             "experts": "picks + experts in VRAM", "wait_inner": "CPU wait, loop layers (6-17)",
             "wait_outer": "CPU wait, outer layers (0-5, 18-23)", "tail": "tail (out projection, FFN mix)",
             "gaps": "between visits", "graph": "the graph, first visit to last"}
    for k, label in names.items():
        print(f"  {label:52s} {m[k]:6.2f}")
    if a.step_ms:
        print(f"  {'outside the graph (output, sampling, host), step - graph':52s} {a.step_ms - m['graph']:6.2f}")


if __name__ == "__main__":
    main()
