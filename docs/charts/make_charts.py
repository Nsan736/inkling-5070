"""Draws the README's charts from measurements.json (the measured numbers, with how they were measured), in English
for README.en.md and in Japanese for README.md.

    python docs/charts/make_charts.py      -> docs/charts/{decode,reading,quality,breakdown}_{en,ja}.png

Needs matplotlib (pip install matplotlib).  The Japanese charts use Yu Gothic, else Meiryo (both come with Windows).
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
GRID, FRAME, TEXT, NOTE = "#ececec", "#d0d0d0", "#333333", "#777777"
OLD, OLD_L, NEW, NEW_L = "#9a9a9a", "#c8c8c8", "#3b6ea5", "#9bb8d6"
PARTS = {"head": "#3b6ea5", "route": "#d9822b", "experts": "#7fa65a", "cpu_wait": "#c0504d", "outside": "#9a9a9a"}

WORDS = {
    "en": {
        "decode": "Writing speed by context (0.3.2 and 0.3.4)", "tps": "tokens/s",
        "lookup": "lookup", "no_lookup": "no lookup",
        "reading": "Reading a prompt from the start", "quality": "Quality at 2K against Q8_K_XL (lower KLD is better)",
        "kld": "mean KLD", "top1": "top-1", "breakdown": "One token at 100K, where the time goes (ms)", "ms": "ms",
    },
    "ja": {
        "decode": "文脈の長さごとの生成の速さ（0.3.2 と 0.3.4）", "tps": "トークン/秒",
        "lookup": "lookup あり", "no_lookup": "lookup なし",
        "reading": "最初から読む速さ", "quality": "2K の品質（Q8_K_XL 基準、KLD は小さいほどよい）",
        "kld": "KLD 平均", "top1": "上位 1 位の一致", "breakdown": "100K の 1 トークンの内訳（ms）", "ms": "ms",
    },
}


def use_font(lang):
    if lang != "ja":
        plt.rcParams["font.family"] = "DejaVu Sans"
        return
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Yu Gothic", "Meiryo"):
        if name in have:
            plt.rcParams["font.family"] = name
            return
    raise SystemExit("Yu Gothic or Meiryo is needed for the Japanese charts")


def frame(ax, title, ylabel):
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT, fontsize=9)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(FRAME)


def label_bars(ax, bars, fmt, size=7):
    for r in bars:
        h = r.get_height()
        ax.annotate(fmt.format(h), (r.get_x() + r.get_width() / 2, h), ha="center", va="bottom", fontsize=size,
                    color=TEXT, xytext=(0, 2), textcoords="offset points")


def save(fig, name, note):
    fig.text(0.01, 0.01, note, fontsize=6.5, color=NOTE, ha="left", va="bottom", wrap=True)
    fig.patch.set_facecolor("white")
    fig.patch.set_edgecolor(FRAME)   # a light frame around the chart
    fig.patch.set_linewidth(1.5)
    fig.savefig(HERE / name, dpi=150, facecolor="white", edgecolor=FRAME)
    plt.close(fig)
    return HERE / name


def decode(m, lang, w):
    d = m["decode"]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    n, width = len(d["contexts"]), 0.2
    colors = {("0.3.2", False): OLD_L, ("0.3.2", True): OLD, ("0.3.4", False): NEW_L, ("0.3.4", True): NEW}
    for k, s in enumerate(d["series"]):
        xs = [i + (k - 1.5) * width for i in range(n)]
        lab = f"{s['version']} {w['lookup'] if s['lookup'] else w['no_lookup']}"
        label_bars(ax, ax.bar(xs, s["values"], width, color=colors[(s["version"], s["lookup"])], label=lab), "{:.1f}", 6.5)
    ax.set_xticks(range(n), d["contexts"])
    ax.set_ylim(0, max(max(s["values"]) for s in d["series"]) * 1.2)
    frame(ax, w["decode"], w["tps"])
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    return save(fig, f"decode_{lang}.png", d["note"][lang] + "\n" + m["machine"][lang])


def reading(m, lang, w):
    r = m["reading"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    n, width = len(r["contexts"]), 0.36
    for k, s in enumerate(r["series"]):
        xs = [i + (k - 0.5) * width for i in range(n)]
        label_bars(ax, ax.bar(xs, s["values"], width, color=OLD if s["version"] == "0.3.2" else NEW,
                              label=s["version"]), "{:,.0f}")
    ax.set_xticks(range(n), r["contexts"])
    ax.set_ylim(0, max(max(s["values"]) for s in r["series"]) * 1.2)
    frame(ax, w["reading"], w["tps"])
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    return save(fig, f"reading_{lang}.png", r["note"][lang] + "\n" + m["machine"][lang])


def quality(m, lang, w):
    q = m["quality"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    labels = [f"{x['model']}\n{w['top1']} {x['top1']:.1f}%" for x in q["rows"]]
    colors = [NEW if x["model"].startswith("iq4_xs 0.3.4") else OLD if x["model"].startswith("iq4_xs") else NEW_L
              for x in q["rows"]]
    label_bars(ax, ax.bar(labels, [x["kld"] for x in q["rows"]], 0.6, color=colors), "{:.4f}", 7.5)
    ax.set_ylim(0, max(x["kld"] for x in q["rows"]) * 1.2)
    frame(ax, w["quality"], w["kld"])
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return save(fig, f"quality_{lang}.png", q["note"][lang])


def breakdown(m, lang, w):
    b = m["breakdown"]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    names = b["names"][lang]
    rows = b["rows"]
    left = [0.0] * len(rows)
    ys = list(range(len(rows)))[::-1]
    for part in b["parts"]:
        vals = [r[part] for r in rows]
        bars = ax.barh(ys, vals, left=left, color=PARTS[part], height=0.55, label=names[part])
        for bar, v, l0 in zip(bars, vals, left):
            if v >= 0.8:
                ax.text(l0 + v / 2, bar.get_y() + bar.get_height() / 2, f"{v:.1f}", ha="center", va="center",
                        fontsize=7.5, color="white")
        left = [l0 + v for l0, v in zip(left, vals)]
    for y, total in zip(ys, left):
        ax.text(total + 0.2, y, f"{total:.1f} ms", va="center", fontsize=8, color=TEXT)
    ax.set_yticks(ys, [r["version"] for r in rows])
    ax.set_xlim(0, max(left) * 1.15)
    ax.set_title(w["breakdown"], color=TEXT, fontsize=11, loc="left")
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(FRAME)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    return save(fig, f"breakdown_{lang}.png", b["note"][lang])


def main():
    m = json.loads((HERE / "measurements.json").read_text(encoding="utf-8"))
    out = []
    for lang in ("en", "ja"):
        use_font(lang)
        w = WORDS[lang]
        out += [decode(m, lang, w), reading(m, lang, w), quality(m, lang, w), breakdown(m, lang, w)]
    print("wrote", ", ".join(str(p) for p in out))


if __name__ == "__main__":
    main()
