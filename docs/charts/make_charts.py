"""Draws the README's charts from measurements.json (the measured numbers, with how they were measured).

    python docs/charts/make_charts.py      -> docs/charts/writing.png, reading.png, quality.png

Needs matplotlib (pip install matplotlib).
"""
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
BLUE, LIGHT, GRID, FRAME, TEXT = "#3b6ea5", "#9bb8d6", "#ececec", "#d0d0d0", "#333333"


def frame(ax, title, ylabel):
    ax.set_title(title, color=TEXT, fontsize=11, loc="left")
    ax.set_ylabel(ylabel, color=TEXT, fontsize=9)
    ax.tick_params(colors=TEXT, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_color(FRAME)


def bars(ax, labels, values, color, fmt):
    b = ax.bar(labels, values, color=color, width=0.6)
    for r, v in zip(b, values):
        ax.annotate(fmt.format(v), (r.get_x() + r.get_width() / 2, r.get_height()), ha="center", va="bottom",
                    fontsize=8, color=TEXT, xytext=(0, 2), textcoords="offset points")
    ax.set_ylim(0, max(values) * 1.18)


def save(fig, name, note):
    fig.text(0.01, 0.01, note, fontsize=7, color="#777777", ha="left", va="bottom")
    fig.patch.set_facecolor("white")
    fig.patch.set_edgecolor(FRAME)   # a light frame around the chart
    fig.patch.set_linewidth(1.5)
    fig.savefig(HERE / name, dpi=150, facecolor="white", edgecolor=FRAME)
    plt.close(fig)


def main():
    m = json.loads((HERE / "measurements.json").read_text(encoding="utf-8"))
    machine = m["machine"]

    w = m["writing"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    labels = [r["context"] for r in w["rows"]]
    x = range(len(labels))
    prose = [r.get("prose") for r in w["rows"]]
    code = [r["code"] for r in w["rows"]]
    width = 0.38
    have = [i for i, v in enumerate(prose) if v]
    b1 = ax.bar([i - width / 2 for i in have], [prose[i] for i in have], width, color=LIGHT, label="prose")
    b2 = ax.bar([i + (width / 2 if i in have else 0) for i in x], code, width, color=BLUE, label="code")
    for b in (b1, b2):
        for r in b:
            ax.annotate(f"{r.get_height():.1f}", (r.get_x() + r.get_width() / 2, r.get_height()), ha="center",
                        va="bottom", fontsize=8, color=TEXT, xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, max(code + [v for v in prose if v]) * 1.2)
    frame(ax, w["title"], "tokens/s")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, "writing.png", w["note"] + "\n" + machine)

    r = m["reading"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bars(ax, [x["length"] for x in r["rows"]], [x["tps"] for x in r["rows"]], BLUE, "{:,.0f}")
    frame(ax, r["title"], "tokens/s")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, "reading.png", r["note"] + "\n" + machine)

    q = m["quality"]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bars(ax, [f"{x['model']}\ntop-1 {x['top1']:.1f}%" for x in q["rows"]], [x["kld"] for x in q["rows"]], BLUE,
         "{:.4f}")
    frame(ax, q["title"], "mean KLD (lower is better)")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, "quality.png", q["note"])
    print("wrote", ", ".join(str(HERE / n) for n in ("writing.png", "reading.png", "quality.png")))


if __name__ == "__main__":
    main()
