"""Results figure for the README (light + dark).

Numbers from experiments/kill_v1/RESULTS.md, experiments/kill_v2/RESULTS.md and the two
data/analysis_summary.json files. Every value is a count divided by the true number of errors,
so 1.0 means exact. Run: python make_figure.py
"""
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
CLIP = 2.0
# rows: label, estimate, range lo, range hi, union (what the detectors found)
# Real-error rows: the 95% interval. Planted-error rows: middle 80% of the 200 replicate estimates.
ROWS = [
    ("Older judges\nreal errors", 181.9 / 181, 171.0 / 181, 218.3 / 181, 171 / 181),
    ("Older judges\nplanted errors", 1.000, 0.991, 1.014, 0.993),
    ("Newer judges, 5% budget\nreal errors", 146.3 / 181, 102.3 / 181, 796.9 / 181, 100 / 181),
    ("Newer judges, 5% budget\nplanted errors", 1.129, 0.936, 1.805, 0.827),
]
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e6e5e0",
                  ours="#2a78d6", union="#eb6834"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#3a3936",
                 ours="#3987e5", union="#d95926"),
}


def draw(theme, path):
    t = THEMES[theme]
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=200)
    fig.patch.set_facecolor(t["surface"]); ax.set_facecolor(t["surface"])
    n = len(ROWS)
    ax.axvline(1.0, color=t["ink"], lw=2.5, zorder=2)
    for i, (name, est, lo, hi, union) in enumerate(ROWS):
        y = n - 1 - i
        ax.plot([lo, min(hi, CLIP)], [y, y], color=t["ours"], lw=2, solid_capstyle="round", zorder=3)
        if hi > CLIP:
            ax.annotate("", xy=(CLIP + 0.04, y), xytext=(CLIP - 0.02, y),
                        arrowprops=dict(arrowstyle="-|>", color=t["ours"], lw=2), zorder=3)
            ax.text(CLIP + 0.06, y, f"to {hi:.1f}", va="center", color=t["ink2"], fontsize=8)
        ax.scatter([est], [y], s=70, color=t["ours"], edgecolor=t["surface"], linewidth=2, zorder=4)
        ax.scatter([union], [y - 0.24], s=58, color=t["union"], edgecolor=t["surface"], linewidth=2,
                   zorder=4, marker="s")
    ax.set_yticks(range(n)); ax.set_yticklabels([r[0] for r in ROWS][::-1], color=t["ink"], fontsize=9.5)
    ax.set_ylim(-0.6, n - 0.4); ax.set_xlim(0.4, CLIP + 0.3)
    ax.set_xticks([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    ax.set_xticklabels(["0.5×", "0.75×", "exact", "1.25×", "1.5×", "1.75×", "2×"], color=t["ink2"], fontsize=9)
    ax.set_xlabel("Count compared with the true number of label errors", color=t["ink2"], fontsize=9.5)
    ax.grid(axis="x", color=t["grid"], lw=0.8, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"]); ax.tick_params(length=0)
    handles = [
        plt.Line2D([], [], color=t["ink"], lw=2.5, label="Truth"),
        plt.Line2D([], [], color=t["ours"], marker="o", lw=2, markersize=7, label="Capture–recapture estimate (likely range)"),
        plt.Line2D([], [], color=t["union"], marker="s", lw=0, markersize=7, label="Errors the detectors actually found"),
    ]
    leg = ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False,
                    fontsize=8.5, handlelength=1.6, columnspacing=1.4)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])
    fig.suptitle("Can the overlap between detectors count the errors nobody found?", x=0.015, ha="left",
                 y=0.985, color=t["ink"], fontsize=12.5, fontweight="bold")
    fig.text(0.015, 0.905, "MMLU-Redux: 181 expert-confirmed wrong answer keys, plus 200 sets of 266 planted errors. "
             "Verdicts: NO-GO, then PARTIAL.", color=t["ink2"], fontsize=8.8)
    fig.subplots_adjust(left=0.23, right=0.975, bottom=0.12, top=0.80)
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)


if __name__ == "__main__":
    draw("light", HERE / "results.png")
    draw("dark", HERE / "results-dark.png")
    print("wrote results.png and results-dark.png")
