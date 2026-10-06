"""Stage 5a figure: what each escaper did after leaving, one row per object.

Reads data/runs/<out_id>/{fates,tracks}/ written by s5_escaper_fates.py and writes
campaign/reports/<out_id>/fig_fates.png:
  - one row per escaper, grouped by group (Trojan, interloper) and swarm, sorted by
    how long it lasted after escape;
  - colour = orbital class in each 1 kyr sample (JFC, Centaur, beyond Neptune; grey
    for everything else);
  - black bars = co-orbital phases with Jupiter (recaptures), from the exact episodes;
  - end marker: x = hyperbolic ejection, o = still bound past 1000 au.

    python campaign/scripts/s5_fates_report.py --out-id escaper_fates_pilot
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from trojans import paths

# Reference palette (dataviz skill). Slots 3, 4, 7 validate all-pairs; blue and orange
# are left to mean L4 and L5 elsewhere in the project.
CLASS_COLOR = {"jfc": "#1baf7a", "centaur": "#eda100", "outer": "#4a3aa7"}
OTHER = "#d6d5ce"
SWARM_COLOR = {"L4": "#2a78d6", "L5": "#eb6834"}
SURFACE, TEXT, TEXT_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
X_MIN, X_MAX = 1.0e2, 3.0e7
GROUPS = [("trojan", "L4"), ("trojan", "L5"), ("interloper", "L4"), ("interloper", "L5")]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "text.color": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 11, "legend.frameon": False,
})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-id", default="escaper_fates_pilot")
    args = parser.parse_args()
    run = paths.run_dir(args.out_id)
    fates = [json.loads(p.read_text()) for p in sorted((run / "fates").glob("*.json"))]

    rows = []
    for group, swarm in GROUPS:
        sel = sorted((f for f in fates if f["group"] == group and f["swarm"] == swarm),
                     key=lambda f: f["t_fate_after_escape_yr"])
        rows.append(("header", group, swarm, len(sel)))
        rows += [("object", f) for f in sel]

    fig, ax = plt.subplots(figsize=(11, 0.24 * len(rows) + 1.8), constrained_layout=True)
    y = 0
    yticks, ylabels = [], []
    for row in rows[::-1]:
        if row[0] == "header":
            _, group, swarm, n = row
            label = f"{'Trojans' if group == 'trojan' else 'Interlopers'} escaped from {swarm} ({n})"
            ax.annotate(label, (X_MIN, y), xytext=(2, 0), textcoords="offset points", va="center",
                        fontsize=10, color=SWARM_COLOR[swarm], fontweight="bold")
            y += 1.6
            continue
        f = row[1]
        track = pd.read_parquet(run / "tracks" / f"{f['spkid']}.parquet")
        t = np.maximum(track["t_after_escape_yr"].to_numpy(), X_MIN)
        t_end = max(f["t_fate_after_escape_yr"], X_MIN)
        edges = np.r_[t, t_end]
        for cls, left, right in zip(track["cls"], edges[:-1], edges[1:]):
            if right > left:
                ax.barh(y, right - left, left=left, height=0.62, color=CLASS_COLOR.get(cls, OTHER), lw=0)
        for e in f["episodes"]:
            s = max(e["t_start_after_escape_yr"], X_MIN)
            ax.barh(y, e["duration_yr"], left=s, height=0.95, color=TEXT, lw=0)
        marker = "x" if f["fate"] == "ejected" else "o"
        ax.plot(t_end * 1.12, y, marker, ms=4.5, color=TEXT_2, mfc="none" if marker == "o" else TEXT_2, mew=1.1)
        yticks.append(y)
        ylabels.append(str(f["spkid"]))
        y += 1
    ax.set_xscale("log")
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(-1, y - 1.0)
    ax.set_yticks(yticks, ylabels, fontsize=8.5)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("time after escape [yr]  (escape = first pass within 2.1 Hill radii of a giant)")
    handles = [Patch(color=CLASS_COLOR["jfc"], label="Jupiter-family comet (2 < T_J < 3)"),
               Patch(color=CLASS_COLOR["centaur"], label="Centaur (q > a_J, a < a_N)"),
               Patch(color=CLASS_COLOR["outer"], label="beyond Neptune (a > a_N)"),
               Patch(color=OTHER, label="other"),
               Patch(color=TEXT, label="co-orbital with Jupiter (recapture)"),
               Line2D([], [], marker="x", ls="none", color=TEXT_2, label="ejected (hyperbolic)"),
               Line2D([], [], marker="o", ls="none", color=TEXT_2, mfc="none", label="bound, past 1000 au")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=9,
              handlelength=1.4, columnspacing=1.2)
    out = paths.report_dir(args.out_id)
    fig.savefig(out / "fig_fates.png", dpi=200)
    plt.close(fig)
    print(out / "fig_fates.png")


if __name__ == "__main__":
    main()
