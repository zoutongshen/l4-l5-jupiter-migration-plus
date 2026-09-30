"""Stage 3: the headline escape curve on a log clock, and the U ladder that tests the U <= 4 cut.

Reads a merged run and writes to campaign/reports/<run_id>/:
  fig_grand.png     cumulative % of each swarm escaped vs log time, with the running
                    escape ratio L5/L4 underneath (V24-selected, U <= 4)
  fig_uladder.png   escape fraction per U value (all H, and 14 < H < 15.5 to hold size
                    fixed), per cumulative cut U <= k, the ratio L5/L4 per cut, and the
                    number of objects per U
  uladder.csv       the numbers behind fig_uladder

U is the JPL orbit condition code (the MPC uncertainty parameter), see
docs/CATALOGUE_DEFINITIONS.md. Escape is the first entry within 2.1 Hill radii of a
giant. Fractions carry Wilson intervals at 1 sigma; the ratio carries binomial errors
propagated in quadrature, as in s3_report.py.

    python campaign/scripts/s3_uladder.py --run-id pilot_10myr
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trojans import paths

SWARMS = ("L4", "L5")
U_VALUES = np.arange(10)
H_FIXED = (14.0, 15.5)  # one H bin that holds every U value in numbers
MIN_ESCAPERS_FOR_RATIO = 3

# Reference palette (dataviz skill), as in s3_report.py.
SWARM_COLOR = {"L4": "#2a78d6", "L5": "#eb6834"}
SURFACE, TEXT, TEXT_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
})


def wilson(k, n, z=1.0):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def ratio(k4, n4, k5, n5):
    """Escape ratio f5/f4 with binomial errors in quadrature (s3_report.ratio_table)."""
    f4, f5 = k4 / n4, k5 / n5
    if k4 == 0 or k5 == 0:
        return np.nan, np.nan
    r = f5 / f4
    err = r * np.sqrt((1 - f4) / (n4 * f4) + (1 - f5) / (n5 * f5))
    return r, err


def load(run_id):
    ev = pd.read_parquet(paths.merged_dir(run_id) / "events.parquet")
    ev = ev[ev["v24_selected"]].copy()
    ev["U"] = pd.to_numeric(ev["condition_code"], errors="coerce").astype(int)
    ev["esc"] = ev["t_hill_record"].notna()
    t_end = json.loads((paths.run_dir(run_id) / "settings.json").read_text())["t_end_yr"]
    return ev, t_end


def ladder(ev) -> pd.DataFrame:
    fixed_h = ev["H"].between(*H_FIXED)
    rows = []
    for u in U_VALUES:
        row = {"U": u}
        for s in SWARMS:
            sw = ev["swarm"] == s
            for tag, mask in [("", sw & (ev["U"] == u)), ("Hfix_", sw & fixed_h & (ev["U"] == u)),
                              ("cum_", sw & (ev["U"] <= u))]:
                n, k = int(mask.sum()), int(ev.loc[mask, "esc"].sum())
                f, lo, hi = wilson(k, n)
                row.update({f"{tag}n_{s}": n, f"{tag}k_{s}": k, f"{tag}f_{s}": f, f"{tag}lo_{s}": lo, f"{tag}hi_{s}": hi})
        row["cum_ratio"], row["cum_ratio_err"] = ratio(row["cum_k_L4"], row["cum_n_L4"], row["cum_k_L5"], row["cum_n_L5"])
        rows.append(row)
    return pd.DataFrame(rows)


def fig_grand(ev, t_end, out):
    sel = ev[ev["U"] <= 4]
    grid = np.logspace(0, np.log10(t_end), 600)
    fig, (top, bot) = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True, height_ratios=[3, 2],
                                   constrained_layout=True)
    curves = {}
    for s in SWARMS:
        g = sel[sel["swarm"] == s]
        n = len(g)
        times = np.sort(g["t_hill_record"].dropna().to_numpy())
        k = np.searchsorted(times, grid, side="right")
        band = np.array([wilson(ki, n) for ki in k])
        curves[s] = (k, n)
        top.fill_between(grid, 100 * band[:, 1], 100 * band[:, 2], step="post", color=SWARM_COLOR[s], alpha=0.15, lw=0)
        top.step(grid, 100 * k / n, where="post", color=SWARM_COLOR[s], lw=2,
                 label=f"{s}: {k[-1]} of {n:,} escaped ({100 * k[-1] / n:.2f}%)")
        top.annotate(s, (grid[-1], 100 * k[-1] / n), xytext=(5, 0), textcoords="offset points",
                     va="center", color=TEXT_2, fontsize=9)
    top.set_ylabel("escaped so far [% of swarm]")
    top.set_title("V24-selected Trojans with well-determined orbits (U ≤ 4): cumulative escapes", loc="left")
    top.legend(loc="upper left")
    top.set_ylim(bottom=0)

    (k4, n4), (k5, n5) = curves["L4"], curves["L5"]
    ok = (k4 >= MIN_ESCAPERS_FOR_RATIO) & (k5 >= MIN_ESCAPERS_FOR_RATIO)
    r = np.full(grid.size, np.nan)
    e = np.full(grid.size, np.nan)
    for i in np.flatnonzero(ok):
        r[i], e[i] = ratio(k4[i], n4, k5[i], n5)
    bot.fill_between(grid, r - e, r + e, step="post", color=MUTED, alpha=0.2, lw=0)
    bot.step(grid, r, where="post", color=TEXT_2, lw=2)
    bot.axhline(1, color=BASELINE, lw=1.2, ls="--")
    bot.annotate("no asymmetry", (grid[1], 1), xytext=(2, 3), textcoords="offset points", color=MUTED, fontsize=8)
    bot.annotate(f"{r[-1]:.2f} ± {e[-1]:.2f} at {t_end / 1e6:.0f} Myr", (grid[-1], r[-1]), xytext=(-4, 8),
                 textcoords="offset points", ha="right", color=TEXT_2, fontsize=8)
    bot.set_ylabel("escape ratio L5 / L4\n(fraction escaped)")
    bot.set_ylim(0, max(4, np.nanmax(r + e) * 1.05))
    bot.set_xscale("log")
    bot.set_xlim(grid[0], grid[-1])
    bot.set_xlabel("time since today's orbits [yr]")
    bot.set_title(f"shown once both swarms have ≥ {MIN_ESCAPERS_FOR_RATIO} escapers; band = 1σ", loc="left",
                  color=TEXT_2, fontsize=8)
    fig.savefig(out / "fig_grand.png", dpi=200)
    plt.close(fig)
    return {"L4": (int(k4[-1]), n4), "L5": (int(k5[-1]), n5), "ratio": (float(r[-1]), float(e[-1]))}


def _rate_panel(ax, lad, tag, title):
    offset = {"L4": -0.12, "L5": 0.12}
    floor = 3e-5
    for s in SWARMS:
        x = lad["U"] + offset[s]
        f, lo, hi = lad[f"{tag}f_{s}"], lad[f"{tag}lo_{s}"], lad[f"{tag}hi_{s}"]
        some = lad[f"{tag}k_{s}"] > 0
        ax.errorbar(x[some], f[some], yerr=[f[some] - np.maximum(lo[some], floor), hi[some] - f[some]], fmt="o",
                    ms=6, color=SWARM_COLOR[s], mec=SURFACE, mew=1, elinewidth=1.2, capsize=0, label=s)
        none = (~some) & (lad[f"{tag}n_{s}"] > 0)
        ax.errorbar(x[none], np.full(none.sum(), floor), yerr=[np.zeros(none.sum()), hi[none] - floor], fmt="o",
                    ms=6, mfc=SURFACE, color=SWARM_COLOR[s], elinewidth=1.2, capsize=0)
    ax.set_yscale("log")
    ax.set_ylim(floor * 0.7, 1)
    ax.set_ylabel("fraction escaped\nin 10 Myr")
    ax.set_title(title, loc="left")


def fig_uladder(lad, out):
    fig, axes = plt.subplots(5, 1, figsize=(7.2, 11.5), sharex=True, height_ratios=[3, 3, 3, 2.2, 2],
                             constrained_layout=True)
    _rate_panel(axes[0], lad, "", "(a) each U value on its own, all H  (open = no escaper; bar = 1σ upper limit)")
    axes[0].legend(loc="upper left")
    _rate_panel(axes[1], lad, "Hfix_", f"(b) same, only {H_FIXED[0]} < H < {H_FIXED[1]}: size held fixed")
    _rate_panel(axes[2], lad, "cum_", "(c) cumulative cut: everything with U ≤ k")

    ax = axes[3]
    ok = lad["cum_ratio"].notna()
    ax.errorbar(lad["U"][ok], lad["cum_ratio"][ok], yerr=lad["cum_ratio_err"][ok], fmt="o", ms=6, color=TEXT_2,
                mec=SURFACE, mew=1, elinewidth=1.2, capsize=0)
    ax.axhline(1, color=BASELINE, lw=1.2, ls="--")
    ax.set_ylabel("escape ratio L5 / L4\nfor U ≤ k")
    ax.set_title("(d) the asymmetry each cut would report", loc="left")

    ax = axes[4]
    w = 0.36
    for s, dx in zip(SWARMS, (-w / 2, w / 2)):
        ax.bar(lad["U"] + dx, lad[f"n_{s}"], width=w - 0.04, color=SWARM_COLOR[s], label=s)
    ax.set_yscale("log")
    ax.set_ylabel("objects")
    ax.set_title("(e) how many objects have each U", loc="left")
    ax.set_xticks(U_VALUES)
    ax.set_xlabel("U (orbit uncertainty, 0 = best, 9 = worst)   ·   (c), (d): the cut U ≤ k")
    for a in axes:
        a.axvspan(4.5, 9.6, color=GRID, alpha=0.35, lw=0, zorder=0)
    axes[0].annotate("excluded by the current cut", (4.6, 0.5), color=MUTED, fontsize=8, va="top")
    fig.savefig(out / "fig_uladder.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default="pilot_10myr")
    args = parser.parse_args()

    ev, t_end = load(args.run_id)
    out = paths.report_dir(args.run_id)
    lad = ladder(ev)
    lad.to_csv(out / "uladder.csv", index=False, float_format="%.5g")
    grand = fig_grand(ev, t_end, out)
    fig_uladder(lad, out)

    print(f"grand (U <= 4): {grand}")
    cols = ["U", "n_L4", "k_L4", "n_L5", "k_L5", "Hfix_n_L4", "Hfix_k_L4", "Hfix_n_L5", "Hfix_k_L5",
            "cum_f_L4", "cum_f_L5", "cum_ratio", "cum_ratio_err"]
    print(lad[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
