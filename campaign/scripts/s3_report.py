"""Stage 3.4: small tracked report for a merged run, in campaign/reports/<run_id>/.

  report.md             headline tables: escapes per sample and swarm, the 1 Myr
                        libration check per label region, cost
  escapes.csv           the escape table behind report.md
  cleaning.csv          the 1 Myr check behind report.md
  fig_escape.png        cumulative fraction with a first 2.1 R_H encounter vs time
  fig_libration.png     fraction no longer a tadpole of its own swarm vs time
  fig_fate_map.png      (B, C) and (A, C) at t0, escapers highlighted

Escape fractions carry binomial errors over objects (no clones), so each
object's single trajectory includes its chaotic scatter.
"""

import argparse
import json

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trojans import integrate, paths
from trojans.outputs import write_json

# Reference palette (dataviz skill): categorical slots 1-2 validated for L4/L5.
SWARM_COLOR = {"L4": "#2a78d6", "L5": "#eb6834"}
HIGHLIGHT = "#4a3aa7"
SURFACE, TEXT, TEXT_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CLEANING_SPAN_YR = 1.0e6

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
})


def well_determined(events: pd.DataFrame) -> np.ndarray:
    """JPL orbit condition code <= 4. Poorer orbits escape far more often, and the
    two swarms hold different shares of them, so mixing them in biases the comparison."""
    return (pd.to_numeric(events["condition_code"], errors="coerce") <= 4).to_numpy()


def samples(events: pd.DataFrame) -> dict:
    v24 = events["v24_selected"].to_numpy()
    good = well_determined(events)
    return {
        "all integrated": np.ones(len(events), bool),
        "V24-selected": v24,
        "V24-selected, well determined (U <= 4)": v24 & good,
        "complete (H <= 13.6), V24-selected": (events["complete"] & events["v24_selected"]).to_numpy(),
        "Di Sisto 2013 numbered": events["disisto2013"].to_numpy(),
    }


def binomial(k, n):
    f = k / n if n else np.nan
    return f, (np.sqrt(f * (1 - f) / n) if n else np.nan)


def escape_table(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, mask in samples(events).items():
        for swarm in ("L4", "L5"):
            sel = events[mask & (events["swarm"] == swarm).to_numpy()]
            n = len(sel)
            k21 = int(sel["t_hill_record"].notna().sum())
            k1 = int((sel["removal_reason"] == "hill").sum())
            f21, e21 = binomial(k21, n)
            f1, e1 = binomial(k1, n)
            rows.append({"sample": name, "swarm": swarm, "n": n,
                         "encounter_2.1RH": k21, "frac_2.1RH": f21, "err_2.1RH": e21,
                         "removed_1RH": k1, "frac_1RH": f1, "err_1RH": e1,
                         "removed_other": int(((sel["removal_reason"] != "alive") & (sel["removal_reason"] != "hill")).sum())})
    return pd.DataFrame(rows)


def ratio_table(escapes: pd.DataFrame) -> pd.DataFrame:
    """The asymmetry: escape-fraction ratio, Di Sisto's erosion amplifier, and N4/N5.

    amplifier = S4/S5 = (1 - f4) / (1 - f5), the factor by which erosion multiplies
    N4/N5 over the run. Errors are binomial and propagated in quadrature; with only
    a handful of escapers they are indicative, not exact.
    """
    rows = []
    for sample, group in escapes.groupby("sample", sort=False):
        g = group.set_index("swarm")
        n4, n5 = int(g.at["L4", "n"]), int(g.at["L5", "n"])
        f4, f5 = g.at["L4", "frac_2.1RH"], g.at["L5", "frac_2.1RH"]
        e4, e5 = g.at["L4", "err_2.1RH"], g.at["L5", "err_2.1RH"]
        with np.errstate(divide="ignore", invalid="ignore"):
            escape_ratio = f5 / f4
            escape_ratio_err = escape_ratio * np.sqrt((e4 / f4) ** 2 + (e5 / f5) ** 2)
        amplifier = (1 - f4) / (1 - f5)
        amplifier_err = amplifier * np.sqrt((e4 / (1 - f4)) ** 2 + (e5 / (1 - f5)) ** 2)
        rows.append({
            "sample": sample,
            "N4/N5 at t0": n4 / n5,
            "N4/N5 at t_end": (n4 * (1 - f4)) / (n5 * (1 - f5)),
            "escape ratio L5/L4": escape_ratio,
            "escape ratio err": escape_ratio_err,
            "amplifier S4/S5": amplifier,
            "amplifier err": amplifier_err,
            "n_escapers L4+L5": int(g.at["L4", "encounter_2.1RH"] + g.at["L5", "encounter_2.1RH"]),
        })
    return pd.DataFrame(rows)


def not_own_tadpole(cls: np.ndarray, swarm: np.ndarray) -> np.ndarray:
    own = np.where(swarm == "L4", integrate.TADPOLE_L4, integrate.TADPOLE_L5)
    return cls != own[None, :]


def cleaning_table(events, cls, t_end) -> pd.DataFrame:
    early = t_end <= CLEANING_SPAN_YR + 1e-6
    failed = not_own_tadpole(cls[early], events["swarm"].to_numpy()).any(axis=0)
    t = events.assign(failed_1myr=failed)
    return (t.groupby(["label_region", "swarm"])
            .agg(n=("spkid", "size"), failed_1myr=("failed_1myr", "sum"),
                 of_which_removed=("t_removed", lambda s: int((s <= CLEANING_SPAN_YR).sum())))
            .reset_index())


def _direct_labels(ax, x, ends: dict):
    """End-of-line labels for L4/L5, pushed apart vertically when the lines end together."""
    close = abs(ends["L4"] - ends["L5"]) < 0.04 * max(abs(np.diff(ax.get_ylim())[0]), 1e-9)
    for swarm, y in ends.items():
        dy = (5 if swarm == "L4" else -5) if close else 0
        ax.annotate(swarm, (x, y), xytext=(5, dy), textcoords="offset points", va="center", color=TEXT_2, fontsize=8)
        ax.plot([x], [y], "o", ms=3, color=SWARM_COLOR[swarm])


def fig_escape(events, t_run_end, out):
    names = ["V24-selected, well determined (U <= 4)", "complete (H <= 13.6), V24-selected"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True, constrained_layout=True)
    grid = np.linspace(0, t_run_end, 400)
    for ax, name in zip(axes, names):
        mask = samples(events)[name]
        ends = {}
        for swarm in ("L4", "L5"):
            sel = events[mask & (events["swarm"] == swarm).to_numpy()]
            n = len(sel)
            times = np.sort(sel["t_hill_record"].dropna().to_numpy())
            frac = np.searchsorted(times, grid, side="right") / max(n, 1)
            err = np.sqrt(frac * (1 - frac) / max(n, 1))
            color = SWARM_COLOR[swarm]
            ax.fill_between(grid / 1e6, 100 * (frac - err), 100 * (frac + err), color=color, alpha=0.15, lw=0)
            ax.plot(grid / 1e6, 100 * frac, color=color, lw=1.5, label=f"{swarm} (n = {n:,})")
            ends[swarm] = 100 * frac[-1]
        _direct_labels(ax, grid[-1] / 1e6, ends)
        ax.set_title(name, loc="left")
        ax.set_xlabel("time since t0 [Myr]")
    axes[0].set_ylabel("first encounter within 2.1 R_H [% of swarm]")
    axes[0].legend(loc="upper left")
    fig.savefig(out / "fig_escape.png", dpi=200)
    plt.close(fig)


def fig_libration(events, cls, t_end, out):
    names = ["V24-selected, well determined (U <= 4)", "complete (H <= 13.6), V24-selected"]
    swarm = events["swarm"].to_numpy()
    off = not_own_tadpole(cls, swarm)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True, constrained_layout=True)
    for ax, name in zip(axes, names):
        mask = samples(events)[name]
        ends = {}
        for s in ("L4", "L5"):
            cols = mask & (swarm == s)
            frac = 100 * off[:, cols].mean(axis=1)
            ax.plot(t_end / 1e6, frac, color=SWARM_COLOR[s], lw=1.5, label=f"{s} (n = {cols.sum():,})")
            ends[s] = frac[-1]
        _direct_labels(ax, t_end[-1] / 1e6, ends)
        ax.set_title(name, loc="left")
        ax.set_xlabel("time since t0 [Myr]")
    axes[0].set_ylabel("not a tadpole of its swarm in window [%]")
    axes[0].legend(loc="upper left")
    fig.savefig(out / "fig_libration.png", dpi=200)
    plt.close(fig)


def fig_fate_map(events, out):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), sharex=True, constrained_layout=True)
    escaped = events["t_hill_record"].notna().to_numpy()
    for j, swarm in enumerate(("L4", "L5")):
        in_swarm = (events["swarm"] == swarm).to_numpy() & events["v24_selected"].to_numpy()
        for i, (ycol, ylabel) in enumerate([("B", "B (eccentricity amplitude)"), ("A", "A (inclination amplitude)")]):
            ax = axes[i, j]
            stay, go = in_swarm & ~escaped, in_swarm & escaped
            ax.scatter(events.loc[stay, "C"], events.loc[stay, ycol], s=2, color=BASELINE, lw=0,
                       label=f"no encounter ({stay.sum():,})", rasterized=True)
            ax.scatter(events.loc[go, "C"], events.loc[go, ycol], s=9, color=HIGHLIGHT, lw=0,
                       label=f"encounter within 2.1 R_H ({go.sum():,})")
            ax.set_xlim(1.45, 2.6)
            ax.set_ylabel(ylabel)
            if i == 0:
                ax.set_title(f"{swarm}, V24-selected, at t0", loc="left")
                ax.legend(loc="upper right", markerscale=2)
            else:
                ax.set_xlabel("C (libration integral; 1.5 = Lagrange point)")
    fig.savefig(out / "fig_fate_map.png", dpi=200)
    plt.close(fig)


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = [f"{v:.4f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, (int, np.integer)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true", help="overwrite an existing report directory")
    args = parser.parse_args()

    merged = paths.merged_dir(args.run_id)
    events = pd.read_parquet(merged / "events.parquet")
    with h5py.File(merged / "windows.h5") as f:
        order = pd.Index(f["spkid"][:]).get_indexer(events["spkid"])
        cls = f["class"][:][:, order]
        t_end = f["t_end"][:]
    cost = json.loads((merged / "cost.json").read_text())

    out = paths.make_output_dir(paths.report_dir(args.run_id), args.resume)
    escapes = escape_table(events)
    ratios = ratio_table(escapes)
    cleaning = cleaning_table(events, cls, t_end)
    escapes.to_csv(out / "escapes.csv", index=False, float_format="%.5f")
    ratios.to_csv(out / "ratios.csv", index=False, float_format="%.5f")
    cleaning.to_csv(out / "cleaning.csv", index=False)
    write_json(out / "cost.json", cost)
    fig_escape(events, float(t_end[-1]), out)
    fig_libration(events, cls, t_end, out)
    fig_fate_map(events, out)

    show = escapes[["sample", "swarm", "n", "encounter_2.1RH", "frac_2.1RH", "err_2.1RH", "removed_1RH", "frac_1RH"]]
    (out / "report.md").write_text("\n".join([
        f"# {args.run_id}",
        "",
        f"{cost['n_trojans']:,} Trojans integrated for {cost['t_end_yr'] / 1e6:g} Myr "
        f"(dt = {cost['dt_yr']:.5f} yr, sampled every {cost['check_every_steps']} steps). No clones: "
        "errors are binomial over objects and include each trajectory's chaotic scatter. "
        f"This is a {cost['t_end_yr'] / 1e6:g} Myr run, not a 4.5 Gyr erosion rate.",
        "",
        "## Encounters with the giants",
        "",
        "`encounter_2.1RH`: first entry within 2.1 Hill radii (Di Sisto 2019's escape criterion). "
        "`removed_1RH`: entry within 1 Hill radius (Di Sisto 2014's), where the particle is removed.",
        "",
        "**Read the orbit-quality row, not the raw one.** Objects with JPL condition code >= 5 "
        "escape at percent level here, and L4 carries twice L5's share of them (mostly recent "
        "discoveries with short arcs), so the raw sample shows an L4 excess that is an observational "
        "artefact rather than dynamics.",
        "",
        _md_table(show),
        "",
        "## The asymmetry",
        "",
        "`amplifier S4/S5` = (1 - f4) / (1 - f5) is Di Sisto's erosion amplifier: the factor by "
        "which this run multiplies N4/N5. For reference, over 4.5 Gyr Di Sisto 2014 got "
        "1.071 +- 0.025 and Di Sisto 2019 got 1.020 +- 0.023. Errors are binomial, propagated in "
        "quadrature, and are only indicative where a swarm has a handful of escapers.",
        "",
        _md_table(ratios),
        "",
        "![escape](fig_escape.png)",
        "",
        "## Libration over the first 1 Myr (V24's cleaning step, per label definition)",
        "",
        "`failed_1myr`: at least one 10 kyr window in the first 1 Myr in which the object was not "
        "a tadpole of its t0 swarm (including removal).",
        "",
        _md_table(cleaning),
        "",
        "![libration](fig_libration.png)",
        "",
        "![fate map](fig_fate_map.png)",
        "",
        "## Cost",
        "",
        f"{cost['cpu_s'] / 3600:.1f} CPU-h in total; {cost['cpu_s_per_trojan_myr']:.2f} CPU-s per Trojan-Myr; "
        f"longest chunk session {cost['wall_s_max_chunk'] / 3600:.1f} h wall. "
        f"Naive projection to 4.5 Gyr for this many Trojans (no removals): "
        f"{cost['cpu_s_per_trojan_myr'] * cost['n_trojans'] * 4500 / 3.15576e7:.1f} CPU-yr.",
        "",
    ]))
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
