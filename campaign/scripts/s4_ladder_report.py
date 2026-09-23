"""Stage 4: report the mass ladder -- where the co-orbital threshold actually falls.

Reads data/runs/<run_id>/ladder.parquet (written by s4_run_ladder.py) and writes
campaign/reports/<run_id>/{report.md, ladder.csv, fig_ladder.png}.

The observable is whether sigma keeps librating, not the campaign's Hill-radius
escape: above the threshold a massive Trojan rearranges the co-orbital region
rather than politely leaving it, so "still in the 1:1 resonance at t_end" is the
question that stays meaningful at every rung.

    python campaign/scripts/s4_ladder_report.py --run-id mass_ladder
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trojans import constants, paths
from trojans.outputs import write_json

# Palette and rcParams as s3_report.py -- the dataviz-validated categorical pair
# for L4/L5. Kept in step with that file by hand; the two stages render the same
# swarms and must not drift apart.
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

# The smallest nonzero rung, used to place the massless rung on a log axis.
FLOOR_MJUP = 1e-10


def binomial(k, n):
    """Fraction and its standard error, guarding n = 0 and the 0/1 ends."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    frac = np.divide(k, n, out=np.full_like(k, np.nan), where=n > 0)
    err = np.sqrt(np.clip(frac * (1 - frac), 0, None) / np.where(n > 0, n, 1))
    return frac, np.where(n > 0, err, np.nan)


def rung_table(ladder: pd.DataFrame) -> pd.DataFrame:
    """One row per rung: how many survived, how fast the rest left, how clean the integration was."""
    rows = []
    for m, grp in ladder.groupby("m_jup"):
        alive = grp["fate"] == "librating"
        left = grp["t_leave_yr"].notna()
        row = {
            "m_jup": m, "m_msun": grp["m_msun"].iloc[0], "mu": grp["mu"].iloc[0],
            "n": len(grp), "n_librating": int(alive.sum()),
            "n_left": int((grp["fate"] == "left").sum()),
            "n_removed": int((grp["fate"] == "removed").sum()),
        }
        # How they leave, not just that they do: tadpole -> horseshoe -> circulating.
        for name in ("horseshoe", "circulating", "quasi_satellite"):
            row[f"n_{name}"] = int((grp["libration_class"] == name).sum())
        row["f_librating"], row["f_err"] = (float(x) for x in binomial(alive.sum(), len(grp)))
        for swarm in ("L4", "L5"):
            sub = grp[grp["swarm"] == swarm]
            f, e = binomial((sub["fate"] == "librating").sum(), len(sub))
            row[f"n_{swarm}"], row[f"f_{swarm}"], row[f"f_{swarm}_err"] = len(sub), float(f), float(e)
        row["median_t_leave_yr"] = float(grp.loc[left, "t_leave_yr"].median()) if left.any() else np.nan
        row["max_dE_rel"] = float(grp["dE_rel"].max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("m_jup").reset_index(drop=True)


def threshold_bracket(table: pd.DataFrame, drop: float = 0.5) -> tuple:
    """The two rungs the survival fraction falls across: the empirical threshold."""
    below = table[table["f_librating"] >= drop]
    above = table[table["f_librating"] < drop]
    if below.empty or above.empty:
        return np.nan, np.nan
    return float(below["m_jup"].max()), float(above["m_jup"].min())


def fig_ladder(table: pd.DataFrame, out) -> None:
    x = table["m_jup"].replace(0.0, FLOOR_MJUP)
    massless = table[table["m_jup"] == 0.0]
    routh = constants.routh_trojan_mass_limit() / constants.GIANT_MASS["jupiter"]
    lw_mjup = 24.0

    # Marker shape carries identity alongside colour, so the swarms stay separable
    # where their curves coincide (they do, over most of the ladder) and in print.
    # L4's circle is drawn first and larger, so its rim still shows where the two
    # curves coincide exactly -- which they do over most of the ladder.
    marker, size = {"L4": "o", "L5": "s"}, {"L4": 6.5, "L5": 4.0}

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)

    ax = axes[0]
    for swarm in ("L4", "L5"):
        f, e = table[f"f_{swarm}"] * 100, table[f"f_{swarm}_err"] * 100
        ax.fill_between(x, f - e, f + e, color=SWARM_COLOR[swarm], alpha=0.15, lw=0)
        ax.plot(x, f, color=SWARM_COLOR[swarm], lw=1.5, marker=marker[swarm], ms=size[swarm],
                mec=SURFACE, mew=0.8, label=f"{swarm} (n = {int(table[f'n_{swarm}'].iloc[0])})")
    ends = {s: table[f"f_{s}"].iloc[-1] * 100 for s in ("L4", "L5")}
    apart = abs(ends["L4"] - ends["L5"]) > 6
    for swarm, dy in (("L4", 0 if apart else 6), ("L5", 0 if apart else -6)):
        ax.annotate(swarm, (x.iloc[-1], ends[swarm]), xytext=(7, dy), textcoords="offset points",
                    va="center", color=TEXT_2, fontsize=8)
    if not massless.empty:
        ax.axhline(massless["f_librating"].iloc[0] * 100, color=BASELINE, lw=1, ls=":")
        ax.annotate("massless", (FLOOR_MJUP, massless["f_librating"].iloc[0] * 100),
                    xytext=(2, -11), textcoords="offset points", color=MUTED, fontsize=7.5)
    ax.set_ylabel("still librating at $t_{\\rm end}$ (%)")
    ax.set_ylim(-5, 108)
    ax.legend(loc="lower left")

    ax = axes[1]
    ok = table["median_t_leave_yr"].notna()
    ax.plot(x[ok], table.loc[ok, "median_t_leave_yr"], color=TEXT_2, lw=1.5, marker="D", ms=4,
            mec=SURFACE, mew=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("median time to leave the resonance (yr)")

    # One x-range for both panels, so the two views line up mass for mass.
    lo, hi = FLOOR_MJUP / 3, max(x.max(), routh) * 3
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.set_xlabel("Trojan mass ($M_{\\rm Jup}$)")
        top = ax.get_ylim()[1]
        # Alternate the label heights: 24 and 40.9 M_Jup are close on a log axis.
        for (value, label), dy in (((lw_mjup, "Liberato & Winter"), -9), ((routh, "Routh"), -78)):
            ax.axvline(value, color=BASELINE, lw=1, ls="--")
            ax.annotate(label, (value, top), xytext=(-3, dy), textcoords="offset points",
                        rotation=90, ha="right", va="top", color=MUTED, fontsize=7.5)

    fig.suptitle("Jupiter Trojans: survival against the mass they carry", color=TEXT, fontsize=11)
    fig.savefig(out / "fig_ladder.png", dpi=200)
    plt.close(fig)


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = [f"{v:.4g}" if isinstance(v, float) else f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)
                 for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default="mass_ladder")
    parser.add_argument("--resume", action="store_true", help="overwrite an existing report directory")
    args = parser.parse_args()

    run = paths.run_dir(args.run_id)
    ladder = pd.read_parquet(run / "ladder.parquet")
    settings = json.loads((run / "settings.json").read_text())

    out = paths.make_output_dir(paths.report_dir(args.run_id), args.resume)
    table = rung_table(ladder)
    table.to_csv(out / "ladder.csv", index=False, float_format="%.6g")
    fig_ladder(table, out)

    routh = constants.routh_trojan_mass_limit() / constants.GIANT_MASS["jupiter"]
    lo, hi = threshold_bracket(table)
    cpu_h = ladder["cpu_s"].sum() / 3600.0
    top = table.iloc[-1]
    massless = table[table["m_jup"] == 0.0]
    hektor = table[np.isclose(table["m_jup"], 1e-9)]

    shown = table[["m_jup", "mu", "n", "n_librating", "f_librating", "f_L4", "f_L5",
                   "n_horseshoe", "n_circulating", "n_removed", "median_t_leave_yr", "max_dE_rel"]]

    lines = [
        f"# Mass ladder ({args.run_id})",
        "",
        f"{settings['n_objects']} catalogue Trojans, {len(settings['rungs_mjup'])} masses, "
        f"{settings['t_end_yr']:.0e} yr each, IAS15, one massive Trojan per simulation. "
        f"Same objects at every rung (ic `{settings['ic_id']}`, seed {settings['seed']}).",
        "",
        "## Where the threshold falls",
        "",
        f"- Empirical: survival drops below 50% between **{lo:g} and {hi:g} M_Jup**.",
        f"- Routh (1875) linear limit for this model: **{routh:.1f} M_Jup** "
        f"(mu = {constants.coorbital_mu(constants.routh_trojan_mass_limit()):.4f}).",
        f"- Liberato & Winter (2020): mu = {settings['liberato_winter_mu']}, i.e. **24.0 M_Jup**.",
        "",
        f"The control rung above Routh ({top['m_jup']:g} M_Jup) leaves "
        f"{top['f_librating']:.0%} librating"
        + (" -- as it must, or nothing below it is trustworthy." if top["f_librating"] < 0.05
           else " -- **this should be ~0%; the setup is suspect.**"),
        "",
        "## Where the real Trojans sit",
        "",
        f"(624) Hektor, the largest known Trojan, is ~1e-9 M_Jup. At that rung "
        f"{hektor['f_librating'].iloc[0]:.0%} still librate, against "
        f"{massless['f_librating'].iloc[0]:.0%} for massless test particles"
        if not hektor.empty and not massless.empty else "",
        "",
        "Real Trojan masses are ten orders of magnitude below the threshold, so treating them as "
        "massless costs nothing for single-body dynamics. Whether their *mutual* gravity matters is a "
        "separate question -- see the self-gravitating swarm run.",
        "",
        "## Per rung",
        "",
        _md_table(shown),
        "",
        "`f_L4` and `f_L5` are the surviving fractions in each swarm; a mass dependence in their "
        "ratio would be a mass axis on the asymmetry question.",
        "",
        "## Cost",
        "",
        f"{len(ladder):,} integrations, {cpu_h:.2f} CPU-h, "
        f"{ladder['cpu_s'].mean():.1f} CPU-s each. Worst |dE/E| over the whole ladder: "
        f"{ladder['dE_rel'].max():.1e}.",
        "",
    ]
    (out / "report.md").write_text("\n".join(x for x in lines if x is not None) + "\n")
    write_json(out / "summary.json", {
        "threshold_bracket_mjup": [lo, hi], "routh_limit_mjup": routh,
        "n_runs": len(ladder), "cpu_h": cpu_h, "max_dE_rel": float(ladder["dE_rel"].max()),
    })
    print(f"wrote {out}/report.md, ladder.csv, fig_ladder.png")
    print(f"threshold between {lo:g} and {hi:g} M_Jup (Routh: {routh:.1f})")


if __name__ == "__main__":
    main()
