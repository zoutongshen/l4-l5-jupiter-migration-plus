"""Stage 4: report the self-gravitating swarm against its massless twin.

Reads data/runs/<run_id>_{selfgrav,massless}/ (written by s4_run_swarm.py) and
writes campaign/reports/<run_id>/{report.md, swarm.csv, fig_swarm.png}.

The two runs hold the identical objects on identical initial conditions and
differ only in whether the Trojans have mass and feel each other. Any difference
between them is the size of the assumption that every long-term Trojan study
makes. The expected answer is "smaller than the integration error"; the point is
to put a number on it rather than assume it.

    python campaign/scripts/s4_swarm_report.py --run-id swarm
"""

import argparse
import json

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trojans import constants, integrate, paths
from trojans.outputs import write_json

# As s3_report.py -- the dataviz-validated categorical pair for the two swarms.
SWARM_COLOR = {"L4": "#2a78d6", "L5": "#eb6834"}
HIGHLIGHT = "#4a3aa7"
SURFACE, TEXT, TEXT_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
})


def load_run(run_id: str) -> dict:
    """Events, per-window series and the selection for one arm of the comparison."""
    run = paths.run_dir(run_id)
    events = pd.read_parquet(run / "events.parquet")
    with h5py.File(run / "windows.h5") as f:
        order = pd.Index(f["spkid"][:]).get_indexer(events["spkid"])
        series = {name: f[name][:][:, order] for name in ["a", "e", "inc", "sigma_min", "sigma_max"]}
        series["class"] = f["class"][:][:, order]
        t_end = f["t_end"][:]
    return {
        "events": events, "series": series, "t_end": t_end,
        "selection": pd.read_parquet(run / "selection.parquet"),
        "settings": json.loads((run / "settings.json").read_text()),
    }


def compare(grav: dict, free: dict) -> pd.DataFrame:
    """Per object: how far the two arms drifted apart by the end of the run."""
    assert list(grav["events"]["spkid"]) == list(free["events"]["spkid"]), "arms hold different objects"
    sel = free["selection"].set_index("spkid")
    amp = {k: 0.5 * (v["series"]["sigma_max"] - v["series"]["sigma_min"]) for k, v in
           (("grav", grav), ("free", free))}
    out = pd.DataFrame({
        "spkid": grav["events"]["spkid"],
        "H": sel.loc[grav["events"]["spkid"], "H"].to_numpy(),
        "swarm": sel.loc[grav["events"]["spkid"], "swarm"].to_numpy(),
        "mass_msun": [constants.mass_from_H(h) for h in sel.loc[grav["events"]["spkid"], "H"]],
        "alive_grav": grav["events"]["removal_reason"].astype(str) == "alive",
        "alive_free": free["events"]["removal_reason"].astype(str) == "alive",
        "da_au": grav["series"]["a"][-1] - free["series"]["a"][-1],
        "de": grav["series"]["e"][-1] - free["series"]["e"][-1],
        "dinc_deg": grav["series"]["inc"][-1] - free["series"]["inc"][-1],
        "damp_deg": amp["grav"][-1] - amp["free"][-1],
        "amp_free_deg": amp["free"][-1],
        "class_grav": grav["series"]["class"][-1],
        "class_free": free["series"]["class"][-1],
    })
    out["class_changed"] = out["class_grav"] != out["class_free"]
    return out


def fig_swarm(diff: pd.DataFrame, out) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)

    ax = axes[0]
    for swarm in ("L4", "L5"):
        sub = diff[diff["swarm"] == swarm]
        ax.scatter(sub["mass_msun"], np.abs(sub["da_au"]), s=14, lw=0.5, edgecolor=SURFACE,
                   color=SWARM_COLOR[swarm], label=f"{swarm} (n = {len(sub)})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Trojan mass ($M_\\odot$)")
    ax.set_ylabel("$|\\Delta a|$ between the two runs (au)")
    ax.legend(loc="upper left")

    ax = axes[1]
    lim = max(1e-4, float(np.abs(diff["damp_deg"]).max()) * 1.1)
    for swarm in ("L4", "L5"):
        sub = diff[diff["swarm"] == swarm]
        ax.scatter(sub["amp_free_deg"], sub["damp_deg"], s=14, lw=0.5, edgecolor=SURFACE,
                   color=SWARM_COLOR[swarm])
    ax.axhline(0.0, color=BASELINE, lw=1)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("libration amplitude, massless run (deg)")
    ax.set_ylabel("amplitude shift from self-gravity (deg)")

    fig.suptitle("Self-gravitating Trojans against the massless assumption", color=TEXT, fontsize=11)
    fig.savefig(out / "fig_swarm.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default="swarm")
    parser.add_argument("--resume", action="store_true", help="overwrite an existing report directory")
    args = parser.parse_args()

    grav = load_run(f"{args.run_id}_selfgrav")
    free = load_run(f"{args.run_id}_massless")
    diff = compare(grav, free)

    out = paths.make_output_dir(paths.report_dir(args.run_id), args.resume)
    diff.to_csv(out / "swarm.csv", index=False, float_format="%.6g")
    fig_swarm(diff, out)

    total = grav["settings"]["total_mass_msun"]
    t_end = float(grav["t_end"][-1])
    worst = diff.loc[np.abs(diff["da_au"]).idxmax()]
    summary = {
        "n": len(diff), "t_end_yr": t_end, "total_mass_msun": total,
        "total_mass_mearth": total * constants.MSUN_KG / 5.972e24,
        "max_abs_da_au": float(np.abs(diff["da_au"]).max()),
        "median_abs_da_au": float(np.abs(diff["da_au"]).median()),
        "max_abs_damp_deg": float(np.abs(diff["damp_deg"]).max()),
        "n_class_changed": int(diff["class_changed"].sum()),
        "n_alive_grav": int(diff["alive_grav"].sum()), "n_alive_free": int(diff["alive_free"].sum()),
    }

    verdict = ("No measurable effect: self-gravity moves nothing further than the integrator does."
               if summary["max_abs_damp_deg"] < 0.1 and summary["n_class_changed"] == 0 else
               "**Self-gravity moves things.** Worth a closer look before the campaign leans on massless particles.")

    lines = [
        f"# Self-gravitating Trojan swarm ({args.run_id})",
        "",
        f"{summary['n']} Trojans -- the largest per swarm, which carry the swarm's mass -- "
        f"for {t_end/1e6:.2f} Myr, run twice: once with real H-derived masses and full mutual "
        f"gravity, once massless. Identical objects, identical initial conditions, WHFast, "
        f"dt = 1% P_J.",
        "",
        f"Total mass carried: **{total:.3e} Msun = {summary['total_mass_mearth']:.2e} M_Earth** "
        f"(albedo {grav['settings']['albedo']}, density {grav['settings']['density_kg_m3']:.0f} kg/m^3).",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        f"- Largest |Δa| between the two runs: **{summary['max_abs_da_au']:.3e} au** "
        f"(median {summary['median_abs_da_au']:.3e}), on spkid {int(worst['spkid'])}.",
        f"- Largest libration-amplitude shift: **{summary['max_abs_damp_deg']:.3e} deg**.",
        f"- Libration class changed for **{summary['n_class_changed']}/{summary['n']}** objects.",
        f"- Survivors: {summary['n_alive_grav']} self-gravitating vs {summary['n_alive_free']} massless.",
        "",
        "## Caveats",
        "",
        f"- This is the top {grav['settings']['n_per_swarm']} per swarm, not all 16,445. They carry most "
        "of the mass but not all of it, and the many small bodies are not represented.",
        "- Masses come from one albedo and one density; good to a factor of a few per object.",
        f"- {t_end/1e6:.2f} Myr, not 4.5 Gyr. A null here bounds the short-term effect, and mutual "
        "scattering accumulates, so it does not licence the assumption over the full campaign.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    write_json(out / "summary.json", summary)
    print(f"wrote {out}/report.md, swarm.csv, fig_swarm.png")
    print(verdict)


if __name__ == "__main__":
    main()
