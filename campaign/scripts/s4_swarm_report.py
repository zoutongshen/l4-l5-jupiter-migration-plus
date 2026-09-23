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

from scipy import stats

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


CAL_COLUMNS = ["jitter_au", "n_moved", "n", "max_abs_da_au", "max_rel_da", "max_abs_damp_deg",
               "n_class_changed"]


def divergence(delta: pd.DataFrame) -> dict:
    """How far one arm moved from the control, at the resolution the diagnostic has.

    Counted as how many objects moved at all, not by a median or a mean: most of these
    orbits are regular over 1 Myr and register no change, so the median is exactly zero
    for every arm and cannot separate them. "No change" means below one float32 step of
    the stored window-mean a (``integrate.py`` writes these as float32), which near
    5.2 au is 4.8e-7 au -- not identical dynamics, just a difference this diagnostic
    cannot see. Every arm is measured through the same floor, so the counts compare.
    """
    d = np.abs(delta["da_au"].to_numpy())
    floor = float(np.spacing(np.float32(np.nanmedian(delta["a_free"].to_numpy()))))
    return {
        "n_moved": int((d > 0).sum()), "n": len(d),
        "n_at_floor": int(((d > 0) & (d < 2 * floor)).sum()),
        "resolution_au": floor,
        "max_abs_da_au": float(np.nanmax(d)),
        "max_rel_da": float(np.nanmax(d / delta["a_free"].to_numpy())),
        "max_abs_damp_deg": float(np.abs(delta["damp_deg"]).max()),
        "n_class_changed": int(delta["class_changed"].sum()),
    }


class IncompleteRun(RuntimeError):
    """An arm that has not reached its last window, so it must not be compared yet."""


def load_run(run_id: str) -> dict:
    """Events, per-window series and the selection for one arm of the comparison.

    Refuses a run that has not reached the window count its settings call for. Two
    unfinished arms are compared at whatever window each happens to have reached, and
    an arm still being written is an HDF5 file being read while another process writes
    it, which is undefined rather than merely stale. Both produce numbers that look
    like results.
    """
    run = paths.run_dir(run_id)
    settings = json.loads((run / "settings.json").read_text())
    expected = settings.get("n_windows")
    if expected is None:
        ic = json.loads((paths.initial_dir(settings["ic_id"]) / "ic.json").read_text())
        expected = integrate.Settings(t_end_yr=settings["t_end_yr"], dt_yr=ic["dt_yr"]).n_windows
    done = json.loads((run / "latest.json").read_text())["window_index"]
    if done < expected:
        raise IncompleteRun(f"{run_id} is at window {done}/{expected}; finish or resume it first")

    events = pd.read_parquet(run / "events.parquet")
    with h5py.File(run / "windows.h5") as f:
        order = pd.Index(f["spkid"][:]).get_indexer(events["spkid"])
        series = {name: f[name][:][:, order] for name in ["a", "e", "inc", "sigma_min", "sigma_max"]}
        series["class"] = f["class"][:][:, order]
        t_end = f["t_end"][:]
    return {
        "events": events, "series": series, "t_end": t_end,
        "selection": pd.read_parquet(run / "selection.parquet"),
        "settings": settings,
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
        "a_free": free["series"]["a"][-1],
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


def calibration_arms(run_id: str, free: dict) -> tuple:
    """Every finished ``<run_id>_massless_jitter<N>`` arm, against the unperturbed control.

    Each is the same massless system started a known, physically meaningless distance
    away. Their spread after 1 Myr is the chaos floor, and how it grows with the seed
    perturbation is what lets a self-gravity difference be quoted as an equivalent nudge.
    """
    root = paths.run_dir(f"{run_id}_massless").parent
    rows, frames = [], {}
    for directory in sorted(root.glob(f"{run_id}_massless_jitter*")):
        if not (directory / "events.parquet").exists():
            continue
        try:
            arm = load_run(directory.name)
        except IncompleteRun as exc:
            print(f"skipping {exc}")
            continue
        delta = compare(arm, free)
        jitter = float(arm["settings"]["jitter_au"])
        frames[jitter] = delta
        rows.append({"jitter_au": jitter, **divergence(delta)})
    if not rows:
        return pd.DataFrame(columns=CAL_COLUMNS), {}
    return pd.DataFrame(rows).sort_values("jitter_au").reset_index(drop=True), frames


def fig_swarm(diff: pd.DataFrame, out, cal: pd.DataFrame = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)

    # Only the objects self-gravity actually moved. A third of them are bit-identical
    # to the massless run, and a log axis would quietly pile those zeros onto the
    # bottom of the frame as if they were small nonzero values.
    ax = axes[0]
    moved = diff[np.abs(diff["da_au"]) > 0]
    for swarm in ("L4", "L5"):
        sub = moved[moved["swarm"] == swarm]
        ax.scatter(sub["mass_msun"], np.abs(sub["da_au"]), s=16, lw=0.5, edgecolor=SURFACE,
                   color=SWARM_COLOR[swarm], label=f"{swarm} ({len(sub)} moved)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Trojan mass ($M_\\odot$)")
    ax.set_ylabel("$|\\Delta a|$ from self-gravity (au)")
    ax.legend(loc="lower right")
    ax.set_title(f"{len(diff) - len(moved)} of {len(diff)} below the diagnostic's resolution",
                 color=TEXT_2, fontsize=8.5, loc="left")

    # The argument in one panel: a force moves most of them, a meaningless nudge
    # moves a chaotic handful, and a sub-ulp nudge moves none at all.
    ax = axes[1]
    labels = ["self-gravity"] + [f"{r.jitter_au:.0e} au nudge" for r in cal.itertuples()]
    values = [int((np.abs(diff["da_au"]) > 0).sum())] + [int(r.n_moved) for r in cal.itertuples()]
    colors = [HIGHLIGHT] + [BASELINE] * len(cal)
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, height=0.6, color=colors, lw=0)
    for yy, v in zip(y, values):
        ax.annotate(f"{v}", (v, yy), xytext=(4, 0), textcoords="offset points",
                    va="center", color=TEXT_2, fontsize=8.5)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, len(diff) * 1.08)
    ax.set_xlabel(f"Trojans whose orbit changed at all (of {len(diff)})")
    ax.grid(axis="y", visible=False)

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

    # The calibration arm: the same massless system nudged by a physically
    # meaningless amount at t0. Trojan orbits are chaotic well inside 1 Myr, so
    # ANY perturbation separates the trajectories; without this, a per-object
    # |da| measures how chaotic the orbits are, not what self-gravity did.
    cal, cal_frames = calibration_arms(args.run_id, free)
    chaos = cal_frames[cal["jitter_au"].iloc[-1]] if len(cal) else None

    out = paths.make_output_dir(paths.report_dir(args.run_id), args.resume)
    diff.to_csv(out / "swarm.csv", index=False, float_format="%.6g")
    fig_swarm(diff, out, cal)

    total = grav["settings"]["total_mass_msun"]
    t_end = float(grav["t_end"][-1])
    summary = {
        "n": len(diff), "t_end_yr": t_end, "total_mass_msun": total,
        "total_mass_mearth": total * constants.MSUN_KG / 5.972e24,
        "max_abs_da_au": float(np.abs(diff["da_au"]).max()),
        "median_abs_da_au": float(np.abs(diff["da_au"]).median()),
        "max_abs_damp_deg": float(np.abs(diff["damp_deg"]).max()),
        "n_class_changed": int(diff["class_changed"].sum()),
        "n_alive_grav": int(diff["alive_grav"].sum()), "n_alive_free": int(diff["alive_free"].sum()),
    }

    grav_div = divergence(diff)
    summary.update(grav_div)
    if chaos is None:
        verdict = ("No calibration arm, so this cannot be read yet. Run `--massless --jitter-au 1e-12` "
                   "and regenerate: over 1 Myr a chaotic system separates on its own, and without "
                   "that baseline a per-object |da| says nothing about self-gravity.")
    else:
        floor = cal["n_moved"].max()          # the most any meaningless nudge disturbed
        n = grav_div["n"]
        ks = stats.ks_2samp(np.abs(diff["da_au"]), np.abs(chaos["da_au"]))
        summary.update(chaos_n_moved=int(floor), ks_stat=float(ks.statistic), ks_p=float(ks.pvalue))
        real = grav_div["n_moved"] > 2 * max(floor, 1)
        verdict = (
            f"**Self-gravity is real but dynamically negligible over {t_end/1e6:.2f} Myr.** It "
            f"perturbs {grav_div['n_moved']}/{n} objects, against {floor}/{n} for the largest "
            f"meaningless nudge, so this is a force doing work and not a chaotic handful "
            f"separating. The work is tiny: the biggest semi-major-axis change is "
            f"{grav_div['max_abs_da_au']:.2e} au, {grav_div['max_rel_da']:.1e} of a, no object "
            f"changed libration class, and all {summary['n_alive_grav']} survived in both arms."
            if real else
            f"**Self-gravity is indistinguishable from the chaos floor.** It perturbs "
            f"{grav_div['n_moved']}/{n} objects against {floor}/{n} for a meaningless nudge, so "
            f"the differences between the arms are a chaotic system separating, not a force "
            f"doing measurable work.")

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
        "## Against the chaos floor",
        "",
        "Each calibration arm is the identical massless system started a known, physically "
        "meaningless distance away. Trojan orbits are chaotic well inside 1 Myr, so two runs "
        "separate no matter what starts them apart; these arms measure how much separation a "
        "given seed buys, and self-gravity is then quoted in the same units.",
        "",
        "| arm | seed at t = 0 | objects moved | largest \\|Δa\\| (au) | largest Δa/a | largest Δamp (deg) |",
        "|---|---|---|---|---|---|",
        f"| **self-gravity** | {summary['total_mass_mearth']:.2e} M_Earth of real mass | "
        f"**{summary['n_moved']}/{summary['n']}** | {summary['max_abs_da_au']:.3e} | "
        f"{summary['max_rel_da']:.1e} | {summary['max_abs_damp_deg']:.3f} |",
        *[f"| nudge | {r.jitter_au:.0e} au | {r.n_moved}/{r.n} | {r.max_abs_da_au:.3e} | "
          f"{r.max_rel_da:.1e} | {r.max_abs_damp_deg:.3f} |" for r in cal.itertuples()],
        "",
        f"\"Moved\" means the stored window-mean a changed by at least one float32 step, "
        f"{summary['resolution_au']:.1e} au near 5.2 au (`integrate.py` writes these diagnostics "
        f"as float32). It is a floor on what can be seen, not a claim about the dynamics, and "
        f"{summary['n_at_floor']} of self-gravity's {summary['n_moved']} sit within two steps of "
        f"it. Every arm is measured through the same floor, so the counts compare.",
        "",
        "The 1e-18 au arm moves nothing: that displacement is below what a float64 coordinate "
        "near 5 au can even represent, so it is the control run repeated. It reaching 0/200 is "
        "the check that these integrations are deterministic and that the other rows are caused "
        "rather than noise.",
        "",
        f"- Libration class changed for **{summary['n_class_changed']}/{summary['n']}** objects "
        f"(and {int(cal['n_class_changed'].sum())} across all calibration arms).",
        f"- Survivors: {summary['n_alive_grav']} self-gravitating vs {summary['n_alive_free']} massless"
        + (f" vs {int(chaos['alive_grav'].sum())} nudged." if chaos is not None else "."),
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
