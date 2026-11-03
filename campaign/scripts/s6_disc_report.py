"""Stage 6 report: who ends up as an L4 or L5 Trojan, binned in D, I and e.

A particle counts as a Trojan of a swarm if it is alive at the end and every
10 kyr window in the last --classify-yr is a tadpole of that swarm (the
docs/TROJAN_DEFINITION.md classifier). D, I and e are means over those windows;
D is the libration half-amplitude (sigma_max - sigma_min) / 2, as in
s3_distributions.py. Counts are also given as fractions of the whole disc, so
"more L4" can be told apart from "fewer L5".

Outputs (campaign/reports/<run-id>/):
  binned.csv         N4, N5 per bin of final D, I, e and of starting e0, I0; uniform disc
  binned_physical.csv  the same, reweighted to a Rayleigh disc (see PHYSICAL)
  formation.csv      per window: how many particles are L4 / L5 tadpoles in that window
  summary.json, particles.parquet

    python campaign/scripts/s6_disc_report.py --run-id disc_grow
"""

import argparse
import json
import warnings

import h5py
import numpy as np
import pandas as pd

from trojans import constants, diagnostics, integrate, paths
from trojans.outputs import write_json

# e and D edges as in s3_distributions.py; I split at 15 deg like ratio_by_inclination.csv.
EDGES = {
    "D": [0, 10, 20, 30, 60, 181],
    "inc": [0, 5, 10, 15, 20, 30, 40, 90],
    "e": [0, 0.05, 0.1, 0.15, 0.25, 1],
    "inc_split": [0, 15, 90],
    "e0": [0, 0.05, 0.1, 0.15, 0.2, 0.3],
    "i0": [0, 5, 10, 15, 20, 30, 40],
}

# The reweighted "physical" view: Pirani+2019b's pre-stirred asteroid-belt disc (§2.4,
# after Minton & Malhotra 2010), Gaussians cut at zero, here also cut at the uniform
# disc's box (e <= 0.3, I <= 40 deg). Weights = target density / uniform density.
PHYSICAL = {"source": "Pirani+2019b pre-stirred (asteroid belt)",
            "mu_e": 0.15, "sigma_e": 0.07, "mu_i_deg": 8.5, "sigma_i_deg": 7.0}


def load_run(run_id: str, classify_yr: float):
    # Removed particles have empty windows; their NaN means are expected.
    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
    rows, formation = [], []
    for chunk in sorted((paths.run_dir(run_id) / "chunks").iterdir()):
        with h5py.File(chunk / "windows.h5", "r") as f:
            t_end = f["t_end"][:]
            cls_all = f["class"][:]
            last = t_end > t_end[-1] - classify_yr
            cls = cls_all[last, :]
            ident = f["spkid"][:]
            half = 0.5 * (f["sigma_max"][last, :] - f["sigma_min"][last, :])
            means = {k: np.nanmean(f[k][last, :], axis=0) for k in ("inc", "e", "a")}
        formation.append(pd.DataFrame({"t_end_yr": t_end,
                                       "L4": (cls_all == integrate.TADPOLE_L4).sum(axis=1),
                                       "L5": (cls_all == integrate.TADPOLE_L5).sum(axis=1)}))
        ev = pd.read_parquet(chunk / "events.parquet").set_index("spkid").loc[ident]
        alive = (ev["removal_reason"] == "alive").to_numpy()
        swarm = np.where(alive & np.all(cls == integrate.TADPOLE_L4, axis=0), "L4",
                         np.where(alive & np.all(cls == integrate.TADPOLE_L5, axis=0), "L5", ""))
        rows.append(pd.DataFrame({"id": ident, "swarm": swarm, "D": np.nanmean(half, axis=0), **means,
                                  "n_windows": int(last.sum())}))
    formation = pd.concat(formation).groupby("t_end_yr", as_index=False).sum()
    return pd.concat(rows, ignore_index=True), formation


def starting_elements(run_id: str) -> pd.DataFrame:
    initial = np.load(paths.run_dir(run_id) / "disc_initial.npz")
    el = diagnostics.elements(initial["pos"], initial["vel"], constants.G * constants.CENTRAL_MASS)
    return pd.DataFrame({"id": initial["id"], "a0": el["a"], "e0": el["e"], "i0": el["inc"]})


def physical_weights(particles: pd.DataFrame, disc: dict) -> np.ndarray:
    """Gaussian(e) x Gaussian(I) on the disc's box, divided by the uniform density; mean 1."""
    p = PHYSICAL
    e, i = particles["e0"].to_numpy(), particles["i0"].to_numpy()
    w = np.exp(-(e - p["mu_e"])**2 / (2 * p["sigma_e"]**2)) * np.exp(-(i - p["mu_i_deg"])**2 / (2 * p["sigma_i_deg"]**2))
    inside = (e <= disc["e_max"]) & (i <= disc["inc_max_deg"])
    w = np.where(inside, w, 0.0)
    return w / w.mean()


def binned(particles: pd.DataFrame, column: str, edges, weights=None) -> pd.DataFrame:
    """Per bin: N4, N5 (weighted sums if weights), their ratio, and fractions of the disc.

    With weights, the error uses the effective counts n_eff = (sum w)^2 / sum w^2.
    """
    w = np.ones(len(particles)) if weights is None else weights
    frame = particles.assign(w=w, w2=w**2, bin=pd.cut(particles[column], edges, right=False))
    n_disc = w.sum()
    trojans = frame[frame["swarm"] != ""]
    sums = trojans.groupby(["bin", "swarm"], observed=False)[["w", "w2"]].sum().unstack(fill_value=0)
    sums = sums.reindex(columns=pd.MultiIndex.from_product([["w", "w2"], ["L4", "L5"]]), fill_value=0)
    in_bin = frame.groupby("bin", observed=False)["w"].sum()
    n4, n5 = sums[("w", "L4")], sums[("w", "L5")]
    with np.errstate(divide="ignore", invalid="ignore"):
        eff4 = n4**2 / sums[("w2", "L4")]
        eff5 = n5**2 / sums[("w2", "L5")]
        table = pd.DataFrame({
            "variable": column, "bin": sums.index.astype(str), "L4": n4.to_numpy(), "L5": n5.to_numpy(),
            "N4/N5": (n4 / n5).to_numpy(), "err": (n4 / n5 * np.sqrt(1 / eff4 + 1 / eff5)).to_numpy(),
            "f4_per_1e4_disc": (1e4 * n4 / n_disc).to_numpy(), "f5_per_1e4_disc": (1e4 * n5 / n_disc).to_numpy(),
        })
        if column in ("e0", "i0"):
            # Capture probability of a particle that STARTS in this bin.
            table["p_capture"] = ((n4 + n5) / in_bin).to_numpy()
    return table


def all_tables(particles: pd.DataFrame, weights=None) -> pd.DataFrame:
    p = particles.assign(inc_split=particles["inc"])
    return pd.concat([binned(p, col, edges, weights) for col, edges in EDGES.items()], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--classify-yr", type=float, default=1.0e6)
    args = parser.parse_args()

    settings = json.loads((paths.run_dir(args.run_id) / "settings.json").read_text())
    particles, formation = load_run(args.run_id, args.classify_yr)
    particles = particles.merge(starting_elements(args.run_id), on="id", how="left")
    trojans = particles[particles["swarm"] != ""]
    n_disc = len(particles)

    # Twins: particle 2k and 2k + 1 start as mirror images. Cross-tabulate their fates.
    pair = (particles["id"] - particles["id"].min()) // 2
    fates = particles.assign(pair=pair, fate=particles["swarm"].replace("", "none"))
    twin = fates.sort_values("id").groupby("pair")["fate"].agg(lambda s: "+".join(sorted(s)))

    tables = all_tables(particles)
    weights = physical_weights(particles, settings["disc"])
    physical = all_tables(particles, weights)

    out = paths.report_dir(args.run_id)
    out.mkdir(parents=True, exist_ok=True)
    tables.to_csv(out / "binned.csv", index=False)
    physical.to_csv(out / "binned_physical.csv", index=False)
    formation.to_csv(out / "formation.csv", index=False)
    particles.to_parquet(out / "particles.parquet", index=False)

    n4, n5 = int((trojans["swarm"] == "L4").sum()), int((trojans["swarm"] == "L5").sum())
    is4, is5 = (particles["swarm"] == "L4").to_numpy(), (particles["swarm"] == "L5").to_numpy()
    summary = {
        "run_id": args.run_id, "arm": settings["arm"], "n_disc": n_disc, "N4": n4, "N5": n5,
        "N4/N5": n4 / n5 if n5 else None,
        "err": (n4 / n5) * np.sqrt(1 / n4 + 1 / n5) if n4 and n5 else None,
        "capture_fraction": (n4 + n5) / n_disc,
        "classify_windows": int(particles["n_windows"].iloc[0]),
        "twin_fates": twin.value_counts().to_dict(),
        "physical_disc": {**PHYSICAL,
                          "N4/N5": float(weights[is4].sum() / weights[is5].sum()) if is5.any() else None,
                          "capture_fraction": float(weights[is4 | is5].sum() / weights.sum()),
                          "n_eff_trojans": float(weights[is4 | is5].sum() ** 2 / (weights[is4 | is5] ** 2).sum())
                          if (is4 | is5).any() else 0.0},
    }
    write_json(out / "summary.json", summary)

    print(json.dumps(summary, indent=1))
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print(tables.round(3).to_string(index=False))
        print("\nformation (every 10th window):")
        print(formation.iloc[::10].to_string(index=False))


if __name__ == "__main__":
    main()
