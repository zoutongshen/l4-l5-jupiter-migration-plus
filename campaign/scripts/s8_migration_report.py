"""Stage 8 report: N4/N5 after Jupiter grew and migrated, binned in D, I, e and by origin.

The Trojan rule and the D, I, e, e0, I0 tables are those of s6_disc_report.py
(alive at the end, a tadpole of one swarm in every 10 kyr window of the last
--classify-yr). Added for stage 8:
  - a0 bins (where in the disc captures come from) and an I0 x e0 table;
  - Pirani's count at the last snapshot: alive, perihelion and aphelion within
    a_J +/- 2.5 R_H, split by the sign of sigma (Pirani+2019a p. 8). It includes
    horseshoes and quasi-satellites, so it is reported for comparison with her
    numbers only;
  - coorbital.csv: per snapshot, how many particles with |a - a_J| < 2.5 R_H sit at
    sigma > 0 and sigma < 0 (the horseshoe-side excess of her Fig. 11), with
    Jupiter's a, m and nu;
  - twin asymmetry: pairs that end L4+none against L5+none, and L4+L4 against L5+L5.

Outputs (campaign/reports/<run-id>/): summary.json, binned.csv, binned_physical.csv
(warm discs only), coorbital.csv, formation.csv, particles.parquet.

    python campaign/scripts/s8_migration_report.py --run-id mig_v2_pirani
"""

import argparse
import json

import h5py
import numpy as np
import pandas as pd

import s6_disc_report as s6
from trojans import disc, paths
from trojans.history import JupiterHistory
from trojans.outputs import write_json

PIRANI_HILL_RADII = 2.5


def history_of(settings: dict) -> JupiterHistory:
    fields = {k: v for k, v in settings["history"].items() if k in JupiterHistory.__dataclass_fields__}
    return JupiterHistory(**fields)


def snapshot_tables(run_id: str, history: JupiterHistory):
    """Co-orbital side counts per snapshot, and every particle's last snapshot."""
    side_rows, last = [], []
    for chunk in sorted((paths.run_dir(run_id) / "chunks").iterdir()):
        with h5py.File(chunk / "snapshots.h5", "r") as f:
            t, a_j, m_j = f["t"][:], f["jupiter_a"][:], f["jupiter_m"][:]
            a, sigma = f["a"], f["sigma"]
            for i in range(len(t)):
                rh = disc.hill_radius(a_j[i], m_j[i])
                ai, si = a[i], sigma[i]
                co = np.abs(ai - a_j[i]) < PIRANI_HILL_RADII * rh
                side_rows.append((t[i], int(co.sum()), int((co & (si > 0)).sum()), int((co & (si < 0)).sum())))
            last.append(pd.DataFrame({"id": f["spkid"][:], **{k: f[k][-1] for k in ("a", "e", "inc", "sigma")},
                                      "t_last": t[-1], "jupiter_a": a_j[-1], "jupiter_m": m_j[-1]}))
    sides = pd.DataFrame(side_rows, columns=["t_yr", "n_coorbital", "n_sigma_pos", "n_sigma_neg"])
    sides = sides.groupby("t_yr", as_index=False).sum()
    sides["jupiter_a"] = [history.a(t) for t in sides["t_yr"]]
    sides["jupiter_m_earth"] = [history.mass(t) / disc.M_EARTH for t in sides["t_yr"]]
    sides["nu"] = [history.nu(t) for t in sides["t_yr"]]
    with np.errstate(divide="ignore", invalid="ignore"):
        sides["pos/neg"] = sides["n_sigma_pos"] / sides["n_sigma_neg"]
    return sides, pd.concat(last, ignore_index=True)


def pirani_count(last: pd.DataFrame) -> dict:
    rh = disc.hill_radius(last["jupiter_a"].iloc[0], last["jupiter_m"].iloc[0])
    a_j = last["jupiter_a"].iloc[0]
    q, Q = last["a"] * (1 - last["e"]), last["a"] * (1 + last["e"])
    inside = (q >= a_j - PIRANI_HILL_RADII * rh) & (Q <= a_j + PIRANI_HILL_RADII * rh)   # NaN (removed) -> False
    n4, n5 = int((inside & (last["sigma"] > 0)).sum()), int((inside & (last["sigma"] < 0)).sum())
    return {"t_yr": float(last["t_last"].iloc[0]), "N4": n4, "N5": n5,
            "N4/N5": n4 / n5 if n5 else None,
            "err": n4 / n5 * np.sqrt(1 / n4 + 1 / n5) if n4 and n5 else None}


def twin_table(particles: pd.DataFrame) -> dict:
    pair = (particles["id"] - particles["id"].min()) // 2
    fates = particles.assign(pair=pair, fate=particles["swarm"].replace("", "none"))
    twin = fates.sort_values("id").groupby("pair")["fate"].agg(lambda s: "+".join(sorted(s)))
    return twin.value_counts().to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--classify-yr", type=float, default=1.0e6)
    args = parser.parse_args()

    settings = json.loads((paths.run_dir(args.run_id) / "settings.json").read_text())
    history = history_of(settings)
    the_disc = settings["disc"]
    particles, formation = s6.load_run(args.run_id, args.classify_yr)
    particles = particles.merge(s6.starting_elements(args.run_id), on="id", how="left")
    is4, is5 = (particles["swarm"] == "L4").to_numpy(), (particles["swarm"] == "L5").to_numpy()
    n4, n5 = int(is4.sum()), int(is5.sum())

    edges = dict(s6.EDGES)
    edges["a0"] = list(np.round(np.linspace(the_disc["a_min"], the_disc["a_max"], 12), 3))
    warm = the_disc["e_max"] >= 0.05 and the_disc["inc_max_deg"] >= 5.0
    if not warm:          # flat disc: e0 and I0 bins are empty
        edges.pop("e0"), edges.pop("i0")
        edges["e"] = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 1]
        edges["inc"] = [0, 0.01, 0.1, 1, 5, 90]
    p = particles.assign(inc_split=particles["inc"])
    tables = pd.concat([s6.binned(p, col, e) for col, e in edges.items()], ignore_index=True)
    if warm:
        grid = particles.assign(i0_bin=pd.cut(particles["i0"], [0, 10, 20, 40], right=False),
                                e0_bin=pd.cut(particles["e0"], [0, 0.1, 0.2, 0.3], right=False))
        cross = (grid[grid["swarm"] != ""].groupby(["i0_bin", "e0_bin", "swarm"], observed=False).size()
                 .unstack(fill_value=0).reset_index())
        cross["N4/N5"] = cross["L4"] / cross["L5"].replace(0, np.nan)
        cross["err"] = cross["N4/N5"] * np.sqrt(1 / cross["L4"] + 1 / cross["L5"])
        cross = cross.astype({"i0_bin": str, "e0_bin": str})

    sides, last = snapshot_tables(args.run_id, history)

    out = paths.report_dir(args.run_id)
    out.mkdir(parents=True, exist_ok=True)
    tables.to_csv(out / "binned.csv", index=False)
    formation.to_csv(out / "formation.csv", index=False)
    sides.to_csv(out / "coorbital.csv", index=False)
    particles.to_parquet(out / "particles.parquet", index=False)
    physical = None
    if warm:
        cross.to_csv(out / "binned_i0_e0.csv", index=False)
        weights = s6.physical_weights(particles, {"e_max": the_disc["e_max"], "inc_max_deg": the_disc["inc_max_deg"]})
        s6.all_tables(particles, weights).to_csv(out / "binned_physical.csv", index=False)
        both = is4 | is5
        physical = {**s6.PHYSICAL, "N4/N5": float(weights[is4].sum() / weights[is5].sum()) if n5 else None,
                    "n_eff_trojans": float(weights[both].sum() ** 2 / (weights[both] ** 2).sum()) if both.any() else 0.0}

    cold = tables[(tables["variable"] == "inc_split")].set_index("bin")
    summary = {
        "run_id": args.run_id, "history": settings["history"], "disc": the_disc,
        "n_disc": len(particles), "N4": n4, "N5": n5, "N4/N5": n4 / n5 if n5 else None,
        "err": n4 / n5 * np.sqrt(1 / n4 + 1 / n5) if n4 and n5 else None,
        "capture_fraction": (n4 + n5) / len(particles),
        "inc_split": {b: {"L4": float(r["L4"]), "L5": float(r["L5"]), "N4/N5": float(r["N4/N5"]), "err": float(r["err"])}
                      for b, r in cold.iterrows()},
        "pirani_count": pirani_count(last),
        "twin_fates": twin_table(particles),
        "classify_windows": int(particles["n_windows"].iloc[0]),
        "physical_disc": physical,
        "max_coorbital_side_ratio": {"pos/neg": float(sides["pos/neg"].replace(np.inf, np.nan).max()),
                                     "at_t_yr": float(sides.loc[sides["pos/neg"].replace(np.inf, np.nan).idxmax(), "t_yr"])}
        if sides["pos/neg"].notna().any() else None,
    }
    write_json(out / "summary.json", summary)

    print(json.dumps({k: v for k, v in summary.items() if k not in ("history", "disc")}, indent=1, default=str))
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print(tables.round(3).to_string(index=False))
        if warm:
            print("\nI0 x e0:\n" + cross.round(3).to_string(index=False))
        print("\nco-orbital sides (every ~20th snapshot):")
        print(sides.iloc[:: max(1, len(sides) // 20)].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
