"""Stage 0.4: propagate the giants and bright Trojans +-span from t0 and compare with Horizons.

Our model (folded Sun, four giants, WHFast at dt = 1% P_J) is not JPL's, so exact
agreement is not expected; the target is Delta sigma well below the libration
amplitude (< 1 deg) over 400 yr.

Horizons states go to data/raw/horizons_check/<date>/; the comparison table and
summary go to campaign/reports/horizons_check_<date>/.
"""

import argparse
import datetime
import json

import numpy as np
import pandas as pd

from trojans import catalogue, constants, diagnostics, fetch, initial, paths
from trojans.outputs import Manifest, write_json, write_provenance

SIGMA_TARGET_DEG = 1.0


def pick_objects(table: pd.DataFrame, n_per_swarm: int) -> pd.DataFrame:
    ok = table["in_run"] & table["v24_selected"] & (table["state_source"] == "sbdb_elements") & table["number"].notna()
    return (table[ok].sort_values("H").groupby("swarm").head(n_per_swarm)
            .sort_values(["swarm", "H"]).reset_index(drop=True))


def heliocentric(sim):
    sun = sim.particles[0]
    xyz = np.array([[p.x - sun.x, p.y - sun.y, p.z - sun.z] for p in sim.particles])
    vxyz = np.array([[p.vx - sun.vx, p.vy - sun.vy, p.vz - sun.vz] for p in sim.particles])
    return xyz, vxyz


def compare(sim, horizons: dict, objects: pd.DataFrame, k_steps: int, dt: float) -> list[dict]:
    xyz, vxyz = heliocentric(sim)
    mu_j = constants.G * (constants.CENTRAL_MASS + constants.GIANT_MASS["jupiter"])
    jup_sim = diagnostics.elements(xyz[1:2], vxyz[1:2], mu_j)
    jup_hz = diagnostics.elements(*[np.array([horizons["jupiter"][i]]) for i in (0, 1)], mu_j)
    rows = []
    for j, planet in enumerate(constants.GIANTS, start=1):
        hp, hv = horizons[planet]
        mu = constants.G * (constants.CENTRAL_MASS + constants.GIANT_MASS[planet])
        e_sim = diagnostics.elements(xyz[j:j + 1], vxyz[j:j + 1], mu)
        e_hz = diagnostics.elements(hp[None], hv[None], mu)
        rows.append({"k_steps": k_steps, "t_yr": k_steps * dt, "target": planet,
                     "dr_au": float(np.linalg.norm(xyz[j] - hp)),
                     "dlam_deg": float(diagnostics.wrap180(e_sim["lam"] - e_hz["lam"])[0]),
                     "da_au": float(e_sim["a"][0] - e_hz["a"][0])})
    mu = constants.G * constants.CENTRAL_MASS
    n_active = sim.N_active
    for i, obj in objects.iterrows():
        hp, hv = horizons[int(obj["spkid"])]
        e_sim = diagnostics.elements(xyz[n_active + i:n_active + i + 1], vxyz[n_active + i:n_active + i + 1], mu)
        e_hz = diagnostics.elements(hp[None], hv[None], mu)
        s_sim = diagnostics.resonant_angle(e_sim["lam"], jup_sim["lam"])[0]
        s_hz = diagnostics.resonant_angle(e_hz["lam"], jup_hz["lam"])[0]
        rows.append({"k_steps": k_steps, "t_yr": k_steps * dt, "target": obj["full_name"], "swarm": obj["swarm"],
                     "dr_au": float(np.linalg.norm(xyz[n_active + i] - hp)),
                     "sigma_horizons_deg": float(s_hz), "dsigma_deg": float(diagnostics.wrap180(s_sim - s_hz)),
                     "da_au": float(e_sim["a"][0] - e_hz["a"][0])})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalogue-id", required=True)
    parser.add_argument("--date", default=datetime.date.today().isoformat())
    parser.add_argument("--span-yr", type=float, default=400.0)
    parser.add_argument("--n-per-swarm", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    cat_dir = paths.catalogue_dir(args.catalogue_id)
    table = pd.read_parquet(cat_dir / "master_table.parquet")
    info = json.loads((cat_dir / "summary.json").read_text())
    t0 = info["t0_jd_tdb"]
    objects = pick_objects(table, args.n_per_swarm)

    giants = catalogue.giants_at_t0(paths.raw_dir("horizons", info["raw_date"]), t0)
    states = initial.states_table(giants, objects["spkid"], objects[["x", "y", "z"]].to_numpy(),
                                  objects[["vx", "vy", "vz"]].to_numpy())
    dt = initial.planet_setup(states)["dt_yr"]
    n = int(round(args.span_yr / dt / 2)) * 2
    ks = [-n, -n // 2, n // 2, n]

    raw = paths.make_output_dir(paths.raw_dir("horizons_check", args.date), args.resume)
    manifest = Manifest(raw)
    targets = {p: constants.HORIZONS_ID[p] for p in constants.GIANTS}
    targets.update({int(s): f"DES={int(s)};" for s in objects["spkid"]})
    horizons = {k: {} for k in ks}
    for k in ks:
        jd = t0 + k * dt * constants.JULIAN_YEAR_D
        for key, command in targets.items():
            filename = f"k{k:+d}/{key}.json"
            if not manifest.has(filename):
                manifest.save_file(raw, filename, fetch.horizons_vectors(command, jd), flush=False)
            s = fetch.parse_horizons_vectors((raw / filename).read_bytes())
            horizons[k][key] = (np.array([s["x"], s["y"], s["z"]]),
                                np.array([s["vx"], s["vy"], s["vz"]]) * constants.JULIAN_YEAR_D)
        manifest.flush()
    write_provenance(raw)

    rows = []
    for direction in (+1, -1):
        sim = initial.build_simulation(states, objects["spkid"], dt)
        sim.dt = direction * dt
        for k in sorted((k for k in ks if np.sign(k) == direction), key=abs):
            sim.steps(abs(k) - int(round(abs(sim.t) / dt)))
            rows += compare(sim, horizons[k], objects, k, dt)
    result = pd.DataFrame(rows).sort_values(["k_steps", "target"])

    trojans = result[result["dsigma_deg"].notna()]
    worst = trojans.groupby("t_yr")["dsigma_deg"].apply(lambda s: float(np.max(np.abs(s))))
    summary = {
        "t0_jd_tdb": t0, "dt_yr": dt, "span_yr": n * dt,
        "max_abs_dsigma_deg_by_t_yr": {f"{t:+.1f}": v for t, v in worst.items()},
        "max_giant_dr_au": float(result.loc[result["dsigma_deg"].isna(), "dr_au"].max()),
        "target_dsigma_deg": SIGMA_TARGET_DEG,
        "pass": bool(worst.max() < SIGMA_TARGET_DEG),
    }
    report = paths.make_output_dir(paths.report_dir(f"horizons_check_{args.date}"), args.resume)
    result.to_csv(report / "comparison.csv", index=False, float_format="%.6g")
    write_json(report / "summary.json", summary)
    print(result.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
