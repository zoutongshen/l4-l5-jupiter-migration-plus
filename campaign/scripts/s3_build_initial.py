"""Stage 0.3: initial conditions at t0 into data/initial/<ic_id>/.

  states.parquet  folded Sun, giant-planet system barycentres, and every Trojan in
                  the run (union of the V24, JPL and MPC definitions); heliocentric,
                  ecliptic J2000, au and au/yr
  ic.json         t0, masses, G, Jupiter's period, the time step, Hill radii
"""

import argparse
import json

import pandas as pd

from trojans import catalogue, constants, initial, paths
from trojans.outputs import write_json, write_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalogue-id", required=True)
    parser.add_argument("--ic-id", default=None, help="defaults to <catalogue-id>_giants")
    args = parser.parse_args()
    ic_id = args.ic_id or f"{args.catalogue_id}_giants"

    cat_dir = paths.catalogue_dir(args.catalogue_id)
    table = pd.read_parquet(cat_dir / "master_table.parquet")
    info = json.loads((cat_dir / "summary.json").read_text())
    t0 = info["t0_jd_tdb"]
    run = table[table["in_run"]]

    giants = catalogue.giants_at_t0(paths.raw_dir("horizons", info["raw_date"]), t0)
    states = initial.states_table(giants, run["spkid"], run[["x", "y", "z"]].to_numpy(), run[["vx", "vy", "vz"]].to_numpy())
    setup = initial.planet_setup(states)

    out = paths.make_output_dir(paths.initial_dir(ic_id))
    states.to_parquet(out / "states.parquet", index=False)
    write_json(out / "ic.json", {
        "ic_id": ic_id,
        "catalogue_id": args.catalogue_id,
        "t0_jd_tdb": t0,
        "frame": "heliocentric ecliptic J2000; au, Julian yr, Msun",
        "G": constants.G,
        "central_mass": constants.CENTRAL_MASS,
        "central_mass_includes": list(constants.INNER_BODIES),
        "giant_mass": constants.GIANT_MASS,
        "n_trojans": int(len(run)),
        **setup,
    })
    write_provenance(out)
    print(f"{ic_id}: {len(run)} Trojans; P_J = {setup['jupiter_period_yr']:.4f} yr, dt = {setup['dt_yr']:.5f} yr")


if __name__ == "__main__":
    main()
