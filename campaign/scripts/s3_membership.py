"""Stage 3: how many catalogue "Trojans" are tadpoles of their labelled swarm.

For every label (V24 box, MPC list, JPL TJN) and the report samples, counts the
objects whose every 10 kyr window is a tadpole of their t0 swarm (sigma crosses
neither 0 nor 180 deg; swarm from the circular mean of sigma), over the first
1 Myr and over the whole run. Removal counts as failure. Writes
campaign/reports/<run_id>/membership_by_label.csv.

    python campaign/scripts/s3_membership.py --run-id pilot_10myr
"""

import argparse

import h5py
import numpy as np
import pandas as pd

from trojans import integrate, paths
from s3_report import CLEANING_SPAN_YR, samples, well_determined


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default="pilot_10myr")
    args = parser.parse_args()

    merged = paths.run_dir(args.run_id) / "merged"
    events = pd.read_parquet(merged / "events.parquet")
    with h5py.File(merged / "windows.h5") as f:
        order = pd.Index(f["spkid"][:]).get_indexer(events["spkid"])
        cls = f["class"][:][:, order]
        t_end = f["t_end"][:]

    swarm = events["swarm"].to_numpy()
    own = np.where(swarm == "L4", integrate.TADPOLE_L4, integrate.TADPOLE_L5)
    tadpole = cls == own[None, :]
    early = t_end <= CLEANING_SPAN_YR + 1e-6

    groups = samples(events)
    good = well_determined(events)
    groups["MPC list"] = events["mpc_trojan"].to_numpy()
    groups["MPC list, U <= 4"] = groups["MPC list"] & good
    groups["JPL TJN"] = events["jpl_tjn"].to_numpy()
    groups["JPL TJN, U <= 4"] = groups["JPL TJN"] & good

    rows = []
    for name, mask in groups.items():
        for s in ("L4", "L5"):
            sel = mask & (swarm == s)
            n = int(sel.sum())
            k1 = int(tadpole[early][:, sel].all(axis=0).sum())
            k10 = int(tadpole[:, sel].all(axis=0).sum())
            rows.append(dict(sample=name, swarm=s, n=n,
                             tadpole_1myr=k1, frac_tadpole_1myr=k1 / n,
                             tadpole_run=k10, frac_tadpole_run=k10 / n, not_tadpole_run=n - k10))
    table = pd.DataFrame(rows)
    out = paths.report_dir(args.run_id) / "membership_by_label.csv"
    table.to_csv(out, index=False, float_format="%.4f")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
