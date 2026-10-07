"""Stage 3: the loss rate Ndot/N per swarm and per U class ("Trojan dot per U").

Reads campaign/reports/<run_id>/uladder.csv (written by s3_uladder.py) and writes
campaign/reports/<run_id>/ndot_by_u.csv. No run data is needed.

Ndot/N is the escape fraction over the run divided by its length, in escapes per 10^4
Trojans per Myr. The ratio L5/L4 is quoted only for groups with escapers in both swarms.
Sample, escape criterion and errors as in s3_uladder.py.

    python campaign/scripts/s3_ndot_by_u.py --run-id pilot_10myr
"""

import argparse

import pandas as pd

from trojans import paths
from s3_uladder import ratio, wilson

T_RUN_MYR = 10.0002  # pilot_10myr length (reports/pilot_10myr/report.md)
GROUPS = {"0": [0], "1-4": [1, 2, 3, 4], "<=4": [0, 1, 2, 3, 4], "5-7": [5, 6, 7], "8": [8], "9": [9]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default="pilot_10myr")
    args = parser.parse_args()

    out_dir = paths.report_dir(args.run_id)
    lad = pd.read_csv(out_dir / "uladder.csv").set_index("U")
    rows = []
    for name, us in GROUPS.items():
        row = {"U": name}
        for s in ("L4", "L5"):
            n, k = int(lad.loc[us, f"n_{s}"].sum()), int(lad.loc[us, f"k_{s}"].sum())
            f, lo, hi = wilson(k, n)
            scale = 1e4 / T_RUN_MYR
            row.update({f"n_{s}": n, f"k_{s}": k, f"ndot_{s}": f * scale,
                        f"lo_{s}": lo * scale, f"hi_{s}": hi * scale})
        r, err = ratio(row["k_L4"], row["n_L4"], row["k_L5"], row["n_L5"])
        row.update({"ratio_L5_L4": r, "ratio_err": err})
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "ndot_by_u.csv", index=False, float_format="%.4g")
    print(table.to_string(index=False, float_format="%.3g"))


if __name__ == "__main__":
    main()
