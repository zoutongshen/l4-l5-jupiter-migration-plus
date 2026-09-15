"""Stages 1.2-1.3: select Trojans, compare labels, and build the master table.

Reads data/raw/{sbdb,mpc,v24,horizons}/<date>/ and writes
data/catalogue/<catalogue_id>/:
  master_table.parquet        one row per candidate (see catalogue.build_master_table)
  v24_family_members.parquet  every (object, V24 family list) pair
  label_comparison.csv        counts per label region and swarm
  summary.json                headline counts and t0
plus a small tracked copy of the summary and label table in
campaign/reports/catalogue_<catalogue_id>/.
"""

import argparse
import datetime
import json

import numpy as np
from scipy.stats import spearmanr

from trojans import catalogue, paths
from trojans.outputs import write_json, write_provenance


def summary(table, t0, jupiter, raw_date) -> dict:
    run = table[table["in_run"]]

    def per_swarm(mask):
        return {s: int(n) for s, n in table.loc[mask, "swarm"].value_counts().sort_index().items()}

    return {
        "raw_date": raw_date,
        "t0_jd_tdb": t0,
        "jupiter_at_t0": {k: round(v, 6) for k, v in jupiter.items()},
        "n_candidates": len(table),
        "n_in_run": int(table["in_run"].sum()),
        "in_run_by_swarm": per_swarm(table["in_run"]),
        "v24_selected_by_swarm": per_swarm(table["v24_selected"]),
        "jpl_tjn_by_swarm": per_swarm(table["jpl_tjn"]),
        "mpc_by_its_own_label": {s: int(n) for s, n in table["mpc_ln"].value_counts().sort_index().items()},
        "complete_v24_by_swarm": per_swarm(table["v24_selected"] & table["complete"]),
        "disisto2013_by_swarm": per_swarm(table["disisto2013"]),
        "off_epoch_in_run": int((run["state_source"] == "horizons").sum()),
        "v24_proper_matched_in_run": int(run["v24_desig"].notna().sum()),
        "v24_family_members_in_run": int(run["v24_family"].notna().sum()),
        "v24_family_overlaps": int(run["v24_family"].fillna("").str.contains(";").sum()),
        "abc_at_t0_vs_v24_proper_spearman": abc_check(table),
    }


def abc_check(table) -> dict:
    """Stage 2.2 check: osculating A, B, C at t0 against V24's synthetic proper elements."""
    v = table[table["v24_selected"] & table["v24_sini_p"].notna()]
    a_jupiter = 5.2026  # only a rank correlation is taken, so the exact reference value is immaterial
    return {
        "n": len(v),
        "A_vs_sini_p": float(spearmanr(v["A"], v["v24_sini_p"]).correlation),
        "B_vs_e_p": float(spearmanr(v["B"], v["v24_e_p"]).correlation),
        "sqrt(C-1.5)_vs_da_p": float(spearmanr(np.sqrt(np.clip(v["C"] - 1.5, 0, None)),
                                                np.abs(v["v24_a_p"] - a_jupiter)).correlation),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--date", default=datetime.date.today().isoformat(), help="raw download date")
    parser.add_argument("--catalogue-id", default=None, help="defaults to the raw download date")
    parser.add_argument("--resume", action="store_true", help="overwrite an existing catalogue directory")
    args = parser.parse_args()
    catalogue_id = args.catalogue_id or args.date

    sbdb_raw, mpc_raw = paths.raw_dir("sbdb", args.date), paths.raw_dir("mpc", args.date)
    horizons_raw, v24_raw = paths.raw_dir("horizons", args.date), paths.raw_dir("v24", args.date)

    candidates = catalogue.load_candidates(sbdb_raw, mpc_raw)
    t0 = json.loads((horizons_raw / "epoch.json").read_text())["t0_jd_tdb"]
    pos, vel, source = catalogue.states_at_t0(candidates, horizons_raw, t0)
    giants = catalogue.giants_at_t0(horizons_raw, t0)
    families = catalogue.load_v24_families(v24_raw)
    table, jupiter = catalogue.build_master_table(
        candidates, pos, vel, source, giants["jupiter"], catalogue.load_v24_proper(v24_raw), families
    )
    labels = catalogue.label_comparison(table)
    info = summary(table, t0, jupiter, args.date)

    out = paths.make_output_dir(paths.catalogue_dir(catalogue_id), args.resume)
    table.to_parquet(out / "master_table.parquet", index=False)
    families.to_parquet(out / "v24_family_members.parquet", index=False)
    labels.to_csv(out / "label_comparison.csv", index=False)
    write_json(out / "summary.json", info)
    write_provenance(out)

    report = paths.make_output_dir(paths.report_dir(f"catalogue_{catalogue_id}"), args.resume)
    labels.to_csv(report / "label_comparison.csv", index=False)
    write_json(report / "summary.json", info)

    print(json.dumps(info, indent=2))
    print(labels.to_string(index=False))


if __name__ == "__main__":
    main()
