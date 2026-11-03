"""Stage 7: apply the learned Trojan rule to every known asteroid and comet, and list what the catalogues miss.

Every SBDB orbit (s7_fetch_population.py) is moved to the pilot's t0 on its
Keplerian ellipse (M advanced by n dt), turned into the same snapshot features as
the training set, and scored by the saved rule and by the catalogue boxes. Kepler
propagation is a screen, not a verdict: every candidate is then integrated from its
Horizons state at t0 (s7_run_map.py --source spkids).

Outputs:
  data/runs/<out-id>/screen_k<k>.parquet  every object near Jupiter's orbit, with scores
  campaign/reports/<out-id>/screen_*.csv  candidates and the counts behind them

The rule was trained on a 4.6-5.8 au, e < 0.6, I < 60 deg; orbits near a_J outside
that domain (high e, high or retrograde I) are listed separately, not scored.

    python campaign/scripts/s7_screen.py --out-id definition_v1 --population-date 2026-10-27
"""

import argparse
import json
import pickle

import numpy as np
import pandas as pd

from trojans import constants, coorbital, definition, paths, runs
from trojans.outputs import provenance, write_json

RULE = "boosted · minimal"
DOMAIN = coorbital.MapGrid(n=0)
NEAR_A = (4.0, 6.5)       # au; nothing outside can be a Jupiter co-orbital


def load_population(raw) -> pd.DataFrame:
    frames = []
    for kind, filename in (("asteroid", "asteroids.json"), ("comet", "comets.json")):
        payload = json.loads((raw / filename).read_text())
        df = pd.DataFrame(payload["data"], columns=payload["fields"])
        df["kind"] = kind
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    out = pd.DataFrame({"spkid": df["spkid"].astype("int64"), "full_name": df["full_name"].str.strip(),
                        "kind": df["kind"], "sbdb_class": df["class"], "condition_code": df["condition_code"],
                        "first_obs": df["first_obs"]})
    for col in ["epoch", "a", "e", "i", "om", "w", "ma", "tp", "H", "data_arc"]:
        out[col] = pd.to_numeric(df[col], errors="coerce") if col in df else np.nan
    return out


def kepler_to(pop: pd.DataFrame, t0_jd: float) -> pd.DataFrame:
    """Elements at t0: M advanced along the osculating ellipse (mu = G M_sun, as SBDB)."""
    n_deg_day = np.degrees(np.sqrt(constants.G / pop["a"] ** 3)) / 365.25
    M_epoch = pop["ma"].where(pop["ma"].notna(), n_deg_day * (pop["epoch"] - pop["tp"]))
    el = pd.DataFrame({"a": pop["a"], "e": pop["e"], "inc": pop["i"], "Omega": pop["om"], "omega": pop["w"],
                       "M": np.mod(M_epoch + n_deg_day * (t0_jd - pop["epoch"]), 360.0)})
    el["dt_days"] = t0_jd - pop["epoch"]
    return el


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-id", required=True)
    parser.add_argument("--population-date", required=True)
    parser.add_argument("--catalogue-id", default="2026-09-15")
    parser.add_argument("--ic-id", default="2026-09-15_giants")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    states, ic = runs.load_ic(args.ic_id)
    t0 = ic["t0_jd_tdb"]
    jupiter = coorbital.jupiter_at_t0(states)
    with open(paths.run_dir(args.out_id) / f"models_k{args.k}.pkl", "rb") as f:
        rule = pickle.load(f)["models"][RULE]

    pop = load_population(paths.raw_dir("sbdb_population", args.population_date))
    n_all = len(pop)
    bound = (pop["e"] < 1) & (pop["a"] > 0)
    near = pop[bound & pop["a"].between(*NEAR_A)].reset_index(drop=True)
    el = kepler_to(near, t0)
    f = definition.features(el, jupiter)
    table = pd.concat([near, f[["sigma", "da"] + [c for c in rule["features"] if c not in ("da",)]], el[["M", "dt_days"]]],
                      axis=1).rename(columns={"M": "M_t0"})
    table = table.loc[:, ~table.columns.duplicated()]

    in_domain = (el["a"].between(DOMAIN.a_min, DOMAIN.a_max) & (el["e"] < DOMAIN.e_max)
                 & (el["inc"] < DOMAIN.inc_max_deg)).to_numpy()
    table["in_domain"] = in_domain
    proba = np.full((len(table), 3), np.nan)
    proba[in_domain] = rule["model"].predict_proba(f.loc[in_domain, rule["features"]])
    classes = list(rule["model"].classes_)
    table["p_trojan"] = proba[:, classes.index("L4")] + proba[:, classes.index("L5")]
    pred = np.full(len(table), None, dtype=object)
    pred[in_domain] = rule["model"].predict(f.loc[in_domain, rule["features"]])
    table["rule"] = pred
    boxes_f = el.assign(sigma=f["sigma"])
    for name, box in definition.BOXES.items():
        table[f"box {name}"] = definition.box_label(boxes_f, box)

    master = pd.read_parquet(paths.catalogue_dir(args.catalogue_id) / "master_table.parquet")
    table["jpl_tjn_now"] = table["sbdb_class"].eq("TJN")
    table["in_sept_catalogue"] = table["spkid"].isin(master["spkid"])
    table["labelled_sept"] = table["spkid"].isin(master.loc[master["in_run"], "spkid"])
    table["labelled"] = table["jpl_tjn_now"] | table["labelled_sept"]
    table["rule_trojan"] = table["rule"].isin(["L4", "L5"])

    run = paths.run_dir(args.out_id)
    table.to_parquet(run / f"screen_k{args.k}.parquet", index=False)
    out = paths.report_dir(args.out_id)
    out.mkdir(parents=True, exist_ok=True)
    show = ["spkid", "full_name", "kind", "sbdb_class", "H", "condition_code", "first_obs", "dt_days", "a", "e", "i",
            "sigma", "da", "p_trojan", "rule", "box JPL TJN", "box MPC (inferred)", "box V24 box",
            "jpl_tjn_now", "labelled_sept", "in_sept_catalogue"]
    missed = table[table["rule_trojan"] & ~table["labelled"]].sort_values("p_trojan", ascending=False)
    borderline = table[table["p_trojan"].between(0.2, 0.5) & ~table["labelled"]].sort_values("p_trojan", ascending=False)
    rejected = table[table["labelled"] & ~table["rule_trojan"] & table["in_domain"]].sort_values("p_trojan")
    outside = table[~table["in_domain"] & table["a"].between(DOMAIN.a_min, DOMAIN.a_max)]
    missed[show].to_csv(out / f"screen_missed_k{args.k}.csv", index=False, float_format="%.4f")
    borderline[show].to_csv(out / f"screen_borderline_k{args.k}.csv", index=False, float_format="%.4f")
    rejected[show].to_csv(out / f"screen_rejected_k{args.k}.csv", index=False, float_format="%.4f")
    outside[show].to_csv(out / f"screen_outside_domain_k{args.k}.csv", index=False, float_format="%.4f")
    counts = {
        "population": n_all, "near_jupiter_a": len(table), "in_domain": int(in_domain.sum()),
        "rule_trojans": int(table["rule_trojan"].sum()), "labelled": int(table["labelled"].sum()),
        "rule_yes_unlabelled": len(missed), "borderline_unlabelled": len(borderline),
        "labelled_rule_no": len(rejected), "outside_domain_near_aJ": len(outside),
        "rule_yes_unlabelled_new_since_sept": int((~missed["in_sept_catalogue"]).sum()),
    }
    write_json(out / f"screen_summary_k{args.k}.json", {"rule": RULE, "t0_jd_tdb": t0, **counts, **provenance()})
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
