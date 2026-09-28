"""Stage 3 exploration: what does erosion correlate with, and do families contaminate it?

Reads a merged run and writes data/runs/<run_id>/explore/explore.json, the data
behind the interactive pilot explorer (sample-level statistics plus one record per
V24-selected object: flags, starting e, i, D, H and escape time, from which the
page draws erosion curves and snapshots at any time), plus two small tables in
campaign/reports/<run_id>/:
  explore_correlates.csv  escapers vs survivors in e, i, D, H (medians, AUC, p)
  explore_families.csv    escape per family and for the background, per swarm

Definitions
  escape   first entry within 2.1 Hill radii of a giant during the run
  e, i     heliocentric osculating elements at t0
  D        libration half-amplitude, (max sigma - min sigma) / 2 over the first
           10 kyr window in which the object was sampled
  H        JPL absolute magnitude
  samples  'clean' = V24-selected with JPL condition code <= 4;
           'complete' = V24-selected with H <= 13.6; 'v24' = all V24-selected
Error bars are Wilson intervals at 1 sigma (z = 1), which behave at 0 or 1 escaper.
AUC is the probability that a random escaper has a larger value than a random
survivor (0.5 = no relation), from the Mann-Whitney U test.
"""

import argparse
import json

import h5py
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu

from trojans import paths
from trojans.outputs import write_provenance

SWARMS = ("L4", "L5")
PARAMS = {
    "e": {"label": "eccentricity e", "edges": [0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.3, 0.6]},
    "inc": {"label": "inclination i [deg]", "edges": [0, 5, 10, 15, 20, 25, 30, 40, 50]},
    "D": {"label": "libration half-amplitude D [deg]", "edges": [0, 10, 20, 30, 40, 50, 60, 80, 180]},
    "H": {"label": "absolute magnitude H", "edges": [6, 12, 13.6, 14.5, 15, 15.5, 16, 19]},
}
HIST_EDGES = {
    "e": np.linspace(0, 0.3, 31),
    "inc": np.linspace(0, 45, 31),
    "D": np.linspace(0, 90, 31),
    "H": np.linspace(7, 17.5, 43),
}


def wilson(k, n, z=1.0):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def rate(k, n):
    f, lo, hi = wilson(int(k), int(n))
    return {"n": int(n), "k": int(k), "f": f, "lo": lo, "hi": hi}


def clean(obj):
    """JSON-safe: NaN -> None, numpy scalars -> Python."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else round(float(obj), 6)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def load(run_id):
    merged = paths.merged_dir(run_id)
    ev = pd.read_parquet(merged / "events.parquet")
    with h5py.File(merged / "windows.h5") as f:
        order = pd.Index(f["spkid"][:]).get_indexer(ev["spkid"])
        n = f["n_samples"][:][:, order]
        smin = f["sigma_min"][:][:, order]
        smax = f["sigma_max"][:][:, order]
    sampled = n > 0
    first = np.argmax(sampled, axis=0)
    cols = np.arange(n.shape[1])
    d0 = np.where(sampled.any(axis=0), (smax[first, cols] - smin[first, cols]) / 2, np.nan)

    t = pd.DataFrame({
        "spkid": ev["spkid"], "swarm": ev["swarm"], "H": ev["H"], "e": ev["e"], "inc": ev["inc"], "D": d0,
        "family": ev["v24_family"], "v24": ev["v24_selected"], "complete": ev["complete"],
        "well_determined": pd.to_numeric(ev["condition_code"], errors="coerce") <= 4,
        "esc": ev["t_hill_record"].notna(), "t_esc": ev["t_hill_record"],
    })
    t["is_family"] = t["family"].notna()
    return t


def sample_masks(t):
    return {
        "clean": (t["v24"] & t["well_determined"]).to_numpy(),
        "complete": (t["v24"] & t["complete"]).to_numpy(),
        "v24": t["v24"].to_numpy(),
    }


def headline(s):
    out = {"swarm": {}}
    for sw in SWARMS:
        g = s[s["swarm"] == sw]
        out["swarm"][sw] = {
            "all": rate(g["esc"].sum(), len(g)),
            "background": rate(g.loc[~g["is_family"], "esc"].sum(), (~g["is_family"]).sum()),
            "families": rate(g.loc[g["is_family"], "esc"].sum(), g["is_family"].sum()),
            "family_share": g["is_family"].mean(),
        }

    def asymmetry(key):
        a4, a5 = out["swarm"]["L4"][key], out["swarm"]["L5"][key]
        f4, f5 = a4["f"], a5["f"]
        e4 = np.sqrt(f4 * (1 - f4) / a4["n"]) if a4["n"] else np.nan
        e5 = np.sqrt(f5 * (1 - f5) / a5["n"]) if a5["n"] else np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = f5 / f4
            ratio_err = ratio * np.sqrt((e4 / f4) ** 2 + (e5 / f5) ** 2)
        amp = (1 - f4) / (1 - f5)
        amp_err = amp * np.sqrt((e4 / (1 - f4)) ** 2 + (e5 / (1 - f5)) ** 2)
        return {"ratio": ratio, "ratio_err": ratio_err, "amplifier": amp, "amplifier_err": amp_err,
                "n4n5_t0": a4["n"] / a5["n"], "n4n5_end": a4["n"] * (1 - f4) / (a5["n"] * (1 - f5))}

    out["asymmetry"] = {key: asymmetry(key) for key in ("all", "background")}
    return out


def binned(s):
    out = {}
    for p, spec in PARAMS.items():
        edges = np.array(spec["edges"], float)
        series = {}
        for group in (*SWARMS, "both"):
            g = s if group == "both" else s[s["swarm"] == group]
            g = g[g[p].notna()]
            idx = np.clip(np.digitize(g[p], edges) - 1, 0, len(edges) - 2)
            series[group] = [rate(g.loc[idx == b, "esc"].sum(), (idx == b).sum()) for b in range(len(edges) - 1)]
        out[p] = {"edges": edges.tolist(), "series": series}
    return out


def correlates(s):
    rows = []
    for p in PARAMS:
        for group in (*SWARMS, "both"):
            g = s if group == "both" else s[s["swarm"] == group]
            g = g[g[p].notna()]
            esc, surv = g.loc[g["esc"], p], g.loc[~g["esc"], p]
            if len(esc) and len(surv):
                u, pval = mannwhitneyu(esc, surv, alternative="two-sided")
                auc = u / (len(esc) * len(surv))
            else:
                auc, pval = np.nan, np.nan
            rows.append({"param": p, "group": group, "n_esc": len(esc), "n_surv": len(surv),
                         "median_esc": esc.median(), "median_surv": surv.median(), "auc": auc, "p": pval})
    return rows


def families(s):
    hist, ks = {}, []
    for p, edges in HIST_EDGES.items():
        hist[p] = {"edges": edges.tolist()}
        for sw in SWARMS:
            g = s[(s["swarm"] == sw) & s[p].notna()]
            bg, fam = g.loc[~g["is_family"], p], g.loc[g["is_family"], p]
            hist[p][sw] = {
                "background": np.histogram(bg.clip(edges[0], edges[-1]), edges, density=True)[0].tolist(),
                "families": np.histogram(fam.clip(edges[0], edges[-1]), edges, density=True)[0].tolist() if len(fam) else [],
                "n_background": len(bg), "n_families": len(fam),
                "median_background": bg.median(), "median_families": fam.median() if len(fam) else np.nan,
            }
            if len(fam) > 5:
                res = ks_2samp(bg, fam)
                ks.append({"param": p, "swarm": sw, "ks_d": res.statistic, "p": res.pvalue})
    per_family = []
    for (name, sw), g in s[s["is_family"]].groupby(["family", "swarm"]):
        per_family.append({"family": name, "swarm": sw, **rate(g["esc"].sum(), len(g)),
                           "median_e": g["e"].median(), "median_inc": g["inc"].median(),
                           "median_D": g["D"].median(), "median_H": g["H"].median()})
    for sw in SWARMS:
        g = s[(s["swarm"] == sw) & ~s["is_family"]]
        per_family.append({"family": "background", "swarm": sw, **rate(g["esc"].sum(), len(g)),
                           "median_e": g["e"].median(), "median_inc": g["inc"].median(),
                           "median_D": g["D"].median(), "median_H": g["H"].median()})
    per_family.sort(key=lambda r: (r["swarm"], r["family"] == "background", -r["n"]))
    return {"hist": hist, "ks": ks, "table": per_family}


def objects(t):
    """One record per V24-selected object, as parallel arrays, for curves and snapshots in the page."""
    obj = t[t["v24"]]
    return {
        "swarm": obj["swarm"].map({"L4": 4, "L5": 5}).tolist(),
        "family": obj["is_family"].astype(int).tolist(),
        "well_determined": obj["well_determined"].astype(int).tolist(),
        "complete": obj["complete"].astype(int).tolist(),
        "t_esc_yr": obj["t_esc"].round(1).tolist(),
        "e": obj["e"].round(4).tolist(),
        "inc": obj["inc"].round(2).tolist(),
        "D": obj["D"].round(1).tolist(),
        "H": obj["H"].round(2).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default="pilot_10myr")
    parser.add_argument("--resume", action="store_true", help="overwrite existing outputs")
    args = parser.parse_args()

    t = load(args.run_id)
    out = {"run_id": args.run_id, "params": {p: s["label"] for p, s in PARAMS.items()}, "samples": {}}
    corr_rows, fam_rows = [], []
    for name, mask in sample_masks(t).items():
        s = t[mask]
        corr = correlates(s)
        fam = families(s)
        out["samples"][name] = {
            "n": int(mask.sum()),
            "headline": headline(s),
            "binned": binned(s),
            "correlates": corr,
            "families": fam,
        }
        corr_rows += [{"sample": name, **r} for r in corr]
        fam_rows += [{"sample": name, **r} for r in fam["table"]]
    out["objects"] = objects(t)

    explore = paths.make_output_dir(paths.run_dir(args.run_id) / "explore", args.resume)
    (explore / "explore.json").write_text(json.dumps(clean(out), separators=(",", ":")))
    write_provenance(explore)

    report = paths.report_dir(args.run_id)
    pd.DataFrame(corr_rows).to_csv(report / "explore_correlates.csv", index=False, float_format="%.5g")
    pd.DataFrame(fam_rows).to_csv(report / "explore_families.csv", index=False, float_format="%.5g")

    print(f"wrote {explore / 'explore.json'} ({(explore / 'explore.json').stat().st_size / 1e6:.1f} MB)")
    c = pd.DataFrame(corr_rows)
    print(c[c["sample"] == "clean"].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    f = pd.DataFrame(fam_rows)
    print(f[f["sample"] == "clean"][["swarm", "family", "n", "k", "f", "median_e", "median_inc", "median_D", "median_H"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    for name in out["samples"]:
        print(name, json.dumps(clean(out["samples"][name]["headline"]["asymmetry"])))


if __name__ == "__main__":
    main()
