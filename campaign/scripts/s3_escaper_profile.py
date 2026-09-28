"""Stage 3 exploration: what does an escaper look like at t0, and why not families?

Sharpens s3_explore.py on one question. That script reports escapers vs survivors in
e, i, libration amplitude and H; this one adds the parts needed to read those numbers:

  - discrimination (Mann-Whitney AUC) across every starting parameter, including the
    ones that turn out to carry no signal, which is itself the result;
  - escape rate binned in e and in libration amplitude, which shows a threshold
    rather than a gradient;
  - whether the H signal is dynamics or observational selection. It cannot be
    dynamics: the Trojans are massless test particles in a gravity-only integration,
    so the equations of motion contain no size term at all. This tests what H is
    standing in for.
  - why family members never escape, by comparing their tails rather than medians.

Prints tables; writes campaign/reports/<run_id>/escaper_profile.csv.

    python campaign/scripts/s3_escaper_profile.py --run-id pilot_10myr
"""

import argparse

import h5py
import numpy as np
import pandas as pd
from scipy import stats

from trojans import constants, paths

PARAMS = [
    ("B", "V24 B - ecc amplitude"), ("e", "eccentricity"),
    ("lib_amp", "libration half-amp (deg)"), ("C", "V24 C - libration"),
    ("H", "H (mag)"), ("sigma_off", "|sigma|-60 offset (deg)"),
    ("a", "semi-major axis (au)"), ("inc", "inclination (deg)"),
    ("A", "V24 A - incl amplitude"), ("diam_km", "diameter (km)"),
]
E_BINS = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 1.01]
LIB_BINS = [0, 20, 40, 60, 90, 181]


def load(run_id: str, catalogue_id: str) -> pd.DataFrame:
    merged = paths.merged_dir(run_id)
    e = pd.read_parquet(merged / "events.parquet")
    # Arc length and observation count are not carried into events.parquet, and the
    # H signal cannot be read without them.
    cat = pd.read_parquet(paths.catalogue_dir(catalogue_id) / "master_table.parquet")
    cat = cat.set_index("spkid")
    for col in ("data_arc", "n_obs_used"):
        e[col] = pd.to_numeric(cat.loc[e["spkid"], col].to_numpy(), errors="coerce")
    with h5py.File(merged / "windows.h5") as f:
        order = pd.Index(f["spkid"][:]).get_indexer(e["spkid"])
        smin, smax = f["sigma_min"][:][:, order], f["sigma_max"][:][:, order]
        nsamp = f["n_samples"][:][:, order]
    # Libration half-amplitude in the first sampled window, as s3_explore's "D".
    first = np.argmax(nsamp > 0, axis=0)
    e["lib_amp"] = 0.5 * (smax[first, np.arange(len(e))] - smin[first, np.arange(len(e))])
    e["escaped"] = e["removal_reason"].astype(str) != "alive"
    e["in_family"] = e["v24_family"].notna()
    e["U"] = pd.to_numeric(e["condition_code"], errors="coerce")
    e["sigma_off"] = np.abs(np.abs(e["sigma"]) - 60.0)
    e["diam_km"] = constants.diameter_km(e["H"].to_numpy())
    return e


def auc_row(esc: pd.Series, surv: pd.Series) -> tuple:
    a, b = esc.dropna(), surv.dropna()
    if len(a) < 3 or len(b) < 3:
        return np.nan, np.nan, np.nan, np.nan
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(np.median(a)), float(np.median(b)), u / (len(a) * len(b)), p


def rate_table(clean: pd.DataFrame, col: str, edges: list) -> pd.DataFrame:
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = clean[(clean[col] >= lo) & (clean[col] < hi)]
        if len(s):
            rows.append({"param": col, "lo": lo, "hi": hi, "n": len(s),
                         "escapers": int(s.escaped.sum()), "rate": s.escaped.mean()})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-id", default="pilot_10myr")
    parser.add_argument("--catalogue-id", default="2026-09-15")
    parser.add_argument("--e-split", type=float, default=0.2,
                        help="boundary of the high-e tail that supplies most escapers")
    args = parser.parse_args()

    e = load(args.run_id, args.catalogue_id)
    clean = e[e["v24_selected"] & (e["U"] <= 4)].copy()
    esc, surv = clean[clean.escaped], clean[~clean.escaped]
    print(f"V24-selected & U <= 4: n = {len(clean):,}  escapers = {len(esc)}\n")

    out = []
    print(f"{'parameter':<28}{'escapers':>11}{'survivors':>11}{'AUC':>8}{'p':>11}")
    for col, label in PARAMS:
        m_e, m_s, auc, p = auc_row(esc[col], surv[col])
        print(f"{label:<28}{m_e:>11.4g}{m_s:>11.4g}{auc:>8.3f}{p:>11.2e}")
        out.append({"kind": "discrimination", "param": label, "escapers": m_e,
                    "survivors": m_s, "auc": auc, "p": p})

    print("\nEscape rate by bin:")
    bins = pd.concat([rate_table(clean, "e", E_BINS), rate_table(clean, "lib_amp", LIB_BINS)])
    for r in bins.itertuples():
        print(f"  {r.param:<8}{r.lo:>7g} - {r.hi:<7g} n = {r.n:>6,}  escapers = {r.escapers:>3}"
              f"  rate = {100*r.rate:>7.3f}%")

    # --- is the H signal dynamics, or arc length? ---------------------------
    print("\nH cannot act dynamically here (massless, gravity-only). What it tracks:")
    rho, p = stats.spearmanr(clean["H"], clean["data_arc"], nan_policy="omit")
    print(f"  H vs observation arc: Spearman rho = {rho:+.3f}, p = {p:.1e}")
    out.append({"kind": "H_confound", "param": "H vs data_arc (Spearman)",
                "escapers": np.nan, "survivors": np.nan, "auc": rho, "p": p})
    for col in ("data_arc", "n_obs_used"):
        m_e, m_s, auc, pp = auc_row(esc[col], surv[col])
        print(f"  {col}: escapers {m_e:,.0f} vs survivors {m_s:,.0f}  AUC = {auc:.3f}  p = {pp:.1e}")
        out.append({"kind": "H_confound", "param": col, "escapers": m_e,
                    "survivors": m_s, "auc": auc, "p": pp})
    print(f"  escape rate by condition code, inside the cut:")
    for u in sorted(clean["U"].dropna().unique()):
        s = clean[clean["U"] == u]
        print(f"    U = {int(u)}: n = {len(s):>6,}  escapers = {int(s.escaped.sum()):>3}"
              f"  rate = {100*s.escaped.mean():>6.3f}%  median H = {s['H'].median():.2f}")

    lo = clean[clean.e <= args.e_split]
    m_e, m_s, auc, p = auc_row(lo[lo.escaped]["H"], lo[~lo.escaped]["H"])
    print(f"\n  With the high-e tail removed (e <= {args.e_split}): "
          f"{int(lo.escaped.sum())} escapers left, H AUC = {auc:.3f}, p = {p:.3f}")
    out.append({"kind": "H_confound", "param": f"H | e <= {args.e_split}",
                "escapers": m_e, "survivors": m_s, "auc": auc, "p": p})

    hi = clean[clean.e > args.e_split]
    print(f"  The high-e tail: {len(hi)} objects ({100*len(hi)/len(clean):.1f}% of sample) "
          f"supplying {int(hi.escaped.sum())} of {int(clean.escaped.sum())} escapers")
    print(f"    U >= 3: {100*(hi['U'] >= 3).mean():.1f}% of the tail vs "
          f"{100*(lo['U'] >= 3).mean():.1f}% of the bulk")
    print(f"    in a family: {100*hi['in_family'].mean():.1f}% vs {100*lo['in_family'].mean():.1f}%")

    # --- families -----------------------------------------------------------
    fam, bg = clean[clean.in_family], clean[~clean.in_family]
    print(f"\nFamily members ({len(fam):,}) vs background ({len(bg):,}):")
    print(f"{'parameter':<28}{'family':>11}{'background':>12}{'AUC':>8}")
    for col, label in [("e", "eccentricity"), ("inc", "inclination (deg)"),
                       ("lib_amp", "libration half-amp (deg)"), ("H", "H (mag)")]:
        m_f, m_b, auc, _ = auc_row(fam[col], bg[col])
        print(f"{label:<28}{m_f:>11.4g}{m_b:>12.4g}{auc:>8.3f}")
        out.append({"kind": "family_vs_background", "param": label,
                    "escapers": m_f, "survivors": m_b, "auc": auc, "p": np.nan})
    print("  tails, where the escapers actually come from:")
    for col, thr in [("e", args.e_split), ("lib_amp", 60.0)]:
        print(f"    {col} > {thr}: family {100*(fam[col] > thr).mean():.2f}%  "
              f"background {100*(bg[col] > thr).mean():.2f}%")
        out.append({"kind": "family_tail", "param": f"{col} > {thr}",
                    "escapers": float((fam[col] > thr).mean()),
                    "survivors": float((bg[col] > thr).mean()), "auc": np.nan, "p": np.nan})

    report = paths.report_dir(args.run_id)
    frame = pd.concat([pd.DataFrame(out), bins.assign(kind="rate_bin")], ignore_index=True)
    frame.to_csv(report / "escaper_profile.csv", index=False, float_format="%.6g")
    print(f"\nwrote {report}/escaper_profile.csv")


if __name__ == "__main__":
    main()
