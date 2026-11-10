"""Stage 9 report: Pirani+2019a's quantities from one or more repeats of a scenario.

Her definitions (sect. 4.1.1): a Jupiter Trojan is a particle whose perihelion and aphelion
both lie within a_J +/- 2.5 R_H (tadpole + horseshoe + quasi-satellite), counted at 5 Myr;
N4 has sigma = lambda - lambda_J > 0 (leading), N5 sigma < 0. Over her ten repeats she quotes
the mean and the unbiased standard deviation, with the ratio's error propagated.

Per run, from the snapshots (campaign/reports/<scenario>/):
  timeseries_<run>.csv   per snapshot: N4, N5, N_tot (her Fig. 7); co-orbital sides
                         |a - a_J| < 2.5 R_H split by sigma (her Fig. 11); Jupiter a, m
  trojans_<run>.csv      the Trojans at the last snapshot: id, a0 (origin, her Fig. 10), a, e, inc, sigma
  hildas: |a - a_32| < 0.2 au around a_32 = a_J (2/3)^(2/3) with e < 0.6 (her Hilda definition is not
          given, G6: this is a region count, not a libration test)
Across runs: summary.json (per run and mean +/- unbiased std, as her Table 3) and report.png.

    python campaign/scripts/s9_pirani_report.py --runs pir_nominal_r0 [pir_nominal_r1 ...] --name pir_nominal
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from trojans import disc, paths  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "reports"
HILL = 2.5


def load_run(run_id: str):
    root = paths.run_dir(run_id)
    ini = np.load(root / "disc_initial.npz")
    a0 = pd.Series(ini["a0"], index=ini["id"])
    ts_rows, last = [], []
    for chunk in sorted((root / "chunks").iterdir()):
        with h5py.File(chunk / "snapshots.h5", "r") as f:
            t, aj, mj = f["t"][:], f["jupiter_a"][:], f["jupiter_m"][:]
            ids = f["spkid"][:]
            for i in range(len(t)):
                a, e, s = f["a"][i], f["e"][i], f["sigma"][i]
                rh = disc.hill_radius(aj[i], mj[i])
                q, Q = a * (1 - e), a * (1 + e)
                troj = (q >= aj[i] - HILL * rh) & (Q <= aj[i] + HILL * rh)
                co = np.abs(a - aj[i]) < HILL * rh
                a32 = aj[i] * (2 / 3) ** (2 / 3)
                hil = (np.abs(a - a32) < 0.2) & (e < 0.6)
                ts_rows.append((t[i], aj[i], mj[i] / disc.M_EARTH, int((troj & (s > 0)).sum()), int((troj & (s < 0)).sum()),
                                int((co & (s > 0)).sum()), int((co & (s < 0)).sum()), int(hil.sum()),
                                int(np.isfinite(a).sum())))
            i = len(t) - 1
            a, e, s, inc = f["a"][i], f["e"][i], f["sigma"][i], f["inc"][i]
            rh = disc.hill_radius(aj[i], mj[i])
            troj = (a * (1 - e) >= aj[i] - HILL * rh) & (a * (1 + e) <= aj[i] + HILL * rh)
            last.append(pd.DataFrame({"id": ids[troj], "a0": a0.loc[ids[troj]].to_numpy(), "a": a[troj], "e": e[troj],
                                      "inc": inc[troj], "sigma": s[troj]}))
    ts = pd.DataFrame(ts_rows, columns=["t_yr", "jupiter_a", "jupiter_m_earth", "N4", "N5", "co_L4side", "co_L5side",
                                        "hildas", "alive"])
    agg = {c: "sum" for c in ["N4", "N5", "co_L4side", "co_L5side", "hildas", "alive"]}
    agg.update(jupiter_a="first", jupiter_m_earth="first")
    ts = ts.groupby("t_yr", as_index=False).agg(agg)
    ts["N_tot"] = ts["N4"] + ts["N5"]
    return ts, pd.concat(last, ignore_index=True), int(len(a0))


def ratio(n4, n5):
    return (n4 / n5, n4 / n5 * np.sqrt(1 / n4 + 1 / n5)) if n4 and n5 else (None, None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--name", required=True, help="output folder under campaign/reports/")
    args = ap.parse_args()
    out = REPORTS / args.name
    out.mkdir(parents=True, exist_ok=True)
    per_run, series, trojans = {}, {}, {}
    for r in args.runs:
        ts, tr, n = load_run(r)
        ts.to_csv(out / f"timeseries_{r}.csv", index=False)
        tr.to_csv(out / f"trojans_{r}.csv", index=False)
        end = ts.iloc[-1]
        rr, ee = ratio(int(end["N4"]), int(end["N5"]))
        pre = ts[(ts["t_yr"] > 1.5e6) & (ts["t_yr"] < 2.29e6)]
        per_run[r] = {"n_particles": n, "t_end_myr": float(end["t_yr"] / 1e6), "N4": int(end["N4"]), "N5": int(end["N5"]),
                      "N_tot": int(end["N_tot"]), "N4/N5": rr, "err": ee, "hildas": int(end["hildas"]),
                      "alive": int(end["alive"]), "jupiter_a_end": float(end["jupiter_a"]),
                      "N_tot_pre_migration_mean": float(pre["N_tot"].mean()) if len(pre) else None,
                      "N_tot_peak": int(ts["N_tot"].max()), "t_peak_myr": float(ts.loc[ts["N_tot"].idxmax(), "t_yr"] / 1e6),
                      "trojan_origin_median_au": float(tr["a0"].median()) if len(tr) else None}
        series[r], trojans[r] = ts, tr
    vals = [v["N4/N5"] for v in per_run.values() if v["N4/N5"] is not None]
    ntot = [v["N_tot"] for v in per_run.values()]
    summary = {"runs": per_run, "n_runs": len(per_run),
               "N_tot_mean": float(np.mean(ntot)), "N_tot_std": float(np.std(ntot, ddof=1)) if len(ntot) > 1 else None,
               "N4/N5_mean": float(np.mean(vals)) if vals else None,
               "N4/N5_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    figure(series, trojans, out)
    print(json.dumps(summary, indent=1))


def figure(series, trojans, out):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for r, ts in series.items():
        t = ts["t_yr"] / 1e6
        ax[0, 0].plot(t, ts["N_tot"], lw=1, label=r)
        with np.errstate(divide="ignore", invalid="ignore"):
            ax[0, 1].plot(t, ts["N4"] / ts["N5"], lw=1)
        ax[1, 0].plot(t, ts["co_L4side"], color="#2a78d6", lw=1)
        ax[1, 0].plot(t, ts["co_L5side"], color="#eb6834", lw=1)
    ax[0, 0].set(xlabel="t [Myr]", ylabel="N Trojans (q, Q within 2.5 R_H)", title="(a) her Fig. 7 top: Trojan number",
                 xlim=(0, 5))
    ax[0, 0].axvspan(2.3, 3.0, color="0.92", zorder=0)
    ax[0, 0].legend(fontsize=7)
    ax[0, 1].axhline(1, color="0.5", lw=0.8)
    for y, lab in [(1.6, "Szabo+07"), (1.4, "Grav+11"), (1.85, "Nakamura&Yoshida 08")]:
        ax[0, 1].axhline(y, color="0.7", lw=0.6, ls=":")
    ax[0, 1].set(xlabel="t [Myr]", ylabel="N4/N5", ylim=(0.4, 2.4), xlim=(0, 5), title="(b) her Fig. 7 bottom: asymmetry")
    ax[1, 0].set(xlabel="t [Myr]", ylabel="co-orbital particles", xlim=(1.9, 3.4),
                 title="(c) her Fig. 11: L4 side (blue), L5 side (orange)")
    allt = pd.concat(trojans.values())
    if len(allt):
        ax[1, 1].hist(allt["a0"], bins=np.arange(4, 26.5, 0.5), color="#2a78d6")
    ax[1, 1].set(xlabel="starting a [au]", ylabel="Trojans at the end", title="(d) her Fig. 10: where the Trojans formed")
    fig.savefig(out / "report.png", dpi=130)


if __name__ == "__main__":
    main()
