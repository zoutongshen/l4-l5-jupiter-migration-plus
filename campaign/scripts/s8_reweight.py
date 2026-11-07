"""Reweight a stage-8 run's Trojans to a physical planetesimal disc (Block D).

The runs sample a0, e0, i0 uniformly, so every (e0, i0) cell is equally filled. A
disc model is a density p(e0, i0); each particle gets the weight p(e0, i0) / uniform,
and the weighted N4/N5 is computed in bins of FINAL inclination, the observable
closest to proper I. Uncertainties use sum(w^2) per swarm (effective counts).

Disc models (all Rayleigh-type, truncated to the run's e_max, i_max):
  indep   e ~ Rayleigh(sig_e), sin i ~ Rayleigh(sig_i), independent
  mix     cold + hot components, each independent Rayleigh in e and sin i:
          (1 - f_hot) R(sig_e_c) R(sig_i_c) + f_hot R(sig_e_h) R(sig_i_h)
          i.e. cold bodies are also circular, hot ones eccentric (e-I correlated)

    python campaign/scripts/s8_reweight.py mig_d7_c2_warm mig_d8_c1_warm
"""

import argparse

import numpy as np
import pandas as pd

from trojans import paths

OBS = pd.DataFrame({"bin": ["0-10", "10-20", "20-30", ">30"], "N4/N5": [2.15, 1.38, 0.93, 1.12],
                    "err": [0.14, 0.09, 0.08, 0.21]})
I_EDGES = [0, 10, 20, 30, 90]


def rayleigh(x, s):
    return x / s**2 * np.exp(-0.5 * (x / s) ** 2)


def density(e, i_deg, model):
    si = np.sin(np.radians(i_deg))
    jac = np.cos(np.radians(i_deg))                     # d(sin i)/di, uniform sampling is in i
    if model["kind"] == "indep":
        return rayleigh(e, model["sig_e"]) * rayleigh(si, model["sig_i"]) * jac
    cold = rayleigh(e, model["sig_e_c"]) * rayleigh(si, model["sig_i_c"])
    hot = rayleigh(e, model["sig_e_h"]) * rayleigh(si, model["sig_i_h"])
    return ((1 - model["f_hot"]) * cold + model["f_hot"] * hot) * jac


MODELS = {
    "uniform": None,
    "indep_cold": {"kind": "indep", "sig_e": 0.05, "sig_i": np.sin(np.radians(5))},
    "indep_warm": {"kind": "indep", "sig_e": 0.1, "sig_i": np.sin(np.radians(10))},
    "indep_hot": {"kind": "indep", "sig_e": 0.1, "sig_i": np.sin(np.radians(20))},
    "mix_30": {"kind": "mix", "f_hot": 0.3, "sig_e_c": 0.02, "sig_i_c": np.sin(np.radians(5)),
               "sig_e_h": 0.15, "sig_i_h": np.sin(np.radians(20))},
    "mix_50": {"kind": "mix", "f_hot": 0.5, "sig_e_c": 0.02, "sig_i_c": np.sin(np.radians(5)),
               "sig_e_h": 0.15, "sig_i_h": np.sin(np.radians(20))},
    "mix_70": {"kind": "mix", "f_hot": 0.7, "sig_e_c": 0.02, "sig_i_c": np.sin(np.radians(5)),
               "sig_e_h": 0.15, "sig_i_h": np.sin(np.radians(20))},
}


def weighted_ratio(t: pd.DataFrame, w: np.ndarray) -> pd.DataFrame:
    rows = []
    b = pd.cut(t["inc"], I_EDGES, right=False, labels=OBS["bin"])
    for lab in OBS["bin"]:
        m = (b == lab).to_numpy()
        w4, w5 = w[m & (t["swarm"] == "L4").to_numpy()], w[m & (t["swarm"] == "L5").to_numpy()]
        s4, s5 = w4.sum(), w5.sum()
        r = s4 / s5 if s5 > 0 else np.nan
        err = r * np.sqrt((w4**2).sum() / s4**2 + (w5**2).sum() / s5**2) if s4 > 0 and s5 > 0 else np.nan
        rows.append((lab, len(w4), len(w5), r, err))
    return pd.DataFrame(rows, columns=["bin", "n4", "n5", "N4/N5", "err"])


def reweight(run_id: str) -> pd.DataFrame:
    p = pd.read_parquet(paths.report_dir(run_id) / "particles.parquet")
    t = p[p["swarm"].isin(["L4", "L5"])].reset_index(drop=True)
    e_max, i_max = p["e0"].max(), p["i0"].max()
    out = []
    for name, model in MODELS.items():
        w = np.ones(len(t)) if model is None else density(t["e0"].to_numpy(), t["i0"].to_numpy(), model)
        r = weighted_ratio(t, w)
        tot4, tot5 = w[(t["swarm"] == "L4").to_numpy()].sum(), w[(t["swarm"] == "L5").to_numpy()].sum()
        out.append({"model": name, **{f"{b}": f"{x:.2f}±{s:.2f}" for b, x, s in zip(r["bin"], r["N4/N5"], r["err"])},
                    "all": f"{tot4 / tot5:.2f}"})
    print(f"\n{run_id}: {len(t)} Trojans, e0 <= {e_max:.2f}, i0 <= {i_max:.0f} deg; N4/N5 by final I")
    df = pd.DataFrame(out)
    print(df.to_string(index=False))
    print("observed (I_p, families removed): " + "  ".join(f"{b} {x:.2f}±{s:.2f}" for b, x, s in OBS.itertuples(index=False)))
    df.to_csv(paths.report_dir(run_id) / "reweighted.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+")
    for r in ap.parse_args().runs:
        reweight(r)


if __name__ == "__main__":
    main()
