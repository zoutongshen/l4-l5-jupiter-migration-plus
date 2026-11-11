"""Overlay one s9 run's Trojan count and N4/N5 on her Fig. 7 (vector-extracted, one of her 10 nominal runs).
Her N4, N5 are derived from her N_tot and ratio: N4 = N r/(1+r), N5 = N/(1+r).

    python campaign/scripts/s9_vs_fig7.py pir_p1_nominal_r0 pir_nominal_r0
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
rep, run = sys.argv[1], sys.argv[2]
her = pd.read_csv(ROOT / "configs/tracks/pirani19a/fig7_curves.csv")
h = (her.groupby(["panel", "t_myr"]).value.mean().unstack(0).dropna().reset_index())
h["N4"] = h.N_tot * h.ratio / (1 + h.ratio)
h["N5"] = h.N_tot / (1 + h.ratio)
ts = pd.read_csv(ROOT / "reports" / rep / f"timeseries_{run}.csv")
t = ts.t_yr / 1e6
HER, OURS = "#2a78d6", "#eb6834"
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)
ax[0].plot(h.t_myr, h.N_tot, color=HER, lw=2, label="Pirani Fig. 7 (one of her 10 runs)")
ax[0].plot(t, ts.N_tot, color=OURS, lw=1.2, label=f"ours ({run})")
ax[0].set(ylabel="N Trojans", title="Trojan number")
with np.errstate(divide="ignore", invalid="ignore"):
    ax[1].plot(h.t_myr, h.ratio, color=HER, lw=2)
    ax[1].plot(t, ts.N4 / ts.N5, color=OURS, lw=1.2)
ax[1].axhline(1.80, color="0.6", lw=0.8, ls=":")
ax[1].text(4.6, 1.84, "her Table 2 mean, 5 Myr: 1.80 ± 0.26", fontsize=8, color="0.35", ha="right")
ax[1].set(ylabel="N4/N5", ylim=(0.4, 5.5), title="asymmetry")
for col, ls, lab in [("N4", "-", "L4"), ("N5", "--", "L5")]:
    ax[2].plot(h.t_myr, h[col], color=HER, lw=2, ls=ls, label=f"her {lab}")
    ax[2].plot(t, ts[col], color=OURS, lw=1.2, ls=ls, label=f"our {lab}")
ax[2].set(ylabel="N per swarm", title="L4 (solid) and L5 (dashed) separately")
ax[0].legend(fontsize=8, frameon=False)
ax[2].legend(fontsize=8, frameon=False, ncol=2)
for a in ax:
    a.set(xlabel="t [Myr]", xlim=(1.5, 5))
    a.axvspan(2.31, 3.0, color="0.93", zorder=0)
out = ROOT / "reports" / rep / "vs_fig7.png"
fig.savefig(out, dpi=130)
print(out)
for T in (2.35, 2.40, 2.45, 2.5, 2.6, 2.7, 2.8, 3.0, 3.5, 5.0):
    hi = h.iloc[(h.t_myr - T).abs().argmin()]
    oi = ts.iloc[(t - T).abs().argmin()]
    print(f"{T:4.2f}  her N4 {hi.N4:4.0f} N5 {hi.N5:4.0f} r {hi.ratio:4.2f} | ours N4 {oi.N4:4.0f} N5 {oi.N5:4.0f} "
          f"r {oi.N4 / max(oi.N5, 1):4.2f} | J a {oi.jupiter_a:5.2f} m {oi.jupiter_m_earth:5.1f}")
