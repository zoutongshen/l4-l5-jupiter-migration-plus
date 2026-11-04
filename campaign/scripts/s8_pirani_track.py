"""Build Pirani+2019a's x0.25 Jupiter track (a, m against t) from her figures.

Inputs, digitized from the arXiv PDF at 400 dpi (configs/tracks/):
  pirani_fig3_x025.txt   Fig. 3, red curve: mass against semimajor axis
  pirani_fig13_x025.txt  Fig. 13, green curve: migration rate against mass (0.1-100 M_E)

Her figures give no clock, so time is rebuilt in three phases:
  A  seed (her 10^-2 M_E scale, here 0.017 M_E) to 0.6 M_E: exponential growth over T_A
     (not constrained by the figures; a short pebble phase), drifting at Fig. 13's rate
     (extended below 0.1 M_E as a power law), ~0.01 au in all;
  B  0.6 to 100 M_E: dt = da / adot(m(a)) from the two figures (~0.32 Myr);
  C  100 M_E to 1 M_J: the remaining time to her 0.69 Myr (2.31 -> 3.00 Myr), with
     exponential growth, and a(m) still from Fig. 3.
The output table (t, a, m) feeds JupiterHistory(migration_law="table").

    python campaign/scripts/s8_pirani_track.py      # writes configs/tracks/pirani_x025[_outward].csv
"""

from pathlib import Path

import numpy as np

TRACKS = Path(__file__).resolve().parents[1] / "configs" / "tracks"
T_TOTAL = 0.69e6
T_A = 0.03e6
M_SEED, M_A, M_C, M_JUP = 0.017, 0.6, 100.0, 317.83


def smooth_a_of_m(m, a, grid):
    """a(log m): medians in 0.1-dex bins (pixel steps are 0.036 au), linear between them."""
    lm = np.log10(m)
    nodes = np.arange(np.log10(M_SEED), np.log10(M_JUP) + 0.1, 0.1)
    med = np.array([np.median(a[np.abs(lm - g) < 0.05]) if np.any(np.abs(lm - g) < 0.05) else np.nan
                    for g in nodes])
    ok = np.isfinite(med)
    nodes, med = nodes[ok], np.minimum.accumulate(med[ok])      # inward: a never increases with m
    return np.interp(grid, nodes, med)


def build():
    m3, a3 = np.loadtxt(TRACKS / "pirani_fig3_x025.txt").T
    m13, v13 = np.loadtxt(TRACKS / "pirani_fig13_x025.txt").T
    grid = np.linspace(np.log10(M_SEED), np.log10(M_JUP), 2000)
    a_of = smooth_a_of_m(m3, a3, grid)
    m = 10**grid
    t = np.zeros_like(m)
    # A: exponential growth over T_A; the drift uses Fig. 13's rate, extended below
    # 0.1 M_E as a power law (adot ~ m, type I), and ends on Fig. 3's a at M_A.
    A = m <= M_A
    t[A] = T_A * np.log(m[A] / M_SEED) / np.log(M_A / M_SEED)
    lo = m13 < 0.5
    slope, icpt = np.polyfit(np.log10(m13[lo]), np.log10(v13[lo]), 1)
    rate_A = 10**(icpt + slope * grid[A]) / 1e6
    drift = np.concatenate([[0.0], np.cumsum(0.5 * (rate_A[1:] + rate_A[:-1]) * np.diff(t[A]))])
    iA = np.flatnonzero(A)
    a_of[A] = a_of[iA[-1]] + (drift[-1] - drift)
    # B: dt = da / adot
    B = (m > M_A) & (m <= M_C)
    rate = 10**np.interp(grid, np.log10(m13), np.log10(v13)) / 1e6        # au/yr
    iB = np.flatnonzero(B)
    for i in iB:
        da = a_of[i - 1] - a_of[i]
        t[i] = t[i - 1] + da / (0.5 * (rate[i - 1] + rate[i]))
    t_B = t[iB[-1]]
    # C: rest of the time, exponential growth
    C = m > M_C
    T_C = T_TOTAL - t_B
    t[C] = t_B + T_C * np.log(m[C] / M_C) / np.log(M_JUP / M_C)
    t = np.round(t, 3)                      # exact 0 and T_TOTAL at the ends (log roundoff)
    t[0], t[-1] = 0.0, T_TOTAL
    keep = np.concatenate([[True], np.diff(t) > 0])
    return t[keep], a_of[keep], m[keep], t_B - T_A, T_C


def main():
    t, a, m, dur_B, T_C = build()
    out = TRACKS / "pirani_x025.csv"
    np.savetxt(out, np.c_[t, a, m], delimiter=",", header="t_yr,a_au,m_earth", comments="")
    # The sign check: the same clock and masses, with the drift reversed (a_end -> a_start).
    a_out = a[-1] + (a[0] - a)
    np.savetxt(TRACKS / "pirani_x025_outward.csv", np.c_[t, a_out, m], delimiter=",",
               header="t_yr,a_au,m_earth", comments="")
    print(f"wrote {out}: {len(t)} rows; phase B {dur_B / 1e3:.0f} kyr, phase C {T_C / 1e3:.0f} kyr; "
          f"a {a[0]:.3f} -> {a[-1]:.3f} au")


if __name__ == "__main__":
    main()
