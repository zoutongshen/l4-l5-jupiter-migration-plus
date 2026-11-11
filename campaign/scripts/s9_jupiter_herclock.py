"""Stage 9: Jupiter's nominal track on HER clock at all masses (seed -> 100 M_E), not our growth model's.

Her Fig. 1 gives the path a(m) and her Fig. 13 the speed |adot|(m); together they fix the clock,
t(m) = t0 + int |da/dm| / |adot| dm. The default track (s9_pirani_tracks.py) uses this only on
3-100 M_E and our pebble-accretion clock below 3 M_E, which grows 0.05 -> 3 M_E ~1.4x faster than her
figures imply and so starts the seed at 18.22 au instead of her 18.39. Here:
  - path: her Fig. 1 (smoothed as in s9_pirani_tracks.load), seed at her start a;
  - clock: her Figs. 1 + 13 from the seed to 100 M_E (Fig. 13 starts at 0.2 M_E; below it her lowest
    speed, 1.33 au/Myr, is held, as her curve is flat there); above 100 M_E the model clock is stretched
    to end at 3 Myr (unchanged from the default track);
  - t0 from her Fig. 4 snapshots at 2.5 and 2.6 Myr, as the default track.
Writes configs/tracks/pirani19a/track_nominal_jupiter_herclock.csv.

    python campaign/scripts/s9_jupiter_herclock.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s9_pirani_tracks as T  # noqa: E402


def her_full(t0):
    m, a, _ = T.load("fig1_jupiter")
    r = T.pt.integrate(T.params(t0, m[0]), path=(m, a))
    return T.her_window_clock(r, m, a, lo=float(m[0]) * 1.0001, hi=100.0)


def start():
    snap = np.loadtxt(T.DIG / "fig4_planets.csv", delimiter=",", skiprows=1)
    clean = [(t, a) for t, a in zip(snap[:, 0], snap[:, 1]) if t in (2.5, 2.6)]
    f = lambda t0: np.mean([her_full(t0)["t"][np.argmax(her_full(t0)["a"] < a)] / 1e6 - t for t, a in clean])
    return brentq(f, 2.2e6, 2.4e6, xtol=200.0), clean


def main():
    t0, clean = start()
    r = her_full(t0)
    t, a, m = T.table(r["t"], r["a"], r["m"])
    name = T.write("nominal_jupiter_herclock", t, a, m, "nominal jupiter, Fig. 1 path, her Figs. 1+13 clock from the seed")
    old = np.loadtxt(T.OUT_TRACKS / "track_nominal_jupiter.csv", delimiter=",", skiprows=1)
    print(f"{name}: t0 = {t0 / 1e6:.4f} Myr (default track {old[1, 0] / 1e6:.4f}), seed a = {a[1]:.2f} au "
          f"(default {old[1, 1]:.2f}); m(3 Myr) = {m[-1]:.1f}")
    for ts, asnap in clean:
        print(f"  her Fig. 4 at {ts} Myr: a = {asnap:.2f}; this track {np.interp(ts * 1e6, t, a):.2f}, "
              f"default {np.interp(ts * 1e6, old[:, 0], old[:, 1]):.2f}")
    print(" m     t her-clock   t default   a her-clock  a default")
    for mm in (0.05, 0.2, 0.5, 1, 2, 3, 6, 10, 30, 100, 300):
        i, j = np.argmin(abs(m - mm)), np.argmin(abs(old[:, 2] - mm))
        print(f"{mm:6.2f}   {t[i] / 1e6:.4f}      {old[j, 0] / 1e6:.4f}      {a[i]:6.2f}     {old[j, 1]:6.2f}")


if __name__ == "__main__":
    main()
