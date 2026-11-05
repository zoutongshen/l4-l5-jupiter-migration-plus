"""Physical growth-and-migration tracks for the giants (stage 8, Block B/C).

Replaces the single-exponential growth law, which the literature rules out
(docs/RUN_migration_capture.md, "Which growth law is physical"). A track has:

Mass, three phases (durations are inputs; the planet sits as a seed before t_start):
  core    seed -> M_iso by pebble accretion, dM/dt ~ M^(2/3): M^(1/3) linear in t
          (Johansen, Ida & Brasser 2019 eq. 1; Bitsch+2015 eqs. 7-8)
  slow    M_iso -> 2 M_iso, envelope contraction until M_env = M_core: linear in t
          (Bitsch+2015 eq. 17; Pirani+2019a section 3.1)
  gas     2 M_iso -> M_final, runaway gas accretion limited by the disc's supply:
          linear in t (Tanigawa & Tanaka 2016; JIB19 eq. 40)
  A planet whose final mass is below 2 M_iso (Uranus, Neptune) only has the core
  phase, stretched over the whole window, ending at its final mass.

Migration: da/dt = -/+ nu(m) q Omega a with
  nu(m) = nu_I / (1 + (m / M_gap)^p)
nu_I is the type I value, independent of the planet's mass and set by the disc
(JIB19 eq. 3; Tanaka+2020 eq. 8). Above M_gap the gap deepens (Kanagawa+2016:
Sigma_gap/Sigma = 1/(1 + 0.04 K), K ~ q^2, so p = 2) and the planet slows to type II.
p = 2 is that scaling; Pirani's x0.25 track falls faster (her Fig. 13: rate ~ m^-2.4
above 30 M_E, i.e. p ~ 3.4), so p is an input.

The track is fixed by its END position (the giants' configuration when the gas
disc goes); the start is solved for, so Delta a is an outcome of nu_I, M_gap, p
and the durations, as in a real disc.

    python campaign/scripts/s8_tracks.py --name jupiter_c0 --planet jupiter --a-end 5.4
    python campaign/scripts/s8_tracks.py --calibrate      # fit nu_I, M_gap, p to Pirani's x0.25 track
"""

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from trojans import constants
from trojans.disc import M_EARTH

TRACKS = Path(__file__).resolve().parents[1] / "configs" / "tracks"
GM_SUN = constants.G * constants.CENTRAL_MASS
M_FINAL_EARTH = {g: constants.GIANT_MASS[g] / M_EARTH for g in constants.GIANTS}


@dataclass(frozen=True)
class Track:
    a_end_au: float
    m_final_earth: float
    direction: str = "inward"          # inward | outward
    m_seed_earth: float = 0.017        # Pirani's seeds are ~1e-2 M_E
    m_iso_earth: float = 20.0
    nu_I: float = 0.3
    m_gap_earth: float = 30.0
    gap_power: float = 2.0
    t_start_yr: float = 0.0
    t_core_yr: float = 2.0e5
    t_slow_yr: float = 0.5e5
    t_gas_yr: float = 4.4e5
    n_rows: int = 3000

    @property
    def t_end_yr(self) -> float:
        return self.t_start_yr + self.t_core_yr + self.t_slow_yr + self.t_gas_yr

    def mass_earth(self, t):
        """Mass in M_E on an array of times (seed before t_start, final after the end)."""
        t = np.asarray(t, dtype=float) - self.t_start_yr
        m0, mi, mf = self.m_seed_earth, self.m_iso_earth, self.m_final_earth
        if mf <= 2.0 * mi:                       # ice giant: core phase over the whole window
            T = self.t_core_yr + self.t_slow_yr + self.t_gas_yr
            u = np.clip(t / T, 0.0, 1.0)
            return (m0 ** (1 / 3) + (mf ** (1 / 3) - m0 ** (1 / 3)) * u) ** 3
        t1, t2 = self.t_core_yr, self.t_core_yr + self.t_slow_yr
        core = (m0 ** (1 / 3) + (mi ** (1 / 3) - m0 ** (1 / 3)) * np.clip(t / t1, 0, 1)) ** 3
        slow = mi + mi * np.clip((t - t1) / self.t_slow_yr, 0, 1)
        gas = 2 * mi + (mf - 2 * mi) * np.clip((t - t2) / self.t_gas_yr, 0, 1)
        return np.where(t < t1, core, np.where(t < t2, slow, gas))

    def nu(self, m_earth):
        return self.nu_I / (1.0 + (np.asarray(m_earth) / self.m_gap_earth) ** self.gap_power)

    def times(self):
        """Seed, then a dense grid over the growth window (denser where the core is light)."""
        u = np.linspace(0.0, 1.0, self.n_rows)
        grid = self.t_start_yr + (self.t_end_yr - self.t_start_yr) * u
        head = [0.0] if self.t_start_yr > 0 else []
        return np.unique(np.concatenate([head, grid]))

    def build(self):
        """(t, a, m_earth) with a(t_end) = a_end exactly."""
        t = self.times()
        m = self.mass_earth(t)
        q = m * M_EARTH / constants.CENTRAL_MASS
        # da/dt = -/+ nu q sqrt(G M_sun / a)  <=>  d(a^(3/2))/dt = -/+ (3/2) nu q sqrt(G M_sun)
        rate = self.nu(m) * q
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(t))])
        sign = -1.0 if self.direction == "inward" else 1.0
        # a^(3/2)(t) = a_end^(3/2) - sign (3/2) sqrt(GM) (cum_end - cum(t))
        w = self.a_end_au**1.5 - sign * 1.5 * np.sqrt(GM_SUN) * (cum[-1] - cum)
        if np.any(w <= 0):
            raise ValueError("track reaches a <= 0: nu_I or the durations are too large")
        a = w ** (2.0 / 3.0)
        a[-1] = self.a_end_au
        return t, a, m


def nu_along(t, a, m_earth):
    adot = np.gradient(a, t)
    q = m_earth * M_EARTH / constants.CENTRAL_MASS
    return np.abs(adot) / (q * np.sqrt(GM_SUN / a))


def write(track: Track, name: str) -> Path:
    t, a, m = track.build()
    out = TRACKS / f"{name}.csv"
    header = "t_yr,a_au,m_earth  # " + " ".join(f"{k}={v}" for k, v in asdict(track).items())
    np.savetxt(out, np.c_[t, a, m], delimiter=",", header=header, comments="")
    return out


def calibrate():
    """Fit nu_I, M_gap and p so a(m) matches Pirani's x0.25 track (Fig. 3, digitized)."""
    m3, a3 = np.loadtxt(TRACKS / "pirani_fig3_x025.txt").T
    t_p, a_p, m_p = np.loadtxt(TRACKS / "pirani_x025.csv", delimiter=",", skiprows=1, unpack=True)
    ok = (m3 > 1.0) & (m3 < 290.0)
    best = None
    for nu_I in np.arange(0.15, 0.51, 0.025):
        for m_gap in (15, 20, 25, 30, 40, 50, 60):
            for p in (2.0, 2.5, 3.0, 3.5, 4.0):
                tr = Track(a_end_au=a_p[-1], m_final_earth=M_FINAL_EARTH["jupiter"], nu_I=nu_I,
                           m_gap_earth=m_gap, gap_power=p)
                t, a, m = tr.build()
                model = np.interp(np.log(m3[ok]), np.log(m), a)
                err = np.sqrt(np.mean((model - a3[ok]) ** 2))
                if best is None or err < best[0]:
                    best = (err, nu_I, m_gap, p, a[0])
    err, nu_I, m_gap, p, a0 = best
    print(f"best fit to Pirani x0.25 a(m): nu_I = {nu_I:.3f}, M_gap = {m_gap} M_E, p = {p}; "
          f"rms {err:.3f} au; start {a0:.2f} au (hers {a_p[0]:.2f})")
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--reverse", metavar="NAME", help="write NAME_reversed.csv: same clock and masses, drift reversed")
    ap.add_argument("--name")
    ap.add_argument("--planet", choices=constants.GIANTS)
    ap.add_argument("--a-end", type=float)
    for f in ("direction", "m_seed_earth", "m_iso_earth", "nu_I", "m_gap_earth", "gap_power",
              "t_start_yr", "t_core_yr", "t_slow_yr", "t_gas_yr"):
        ap.add_argument("--" + f.replace("_", "-"), dest=f, type=str if f == "direction" else float)
    args = ap.parse_args()
    if args.calibrate:
        calibrate()
        return
    if args.reverse:
        t, a, m = np.loadtxt(TRACKS / f"{args.reverse}.csv", delimiter=",", skiprows=1, unpack=True)
        out = TRACKS / f"{args.reverse}_reversed.csv"
        np.savetxt(out, np.c_[t, a[-1] + a[0] - a, m], delimiter=",",
                   header=f"t_yr,a_au,m_earth  # {args.reverse} with the drift reversed", comments="")
        print(f"wrote {out}: {a[-1]:.3f} -> {a[0]:.3f} au")
        return
    kw = {k: v for k, v in vars(args).items() if k in Track.__dataclass_fields__ and v is not None}
    track = Track(a_end_au=args.a_end, m_final_earth=M_FINAL_EARTH[args.planet], **kw)
    out = write(track, args.name)
    t, a, m = track.build()
    nu = nu_along(t, a, m)
    print(f"wrote {out}: {a[0]:.3f} -> {a[-1]:.3f} au, {m[0]:.3g} -> {m[-1]:.1f} M_E over "
          f"[{track.t_start_yr / 1e3:.0f}, {track.t_end_yr / 1e3:.0f}] kyr; nu at 1/10/30/100 M_E: "
          + " ".join(f"{np.interp(x, m, nu):.3f}" for x in (1, 10, 30, 100)))


if __name__ == "__main__":
    main()
