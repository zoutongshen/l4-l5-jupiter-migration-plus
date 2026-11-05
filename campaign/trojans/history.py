"""Jupiter's growth and migration history for the capture runs (stage 8).

Jupiter's semimajor axis a(t) and mass m(t) are pure functions of time, and
``HistoryForcing`` imposes them on a circular, planar Jupiter after every block
of steps. Because nothing depends on the previous call, a resumed run continues
the history exactly (the ``ChunkRunner`` forcing contract).

The rate is described by the dimensionless migration speed

    nu = |da/dt| / (q Omega a),    q = m / M_sun,  Omega a = sqrt(G (M_sun + m) / a),

the variable in which Pirani+2019a (p. 11) put the asymmetry window:
nu > 0.17 (v_lim, below which the horseshoe asymmetry is too weak) and
nu < 1.45 (v_crit, Ogilvie & Lubow 2006, where L4 merges with L3 and inward
migration can hold no L4 tadpoles).

Migration laws, for s = (t - t_mig_start) / t_mig clipped to [0, 1]:
  const_nu     a^(3/2) linear in s: constant nu at a fixed mass (the default)
  linear       a linear in s: constant da/dt
  exponential  ln a linear in s: constant (da/dt)/a
  table        a(t) and m(t) both interpolated from a table (``from_table``; e.g. a
               track digitized from Pirani+2019a): a linear in t, ln m linear in t.
               The growth fields then only mirror the table's ends.

Growth laws, for u = (t - t_grow_start) / t_grow clipped to [0, 1]:
  exponential  ln m linear in u (dm/dt proportional to m, runaway-like)
  linear       m linear in u
``t_grow = 0`` means full mass from ``t_grow_start`` on (and m0 before it).
"""

import math
from dataclasses import asdict, dataclass

import numpy as np
import rebound

from trojans import constants
from trojans.disc import M_EARTH, hill_radius

MIGRATION_LAWS = ("const_nu", "linear", "exponential", "table")
GROWTH_LAWS = ("exponential", "linear")
GM_SUN = constants.G * constants.CENTRAL_MASS

# The forcing leaves Jupiter alone when it is already where the history says.
# WHFast keeps a circular two-body orbit's radius to ~1e-14; a migrating Jupiter
# moves by > 1e-6 of a per check at the slowest rate we use.
RADIUS_TOLERANCE = 1e-10


def _clip(x: float) -> float:
    return min(max(x, 0.0), 1.0)


@dataclass(frozen=True)
class JupiterHistory:
    a_start_au: float
    a_end_au: float
    t_mig_yr: float = 0.0
    t_mig_start_yr: float = 0.0
    migration_law: str = "const_nu"
    m0_msun: float = 10.0 * M_EARTH
    m_final_msun: float = constants.GIANT_MASS["jupiter"]
    t_grow_start_yr: float = 0.0
    t_grow_yr: float = 0.0
    growth_law: str = "exponential"
    table_t_yr: tuple = ()
    table_a_au: tuple = ()
    table_m_msun: tuple = ()

    def __post_init__(self):
        for name in ("table_t_yr", "table_a_au", "table_m_msun"):     # lists from JSON
            object.__setattr__(self, name, tuple(float(x) for x in getattr(self, name)))
        if self.migration_law == "table":
            t, a, m = self.table_t_yr, self.table_a_au, self.table_m_msun
            if not (len(t) == len(a) == len(m) >= 2) or t[0] != 0.0 or not np.all(np.diff(t) > 0):
                raise ValueError("table needs equal-length t, a, m with t[0] = 0 and t increasing")
            expect = {"a_start_au": a[0], "a_end_au": a[-1], "t_mig_yr": t[-1], "t_mig_start_yr": 0.0,
                      "m0_msun": m[0], "m_final_msun": m[-1], "t_grow_start_yr": 0.0, "t_grow_yr": t[-1]}
            for k, v in expect.items():
                if getattr(self, k) != v:
                    raise ValueError(f"table history: {k} must be {v} (use from_table)")
        elif self.table_t_yr:
            raise ValueError("table_* fields need migration_law='table'")
        if self.migration_law not in MIGRATION_LAWS:
            raise ValueError(f"migration_law must be one of {MIGRATION_LAWS}")
        if self.growth_law not in GROWTH_LAWS:
            raise ValueError(f"growth_law must be one of {GROWTH_LAWS}")
        if self.migrates and self.t_mig_yr <= 0.0:
            raise ValueError("a_start != a_end needs t_mig_yr > 0")
        if min(self.a_start_au, self.a_end_au, self.m0_msun, self.m_final_msun) <= 0.0:
            raise ValueError("a and m must be positive")

    # --- construction from a rate -------------------------------------------

    @classmethod
    def from_nu(cls, nu0: float, a_start_au: float, a_end_au: float, **kwargs) -> "JupiterHistory":
        """``const_nu`` migration whose speed is ``nu0`` for a planet of mass m0.

        With q0 = m0 / M_sun fixed, da/dt = -/+ nu0 q0 sqrt(G M' / a) integrates to
        a^(3/2) changing by (3/2) nu0 q0 sqrt(G M') per unit time, M' = M_sun + m0.
        Once Jupiter grows, the same da/dt is a smaller nu (see ``nu``).
        """
        m0 = kwargs.get("m0_msun", cls.m0_msun)
        q0 = m0 / constants.CENTRAL_MASS
        rate = 1.5 * nu0 * q0 * math.sqrt(constants.G * (constants.CENTRAL_MASS + m0))
        t_mig = abs(a_end_au**1.5 - a_start_au**1.5) / rate
        return cls(a_start_au=a_start_au, a_end_au=a_end_au, t_mig_yr=t_mig,
                   migration_law="const_nu", **kwargs)

    @classmethod
    def from_table(cls, t_yr, a_au, m_msun) -> "JupiterHistory":
        """A history interpolated from a table of (t, a, m), starting at t = 0."""
        t, a, m = (tuple(float(x) for x in v) for v in (t_yr, a_au, m_msun))
        return cls(a_start_au=a[0], a_end_au=a[-1], t_mig_yr=t[-1], migration_law="table",
                   m0_msun=m[0], m_final_msun=m[-1], t_grow_yr=t[-1],
                   table_t_yr=t, table_a_au=a, table_m_msun=m)

    # --- the history ----------------------------------------------------------

    @property
    def migrates(self) -> bool:
        return self.a_start_au != self.a_end_au

    @property
    def direction(self) -> str:
        if not self.migrates:
            return "in_situ"
        return "inward" if self.a_end_au < self.a_start_au else "outward"

    @property
    def a_min_au(self) -> float:
        return min(self.a_start_au, self.a_end_au)

    @property
    def t_mig_end_yr(self) -> float:
        return self.t_mig_start_yr + self.t_mig_yr if self.migrates else 0.0

    @property
    def t_grow_end_yr(self) -> float:
        return self.t_grow_start_yr + self.t_grow_yr

    @property
    def t_forcing_end_yr(self) -> float:
        """After this Jupiter is fixed: full mass, final a."""
        return max(self.t_mig_end_yr, self.t_grow_end_yr)

    def _s(self, t: float) -> float:
        return _clip((t - self.t_mig_start_yr) / self.t_mig_yr) if self.migrates else 1.0

    def a(self, t: float) -> float:
        if self.migration_law == "table":
            return float(np.interp(t, self.table_t_yr, self.table_a_au))
        if not self.migrates:
            return self.a_end_au
        s, a0, a1 = self._s(t), self.a_start_au, self.a_end_au
        if s <= 0.0:
            return a0
        if s >= 1.0:
            return a1
        if self.migration_law == "linear":
            return a0 + (a1 - a0) * s
        if self.migration_law == "exponential":
            return a0 * (a1 / a0) ** s
        return (a0**1.5 + (a1**1.5 - a0**1.5) * s) ** (2.0 / 3.0)

    def adot(self, t: float) -> float:
        """da/dt in au/yr (signed; negative inward); zero outside the migration."""
        if not self.migrates or not (self.t_mig_start_yr <= t < self.t_mig_end_yr):
            return 0.0
        a0, a1, T = self.a_start_au, self.a_end_au, self.t_mig_yr
        if self.migration_law == "table":
            tt, aa = self.table_t_yr, self.table_a_au
            i = min(int(np.searchsorted(tt, t, side="right")) - 1, len(tt) - 2)
            return (aa[i + 1] - aa[i]) / (tt[i + 1] - tt[i])
        if self.migration_law == "linear":
            return (a1 - a0) / T
        if self.migration_law == "exponential":
            return self.a(t) * math.log(a1 / a0) / T
        return (2.0 / 3.0) * (a1**1.5 - a0**1.5) / T / math.sqrt(self.a(t))

    def mass(self, t: float) -> float:
        """Jupiter's mass in M_sun at time t (named like ``disc.Growth.mass``)."""
        m0, m1 = self.m0_msun, self.m_final_msun
        if self.migration_law == "table":
            if t >= self.table_t_yr[-1]:
                return m1
            return float(np.exp(np.interp(t, self.table_t_yr, np.log(self.table_m_msun))))
        if self.t_grow_yr <= 0.0:
            return m1 if t >= self.t_grow_start_yr else m0
        u = _clip((t - self.t_grow_start_yr) / self.t_grow_yr)
        if u >= 1.0:
            return m1
        if self.growth_law == "linear":
            return m0 + (m1 - m0) * u
        return m0 * (m1 / m0) ** u

    def nu(self, t: float) -> float:
        """|da/dt| / (q Omega a) at time t, with the current mass."""
        m, a = self.mass(t), self.a(t)
        q = m / constants.CENTRAL_MASS
        return abs(self.adot(t)) / (q * math.sqrt(constants.G * (constants.CENTRAL_MASS + m) / a))

    def summary(self) -> dict:
        """The history plus derived numbers, for settings.json."""
        mid = self.t_mig_start_yr + 0.5 * self.t_mig_yr
        record = asdict(self)
        if self.migration_law == "table":
            record["table_rows"] = len(self.table_t_yr)
        return {**record, "direction": self.direction,
                "m0_earth": self.m0_msun / M_EARTH, "m_final_earth": self.m_final_msun / M_EARTH,
                "t_forcing_end_yr": self.t_forcing_end_yr,
                "mean_rate_au_per_myr": (self.a_end_au - self.a_start_au) / self.t_mig_yr * 1e6
                if self.migrates else 0.0,
                "nu_start": self.nu(self.t_mig_start_yr), "nu_mid": self.nu(mid),
                "nu_end": self.nu(max(self.t_mig_end_yr - 1e-9 * max(self.t_mig_yr, 1.0), 0.0))}


class HistoryForcing:
    """``ChunkRunner`` forcing: put Jupiter on the circular orbit of a(t) with mass m(t).

    Jupiter keeps its current direction from the Sun; its distance is set to a(t)
    and its velocity to the circular speed, prograde in the reference plane.
    Nothing is changed when Jupiter already has that mass and radius, so a
    non-migrating history with the same growth reproduces ``disc.GrowthForcing``
    exactly, and calling the forcing twice is the same as calling it once.
    """

    def __init__(self, history: JupiterHistory):
        self.history = history

    def __call__(self, sim: rebound.Simulation, runner) -> None:
        t = sim.t
        a, m = self.history.a(t), self.history.mass(t)
        sun, jup = sim.particles[0], sim.particles[1]
        rel_x = np.array(jup.xyz) - np.array(sun.xyz)
        r = float(np.linalg.norm(rel_x))
        move = abs(r - a) > RADIUS_TOLERANCE * a
        if m != jup.m or move:
            # Without a move the position is left untouched (as in GrowthForcing).
            radius = a if move else r
            speed = np.sqrt(constants.G * (constants.CENTRAL_MASS + m) / radius)
            tangent = np.array([-rel_x[1], rel_x[0], 0.0]) / np.hypot(rel_x[0], rel_x[1])
            jup.m = m
            if move:
                jup.xyz = tuple(np.array(sun.xyz) + a * rel_x / r)
            jup.vxyz = tuple(np.array(sun.vxyz) + speed * tangent)
            sim.move_to_com()
            sim.ri_whfast.recalculate_coordinates_this_timestep = 1
        runner.m_jupiter = m
        runner.hill = np.array([hill_radius(a, m)])


# --- several giants ------------------------------------------------------------

@dataclass(frozen=True)
class SystemHistory:
    """One history per giant, Jupiter first, in simulation order.

    Each giant is forced onto its circular orbit of a(t), m(t) only while its own
    history runs (t < its t_forcing_end_yr). After that it moves freely under the
    N-body forces, so the giants can perturb each other once the gas disc is gone.
    A giant whose history never forces it (a_start = a_end, t_grow_yr = 0, m0 = m_final)
    is free from t = 0.
    """
    planets: tuple
    histories: tuple

    def __post_init__(self):
        object.__setattr__(self, "planets", tuple(self.planets))
        hs = tuple(h if isinstance(h, JupiterHistory) else JupiterHistory(**h) for h in self.histories)
        object.__setattr__(self, "histories", hs)
        if len(self.planets) != len(hs) or not self.planets or self.planets[0] != "jupiter":
            raise ValueError("SystemHistory needs one history per planet, Jupiter first")

    @property
    def jupiter(self) -> JupiterHistory:
        return self.histories[0]

    @property
    def t_forcing_end_yr(self) -> float:
        return max(h.t_forcing_end_yr for h in self.histories)

    @staticmethod
    def forced_until(h: JupiterHistory) -> float:
        """Time after which this giant is left alone (0 for a giant that is fixed from the start)."""
        static = not h.migrates and h.t_grow_yr <= 0.0 and h.m0_msun == h.m_final_msun
        return 0.0 if static else h.t_forcing_end_yr


class SystemForcing:
    """``ChunkRunner`` forcing for several giants (particles 1..n after the Sun).

    While a giant's history runs, it is put on the circular orbit of its a(t), m(t),
    keeping its direction from the Sun (as ``HistoryForcing``). After its history
    ends, at t >= ``SystemHistory.forced_until``, it is not touched again.
    ``runner.hill`` follows each giant's history (its final values once free) and
    ``runner.m_jupiter`` Jupiter's mass. Everything depends on t only, so resumes
    are exact. With Jupiter alone this is ``HistoryForcing`` (bitwise; tested).
    """

    def __init__(self, system: SystemHistory):
        self.system = system
        self.until = [SystemHistory.forced_until(h) for h in system.histories]

    def __call__(self, sim: rebound.Simulation, runner) -> None:
        t = sim.t
        sun = sim.particles[0]
        changed = False
        hill = np.empty(len(self.system.histories))
        for k, h in enumerate(self.system.histories):
            a, m = h.a(t), h.mass(t)
            hill[k] = hill_radius(a, m)
            if t >= self.until[k]:
                continue                     # its history is over: it moves freely
            p = sim.particles[k + 1]
            rel_x = np.array(p.xyz) - np.array(sun.xyz)
            r = float(np.linalg.norm(rel_x))
            move = abs(r - a) > RADIUS_TOLERANCE * a
            if m != p.m or move:
                radius = a if move else r
                speed = np.sqrt(constants.G * (constants.CENTRAL_MASS + m) / radius)
                tangent = np.array([-rel_x[1], rel_x[0], 0.0]) / np.hypot(rel_x[0], rel_x[1])
                p.m = m
                if move:
                    p.xyz = tuple(np.array(sun.xyz) + a * rel_x / r)
                p.vxyz = tuple(np.array(sun.vxyz) + speed * tangent)
                changed = True
        if changed:
            sim.move_to_com()
            sim.ri_whfast.recalculate_coordinates_this_timestep = 1
        runner.m_jupiter = self.system.jupiter.mass(t)
        runner.hill = hill
