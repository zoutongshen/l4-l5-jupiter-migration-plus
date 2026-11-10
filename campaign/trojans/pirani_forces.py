"""Stage 9: the forces of Pirani et al. 2019a, as a ``ChunkRunner`` forcing.

Giants (particles 1..n after the Sun), unlike stage 8 they are never put on a circle:
  - migration: a tangential acceleration a_m = v * adot(t) / (2 a(t)), with adot(t) and
    a(t) from each giant's growth track. This imposes da/dt (Pirani sect. 3.1: the giants
    "migrate following the growth tracks"); planet-planet forces stay free to move a, e
    and i, which is how her Jupiter ends at ~5.0 au and in the 1:2 with Saturn (sect. 4);
  - e and i damping ("tidal gas drag", Pirani App. B): the Cresswell & Nelson 2008
    accelerations (their eqs. 14-16) a_e = -2 (v.r) r / (r^2 t_e), a_i = -2 v_z / t_i z,
    with t_e, t_i from Pirani eqs. B.1-B.2 (CN08 eqs. 11-12) and t_wave from B.3
    (Tanaka & Ward 2004), using the Ida+2016 disc of ``gasdisc``. Only while the gas is
    present and only for a giant below 30 M_E. CN08's eq. 16 has a_i = -v_z / t_i, which
    damps i at 1/(2 t_i) (measured, tests/test_pirani.py); Pirani's B.2 and Tanaka & Ward
    2004 state di/dt = -i/t_i, so the factor 2 is used (as REBOUNDx's modify_orbits);
  - mass m(t) from the track, set at every runner check (positions/velocities untouched).
The accelerations are a compiled REBOUND ``additional_forces`` function over the giants only
(csrc/pirani_forces.c; O(n_giants) per step), with its coefficients (adot/2a, 1/t_e, 2/t_i)
set from Python at every runner check. A Python callback cost ~1 ms per step (measured on
haagsebeek: a fixed ~400 s per chunk per 50 kyr), which made extra workers useless.

Disc particles: aerodynamic gas drag (Pirani App. A), the orbit-averaged rates of
Adachi et al. 1976 (her eqs. A.5-A.7), r_p = 50 km, rho_p = 1 g/cm^3, C_D = 0.44, gas
at the midplane density of the particle's a. Applied as an element update every
``drag_every_yr`` (at checks where t crosses a multiple of it, over exactly that
interval), so it is a deterministic function of the state and resumes are exact.
The drag times are >= 1e5 yr, so a 100 yr cadence is accurate to < 1e-3.

Removal on impact (planet radii from a bulk density) is done by the runner's checks;
REBOUND's own collision search is O(N^2) with test particles (100x slower, measured).
"""

import ctypes
import hashlib
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rebound

from trojans import constants, diagnostics, gasdisc
from trojans.disc import M_EARTH, hill_radius
from trojans.history import SystemHistory

M_DAMP_MAX = 30.0 * M_EARTH              # Pirani App. B: tidal drag only below 30 M_E
_CSRC = Path(__file__).with_name("csrc") / "pirani_forces.c"
_LIB = None


def _forces_lib():
    """Compile (once per source version, atomically, so parallel workers are safe) and load the C force."""
    global _LIB
    if _LIB is not None:
        return _LIB
    src = _CSRC.read_bytes()
    tag = hashlib.sha1(src + rebound.__version__.encode()).hexdigest()[:12]
    build = Path(os.environ.get("TROJANS_BUILD", Path.home() / ".cache" / "trojans"))
    build.mkdir(parents=True, exist_ok=True)
    so = build / f"pirani_forces_{tag}.so"
    if not so.exists():
        tmp = build / f".{so.name}.{os.getpid()}"
        cc = os.environ.get("CC", "cc")
        subprocess.run([cc, "-O3", "-fPIC", "-shared", "-I", str(Path(rebound.__file__).parent), str(_CSRC),
                        "-o", str(tmp)] + (["-undefined", "dynamic_lookup"] if sys.platform == "darwin" else []),
                       check=True)
        os.replace(tmp, so)
    lib = ctypes.CDLL(str(so))
    ns = type("PiraniForcesLib", (), {})()
    ns.cdll = lib
    ns.pf_mig = (ctypes.c_double * 8).in_dll(lib, "pf_mig")
    ns.pf_e = (ctypes.c_double * 8).in_dll(lib, "pf_e")
    ns.pf_i = (ctypes.c_double * 8).in_dll(lib, "pf_i")
    ns.pf_n = ctypes.c_int.in_dll(lib, "pf_n")
    ns.fn = ctypes.cast(lib.pirani_forces, ctypes.c_void_p).value
    _LIB = ns
    return ns
G_CM3 = gasdisc.G_CM3_TO_MSUN_AU3
AU_CM = constants.AU_KM * 1e5


@dataclass(frozen=True)
class DragParams:
    r_p_au: float = 50.0e5 / AU_CM        # 50 km
    rho_p: float = 1.0 * G_CM3            # 1 g/cm^3 in Msun/au^3
    c_d: float = 0.44
    every_yr: float = 100.0
    enabled: bool = True


def t_wave(m, a, sigma, h, mstar=1.0):
    """Tanaka & Ward 2004 eq. 49 (Pirani B.3), in yr."""
    om = math.sqrt(constants.G * mstar / a**3)
    return (mstar / m) * (mstar / (sigma * a * a)) * h**4 / om


def damping_times(m, a, e, inc_rad, t):
    """(t_e, t_i) of Pirani B.1-B.2 (CN08 eqs. 11-12), in yr."""
    h = float(gasdisc.aspect_ratio(a, t))
    tw = t_wave(m, a, float(gasdisc.sigma(a, t)), h)
    x, y = e / h, inc_rad / h
    t_e = tw / 0.780 * (1.0 - 0.14 * x**2 + 0.06 * x**3 + 0.18 * x * y**2)
    t_i = tw / 0.544 * (1.0 - 0.30 * y**2 + 0.24 * y**3 + 0.14 * x**2 * y)
    return t_e, t_i


def adachi_rates(a, e, inc_rad, t, p: DragParams):
    """(adot/a, edot/e, idot/i) per yr, Pirani eqs. A.3, A.5-A.7, for arrays of orbits."""
    rho_g = gasdisc.rho_mid(a, t)
    v_k = np.sqrt(constants.G * constants.CENTRAL_MASS / a)
    with np.errstate(divide="ignore"):
        tau0 = 8.0 * p.rho_p * p.r_p_au / (3.0 * p.c_d * rho_g * v_k)
    eta = gasdisc.eta(a, t)
    s = np.sqrt(eta**2 + 0.625 * e**2 + 0.5 * inc_rad**2)
    return (-2.0 / tau0 * s * (eta + e**2 + 0.125 * inc_rad**2), -s / tau0, -0.5 * s / tau0)


def planet_radius_au(m_msun, density_g_cm3=1.33):
    return (3.0 * m_msun / (4.0 * math.pi * density_g_cm3 * G_CM3)) ** (1.0 / 3.0)


class PiraniForcing:
    """``ChunkRunner`` forcing for stage 9. ``system``: one track per giant, Jupiter first."""

    def __init__(self, system: SystemHistory, drag: DragParams = DragParams(), planet_density_g_cm3: float = 1.33,
                 integrator: str = "mercurius"):
        self.system = system
        self.drag = drag
        self.density = planet_density_g_cm3
        self.integrator = integrator
        self._attached = None
        self._callback = None

    # --- giants: accelerations every step (compiled; coefficients set at each check) ----

    def _set_coefficients(self, sim, t) -> None:
        """adot/(2a), 1/t_e and 2/t_i of each giant for the C force (csrc/pirani_forces.c).

        Updated at every runner check (~1 yr); the damping times are >= 10 yr and adot(t) is
        piecewise linear in a table with ~230 yr rows, so this matches a per-step evaluation.
        """
        lib = _forces_lib()
        gas = gasdisc.gas_present(t)
        sun = sim.particles[0]
        for k, hist in enumerate(self.system.histories):
            p = sim.particles[k + 1]
            mig = e_rate = i_rate = 0.0
            if gas:
                adot = hist.adot(t)
                if adot != 0.0:
                    mig = adot / (2.0 * hist.a(t))
                if p.m < M_DAMP_MAX:
                    orb = p.orbit(primary=sun)
                    t_e, t_i = damping_times(p.m, orb.a, orb.e, orb.inc, t)
                    e_rate, i_rate = 1.0 / t_e, 2.0 / t_i        # 2x CN08 eq. 16, so that di/dt = -i/t_i
            lib.pf_mig[k], lib.pf_e[k], lib.pf_i[k] = mig, e_rate, i_rate

    def _attach(self, sim: rebound.Simulation) -> None:
        if self._attached is sim:
            return
        lib = _forces_lib()
        lib.pf_n.value = len(self.system.histories)
        sim.additional_forces = lib.fn
        sim.force_is_velocity_dependent = 1
        self._attached = sim

    # --- the runner hook ---------------------------------------------------------

    def __call__(self, sim: rebound.Simulation, runner) -> None:
        first = self._attached is not sim           # start, or resume: drag at this t is already done
        self._attach(sim)
        t = sim.t
        changed = False
        hill = np.empty(len(self.system.histories))
        radius = np.empty(len(self.system.histories))
        for k, h in enumerate(self.system.histories):
            m = h.mass(t)
            p = sim.particles[k + 1]
            if m != p.m:
                p.m = m
                changed = True
            hill[k] = hill_radius(h.a(t), m)
            radius[k] = planet_radius_au(m, self.density)
        if self.drag.enabled and gasdisc.gas_present(t) and not first and self._drag_due(sim, runner):
            self._apply_drag(sim, t)
            changed = True
        if changed:
            if self.integrator == "mercurius":
                sim.ri_mercurius.recalculate_r_crit_this_timestep = 1
                sim.ri_mercurius.recalculate_coordinates_this_timestep = 1
            else:
                sim.ri_whfast.recalculate_coordinates_this_timestep = 1
        self._set_coefficients(sim, t)
        runner.m_jupiter = self.system.jupiter.mass(t)
        runner.hill = hill
        runner.planet_radius_au = radius

    def _drag_due(self, sim, runner) -> bool:
        every = self.drag.every_yr
        dt_check = runner.s.check_interval_yr
        return math.floor(sim.t / every) != math.floor((sim.t - dt_check) / every) and sim.t > 0.0

    def _apply_drag(self, sim, t) -> None:
        n_act = 1 + len(self.system.histories)
        n = sim.N
        if n <= n_act:
            return
        xyz = np.empty((n, 3))
        vxyz = np.empty((n, 3))
        sim.serialize_particle_data(xyz=xyz, vxvyvz=vxyz)
        pos = xyz[n_act:] - xyz[0]
        vel = vxyz[n_act:] - vxyz[0]
        mu = constants.G * constants.CENTRAL_MASS
        el = diagnostics.elements(pos, vel, mu)
        ok = (el["e"] < 1.0) & (el["a"] > 0.0) & np.isfinite(el["M"])
        if not ok.any():
            return
        a, e, inc = el["a"][ok], el["e"][ok], np.radians(el["inc"][ok])
        ra, re_, ri = adachi_rates(a, e, inc, t, self.drag)
        dt = self.drag.every_yr
        a_new, e_new, i_new = a * np.exp(ra * dt), e * np.exp(re_ * dt), inc * np.exp(ri * dt)
        p_new, v_new = diagnostics.cartesian(a_new, e_new, np.degrees(i_new), el["Omega"][ok], el["omega"][ok],
                                             el["M"][ok], mu)
        idx = np.flatnonzero(ok) + n_act
        xyz[idx] = xyz[0] + p_new
        vxyz[idx] = vxyz[0] + v_new
        sim.set_serialized_particle_data(xyz=xyz, vxvyvz=vxyz)
