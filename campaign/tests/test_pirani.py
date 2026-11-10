"""Stage 9 (Pirani+2019a reproduction): gas disc, giant forces, particle drag, resume."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import rebound

from trojans import constants, disc, gasdisc, integrate
from trojans.history import JupiterHistory, SystemHistory
from trojans.pirani_forces import DragParams, PiraniForcing, adachi_rates, damping_times

ME = disc.M_EARTH


# --- gas disc -------------------------------------------------------------------

def test_mdot_and_gas_end():
    assert gasdisc.mdot(1e6 - 1e5) == pytest.approx(1e-8)
    assert gasdisc.mdot(3e6 - 1.0) == pytest.approx(2.05e-9, rel=0.01)       # "Mdot <= 2e-9" at 3 Myr
    assert gasdisc.mdot(3e6) == 0.0 and not gasdisc.gas_present(3e6)


@pytest.mark.parametrize("r, regime, expected", [(1.0, "vis", 2.1e3), (18.0, "irr", 2.7e3 * 18 ** (-15 / 14))])
def test_sigma_matches_ida2016(r, regime, expected):
    """Ida+2016 eqs. 12-13 at Mdot = 1e-8 Msun/yr, alpha = 1e-3 (their fits round to 2 digits)."""
    t = 1e6 - 1e5
    h_vis, h_irr = gasdisc.aspect_ratios(r, t)
    assert (h_vis > h_irr) == (regime == "vis")
    assert gasdisc.sigma(r, t) / gasdisc.G_CM2_TO_MSUN_AU2 == pytest.approx(expected, rel=0.05)
    assert gasdisc.eta(r, t) > 0


# --- ring disc ------------------------------------------------------------------

def test_ring_disc_layout():
    d = disc.RingDisc(n_per_ring=50, seed=3)
    pos, vel, a0 = disc.ring_disc_states(d)
    assert d.n_rings == 44 and len(a0) == 44 * 50 == len(pos)
    counts = np.histogram(a0, bins=np.arange(4.0, 26.01, 0.5))[0]
    assert np.all(counts == 50)
    el = disc.diagnostics.elements(pos, vel, constants.G * constants.CENTRAL_MASS)
    assert np.allclose(el["a"], a0) and el["e"].max() <= 0.01 and el["inc"].max() <= 0.01


# --- giants ---------------------------------------------------------------------

def _one_planet(h: JupiterHistory, e0=0.0, inc0=0.0, dt=50 / 365.25):
    system = SystemHistory(planets=("jupiter",), histories=(h,))
    sim = rebound.Simulation()
    sim.G = constants.G
    sim.integrator = "mercurius"
    sim.dt = dt
    sim.add(m=constants.CENTRAL_MASS, hash=disc.SUN_HASH)
    sim.add(m=h.mass(0.0), a=h.a(0.0), e=e0, inc=inc0, primary=sim.particles[0], hash=disc.JUPITER_HASH)
    sim.N_active = 2
    sim.move_to_com()
    forcing = PiraniForcing(system, DragParams(enabled=False))
    runner = SimpleNamespace(s=SimpleNamespace(check_interval_yr=8 * dt))
    forcing(sim, runner)
    return sim, forcing, runner


def _advance(sim, forcing, runner, t_end):
    while sim.t < t_end:
        sim.steps(8)
        forcing(sim, runner)


def test_migration_force_follows_the_track():
    m = 50 * ME                                    # above 30 M_E: migration only, no damping
    h = JupiterHistory(a_start_au=10.0, a_end_au=9.0, t_mig_yr=2000.0, migration_law="linear",
                       m0_msun=m, m_final_msun=m)
    sim, forcing, runner = _one_planet(h)
    _advance(sim, forcing, runner, 1000.0)
    orb = sim.particles[1].orbit(primary=sim.particles[0])
    assert orb.a == pytest.approx(h.a(sim.t), rel=2e-3)
    assert orb.e < 2e-3


def test_damping_rates_below_30_earth_masses():
    m = 10 * ME
    h = JupiterHistory(a_start_au=10.0, a_end_au=10.0, m0_msun=m, m_final_msun=m)
    e0, i0 = 0.02, 0.02
    sim, forcing, runner = _one_planet(h, e0=e0, inc0=i0)
    t_e, t_i = damping_times(m, 10.0, e0, i0, 0.0)
    assert 10.0 < t_e < 1e5
    _advance(sim, forcing, runner, t_e)
    orb = sim.particles[1].orbit(primary=sim.particles[0])
    assert orb.e / e0 == pytest.approx(math.exp(-1.0), rel=0.2)
    assert orb.inc / i0 == pytest.approx(math.exp(-sim.t / t_i), rel=0.2)


def test_no_damping_above_30_earth_masses():
    m = 40 * ME
    h = JupiterHistory(a_start_au=10.0, a_end_au=10.0, m0_msun=m, m_final_msun=m)
    sim, forcing, runner = _one_planet(h, e0=0.02, inc0=0.02)
    _advance(sim, forcing, runner, 300.0)
    orb = sim.particles[1].orbit(primary=sim.particles[0])
    assert orb.e == pytest.approx(0.02, rel=0.02)


# --- particle drag -----------------------------------------------------------------

def test_drag_operator_applies_adachi_rates():
    h = JupiterHistory(a_start_au=5.0, a_end_au=5.0, m0_msun=0.01 * ME, m_final_msun=0.01 * ME)
    sim, forcing, runner = _one_planet(h)
    forcing.drag = DragParams()
    a, e, inc = 18.0, 0.3, 0.1
    sim.add(m=0.0, a=a, e=e, inc=inc, primary=sim.particles[0])
    sim.t = 1.5e6
    forcing._apply_drag(sim, sim.t)
    orb = sim.particles[2].orbit(primary=sim.particles[0])
    ra, re_, ri = adachi_rates(np.array([a]), np.array([e]), np.array([inc]), 1.5e6, forcing.drag)
    dt = forcing.drag.every_yr
    assert ra[0] < 0 and re_[0] < 0 and ri[0] < 0
    assert orb.a == pytest.approx(a * math.exp(ra[0] * dt), rel=1e-9)
    assert orb.e == pytest.approx(e * math.exp(re_[0] * dt), rel=1e-9)
    assert orb.inc == pytest.approx(inc * math.exp(ri[0] * dt), rel=1e-9)


# --- the runner, with everything on ----------------------------------------------------

def test_restart_is_bitwise_with_pirani_forces(tmp_path):
    jup = JupiterHistory(a_start_au=8.0, a_end_au=7.6, t_mig_yr=1200.0, t_mig_start_yr=200.0,
                         m0_msun=1 * ME, m_final_msun=40 * ME, t_grow_start_yr=200.0, t_grow_yr=1200.0)
    sat = JupiterHistory(a_start_au=11.0, a_end_au=11.0, m0_msun=5 * ME, m_final_msun=5 * ME)
    system = SystemHistory(planets=("jupiter", "saturn"), histories=(jup, sat))
    d = disc.RingDisc(a_min=6.0, a_max=9.0, n_per_ring=3, seed=1)
    pos, vel, _ = disc.ring_disc_states(d)
    ids = disc.FIRST_PARTICLE_HASH + np.arange(d.n)

    def runner_for(path, t_end, resume=False):
        settings = integrate.Settings(t_end_yr=t_end, dt_yr=50 / 365.25, check_every_steps=8, window_yr=200.0,
                                      checkpoint_every_windows=2, remove_hill_radii=0.0, integrator="mercurius",
                                      sun_radius_au=0.005, snapshot_every_yr=200.0)
        giants = {p: {"a": h.a(0.0), "hill_radius": disc.hill_radius(h.a(0.0), h.mass(0.0))}
                  for p, h in zip(system.planets, system.histories)}
        forcing = PiraniForcing(system, DragParams(every_yr=50.0))
        if resume:
            return integrate.ChunkRunner.resume(ids, settings, giants, path, system.planets, forcing)
        sim = disc.build_system(pos, vel, ids, system.planets, [8.0, 11.0], [1 * ME, 5 * ME], settings.dt_yr,
                                integrator="mercurius")
        return integrate.ChunkRunner.start(sim, ids, settings, giants, path, system.planets, forcing)

    straight = runner_for(tmp_path / "straight", 1600.0)
    straight.run(log=lambda *_: None)
    runner_for(tmp_path / "split", 800.0).run(log=lambda *_: None)
    resumed = runner_for(tmp_path / "split", 1600.0, resume=True)
    resumed.run(log=lambda *_: None)
    assert np.array_equal(straight._states()[0], resumed._states()[0])


# --- growth tracks ------------------------------------------------------------------

def test_tracks_reproduce_her_final_masses():
    """Along her Fig. 1 paths, the calibrated growth model ends J and S at her masses (+-3%, +-5%)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import s9_pirani_tracks as s9
    if not (s9.DIG / "fig1_jupiter.csv").exists():
        pytest.skip("run s9_digitize_pirani.py first")
    assert s9.clock("jupiter", 2.31e6)["m"][-1] == pytest.approx(306.0, rel=0.03)
    assert s9.clock("saturn", 2.56e6)["m"][-1] == pytest.approx(113.0, rel=0.05)


def test_rate_tracks_are_scaled_nominal():
    from pathlib import Path
    d = Path(__file__).resolve().parents[1] / "configs" / "tracks" / "pirani19a"
    if not (d / "track_rate_x0.25_jupiter.csv").exists():
        pytest.skip("run s9_pirani_tracks.py first")
    t1, a1, m1 = np.loadtxt(d / "track_nominal_jupiter.csv", delimiter=",", skiprows=1).T
    tk, ak, mk = np.loadtxt(d / "track_rate_x0.25_jupiter.csv", delimiter=",", skiprows=1).T
    assert np.array_equal(t1, tk) and np.allclose(m1, mk)
    assert np.allclose(ak - ak[-1], 0.25 * (a1 - a1[-1]), atol=1e-6)


def test_alpha_is_pirani_value_everywhere():
    """Pirani sect. 3.1: Ida+2016 disc with alpha = 1e-3 (BLJ15 had 0.0054)."""
    from trojans import pirani_tracks as pt
    assert gasdisc.ALPHA == 1e-3 and pt.TrackParams().alpha == 1e-3
    with pytest.raises(ValueError):
        pt.integrate(pt.TrackParams(alpha=5.4e-3, t_end_yr=2.32e6))
