"""Jupiter's growth and migration history (stage 8) and the snapshot writer."""

import math
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from trojans import constants, disc, integrate
from trojans.history import HistoryForcing, JupiterHistory

M_J = constants.GIANT_MASS["jupiter"]


# --- the history as a function of time ----------------------------------------

@pytest.mark.parametrize("law", ["const_nu", "linear", "exponential"])
def test_a_endpoints_and_monotonic(law):
    h = JupiterHistory(a_start_au=8.2, a_end_au=5.2, t_mig_yr=1e5, t_mig_start_yr=1e4, migration_law=law)
    assert h.a(0.0) == 8.2 and h.a(1e4) == 8.2
    assert h.a(1.1e5) == 5.2 and h.a(1e7) == 5.2
    t = np.linspace(1e4, 1.1e5, 200)
    a = np.array([h.a(x) for x in t])
    assert np.all(np.diff(a) < 0)
    # adot is the derivative of a
    mid = 6e4
    numeric = (h.a(mid + 1.0) - h.a(mid - 1.0)) / 2.0
    assert h.adot(mid) == pytest.approx(numeric, rel=1e-6)
    assert h.adot(5e3) == 0.0 and h.adot(2e5) == 0.0
    assert h.direction == "inward" and h.a_min_au == 5.2


def test_outward_and_in_situ():
    out = JupiterHistory(a_start_au=4.4, a_end_au=5.2, t_mig_yr=1e5)
    assert out.direction == "outward" and out.adot(5e4) > 0 and out.a_min_au == 4.4
    still = JupiterHistory(a_start_au=5.2, a_end_au=5.2)
    assert still.direction == "in_situ" and still.a(123.0) == 5.2 and still.adot(1.0) == 0.0
    with pytest.raises(ValueError):
        JupiterHistory(a_start_au=8.0, a_end_au=5.2)          # migrating needs t_mig


def test_mass_laws():
    m0 = 10 * disc.M_EARTH
    h = JupiterHistory(a_start_au=5.2, a_end_au=5.2, m0_msun=m0, t_grow_start_yr=1e4, t_grow_yr=1e5)
    assert h.mass(0.0) == m0 and h.mass(1e4) == m0
    assert h.mass(1.1e5) == M_J
    assert h.mass(6e4) == pytest.approx(math.sqrt(m0 * M_J))          # halfway, exponential
    lin = JupiterHistory(a_start_au=5.2, a_end_au=5.2, m0_msun=m0, t_grow_yr=1e5, growth_law="linear")
    assert lin.mass(5e4) == pytest.approx(0.5 * (m0 + M_J))
    fixed = JupiterHistory(a_start_au=5.2, a_end_au=5.2)
    assert fixed.mass(0.0) == M_J
    # Same as the stage-6 Growth law
    g = disc.Growth(m0_msun=m0, t_grow_yr=1e5)
    h0 = JupiterHistory(a_start_au=5.2, a_end_au=5.2, m0_msun=m0, t_grow_yr=1e5)
    for t in (0.0, 3e4, 9.9e4, 2e5):
        assert h0.mass(t) == pytest.approx(g.mass(t), rel=1e-14)


def test_from_nu_gives_nu_at_core_mass():
    m0 = 10 * disc.M_EARTH
    h = JupiterHistory.from_nu(0.5, 8.2, 5.2, m0_msun=m0, t_grow_start_yr=1e9)   # stays a core
    for t in (0.0, 0.3 * h.t_mig_yr, 0.9 * h.t_mig_yr):
        assert h.nu(t) == pytest.approx(0.5, rel=1e-12)
    assert 7e4 < h.t_mig_yr < 9e4          # the plan's nominal: ~80 kyr
    # growing during the migration lowers nu at the same da/dt
    g = JupiterHistory.from_nu(0.5, 8.2, 5.2, m0_msun=m0, t_grow_yr=1e5)
    assert g.nu(0.5 * g.t_mig_yr) < 0.5


# --- the forcing --------------------------------------------------------------

def _sim(history, n_pairs=20, a_min=4.9, a_max=5.5):
    d = disc.Disc(n_pairs=n_pairs, a_min=a_min, a_max=a_max, seed=7)
    pos, vel = disc.disc_states(d)
    return disc.build_simulation(pos, vel, disc.particle_ids(n_pairs), history, history.a_start_au,
                                 constants.timestep(history.a_min_au))


def _stub():
    return SimpleNamespace(m_jupiter=None, hill=None)


def _states(sim):
    return np.array([[p.x, p.y, p.z, p.vx, p.vy, p.vz] for p in sim.particles])


def test_forcing_puts_jupiter_on_the_history_and_is_idempotent():
    h = JupiterHistory.from_nu(0.5, 8.2, 5.2, t_grow_yr=5e4)
    sim = _sim(h)
    forcing, runner = HistoryForcing(h), _stub()
    forcing(sim, runner)
    for _ in range(300):
        sim.steps(8)
        forcing(sim, runner)
        orbit = sim.particles[1].orbit(primary=sim.particles[0])
        assert orbit.a == pytest.approx(h.a(sim.t), rel=1e-9)
        assert orbit.e < 1e-9 and abs(orbit.inc) < 1e-12
        assert sim.particles[1].m == h.mass(sim.t) == runner.m_jupiter
        assert runner.hill[0] == disc.hill_radius(h.a(sim.t), h.mass(sim.t))
    before = _states(sim)
    forcing(sim, runner)
    assert np.array_equal(before, _states(sim))


def test_in_situ_history_reproduces_growth_forcing_exactly():
    m0 = 10 * disc.M_EARTH
    growth = disc.Growth(m0_msun=m0, t_grow_yr=500.0)
    history = JupiterHistory(a_start_au=5.2026, a_end_au=5.2026, m0_msun=m0, t_grow_yr=500.0)
    pos, vel = disc.disc_states(disc.Disc(n_pairs=20, seed=3))
    ids = disc.particle_ids(20)
    dt = constants.timestep(5.2026)
    runs = []
    for forcing in (disc.GrowthForcing(growth, 5.2026), HistoryForcing(history)):
        sim = disc.build_simulation(pos, vel, ids, growth, 5.2026, dt)
        runner = _stub()
        forcing(sim, runner)
        for _ in range(100):
            sim.steps(8)
            forcing(sim, runner)
        runs.append((_states(sim), runner.hill[0], runner.m_jupiter))
    # Growth uses m0 * ratio**u, History the same; any difference would show here.
    assert np.array_equal(runs[0][0], runs[1][0])
    assert runs[0][1:] == runs[1][1:]


# --- the runner with migration and snapshots ---------------------------------

def _runner(tmp, h, t_end, resume=False):
    settings = integrate.Settings(t_end_yr=t_end, dt_yr=constants.timestep(h.a_min_au), check_every_steps=8,
                                  window_yr=200.0, checkpoint_every_windows=2,
                                  snapshot_every_yr=200.0, snapshot_fine_every_yr=20.0,
                                  snapshot_fine_until_yr=500.0)
    ids = disc.particle_ids(10)
    giants = {"jupiter": {"a": h.a_start_au, "hill_radius": disc.hill_radius(h.a_start_au, h.mass(0.0))}}
    forcing = HistoryForcing(h)
    if resume:
        return integrate.ChunkRunner.resume(ids, settings, giants, tmp, ("jupiter",), forcing)
    pos, vel = disc.disc_states(disc.Disc(n_pairs=10, a_min=5.0, a_max=6.5, seed=11))
    sim = disc.build_simulation(pos, vel, ids, h, h.a_start_au, settings.dt_yr)
    return integrate.ChunkRunner.start(sim, ids, settings, giants, tmp, ("jupiter",), forcing)


def test_restart_with_migration_and_snapshots_is_bitwise_identical(tmp_path):
    h = JupiterHistory(a_start_au=6.0, a_end_au=5.2, t_mig_yr=800.0, t_grow_yr=600.0)
    straight = _runner(tmp_path / "straight", h, 1600.0)
    straight.run(log=lambda *_: None)

    first = _runner(tmp_path / "split", h, 800.0)
    first.run(log=lambda *_: None)
    resumed = _runner(tmp_path / "split", h, 1600.0, resume=True)
    resumed.run(log=lambda *_: None)

    assert np.array_equal(straight._states()[0], resumed._states()[0])
    files = {}
    for name in ("straight", "split"):
        with h5py.File(tmp_path / name / "snapshots.h5") as f:
            files[name] = {k: f[k][:] for k in [*integrate.SNAPSHOT_SCALARS, *integrate.SNAPSHOT_FIELDS]}
    for k, v in files["straight"].items():
        assert np.array_equal(v, files["split"][k], equal_nan=True), k

    snaps = files["straight"]
    t = snaps["t"]
    # cadence: 20 yr while fine (t <= 500), one window (200 yr) after, ending on the last check
    assert t[0] == 0.0 and t[-1] == pytest.approx(straight.sim.t)
    fine, coarse = np.diff(t[t <= 500.0]), np.diff(t[t > 500.0])
    assert np.allclose(fine, fine[0], rtol=0.05) and 18 < fine[0] < 22
    assert np.allclose(coarse, 199.1, rtol=0.01)
    # Jupiter in the snapshots follows the history
    expected = np.array([h.a(x) for x in t])
    assert np.allclose(snaps["jupiter_a"], expected, rtol=1e-9)
    assert np.allclose(snaps["jupiter_m"], [h.mass(x) for x in t], rtol=1e-15)
    with h5py.File(tmp_path / "straight" / "windows.h5") as f:
        assert np.allclose(f["jupiter_a"][:], [h.a(x) for x in f["t_end"][:]], rtol=1e-9)


# --- tabulated tracks ---------------------------------------------------------

def _table_from(h, n=4001):
    t = np.linspace(0.0, h.t_forcing_end_yr, n)
    return JupiterHistory.from_table(t, [h.a(x) for x in t], [h.mass(x) for x in t])


def test_table_reproduces_an_analytic_history():
    h = JupiterHistory.from_nu(0.5, 8.2, 5.2, t_grow_yr=8e4, m0_msun=0.5 * disc.M_EARTH)
    tab = _table_from(h)
    assert tab.migration_law == "table" and tab.direction == "inward" and tab.a_min_au == 5.2
    assert tab.t_forcing_end_yr == h.t_forcing_end_yr
    for t in np.linspace(0.0, 1.2 * h.t_forcing_end_yr, 37):
        assert tab.a(t) == pytest.approx(h.a(t), rel=1e-6)
        assert tab.mass(t) == pytest.approx(h.mass(t), rel=1e-6)
    for t in np.linspace(0.05, 0.95, 7) * h.t_mig_yr:
        assert tab.adot(t) == pytest.approx(h.adot(t), rel=1e-3)
        assert tab.nu(t) == pytest.approx(h.nu(t), rel=1e-3)
    assert tab.adot(2 * h.t_forcing_end_yr) == 0.0
    assert tab.mass(2 * h.t_forcing_end_yr) == tab.m_final_msun


def test_table_round_trips_through_json_and_validates():
    import json
    from dataclasses import asdict
    tab = _table_from(JupiterHistory(a_start_au=6.0, a_end_au=5.2, t_mig_yr=800.0, t_grow_yr=600.0), n=50)
    back = JupiterHistory(**json.loads(json.dumps(asdict(tab))))
    assert back == tab
    with pytest.raises(ValueError):
        JupiterHistory(a_start_au=6.0, a_end_au=5.0, t_mig_yr=1.0, migration_law="table",
                       table_t_yr=(0.0, 1.0), table_a_au=(6.0, 5.2), table_m_msun=(1e-5, 1e-3))


def test_restart_with_a_table_is_bitwise_identical(tmp_path):
    h = _table_from(JupiterHistory(a_start_au=6.0, a_end_au=5.2, t_mig_yr=800.0, t_grow_yr=600.0), n=300)
    straight = _runner(tmp_path / "straight", h, 1600.0)
    straight.run(log=lambda *_: None)
    first = _runner(tmp_path / "split", h, 800.0)
    first.run(log=lambda *_: None)
    resumed = _runner(tmp_path / "split", h, 1600.0, resume=True)
    resumed.run(log=lambda *_: None)
    assert np.array_equal(straight._states()[0], resumed._states()[0])
