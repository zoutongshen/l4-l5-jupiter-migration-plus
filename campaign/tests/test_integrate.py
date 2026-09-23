"""Chunk runner: classification, encounters, removal, restart, energy."""

import math

import h5py
import numpy as np
import pandas as pd
import pytest
import rebound

from trojans import constants, initial, integrate

# Rough giant orbits (a au, e, inc deg, Omega, omega, M deg); not an ephemeris.
GIANT_ELEMENTS = {
    "jupiter": (5.2026, 0.0489, 1.30, 100.5, 273.9, 20.0),
    "saturn": (9.5549, 0.0565, 2.49, 113.7, 339.4, 317.0),
    "uranus": (19.2184, 0.0463, 0.77, 74.0, 96.9, 142.0),
    "neptune": (30.1104, 0.0095, 1.77, 131.8, 273.2, 260.0),
}


def _heliocentric(a, e, inc, Omega, omega, M, m):
    sim = rebound.Simulation()
    sim.G = constants.G
    sim.add(m=constants.CENTRAL_MASS)
    sim.add(primary=sim.particles[0], m=m, a=a, e=e, inc=math.radians(inc),
            Omega=math.radians(Omega), omega=math.radians(omega), M=math.radians(M))
    p = sim.particles[1]
    return np.array([p.x, p.y, p.z]), np.array([p.vx, p.vy, p.vz])


def _system(trojans):
    """States table for the giants plus named test particles {spkid: (pos, vel)}."""
    giants = {p: _heliocentric(*GIANT_ELEMENTS[p], m=constants.GIANT_MASS[p]) for p in constants.GIANTS}
    spkids = list(trojans)
    pos = np.array([trojans[s][0] for s in spkids]) if spkids else np.empty((0, 3))
    vel = np.array([trojans[s][1] for s in spkids]) if spkids else np.empty((0, 3))
    return initial.states_table(giants, spkids, pos, vel), giants


def _lagrange(offset_deg):
    a, e, inc, Om, om, M = GIANT_ELEMENTS["jupiter"]
    return _heliocentric(a, e, inc, Om, om, M + offset_deg, m=0.0)


def _runner(states, spkids, tmp_path, **overrides):
    setup = initial.planet_setup(states)
    kwargs = dict(t_end_yr=2000.0, dt_yr=setup["dt_yr"], check_every_steps=4, window_yr=500.0,
                  checkpoint_every_windows=1)
    kwargs.update(overrides)
    settings = integrate.Settings(**kwargs)
    sim = initial.build_simulation(states, spkids, settings.dt_yr)
    return integrate.ChunkRunner.start(sim, spkids, settings, setup["giants"], tmp_path), setup


# --- pure functions --------------------------------------------------------

def test_sigma_crossings():
    prev = np.array([170.0, -5.0, 60.0, np.nan])
    now = np.array([-175.0, 4.0, 65.0, 10.0])
    c180, c0 = integrate.sigma_crossings(prev, now)
    assert c180.tolist() == [True, False, False, False]
    assert c0.tolist() == [False, True, False, False]


def test_classify_windows():
    n = np.array([10, 10, 10, 10, 10, 0])
    crossed0 = np.array([False, False, False, True, True, False])
    crossed180 = np.array([False, False, True, False, True, False])
    cmean = np.array([60.0, -60.0, 180.0, 0.0, 30.0, 0.0])
    cls = integrate.classify_windows(n, crossed0, crossed180, cmean)
    assert [integrate.CLASS_NAMES[c] for c in cls] == [
        "tadpole_L4", "tadpole_L5", "horseshoe", "quasi_satellite", "circulating", "no_samples"]


# --- integrations ------------------------------------------------------------

def test_lagrange_particles_stay_tadpoles(tmp_path):
    states, _ = _system({20000001: _lagrange(60.0), 20000002: _lagrange(-60.0)})
    runner, _ = _runner(states, [20000001, 20000002], tmp_path)
    runner.run(log=lambda *_: None)
    with h5py.File(tmp_path / "windows.h5") as f:
        cls = f["class"][:]
        cmean = f["sigma_cmean"][:]
        C = f["C"][:]
    assert cls.shape == (4, 2)
    assert np.all(cls[:, 0] == integrate.TADPOLE_L4)
    assert np.all(cls[:, 1] == integrate.TADPOLE_L5)
    assert np.all(np.abs(cmean[:, 0] - 60) < 30) and np.all(np.abs(cmean[:, 1] + 60) < 30)
    assert np.all((C > 1.45) & (C < 1.7))
    events = pd.read_parquet(tmp_path / "events.parquet")
    assert (events["removal_reason"] == "alive").all()
    assert events["t_sigma_cross180"].isna().all()


def test_encounter_is_logged_and_removed(tmp_path):
    jx, jv = _heliocentric(*GIANT_ELEMENTS["jupiter"], m=constants.GIANT_MASS["jupiter"])
    rh = constants.hill_radius(GIANT_ELEMENTS["jupiter"][0], "jupiter")
    radial = jx / np.linalg.norm(jx)
    start = jx + 3.0 * rh * radial
    incoming = jv - 1.0 * radial                    # ~4.7 km/s towards Jupiter; passes within 0.13 R_H
    far = (np.array([150.0, 0.0, 0.0]), np.array([0.0, 0.5, 0.0]))
    states, _ = _system({20000010: (start, incoming), 20000011: far, 20000012: _lagrange(60.0)})
    runner, _ = _runner(states, [20000010, 20000011, 20000012], tmp_path,
                        t_end_yr=100.0, window_yr=50.0, check_every_steps=1)
    n_active = runner.sim.N_active
    runner.run(log=lambda *_: None)

    events = pd.read_parquet(tmp_path / "events.parquet").set_index("spkid")
    hit = events.loc[20000010]
    assert hit["removal_reason"] == "hill"
    assert hit["planet_hill_record"] == "jupiter" and hit["planet_hill_remove"] == "jupiter"
    assert 0 < hit["t_hill_record"] < hit["t_hill_remove"] < 2.0
    assert events.loc[20000011, "removal_reason"] == "r_max"
    assert events.loc[20000011, "t_removed"] == 0.0
    assert events.loc[20000012, "removal_reason"] == "alive"
    assert runner.sim.N == n_active + 1
    assert runner.sim.N_active == n_active


def test_restart_is_bitwise_identical(tmp_path):
    trojans = {20000000 + k: _lagrange(60.0 + 8 * k) for k in range(6)}
    spkids = list(trojans)
    states, _ = _system(trojans)

    straight, _ = _runner(states, spkids, tmp_path / "straight", t_end_yr=2000.0)
    straight.run(log=lambda *_: None)

    first, setup = _runner(states, spkids, tmp_path / "split", t_end_yr=1000.0)
    first.run(log=lambda *_: None)
    settings = integrate.Settings(t_end_yr=2000.0, dt_yr=setup["dt_yr"], check_every_steps=4,
                                  window_yr=500.0, checkpoint_every_windows=1)
    resumed = integrate.ChunkRunner.resume(spkids, settings, setup["giants"], tmp_path / "split")
    assert resumed.window_index == 2
    resumed.run(log=lambda *_: None)

    a_xyz, _, a_hash = straight._states()
    b_xyz, _, b_hash = resumed._states()
    assert straight.sim.t == resumed.sim.t
    assert np.array_equal(a_hash, b_hash)
    assert np.array_equal(a_xyz, b_xyz)
    with h5py.File(tmp_path / "straight" / "windows.h5") as fa, h5py.File(tmp_path / "split" / "windows.h5") as fb:
        for name in ["t_end", "class", "n_samples", *integrate.WINDOW_FIELDS]:
            assert np.array_equal(fa[name][:], fb[name][:], equal_nan=name not in ("class", "n_samples")), name
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "straight" / "events.parquet"),
                                  pd.read_parquet(tmp_path / "split" / "events.parquet"))
    assert sorted(p.name for p in (tmp_path / "split").glob("checkpoint_*.bin")) == ["checkpoint_w000004.bin"]


def test_giants_energy_is_conserved():
    states, _ = _system({})
    setup = initial.planet_setup(states)
    sim = initial.build_simulation(states, [], setup["dt_yr"])
    e0 = sim.energy()
    sim.steps(int(1.0e4 / setup["dt_yr"]))
    assert abs((sim.energy() - e0) / e0) < 1e-5
    assert setup["dt_yr"] == pytest.approx(0.01 * setup["jupiter_period_yr"])


# --- Trojan mass (the mass ladder, s4_run_ladder.py) ------------------------

def test_trojans_are_massless_by_default():
    states, _ = _system({100: _lagrange(60.0), 101: _lagrange(-60.0)})
    sim = initial.build_simulation(states, [100, 101], 0.1)
    assert sim.N_active == 5
    assert sim.testparticle_type == 0
    assert [p.m for p in sim.particles[5:]] == [0.0, 0.0]


def test_one_massive_trojan_is_semi_active():
    states, _ = _system({100: _lagrange(60.0)})
    m = constants.GIANT_MASS["jupiter"]
    sim = initial.build_simulation(states, [100], 0.1, trojan_mass=m)
    # N_active stays 5, so "test particles do not act on each other" is vacuous
    # for a single one and testparticle_type=1 is the exact six-body problem.
    assert sim.N_active == 5
    assert sim.testparticle_type == 1
    assert sim.particles[5].m == pytest.approx(m)


def test_massive_trojan_perturbs_jupiter():
    states, _ = _system({100: _lagrange(60.0)})
    jupiter = []
    for mass in (0.0, constants.GIANT_MASS["jupiter"]):
        sim = initial.build_simulation(states, [100], 0.05, trojan_mass=mass)
        sim.integrator = "ias15"
        sim.integrate(500.0)
        jupiter.append(np.array([sim.particles[1].x, sim.particles[1].y, sim.particles[1].z]))
    assert np.linalg.norm(jupiter[0] - jupiter[1]) > 1e-3


def test_several_massive_trojans_need_self_gravity():
    states, _ = _system({100: _lagrange(60.0), 101: _lagrange(62.0)})
    with pytest.raises(ValueError, match="self_gravity"):
        initial.build_simulation(states, [100, 101], 0.1, trojan_mass=1e-9)


def test_self_gravity_makes_every_trojan_active():
    states, _ = _system({100: _lagrange(60.0), 101: _lagrange(62.0)})
    sim = initial.build_simulation(states, [100, 101], 0.1, trojan_mass=1e-9, self_gravity=True)
    assert sim.N_active == sim.N == 7
    assert sim.testparticle_type == 1


def test_per_object_masses_and_negative_masses():
    states, _ = _system({100: _lagrange(60.0), 101: _lagrange(62.0)})
    sim = initial.build_simulation(states, [100, 101], 0.1, trojan_mass=[1e-9, 2e-9], self_gravity=True)
    assert [p.m for p in sim.particles[5:]] == [1e-9, 2e-9]
    with pytest.raises(ValueError, match="non-negative"):
        initial.build_simulation(states, [100], 0.1, trojan_mass=-1.0)


def test_settings_read_the_particles_section():
    base = {"run": {"t_end_yr": 1.0e6}}
    assert integrate.settings_from_config(base, 0.1).trojan_mass_msun == 0.0
    assert integrate.settings_from_config(base, 0.1).self_gravity is False
    loaded = integrate.settings_from_config(
        {**base, "particles": {"trojan_mass_msun": 1.5e-3, "self_gravity": True}}, 0.1)
    assert loaded.trojan_mass_msun == pytest.approx(1.5e-3)
    assert loaded.self_gravity is True
