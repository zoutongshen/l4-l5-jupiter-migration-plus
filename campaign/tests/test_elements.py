import math

import numpy as np
import pytest
import rebound

from trojans import constants as c
from trojans import diagnostics as d


def _random_orbits(n, seed=1):
    rng = np.random.default_rng(seed)
    sim = rebound.Simulation()
    sim.G = c.G
    sim.add(m=1.0)
    for _ in range(n):
        sim.add(
            primary=sim.particles[0],
            a=rng.uniform(4.5, 6.0), e=rng.uniform(0.0, 0.5), inc=rng.uniform(0.0, 1.0),
            Omega=rng.uniform(0, 2 * math.pi), omega=rng.uniform(0, 2 * math.pi),
            M=rng.uniform(0, 2 * math.pi),
        )
    return sim


def _angle_diff(x, y):
    return np.abs(d.wrap180(np.asarray(x) - np.asarray(y)))


def test_elements_match_rebound():
    sim = _random_orbits(200)
    sun = sim.particles[0]
    ps = sim.particles[1:]
    pos = np.array([[p.x, p.y, p.z] for p in ps])
    vel = np.array([[p.vx, p.vy, p.vz] for p in ps])
    el = d.elements(pos, vel, c.G * 1.0)
    orbits = [p.orbit(primary=sun) for p in ps]

    assert el["a"] == pytest.approx([o.a for o in orbits], rel=1e-10)
    assert el["e"] == pytest.approx([o.e for o in orbits], abs=1e-10)
    assert el["inc"] == pytest.approx([math.degrees(o.inc) for o in orbits], abs=1e-8)
    for key, attr in [("Omega", "Omega"), ("omega", "omega"), ("M", "M"), ("lam", "l"), ("varpi", "pomega")]:
        ref = [math.degrees(getattr(o, attr)) for o in orbits]
        assert np.max(_angle_diff(el[key], ref)) < 1e-6, key


def test_cartesian_matches_rebound_and_round_trips():
    rng = np.random.default_rng(7)
    n = 300
    a, e = rng.uniform(4.5, 6.0, n), rng.uniform(0.0, 0.59, n)
    inc, Om, om, M = rng.uniform(0, 50, n), rng.uniform(0, 360, n), rng.uniform(0, 360, n), rng.uniform(0, 360, n)
    pos, vel = d.cartesian(a, e, inc, Om, om, M, c.G)

    sim = rebound.Simulation()
    sim.G = c.G
    sim.add(m=1.0)
    for k in range(n):
        sim.add(primary=sim.particles[0], a=a[k], e=e[k], inc=math.radians(inc[k]),
                Omega=math.radians(Om[k]), omega=math.radians(om[k]), M=math.radians(M[k]))
    ref_pos = np.array([[p.x, p.y, p.z] for p in sim.particles[1:]])
    ref_vel = np.array([[p.vx, p.vy, p.vz] for p in sim.particles[1:]])
    assert np.max(np.abs(pos - ref_pos)) < 1e-10
    assert np.max(np.abs(vel - ref_vel)) < 1e-10

    back = d.elements(pos, vel, c.G)
    assert back["a"] == pytest.approx(a, rel=1e-10)
    assert np.max(_angle_diff(back["lam"], np.mod(Om + om + M, 360))) < 1e-7


def test_planar_orbit_has_well_defined_longitudes():
    sim = rebound.Simulation()
    sim.G = c.G
    sim.add(m=1.0)
    sim.add(primary=sim.particles[0], a=5.2, e=0.05, inc=0.0, Omega=0.0, omega=0.7, M=1.1)
    p = sim.particles[1]
    el = d.elements([[p.x, p.y, p.z]], [[p.vx, p.vy, p.vz]], c.G)
    assert el["lam"][0] == pytest.approx(math.degrees(0.7 + 1.1), abs=1e-8)


def test_wrap180():
    assert d.wrap180([180.0, -180.0, 190.0, -190.0, 0.0]).tolist() == [180.0, 180.0, -170.0, 170.0, 0.0]


def test_resonant_angle_signs():
    assert d.resonant_angle(60.0, 0.0) == pytest.approx(60.0)
    assert d.resonant_angle(10.0, 310.0) == pytest.approx(60.0)
    assert d.resonant_angle(250.0, 310.0) == pytest.approx(-60.0)


def test_abc_at_the_lagrange_point():
    jupiter = {"a": 5.2, "e": 0.048, "inc": 1.3, "Omega": 100.0, "varpi": 15.0}
    mu = c.GIANT_MASS["jupiter"] / (c.CENTRAL_MASS + c.GIANT_MASS["jupiter"])
    # L4: same a and inclination vector as Jupiter, forced eccentricity 60 deg ahead.
    A, B, C = d.abc(5.2, 0.048, 1.3, 100.0, 75.0, 60.0, jupiter, mu)
    assert A == pytest.approx(0.0, abs=1e-12)
    assert B == pytest.approx(0.0, abs=1e-12)
    assert C == pytest.approx(1.5, abs=1e-12)
    # L5 mirror.
    _, B5, C5 = d.abc(5.2, 0.048, 1.3, 100.0, -45.0, -60.0, jupiter, mu)
    assert B5 == pytest.approx(0.0, abs=1e-12)
    assert C5 == pytest.approx(1.5, abs=1e-12)


def test_c_grows_with_distance_from_the_lagrange_point():
    jupiter = {"a": 5.2, "e": 0.0, "inc": 0.0, "Omega": 0.0, "varpi": 0.0}
    mu = 1e-3
    _, _, C = d.abc(5.2 * (1 + 0.02), 0.0, 0.0, 0.0, 0.0, 60.0, jupiter, mu)
    assert C == pytest.approx(1.5 + 3 * 0.02**2 / (8 * mu))
    assert d.libration_integral_f(100.0) > d.libration_integral_f(60.0)
