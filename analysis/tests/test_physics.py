"""
Physics correctness tests for the REBOUND port.

Run from the analysis/ directory:
    conda activate trojan-rb
    pytest tests/ -v
"""

import os
import sys

import numpy as np
import pytest

# Make the `simulation` package importable when running from analysis/ or repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulation import (  # noqa: E402
    SimulationConfig,
    create_solar_system,
    attach_migration,
    compute_trojan_counts,
    compute_jupiter_orbit,
    TrojanCriteria,
)
from simulation.config import MigrationConfig  # noqa: E402


def _planet_only_sim(jupiter_a=5.2):
    """A Sun + planets simulation with no planetesimals (N_active = all)."""
    config = SimulationConfig(jupiter_initial_a=jupiter_a)
    sim, names = create_solar_system(config)
    return sim, names


# --------------------------------------------------------------------------- #
# 1. Energy / angular-momentum conservation (planet-only, no migration)
# --------------------------------------------------------------------------- #

def test_energy_conservation_planets_only():
    sim, _ = _planet_only_sim()
    e0 = sim.energy()
    sim.integrate(10_000.0)  # 10 kyr
    e1 = sim.energy()
    rel = abs((e1 - e0) / e0)
    # WHFast with dt clamped to P_venus/20; expect a small symplectic error.
    assert rel < 1e-4, f"energy drift too large: {rel:.2e}"


def test_angular_momentum_conservation_planets_only():
    sim, _ = _planet_only_sim()

    def lz(s):
        L = 0.0
        for p in s.particles:
            L += p.m * (p.x * p.vy - p.y * p.vx)
        return L

    l0 = lz(sim)
    sim.integrate(10_000.0)
    l1 = lz(sim)
    rel = abs((l1 - l0) / l0)
    assert rel < 1e-8, f"Lz drift too large: {rel:.2e}"


# --------------------------------------------------------------------------- #
# 2. Migration trajectory: a(t) ~ a0 * exp(-t / tau_a)
# --------------------------------------------------------------------------- #

def test_migration_exponential_inward():
    a0 = 5.2
    tau = 10_000.0  # user convention: positive => inward
    config = SimulationConfig(jupiter_initial_a=a0)
    config.migration = MigrationConfig(enabled=True, tau_a=tau,
                                       tau_e=tau / 2, direction='inward')
    sim, _ = create_solar_system(config)
    driver = attach_migration(sim, config)

    t_test = tau / 2  # half an e-fold
    driver.advance(t_test)
    mpos = np.array([[p.x, p.y, p.z] for p in sim.particles])
    mvel = np.array([[p.vx, p.vy, p.vz] for p in sim.particles])
    mass = np.array([p.m for p in sim.particles])

    orbit = compute_jupiter_orbit(mpos, mvel, mass)
    a_expected = a0 * np.exp(-t_test / tau)
    rel = abs(orbit['a'] - a_expected) / a_expected
    assert rel < 0.02, f"a(t)={orbit['a']:.4f} vs expected {a_expected:.4f} (rel {rel:.3f})"


# --------------------------------------------------------------------------- #
# 3. Sign convention: +tau_a shrinks a, -tau_a grows a
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("tau_a,direction,grows", [
    (5000.0, 'inward', False),
    (-5000.0, 'outward', True),
])
def test_migration_sign_convention(tau_a, direction, grows):
    a0 = 5.2
    config = SimulationConfig(jupiter_initial_a=a0)
    config.migration = MigrationConfig(enabled=True, tau_a=tau_a,
                                       tau_e=abs(tau_a) / 2, direction=direction)
    sim, _ = create_solar_system(config)
    driver = attach_migration(sim, config)

    driver.advance(1000.0)
    a_jup = sim.particles[4].orbit(primary=sim.particles[0]).a
    if grows:
        assert a_jup > a0, f"expected outward (a>{a0}), got {a_jup:.4f}"
    else:
        assert a_jup < a0, f"expected inward (a<{a0}), got {a_jup:.4f}"


# --------------------------------------------------------------------------- #
# 4. Trojan classifier: inject particles at known angles relative to Jupiter
# --------------------------------------------------------------------------- #

def test_trojan_classifier_known_angles():
    # Place Jupiter on the +x axis at a_j; build synthetic positions.
    a_j = 5.2
    massive_pos = np.zeros((5, 3))
    massive_pos[4] = [a_j, 0.0, 0.0]  # Jupiter at angle 0

    def at_angle(deg, r=a_j):
        return [r * np.cos(np.deg2rad(deg)), r * np.sin(np.deg2rad(deg)), 0.0]

    crit = TrojanCriteria()
    # +60 -> L4, -60 -> L5, +30 -> neither
    p_l4 = np.array([at_angle(60)])
    p_l5 = np.array([at_angle(-60)])
    p_none = np.array([at_angle(30)])

    assert compute_trojan_counts(massive_pos, p_l4, criteria=crit) == (1, 0)
    assert compute_trojan_counts(massive_pos, p_l5, criteria=crit) == (0, 1)
    assert compute_trojan_counts(massive_pos, p_none, criteria=crit) == (0, 0)


def test_escaper_does_not_nan_the_system():
    """A test particle flung onto an escape trajectory must be removed, not
    poison the shared centre-of-mass with NaN (the bug that killed the first
    full migration run)."""
    import rebound
    from simulation.runner import _integrate_with_escape, ESCAPE_DISTANCE_AU

    cfg = SimulationConfig(jupiter_initial_a=5.2)
    sim, names = create_solar_system(cfg)
    n_massive = len(names)

    # One well-behaved co-orbital + one on a fast radial escape trajectory.
    sim.add(m=0.0, a=5.2, e=0.01, f=1.047, hash=1)              # ~L4-ish
    sim.add(m=0.0, x=5.0, y=0.0, z=0.0, vx=50.0, vy=0.0, vz=0.0, hash=2)  # runaway
    sim.N_active = n_massive
    sim.exit_max_distance = ESCAPE_DISTANCE_AU

    n_removed = _integrate_with_escape(sim, 2000.0, n_massive)

    assert n_removed >= 1, "the escaping particle should have been removed"
    # Everything that remains must be finite (no NaN poisoning).
    for p in sim.particles:
        assert np.isfinite(p.x) and np.isfinite(p.vx), "found non-finite state"
    # Jupiter is still on a sane orbit.
    a_jup = sim.particles[4].orbit(primary=sim.particles[0]).a
    assert 5.0 < a_jup < 5.4


def test_trojan_classifier_coorbital_tolerance():
    # A particle at +60 deg but well outside the co-orbital annulus is not a Trojan.
    a_j = 5.2
    massive_pos = np.zeros((5, 3))
    massive_pos[4] = [a_j, 0.0, 0.0]
    far = np.array([[2.0 * np.cos(np.deg2rad(60)), 2.0 * np.sin(np.deg2rad(60)), 0.0]])
    assert compute_trojan_counts(massive_pos, far) == (0, 0)
