import math

import pytest
import rebound

from trojans import constants as c


def test_G_agrees_with_rebound_units():
    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    # REBOUND uses CODATA G and a nominal solar mass; ours comes from GM_sun.
    assert c.G == pytest.approx(sim.G, rel=1e-4)
    assert c.G == pytest.approx(4 * math.pi**2, rel=1e-4)


def test_inner_planets_are_folded_into_the_sun():
    extra = c.CENTRAL_MASS - 1.0
    assert extra == pytest.approx(5.977e-6, rel=1e-3)


def test_giant_masses():
    assert c.GIANT_MASS["jupiter"] == pytest.approx(9.5479e-4, rel=1e-4)
    assert c.GIANT_MASS["saturn"] == pytest.approx(2.8589e-4, rel=1e-4)


def test_jupiter_period_and_timestep():
    period = c.jupiter_period(5.2026)
    assert period == pytest.approx(11.862, abs=0.01)
    assert c.timestep(5.2026) == pytest.approx(0.01 * period)


def test_jupiter_hill_radius():
    assert c.hill_radius(5.2026, "jupiter") == pytest.approx(0.355, abs=0.002)
