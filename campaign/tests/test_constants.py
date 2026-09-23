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


# --- co-orbital stability limits -------------------------------------------

def test_routh_criterion_matches_the_restricted_value():
    # With a massless third body the criterion reduces to m1/m2 > 24.9599.
    assert c.routh_stable(1.0, 1.0 / 24.96, 0.0)
    assert not c.routh_stable(1.0, 1.0 / 24.95, 0.0)


def test_routh_trojan_mass_limit_is_where_stability_flips():
    limit = c.routh_trojan_mass_limit()
    m1, m2 = c.CENTRAL_MASS, c.GIANT_MASS["jupiter"]
    assert c.routh_stable(m1, m2, limit * 0.999)
    assert not c.routh_stable(m1, m2, limit * 1.001)
    # ~41 Jupiter masses, ten orders of magnitude above the largest real Trojan.
    assert limit / m2 == pytest.approx(40.9, abs=0.1)


def test_coorbital_mu_spans_the_published_limits():
    # A massless Trojan gives the restricted-problem value for Sun-Jupiter.
    assert c.coorbital_mu(0.0) == pytest.approx(9.5388e-4, rel=1e-4)
    # Routh's mu_G = 0.0385209 sits just above the finite-mass limit.
    assert c.coorbital_mu(c.routh_trojan_mass_limit()) == pytest.approx(0.0385209, abs=1e-4)
    # Liberato & Winter's limit is 24.0 Jupiter masses for this model.
    m_24 = 24.0 * c.GIANT_MASS["jupiter"]
    assert c.coorbital_mu(m_24) == pytest.approx(c.LIBERATO_WINTER_MU, rel=1e-3)


# --- real Trojan masses from H ---------------------------------------------

def test_solar_mass_matches_the_accepted_value():
    assert c.MSUN_KG == pytest.approx(1.98841e30, rel=1e-4)


def test_mass_from_H_reproduces_hektor_to_a_factor_of_a_few():
    # (624) Hektor, H = 7.32, measured 7.9e18 kg. A single albedo and density
    # cannot do better than a factor of a few -- Hektor is an elongated binary.
    hektor = c.mass_from_H(7.32) * c.MSUN_KG
    assert 1e18 < hektor < 1.6e19
    assert c.diameter_km(7.32) == pytest.approx(173.0, rel=0.05)


def test_mass_from_H_scales_as_ten_to_the_minus_zero_point_six_H():
    assert c.mass_from_H(15.0) / c.mass_from_H(10.0) == pytest.approx(10.0 ** (-0.6 * 5), rel=1e-9)
    assert c.mass_from_H(10.0) > c.mass_from_H(14.0)


def test_real_trojans_are_far_below_the_threshold():
    # The whole point of the mass ladder: nine to ten orders of magnitude of margin.
    assert c.mass_from_H(7.32) / c.routh_trojan_mass_limit() < 1e-9
