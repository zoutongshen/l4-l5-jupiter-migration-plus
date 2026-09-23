"""Physical constants and the model fixed at the 2026-09-15 meeting.

The massive bodies are the Sun, carrying the masses of Mercury, Venus, the
Earth-Moon system and the Mars system, plus Jupiter, Saturn, Uranus and Neptune
as separate bodies with their system masses (planet plus moons). The time step
is 1% of Jupiter's orbital period.

Simulation units: au, Julian year, solar mass. G is derived from DE440's GM_sun
rather than CODATA G, so that heliocentric elements from JPL convert exactly.
"""

import math

# JPL DE440 (Park et al. 2021), from https://ssd.jpl.nasa.gov/astro_par.html,
# in km^3 s^-2. EMB = Earth 398600.435507 + Moon 4902.800118.
GM_KM3_S2 = {
    "sun": 1.32712440041279419e11,
    "mercury": 22031.868551,
    "venus": 324858.592000,
    "earth_moon": 403503.235625,
    "mars_system": 42828.375816,
    "jupiter_system": 126712764.100000,
    "saturn_system": 37940584.841800,
    "uranus_system": 5794556.400000,
    "neptune_system": 6836527.100580,
}

AU_KM = 149_597_870.700          # IAU 2012 Resolution B1
DAY_S = 86_400.0
JULIAN_YEAR_D = 365.25
YEAR_S = JULIAN_YEAR_D * DAY_S

# G such that G * 1 Msun = GM_sun, in au^3 yr^-2 Msun^-1.
G = GM_KM3_S2["sun"] * YEAR_S**2 / AU_KM**3

INNER_BODIES = ("mercury", "venus", "earth_moon", "mars_system")
GIANTS = ("jupiter", "saturn", "uranus", "neptune")

# Horizons target IDs of the giant-planet system barycentres.
HORIZONS_ID = {"jupiter": "5", "saturn": "6", "uranus": "7", "neptune": "8"}


def _msun(name: str) -> float:
    return GM_KM3_S2[name] / GM_KM3_S2["sun"]


CENTRAL_MASS = 1.0 + sum(_msun(b) for b in INNER_BODIES)
GIANT_MASS = {g: _msun(f"{g}_system") for g in GIANTS}

DT_FRACTION_OF_JUPITER_PERIOD = 0.01


def jupiter_period(a_jupiter_au: float) -> float:
    """Keplerian period of Jupiter about the folded central mass, in years."""
    mu = G * (CENTRAL_MASS + GIANT_MASS["jupiter"])
    return 2.0 * math.pi * math.sqrt(a_jupiter_au**3 / mu)


def timestep(a_jupiter_au: float) -> float:
    """The campaign time step: 1% of Jupiter's period, in years."""
    return DT_FRACTION_OF_JUPITER_PERIOD * jupiter_period(a_jupiter_au)


def hill_radius(a_planet_au: float, planet: str) -> float:
    """Hill radius of a giant about the central mass, in au (circular approximation)."""
    return a_planet_au * (GIANT_MASS[planet] / (3.0 * CENTRAL_MASS)) ** (1.0 / 3.0)


# --- co-orbital stability limits on a finite Trojan mass --------------------
#
# Gascheau (1843) / Routh (1875): the equilateral configuration of three finite
# masses is linearly stable iff 27 (m1 m2 + m2 m3 + m3 m1) < (m1 + m2 + m3)^2.
# Liberato & Winter (2020, MNRAS 496, 3700) integrated the co-orbital region and
# found stability in practice disappears well below that, at mu = 0.023313.

LIBERATO_WINTER_MU = 0.023313     # practical co-orbital limit, Liberato & Winter (2020)


def routh_stable(m1: float, m2: float, m3: float) -> bool:
    """Gascheau-Routh linear stability of the equilateral three-body configuration."""
    return 27.0 * (m1 * m2 + m2 * m3 + m3 * m1) < (m1 + m2 + m3) ** 2


def routh_trojan_mass_limit(m_planet: float = None, m_central: float = None) -> float:
    """Largest Trojan mass, in Msun, that keeps Sun + Jupiter + Trojan linearly stable.

    Writing S = m1 + m2, the criterion rearranges to m3^2 - 25 S m3 + (S^2 - 27 m1 m2) > 0,
    and the configuration is stable below the smaller root. For this project's masses that
    is 0.0390 Msun -- about 41 Jupiter masses, ten orders of magnitude above the largest
    real Trojan.
    """
    m1 = CENTRAL_MASS if m_central is None else m_central
    m2 = GIANT_MASS["jupiter"] if m_planet is None else m_planet
    s = m1 + m2
    b, c = -25.0 * s, s * s - 27.0 * m1 * m2
    return 0.5 * (-b - math.sqrt(b * b - 4.0 * c))


# --- real Trojan masses from absolute magnitude -----------------------------
#
# Only needed for the self-gravitating swarm run; the campaign's Trojans are
# massless. Albedo from Grav et al. (2012), WISE/NEOWISE Jovian Trojan taxonomy;
# density from the two Trojans with measured bulk densities, (624) Hektor and
# (617) Patroclus, both 1.0 +/- 0.3 g/cm^3.

MSUN_KG = GM_KM3_S2["sun"] * 1e9 / 6.67430e-11      # GM_sun (m^3 s^-2) / CODATA G
TROJAN_ALBEDO = 0.07
TROJAN_DENSITY_KG_M3 = 1000.0
_H_TO_DIAMETER_KM = 1329.0                          # standard D = 1329 p_V^-1/2 10^(-H/5) km


def diameter_km(H, albedo: float = TROJAN_ALBEDO):
    """Diameter in km from absolute magnitude, assuming a fixed albedo."""
    return _H_TO_DIAMETER_KM / math.sqrt(albedo) * 10.0 ** (-0.4 * H / 2.0)


def mass_from_H(H, albedo: float = TROJAN_ALBEDO, density_kg_m3: float = TROJAN_DENSITY_KG_M3):
    """Mass in Msun from absolute magnitude, as a uniform sphere.

    Crude by construction -- one albedo and one density for the whole population,
    and mass goes as the cube of a diameter inferred from a single magnitude. Good
    to a factor of a few per object, which is all the self-gravity question needs:
    the answer there turns on orders of magnitude, not on individual masses.
    """
    radius_m = 0.5 * diameter_km(H, albedo) * 1e3
    return density_kg_m3 * (4.0 / 3.0) * math.pi * radius_m**3 / MSUN_KG


def coorbital_mu(m_trojan_msun: float = 0.0) -> float:
    """Co-orbital mass parameter (m_J + m_T) / (M_central + m_J + m_T).

    Reduces to the restricted-problem mu = 9.5385e-4 for a massless Trojan, so it
    is directly comparable to ``LIBERATO_WINTER_MU`` and to Routh's 0.0385.
    """
    m_j = GIANT_MASS["jupiter"]
    return (m_j + m_trojan_msun) / (CENTRAL_MASS + m_j + m_trojan_msun)
