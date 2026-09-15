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
