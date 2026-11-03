"""Test particles around Jupiter's orbit at t0: the map for the Trojan definition (stage 7).

The question is which single-epoch orbits are tadpole librators. Real objects
barely sample the edge of the tadpole region, because the region next to it has
been emptied, so the edge is mapped with test particles instead. They are drawn
in the present-day system (the same giants, epoch and integrator as the pilot),
uniformly in the quantities a snapshot gives, and each is labelled by the window
classifier on its own integration.

The draw is a map, not a population: nothing about how many particles fall in
any region means anything physical. Only where the label flips does.

Elements are heliocentric osculating with mu = G M_sun, the SBDB convention the
catalogue's states use (``catalogue.states_at_t0``), so the map and the real
objects are in the same coordinates. sigma = lambda - lambda_J is drawn directly,
with Jupiter's lambda from the t0 states.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trojans import catalogue, constants, diagnostics

# Disc particles start at 1e8 (trojans.disc); map particles sit above them.
FIRST_MAP_HASH = 200_000_000


@dataclass(frozen=True)
class MapGrid:
    """Uniform draw in (a, e, I, sigma) with Omega and varpi uniform.

    The ranges enclose every catalogue box (V24: 4.95-5.45 au, e < 0.6, I < 50 deg;
    MPC as listed: e <= 0.297, I <= 55.4 deg; JPL: 4.6-5.5 au, e < 0.3), so each
    box's edges fall inside the map.
    """
    n: int
    a_min: float = 4.6
    a_max: float = 5.8
    e_max: float = 0.6
    inc_max_deg: float = 60.0
    seed: int = 20261027


def jupiter_at_t0(states: pd.DataFrame) -> dict:
    row = states.loc[states["body"] == "jupiter"].iloc[0]
    return catalogue.planet_elements(np.array([row.x, row.y, row.z]), np.array([row.vx, row.vy, row.vz]), "jupiter")


def draw_elements(grid: MapGrid, jupiter: dict) -> pd.DataFrame:
    rng = np.random.default_rng(grid.seed)
    n = grid.n
    a = rng.uniform(grid.a_min, grid.a_max, n)
    e = rng.uniform(0.0, grid.e_max, n)
    inc = rng.uniform(0.0, grid.inc_max_deg, n)
    sigma = rng.uniform(-180.0, 180.0, n)
    Omega = rng.uniform(0.0, 360.0, n)
    varpi = rng.uniform(0.0, 360.0, n)
    lam = jupiter["lam"] + sigma
    return pd.DataFrame({
        "spkid": FIRST_MAP_HASH + np.arange(n, dtype=np.int64),
        "a": a, "e": e, "inc": inc,
        "Omega": Omega, "omega": np.mod(varpi - Omega, 360.0), "M": np.mod(lam - varpi, 360.0),
        "sigma": sigma,
    })


def states_with_particles(states: pd.DataFrame, particles: pd.DataFrame) -> pd.DataFrame:
    """The t0 states table with its Trojans replaced by ``particles`` (spkid + heliocentric elements)."""
    pos, vel = diagnostics.cartesian(particles["a"], particles["e"], particles["inc"], particles["Omega"],
                                     particles["omega"], particles["M"], constants.G)
    massive = states[states["body"] != "trojan"]
    added = pd.DataFrame({
        "body": "trojan", "spkid": particles["spkid"].to_numpy(dtype="int64"), "m": 0.0,
        "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2], "vx": vel[:, 0], "vy": vel[:, 1], "vz": vel[:, 2],
    })
    return pd.concat([massive, added], ignore_index=True)
