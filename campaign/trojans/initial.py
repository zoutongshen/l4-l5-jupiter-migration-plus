"""Initial conditions at t0 and the REBOUND simulation built from them.

The initial conditions are a plain table (``states.parquet``): the folded Sun,
the four giant-planet system barycentres, and every Trojan, as heliocentric
positions (au) and velocities (au/yr) at t0. A simulation is rebuilt from that
table deterministically, for any subset of particles.
"""

import numpy as np
import pandas as pd
import rebound

from trojans import constants, diagnostics

# REBOUND hashes of the massive bodies: NAIF IDs. Trojans use their SPK-IDs (>= 2e7).
MASSIVE_HASH = {"sun": 10, "jupiter": 5, "saturn": 6, "uranus": 7, "neptune": 8}


def states_table(giants_pv: dict, spkid, pos, vel) -> pd.DataFrame:
    """The initial-condition table; massive bodies first, in simulation order."""
    rows = [{"body": "sun", "spkid": 0, "m": constants.CENTRAL_MASS, "x": 0.0, "y": 0.0, "z": 0.0,
             "vx": 0.0, "vy": 0.0, "vz": 0.0}]
    for planet in constants.GIANTS:
        p, v = giants_pv[planet]
        rows.append({"body": planet, "spkid": 0, "m": constants.GIANT_MASS[planet],
                     "x": p[0], "y": p[1], "z": p[2], "vx": v[0], "vy": v[1], "vz": v[2]})
    massive = pd.DataFrame(rows)
    particles = pd.DataFrame({
        "body": "trojan", "spkid": np.asarray(spkid, dtype="int64"), "m": 0.0,
        "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2],
        "vx": vel[:, 0], "vy": vel[:, 1], "vz": vel[:, 2],
    })
    return pd.concat([massive, particles], ignore_index=True)


def planet_setup(states: pd.DataFrame) -> dict:
    """Jupiter's orbit, the time step, and each giant's Hill radius, from the t0 states."""
    out = {}
    for planet in constants.GIANTS:
        row = states.loc[states["body"] == planet].iloc[0]
        mu = constants.G * (constants.CENTRAL_MASS + constants.GIANT_MASS[planet])
        el = diagnostics.elements([[row.x, row.y, row.z]], [[row.vx, row.vy, row.vz]], mu)
        a = float(el["a"][0])
        out[planet] = {"a": a, "hill_radius": constants.hill_radius(a, planet)}
    a_j = out["jupiter"]["a"]
    return {
        "giants": out,
        "jupiter_period_yr": constants.jupiter_period(a_j),
        "dt_yr": constants.timestep(a_j),
    }


def build_simulation(states: pd.DataFrame, spkids, dt_yr: float) -> rebound.Simulation:
    """WHFast simulation of the massive bodies plus the requested Trojans (massless)."""
    sim = rebound.Simulation()
    sim.G = constants.G
    sim.integrator = "whfast"
    sim.dt = dt_yr

    massive = states[states["body"] != "trojan"]
    for row in massive.itertuples():
        sim.add(m=row.m, x=row.x, y=row.y, z=row.z, vx=row.vx, vy=row.vy, vz=row.vz,
                hash=MASSIVE_HASH[row.body])
    sim.N_active = len(massive)
    sim.testparticle_type = 0

    trojans = states[states["body"] == "trojan"].set_index("spkid").loc[np.asarray(spkids, dtype="int64")]
    for spkid, row in zip(trojans.index, trojans.itertuples()):
        sim.add(m=0.0, x=row.x, y=row.y, z=row.z, vx=row.vx, vy=row.vy, vz=row.vz, hash=int(spkid))

    sim.move_to_com()
    return sim
