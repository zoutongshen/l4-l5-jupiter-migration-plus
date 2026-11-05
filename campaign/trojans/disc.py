"""A mirror-symmetric planetesimal disc around a fixed or growing Jupiter, no migration.

The capture control (2026-10-23). Nothing here uses the catalogue: the disc is
drawn from simple distributions and Trojans are whatever the window classifier
says librates in one swarm.

The system is the folded Sun plus Jupiter on a circular orbit in the reference
plane at today's semimajor axis. With Jupiter on the +x axis at t = 0, the mirror
map (x, y, z, vx, vy, vz) -> (x, -y, z, -vx, vy, -vz) sends the system to
itself, so every disc particle is drawn together with its mirror twin and the
starting disc is exactly symmetric, not just statistically. Any L4/L5 difference
at the end was made by the dynamics.

Jupiter either keeps its full mass (arm "fixed") or grows exponentially from a
core (arm "grow"). The mirror map reverses time, so a growing Jupiter is mapped
to a shrinking one: the theorem does not promise a symmetric outcome for growth,
which is why it is worth running. After every mass change Jupiter's velocity is
reset to the circular speed at its current radius, so its semimajor axis cannot
drift: "no migration" holds by construction.
"""

from dataclasses import dataclass

import numpy as np
import rebound

from trojans import constants, diagnostics

M_EARTH = constants.GM_KM3_S2["earth_moon"] / constants.GM_KM3_S2["sun"] / (1.0 + 0.0123000371)
JUPITER_HASH = 5
SUN_HASH = 10
FIRST_PARTICLE_HASH = 100_000_000


@dataclass(frozen=True)
class Disc:
    n_pairs: int
    # The smoke runs gave no Trojans from a < 5.0 or a > 5.6 au in either arm.
    a_min: float = 4.9
    a_max: float = 5.5
    e_max: float = 0.3
    inc_max_deg: float = 40.0
    seed: int = 20261023


@dataclass(frozen=True)
class Growth:
    """Exponential growth from ``m0`` to ``m_final`` over ``t_grow_yr``; then constant.

    ``t_grow_yr = 0`` is the fixed arm: full mass from t = 0.
    """
    m0_msun: float
    m_final_msun: float = constants.GIANT_MASS["jupiter"]
    t_grow_yr: float = 0.0

    def mass(self, t_yr: float) -> float:
        if self.t_grow_yr <= 0.0 or t_yr >= self.t_grow_yr:
            return self.m_final_msun
        return self.m0_msun * (self.m_final_msun / self.m0_msun) ** (max(t_yr, 0.0) / self.t_grow_yr)


def hill_radius(a_au: float, m_msun: float) -> float:
    return a_au * (m_msun / (3.0 * constants.CENTRAL_MASS)) ** (1.0 / 3.0)


def disc_states(disc: Disc):
    """Heliocentric positions and velocities, shape (2 n_pairs, 3): each draw, then its twin.

    a, e and I are uniform (not area-weighted) so that every (e, I) bin is equally
    filled; Omega, omega and M are uniform.
    """
    rng = np.random.default_rng(disc.seed)
    n = disc.n_pairs
    a = rng.uniform(disc.a_min, disc.a_max, n)
    e = rng.uniform(0.0, disc.e_max, n)
    inc = rng.uniform(0.0, disc.inc_max_deg, n)
    Omega, omega, M = (rng.uniform(0.0, 360.0, n) for _ in range(3))
    pos, vel = diagnostics.cartesian(a, e, inc, Omega, omega, M, constants.G * constants.CENTRAL_MASS)
    twin_pos = pos * [1.0, -1.0, 1.0]
    twin_vel = vel * [-1.0, 1.0, -1.0]
    out_pos = np.empty((2 * n, 3))
    out_vel = np.empty((2 * n, 3))
    out_pos[0::2], out_pos[1::2] = pos, twin_pos
    out_vel[0::2], out_vel[1::2] = vel, twin_vel
    return out_pos, out_vel


def particle_ids(n_pairs: int) -> np.ndarray:
    """Particle 2k and its twin 2k + 1 share k = (id - FIRST_PARTICLE_HASH) // 2."""
    return FIRST_PARTICLE_HASH + np.arange(2 * n_pairs, dtype=np.int64)


def build_simulation(pos, vel, ids, growth: Growth, a_jupiter: float, dt_yr: float) -> rebound.Simulation:
    sim = rebound.Simulation()
    sim.G = constants.G
    sim.integrator = "whfast"
    sim.dt = dt_yr
    sim.add(m=constants.CENTRAL_MASS, hash=SUN_HASH)
    sim.add(m=growth.mass(0.0), a=a_jupiter, e=0.0, inc=0.0, Omega=0.0, omega=0.0, M=0.0,
            primary=sim.particles[0], hash=JUPITER_HASH)
    sim.N_active = 2
    sim.testparticle_type = 0
    for i, (p, v) in zip(ids, zip(pos, vel)):
        sim.add(m=0.0, x=p[0], y=p[1], z=p[2], vx=v[0], vy=v[1], vz=v[2], hash=int(i))
    sim.move_to_com()
    return sim


class GrowthForcing:
    """``ChunkRunner`` forcing: set Jupiter's mass for ``sim.t``, keep its orbit circular."""

    def __init__(self, growth: Growth, a_jupiter: float):
        self.growth = growth
        self.a_jupiter = a_jupiter

    def __call__(self, sim: rebound.Simulation, runner) -> None:
        m = self.growth.mass(sim.t)
        jup, sun = sim.particles[1], sim.particles[0]
        if m != jup.m:
            rel_x = np.array(jup.xyz) - np.array(sun.xyz)
            r = np.linalg.norm(rel_x)
            speed = np.sqrt(constants.G * (constants.CENTRAL_MASS + m) / r)
            tangent = np.array([-rel_x[1], rel_x[0], 0.0]) / np.hypot(rel_x[0], rel_x[1])
            jup.m = m
            jup.vxyz = tuple(np.array(sun.vxyz) + speed * tangent)
            sim.move_to_com()
            sim.ri_whfast.recalculate_coordinates_this_timestep = 1
        runner.m_jupiter = m
        runner.hill = np.array([hill_radius(self.a_jupiter, m)])


# Giants after Jupiter get hashes 6, 7, 8 (Jupiter is 5, the Sun 10).
GIANT_HASH = {"jupiter": JUPITER_HASH, "saturn": 6, "uranus": 7, "neptune": 8}


def build_system(pos, vel, ids, planets, a0, m0, dt_yr: float) -> rebound.Simulation:
    """Sun, then the giants on circular planar orbits, then the disc particles.

    ``planets`` are names (Jupiter first), ``a0`` and ``m0`` their starting a and mass.
    The giants alternate between longitude 0 and 180 deg (Jupiter on +x), so the
    starting system is still sent to itself by the mirror map and the twin disc stays
    exactly symmetric. With Jupiter alone this is ``build_simulation``.
    """
    sim = rebound.Simulation()
    sim.G = constants.G
    sim.integrator = "whfast"
    sim.dt = dt_yr
    sim.add(m=constants.CENTRAL_MASS, hash=SUN_HASH)
    for k, (name, a, m) in enumerate(zip(planets, a0, m0)):
        sim.add(m=m, a=a, e=0.0, inc=0.0, Omega=0.0, omega=0.0, M=0.0 if k % 2 == 0 else np.pi,
                primary=sim.particles[0], hash=GIANT_HASH[name])
    sim.N_active = 1 + len(planets)
    sim.testparticle_type = 0
    for i, (p, v) in zip(ids, zip(pos, vel)):
        sim.add(m=0.0, x=p[0], y=p[1], z=p[2], vx=v[0], vy=v[1], vz=v[2], hash=int(i))
    sim.move_to_com()
    return sim
