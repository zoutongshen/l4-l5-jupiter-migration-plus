"""
Functions for creating the solar system (Sun + planets) and planetesimals
as a REBOUND simulation.

Units: ('yr', 'AU', 'Msun') so G = 4 pi^2.

Frame convention (matches the original AMUSE setup intent):
- Bodies are added heliocentrically (relative to the Sun at the origin).
- The whole system is moved to the centre-of-mass frame once at the end.
- Planetesimals are massless test particles (m = 0) and do not perturb the
  massive bodies or each other (enforced via sim.N_active).
- Jupiter is kept at index 4 (Sun=0, Venus=1, Earth=2, Mars=3, Jupiter=4),
  preserving the indexing the analysis code relies on.
"""

import numpy as np
import rebound

from .config import SimulationConfig

# 1 Earth mass in solar masses (M_sun / M_earth = 332946.0487)
MEARTH_TO_MSUN = 1.0 / 332946.0487


def _new_sim(config: SimulationConfig) -> rebound.Simulation:
    """Create an empty REBOUND simulation with units and integrator configured."""
    sim = rebound.Simulation()
    sim.units = ('yr', 'AU', 'Msun')
    sim.integrator = 'whfast'
    # dt is finalised in _finalise_integrator() once periods are known.
    sim.dt = config.dt
    return sim


def _finalise_integrator(sim: rebound.Simulation, config: SimulationConfig,
                         verbose: bool = False) -> None:
    """Clamp the WHFast timestep to <= P_min / 20 for the innermost body.

    The original code ran dt = 0.1 yr with Huayno; WHFast is a fixed-step
    integrator so we make sure the innermost planet (Venus, P ~ 0.62 yr) is
    sampled well enough for a stable symplectic integration.
    """
    periods = []
    for p in sim.particles[1:sim.N_active if sim.N_active > 0 else sim.N]:
        try:
            orb = p.orbit(primary=sim.particles[0])
            if orb.P > 0:
                periods.append(orb.P)
        except Exception:
            pass
    if periods:
        p_min = min(periods)
        dt = min(config.dt, p_min / 20.0)
    else:
        dt = config.dt
    sim.dt = dt
    if verbose:
        print(f"  WHFast timestep: {dt:.4f} yr")


def _add_by_elements(sim, m_msun, a, e, inc_deg, Omega_deg, omega_deg, f_deg):
    """Add a particle by heliocentric Keplerian elements (angles in degrees)."""
    sim.add(
        primary=sim.particles[0],
        m=m_msun,
        a=a,
        e=e,
        inc=np.deg2rad(inc_deg),
        Omega=np.deg2rad(Omega_deg),
        omega=np.deg2rad(omega_deg),
        f=np.deg2rad(f_deg),
    )


def create_solar_system(config: SimulationConfig):
    """
    Build Sun + planets as a REBOUND simulation in the COM frame.

    Returns:
        (sim, names): the simulation and the list of massive-body names.
    """
    sim = _new_sim(config)

    # Sun at the origin
    sim.add(m=config.star_mass, x=0, y=0, z=0, vx=0, vy=0, vz=0)
    names = ['Sun']

    # Planets via orbital elements; preserve the original RNG call order/seed.
    np.random.seed(config.random_seed)
    for name in config.active_planet_names:
        data = config.planet_data[name]
        mass_msun = data['mass'] * MEARTH_TO_MSUN
        a = config.jupiter_initial_a if name == 'Jupiter' else data['a']
        e = data['e']
        inc = data['i']

        true_anomaly = np.random.uniform(0, 360)
        arg_perihelion = np.random.uniform(0, 360)
        long_asc_node = np.random.uniform(0, 360)

        _add_by_elements(sim, mass_msun, a, e, inc,
                         long_asc_node, arg_perihelion, true_anomaly)
        names.append(name)

    sim.N_active = len(names)
    sim.move_to_com()
    _finalise_integrator(sim, config, verbose=False)

    return sim, names


def add_planetesimals(sim: rebound.Simulation, config: SimulationConfig,
                      n_massive: int) -> int:
    """
    Add test-particle planetesimals to an existing solar-system simulation.

    Dispatches on config:
    - trojan_emphasis=True  -> structured 4-zone disk (L4/L5 biased co-orbitals)
    - else                  -> uniform disk (burn-in / source-tracking)

    Massless particles (m=0); sim.N_active is set so they perturb nothing.

    Returns:
        Number of planetesimals added.
    """
    if config.trojan_emphasis:
        n_added = _add_structured_disk(sim, config)
    else:
        n_added = _add_uniform_disk(sim, config)

    sim.N_active = n_massive
    _assign_planetesimal_hashes(sim, n_massive)
    _finalise_integrator(sim, config, verbose=False)
    return n_added


def _assign_planetesimal_hashes(sim, n_massive):
    """Tag planetesimal j with hash j+1 so snapshots stay aligned to a fixed
    slot even after escapers are removed mid-run (hash 0 = unset)."""
    for j, p in enumerate(sim.particles[n_massive:]):
        p.hash = j + 1


def _add_structured_disk(sim: rebound.Simulation, config: SimulationConfig) -> int:
    """Structured protoplanetary disk: inner 50%, mid 30%, co-orbital 15%, outer 5%."""
    np.random.seed(config.random_seed + 1)

    n = config.n_planetesimals
    jupiter_a = config.jupiter_initial_a

    def sample_powerlaw(r_min, r_max, count, p=-1.5):
        u = np.random.random(count)
        if abs(p + 1) < 1e-8:
            return r_min * (r_max / r_min) ** u
        term1 = r_min ** (p + 1)
        term2 = r_max ** (p + 1) - r_min ** (p + 1)
        return (term1 + u * term2) ** (1 / (p + 1))

    u_regions = np.random.random(n)
    a_values = np.zeros(n)
    true_anoms = np.random.uniform(0, 360, n)

    mask_inner = u_regions < 0.50
    n_inner = int(np.sum(mask_inner))
    if n_inner > 0:
        a_values[mask_inner] = sample_powerlaw(0.7, 3.0, n_inner)

    mask_mid = (u_regions >= 0.50) & (u_regions < 0.80)
    n_mid = int(np.sum(mask_mid))
    if n_mid > 0:
        a_values[mask_mid] = sample_powerlaw(3.0, 10.0, n_mid)

    mask_coorb = (u_regions >= 0.80) & (u_regions < 0.95)
    n_coorb = int(np.sum(mask_coorb))
    if n_coorb > 0:
        a_values[mask_coorb] = jupiter_a + 0.2 * np.random.randn(n_coorb)
        # Bias co-orbital angles toward L4 (+60) / L5 (-60) of Jupiter's angle.
        centers = np.where(np.random.random(n_coorb) < 0.5, 60, -60)
        offsets = 15 * np.random.uniform(-1, 1, n_coorb)
        jupiter = sim.particles[config.jupiter_index]
        jupiter_angle_deg = np.degrees(np.arctan2(jupiter.y, jupiter.x))
        true_anoms[mask_coorb] = centers + offsets + jupiter_angle_deg

    mask_outer = u_regions >= 0.95
    n_outer = int(np.sum(mask_outer))
    if n_outer > 0:
        a_values[mask_outer] = sample_powerlaw(10.0, 30.0, n_outer, p=-2.0)

    e_values = 0.05 * np.random.random(n)
    i_values = 3.0 * np.random.random(n)
    long_asc = np.random.uniform(0, 360, n)
    arg_peri = np.random.uniform(0, 360, n)

    for k in range(n):
        _add_by_elements(sim, 0.0, a_values[k], e_values[k], i_values[k],
                         long_asc[k], arg_peri[k], true_anoms[k])

    print(f"Created structured disk with {n} particles:")
    print(f"  Inner (0.7-3 AU): {n_inner} ({n_inner/n:.1%})")
    print(f"  Mid (3-10 AU): {n_mid} ({n_mid/n:.1%})")
    print(f"  Co-orbital (~{jupiter_a} AU): {n_coorb} ({n_coorb/n:.1%})")
    print(f"  Outer (10-30 AU): {n_outer} ({n_outer/n:.1%})")
    return n


def _add_uniform_disk(sim: rebound.Simulation, config: SimulationConfig) -> int:
    """Uniform-in-a disk with Rayleigh e/i (burn-in and source-tracking)."""
    np.random.seed(config.random_seed + 1)

    n = config.n_planetesimals
    if config.sim_type == 'source_tracking':
        a_min, a_max = config.disk_a_min, config.disk_a_max
    else:
        a_min, a_max = config.planetesimal_a_min, config.planetesimal_a_max

    e_max = config.planetesimal_e_max
    i_max = config.planetesimal_i_max

    a_values = np.random.uniform(a_min, a_max, n)
    e_values = np.clip(np.random.rayleigh(e_max / 2, n), 0, e_max)
    i_values = np.clip(np.random.rayleigh(i_max / 2, n), 0, i_max)
    true_anomaly = np.random.uniform(0, 360, n)
    arg_perihelion = np.random.uniform(0, 360, n)
    long_asc_node = np.random.uniform(0, 360, n)

    for i in range(n):
        _add_by_elements(sim, 0.0, a_values[i], e_values[i], i_values[i],
                         long_asc_node[i], arg_perihelion[i], true_anomaly[i])
    return n


def add_trojan_disk(sim: rebound.Simulation, config: SimulationConfig,
                    n_l4: int = 5000, n_l5: int = 5000) -> int:
    """
    Add planetesimals specifically in Jupiter's L4 and L5 regions.

    Useful for targeted injection after source-tracking analysis. Massless
    test particles; caller is responsible for sim.N_active if needed (it is
    left unchanged here since the massive set is untouched).
    """
    np.random.seed(config.random_seed + 2)

    jupiter = sim.particles[config.jupiter_index]
    jupiter_a = np.sqrt(jupiter.x ** 2 + jupiter.y ** 2 + jupiter.z ** 2)
    jupiter_angle = np.arctan2(jupiter.y, jupiter.x)

    l4_angle = jupiter_angle + np.deg2rad(60)
    l5_angle = jupiter_angle - np.deg2rad(60)

    def generate(n, center_angle):
        a = np.random.uniform(jupiter_a * 0.93, jupiter_a * 1.07, n)
        angle_offset = np.random.uniform(-25, 25, n)
        angles = center_angle + np.deg2rad(angle_offset)
        e = np.clip(np.random.rayleigh(0.05, n), 0, 0.15)
        inc = np.clip(np.random.rayleigh(2, n), 0, 10)
        return a, angles, e, inc

    a_l4, ang_l4, e_l4, i_l4 = generate(n_l4, l4_angle)
    a_l5, ang_l5, e_l5, i_l5 = generate(n_l5, l5_angle)

    a_all = np.concatenate([a_l4, a_l5])
    ang_all = np.concatenate([ang_l4, ang_l5])
    e_all = np.concatenate([e_l4, e_l5])
    i_all = np.concatenate([i_l4, i_l5])

    for idx in range(n_l4 + n_l5):
        # omega = Omega = 0, so the true anomaly is the position angle directly.
        _add_by_elements(sim, 0.0, a_all[idx], e_all[idx], i_all[idx],
                         0.0, 0.0, np.rad2deg(ang_all[idx]))
    return n_l4 + n_l5


def sim_from_state(massive_pos, massive_vel, massive_mass, names,
                   planetesimal_pos, planetesimal_vel,
                   config: SimulationConfig) -> rebound.Simulation:
    """
    Rebuild a REBOUND simulation from saved Cartesian state arrays.

    Used to continue a run from a saved checkpoint (e.g. migration continuing
    from a burn-in state). Arrays are in AU and AU/yr; masses in Msun.
    """
    sim = _new_sim(config)
    for i in range(len(massive_mass)):
        sim.add(m=float(massive_mass[i]),
                x=float(massive_pos[i, 0]), y=float(massive_pos[i, 1]), z=float(massive_pos[i, 2]),
                vx=float(massive_vel[i, 0]), vy=float(massive_vel[i, 1]), vz=float(massive_vel[i, 2]))
    n_massive = len(massive_mass)

    for j in range(len(planetesimal_pos)):
        sim.add(m=0.0,
                x=float(planetesimal_pos[j, 0]), y=float(planetesimal_pos[j, 1]), z=float(planetesimal_pos[j, 2]),
                vx=float(planetesimal_vel[j, 0]), vy=float(planetesimal_vel[j, 1]), vz=float(planetesimal_vel[j, 2]))

    sim.N_active = n_massive
    _assign_planetesimal_hashes(sim, n_massive)
    _finalise_integrator(sim, config, verbose=False)
    return sim
