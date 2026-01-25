"""
Functions for creating massive bodies (Sun + planets) and planetesimals.
"""

import numpy as np
from amuse.units import units, constants
from amuse.datamodel import Particles
from amuse.ext.orbital_elements import new_binary_from_orbital_elements

from .config import SimulationConfig, DEFAULT_PLANET_DATA


def create_massive_bodies(config: SimulationConfig) -> Particles:
    """
    Create Sun + planets for the solar system.

    Args:
        config: Simulation configuration

    Returns:
        Particles set containing Sun and planets
    """
    star_mass = config.star_mass | units.MSun
    planet_data = config.planet_data

    # Create Sun
    sun = Particles(1)
    sun.mass = star_mass
    sun.position = [0, 0, 0] | units.AU
    sun.velocity = [0, 0, 0] | units.km / units.s
    sun.name = 'Sun'

    # Create planets
    planets = Particles()
    np.random.seed(config.random_seed)

    for name, data in planet_data.items():
        mass = data['mass'] | units.MEarth
        a = data['a'] | units.AU
        e = data['e']
        inc = data['i'] | units.deg

        # Override Jupiter's semi-major axis if specified
        if name == 'Jupiter':
            a = config.jupiter_initial_a | units.AU

        # Random orbital angles
        true_anomaly = np.random.uniform(0, 360) | units.deg
        arg_perihelion = np.random.uniform(0, 360) | units.deg
        long_asc_node = np.random.uniform(0, 360) | units.deg

        # Create planet using orbital elements
        binary = new_binary_from_orbital_elements(
            star_mass,
            mass,
            a,
            e,
            true_anomaly=true_anomaly,
            inclination=inc,
            longitude_of_the_ascending_node=long_asc_node,
            argument_of_periapsis=arg_perihelion
        )

        planet = binary[1]
        planet.name = name
        planets.add_particle(planet)

    # Combine Sun + planets
    bodies = sun + planets

    # Move to center of mass frame
    bodies.move_to_center()

    return bodies


def create_planetesimals(config: SimulationConfig,
                         massive_bodies: Particles) -> Particles:
    """
    Create planetesimals (test particles) for the simulation.

    For 'source_tracking' simulations, uses broader disk parameters.

    Args:
        config: Simulation configuration
        massive_bodies: The massive bodies (for COM velocity)

    Returns:
        Particles set containing planetesimals
    """
    np.random.seed(config.random_seed + 1)  # Different seed from planets

    n = config.n_planetesimals

    # Choose disk extent based on simulation type
    if config.sim_type == 'source_tracking':
        a_min = config.disk_a_min
        a_max = config.disk_a_max
    else:
        a_min = config.planetesimal_a_min
        a_max = config.planetesimal_a_max

    e_max = config.planetesimal_e_max
    i_max = config.planetesimal_i_max

    # Get Sun for orbital calculations
    sun = massive_bodies[0]
    star_mass = sun.mass

    planetesimals = Particles(n)
    planetesimals.mass = 0 | units.kg  # Test particles

    # Generate orbital elements
    # Semi-major axis: uniform in a (or could use a^(-1/2) for surface density)
    a_values = np.random.uniform(a_min, a_max, n) | units.AU

    # Eccentricity: Rayleigh distribution peaked at low e
    e_values = np.random.rayleigh(e_max / 2, n)
    e_values = np.clip(e_values, 0, e_max)

    # Inclination: Rayleigh distribution peaked at low i
    i_values = np.random.rayleigh(i_max / 2, n)
    i_values = np.clip(i_values, 0, i_max) | units.deg

    # Random angles (uniform)
    true_anomaly = np.random.uniform(0, 360, n) | units.deg
    arg_perihelion = np.random.uniform(0, 360, n) | units.deg
    long_asc_node = np.random.uniform(0, 360, n) | units.deg

    # Convert orbital elements to positions and velocities
    for i in range(n):
        binary = new_binary_from_orbital_elements(
            star_mass,
            0 | units.kg,  # Test particle
            a_values[i],
            e_values[i],
            true_anomaly=true_anomaly[i],
            inclination=i_values[i],
            longitude_of_the_ascending_node=long_asc_node[i],
            argument_of_periapsis=arg_perihelion[i]
        )
        planetesimals[i].position = binary[1].position
        planetesimals[i].velocity = binary[1].velocity

    # Adjust for center of mass motion
    com_velocity = massive_bodies.center_of_mass_velocity()
    planetesimals.velocity -= com_velocity

    return planetesimals


def create_trojan_disk(config: SimulationConfig,
                       massive_bodies: Particles,
                       n_l4: int = 5000,
                       n_l5: int = 5000) -> Particles:
    """
    Create planetesimals specifically in Jupiter's L4 and L5 Trojan regions.

    Useful for targeted injection after source tracking analysis.

    Args:
        config: Simulation configuration
        massive_bodies: The massive bodies (need Jupiter's position)
        n_l4: Number of particles near L4
        n_l5: Number of particles near L5

    Returns:
        Particles set containing Trojan-region planetesimals
    """
    np.random.seed(config.random_seed + 2)

    # Find Jupiter
    jupiter = None
    for body in massive_bodies:
        if hasattr(body, 'name') and body.name == 'Jupiter':
            jupiter = body
            break

    if jupiter is None:
        # Fall back to index 4 (typical position)
        jupiter = massive_bodies[4]

    sun = massive_bodies[0]
    star_mass = sun.mass

    # Jupiter's orbital elements
    jupiter_a = jupiter.position.length().value_in(units.AU)
    jupiter_angle = np.arctan2(
        jupiter.y.value_in(units.AU),
        jupiter.x.value_in(units.AU)
    )

    # L4 at +60°, L5 at -60° from Jupiter
    l4_angle = jupiter_angle + np.deg2rad(60)
    l5_angle = jupiter_angle - np.deg2rad(60)

    planetesimals = Particles(n_l4 + n_l5)
    planetesimals.mass = 0 | units.kg

    def generate_trojans(n, center_angle, jupiter_a):
        """Generate trojans near a Lagrange point."""
        # Semi-major axis: within 7% of Jupiter
        a = np.random.uniform(jupiter_a * 0.93, jupiter_a * 1.07, n)

        # Angular offset from L-point: ±25°
        angle_offset = np.random.uniform(-25, 25, n)
        angles = center_angle + np.deg2rad(angle_offset)

        # Small eccentricity and inclination
        e = np.random.rayleigh(0.05, n)
        e = np.clip(e, 0, 0.15)

        i = np.random.rayleigh(2, n)
        i = np.clip(i, 0, 10)

        return a, angles, e, i

    # Generate L4 trojans
    a_l4, angles_l4, e_l4, i_l4 = generate_trojans(n_l4, l4_angle, jupiter_a)

    # Generate L5 trojans
    a_l5, angles_l5, e_l5, i_l5 = generate_trojans(n_l5, l5_angle, jupiter_a)

    # Combine
    a_all = np.concatenate([a_l4, a_l5]) | units.AU
    angles_all = np.concatenate([angles_l4, angles_l5])
    e_all = np.concatenate([e_l4, e_l5])
    i_all = np.concatenate([i_l4, i_l5]) | units.deg

    # Convert to Cartesian (simplified: circular approximation for initial placement)
    for idx in range(n_l4 + n_l5):
        a = a_all[idx]
        angle = angles_all[idx]
        e = e_all[idx]
        inc = i_all[idx]

        # Use orbital elements for proper velocity
        binary = new_binary_from_orbital_elements(
            star_mass,
            0 | units.kg,
            a,
            e,
            true_anomaly=np.rad2deg(angle) | units.deg,
            inclination=inc,
            longitude_of_the_ascending_node=0 | units.deg,
            argument_of_periapsis=0 | units.deg
        )
        planetesimals[idx].position = binary[1].position
        planetesimals[idx].velocity = binary[1].velocity

    # Adjust for COM
    com_velocity = massive_bodies.center_of_mass_velocity()
    planetesimals.velocity -= com_velocity

    return planetesimals
