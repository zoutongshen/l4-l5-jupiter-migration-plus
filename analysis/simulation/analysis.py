"""
Analysis functions for Jupiter Trojan simulations.

- Trojan counting (L4/L5 populations)
- Orbital element calculations
- Population statistics
"""

import numpy as np
from amuse.units import units, constants
from typing import Tuple, Dict, List, Optional

from .config import TrojanCriteria


def compute_trojan_counts(massive_bodies,
                          planetesimals,
                          jupiter_index: int = 4,
                          criteria: Optional[TrojanCriteria] = None) -> Tuple[int, int]:
    """
    Count planetesimals in Jupiter's L4 and L5 Trojan regions.

    Args:
        massive_bodies: Particles set with Sun and planets
        planetesimals: Particles set with test particles
        jupiter_index: Index of Jupiter in massive_bodies
        criteria: Trojan detection criteria (uses defaults if None)

    Returns:
        Tuple of (n_L4, n_L5)
    """
    if criteria is None:
        criteria = TrojanCriteria()

    jupiter = massive_bodies[jupiter_index]

    # Jupiter's position in heliocentric frame
    jupiter_x = jupiter.x.value_in(units.AU)
    jupiter_y = jupiter.y.value_in(units.AU)
    jupiter_z = jupiter.z.value_in(units.AU)

    lambda_j = np.arctan2(jupiter_y, jupiter_x)
    a_j = np.sqrt(jupiter_x ** 2 + jupiter_y ** 2 + jupiter_z ** 2)

    # Vectorized computation for all planetesimals
    p_x = np.array([p.x.value_in(units.AU) for p in planetesimals])
    p_y = np.array([p.y.value_in(units.AU) for p in planetesimals])
    p_z = np.array([p.z.value_in(units.AU) for p in planetesimals])

    lam = np.arctan2(p_y, p_x)
    r = np.sqrt(p_x ** 2 + p_y ** 2 + p_z ** 2)

    # Angular separation from Jupiter (wrapped to [-pi, pi])
    dlam = np.arctan2(np.sin(lam - lambda_j), np.cos(lam - lambda_j))

    # Co-orbital criterion
    is_coorb = np.abs(r / a_j - 1) < criteria.coorbital_tolerance

    # L4: +60° from Jupiter (within sector_half_width)
    l4_min = np.deg2rad(criteria.l4_angle - criteria.sector_half_width)
    l4_max = np.deg2rad(criteria.l4_angle + criteria.sector_half_width)
    is_L4 = (dlam > l4_min) & (dlam < l4_max) & is_coorb

    # L5: -60° from Jupiter (within sector_half_width)
    l5_min = np.deg2rad(criteria.l5_angle - criteria.sector_half_width)
    l5_max = np.deg2rad(criteria.l5_angle + criteria.sector_half_width)
    is_L5 = (dlam > l5_min) & (dlam < l5_max) & is_coorb

    return int(is_L4.sum()), int(is_L5.sum())


def compute_orbital_elements(body, sun) -> Dict:
    """
    Compute orbital elements for a body relative to the Sun.

    Args:
        body: The orbiting body (planet or particle)
        sun: The central body (Sun)

    Returns:
        Dictionary with orbital elements:
        - a: semi-major axis (AU)
        - e: eccentricity
        - i: inclination (degrees)
        - E: specific orbital energy (km^2/s^2)
    """
    r_vec = body.position - sun.position
    v_vec = body.velocity - sun.velocity

    r = r_vec.length()
    v = v_vec.length()

    mu = constants.G * (sun.mass + body.mass)

    # Specific orbital energy
    E = 0.5 * v * v - mu / r

    # Semi-major axis from energy
    a = -mu / (2 * E)

    # Angular momentum
    h_vec = r_vec.cross(v_vec)
    h = h_vec.length()

    # Eccentricity from energy and angular momentum
    e_sq = 1 + (2 * E * h * h) / (mu * mu)

    # Handle unit extraction
    if hasattr(e_sq, "value_in"):
        e_sq_val = e_sq.value_in(1)
    elif hasattr(e_sq, "number"):
        e_sq_val = e_sq.number
    else:
        e_sq_val = float(e_sq)

    e = np.sqrt(max(e_sq_val, 0.0))

    # Inclination
    h_z = h_vec[2]
    i_rad = np.arccos(h_z / h)

    return {
        'a': a.value_in(units.AU),
        'e': e,
        'i': np.degrees(i_rad.value_in(1) if hasattr(i_rad, 'value_in') else i_rad),
        'E': E.value_in(units.km ** 2 / units.s ** 2)
    }


def compute_jupiter_orbit(massive_bodies, jupiter_index: int = 4) -> Dict:
    """
    Compute Jupiter's orbital elements.

    Args:
        massive_bodies: Particles set with Sun and planets
        jupiter_index: Index of Jupiter

    Returns:
        Dictionary with Jupiter's orbital elements
    """
    sun = massive_bodies[0]
    jupiter = massive_bodies[jupiter_index]
    return compute_orbital_elements(jupiter, sun)


def classify_planetesimals(massive_bodies,
                           planetesimals,
                           jupiter_index: int = 4,
                           criteria: Optional[TrojanCriteria] = None) -> Dict[str, List[int]]:
    """
    Classify planetesimals by their location relative to Jupiter.

    Args:
        massive_bodies: Particles set with Sun and planets
        planetesimals: Particles set with test particles
        jupiter_index: Index of Jupiter
        criteria: Trojan detection criteria

    Returns:
        Dictionary mapping region names to lists of particle indices:
        - 'L4': L4 Trojans
        - 'L5': L5 Trojans
        - 'inner': Inside Jupiter's orbit
        - 'outer': Outside Jupiter's orbit
        - 'coorbital_other': Co-orbital but not in L4/L5
    """
    if criteria is None:
        criteria = TrojanCriteria()

    jupiter = massive_bodies[jupiter_index]
    jupiter_x = jupiter.x.value_in(units.AU)
    jupiter_y = jupiter.y.value_in(units.AU)
    jupiter_z = jupiter.z.value_in(units.AU)

    lambda_j = np.arctan2(jupiter_y, jupiter_x)
    a_j = np.sqrt(jupiter_x ** 2 + jupiter_y ** 2 + jupiter_z ** 2)

    result = {
        'L4': [],
        'L5': [],
        'inner': [],
        'outer': [],
        'coorbital_other': []
    }

    for idx, p in enumerate(planetesimals):
        p_x = p.x.value_in(units.AU)
        p_y = p.y.value_in(units.AU)
        p_z = p.z.value_in(units.AU)

        r = np.sqrt(p_x ** 2 + p_y ** 2 + p_z ** 2)
        lam = np.arctan2(p_y, p_x)
        dlam = np.arctan2(np.sin(lam - lambda_j), np.cos(lam - lambda_j))

        is_coorb = abs(r / a_j - 1) < criteria.coorbital_tolerance

        l4_min = np.deg2rad(criteria.l4_angle - criteria.sector_half_width)
        l4_max = np.deg2rad(criteria.l4_angle + criteria.sector_half_width)
        l5_min = np.deg2rad(criteria.l5_angle - criteria.sector_half_width)
        l5_max = np.deg2rad(criteria.l5_angle + criteria.sector_half_width)

        if is_coorb:
            if l4_min < dlam < l4_max:
                result['L4'].append(idx)
            elif l5_min < dlam < l5_max:
                result['L5'].append(idx)
            else:
                result['coorbital_other'].append(idx)
        elif r < a_j * (1 - criteria.coorbital_tolerance):
            result['inner'].append(idx)
        else:
            result['outer'].append(idx)

    return result


def compute_l4_l5_ratio_slope(times: List[float],
                              l4_counts: List[int],
                              l5_counts: List[int]) -> Tuple[float, float]:
    """
    Compute linear slope of L4/L5 ratio over time.

    Args:
        times: List of time values (years)
        l4_counts: List of L4 counts
        l5_counts: List of L5 counts

    Returns:
        Tuple of (slope, intercept) for L4/L5 ratio vs time
    """
    times_array = np.array(times)
    l4_array = np.array(l4_counts)
    l5_array = np.array(l5_counts)

    # Avoid division by zero
    ratio = l4_array / (l5_array + 1e-10)

    # Linear fit
    slope, intercept = np.polyfit(times_array, ratio, 1)

    return slope, intercept


def compute_population_summary(times: List[float],
                               l4_counts: List[int],
                               l5_counts: List[int]) -> Dict:
    """
    Compute summary statistics for Trojan population evolution.

    Args:
        times: List of time values
        l4_counts: List of L4 counts
        l5_counts: List of L5 counts

    Returns:
        Dictionary with summary statistics
    """
    slope, intercept = compute_l4_l5_ratio_slope(times, l4_counts, l5_counts)

    initial_total = l4_counts[0] + l5_counts[0]
    final_total = l4_counts[-1] + l5_counts[-1]

    return {
        'initial_l4': l4_counts[0],
        'initial_l5': l5_counts[0],
        'initial_total': initial_total,
        'final_l4': l4_counts[-1],
        'final_l5': l5_counts[-1],
        'final_total': final_total,
        'initial_ratio': l4_counts[0] / max(l5_counts[0], 1),
        'final_ratio': l4_counts[-1] / max(l5_counts[-1], 1),
        'ratio_slope': slope,
        'ratio_intercept': intercept,
        'retention_percent': 100 * final_total / max(initial_total, 1),
        'time_span': times[-1] - times[0]
    }
