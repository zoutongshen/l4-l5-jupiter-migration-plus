"""
Analysis functions for Jupiter Trojan simulations.

These operate on plain NumPy arrays (positions in AU, velocities in AU/yr,
masses in Msun) so they work identically during a live run and on the saved
snapshot arrays. Jupiter is assumed to be at index 4 of the massive set.

- Trojan counting (L4/L5 populations)
- Orbital element calculations
- Population statistics
"""

import numpy as np
from typing import Tuple, Dict, List, Optional

from .config import TrojanCriteria

# Gravitational constant in AU^3 / (Msun * yr^2)
G = 4.0 * np.pi ** 2


def _jupiter_angle_and_a(massive_pos, jupiter_index):
    """Return Jupiter's heliocentric longitude (rad) and distance (AU)."""
    jx, jy, jz = massive_pos[jupiter_index]
    lambda_j = np.arctan2(jy, jx)
    a_j = np.sqrt(jx ** 2 + jy ** 2 + jz ** 2)
    return lambda_j, a_j


def compute_trojan_counts(massive_pos,
                          planetesimal_pos,
                          jupiter_index: int = 4,
                          criteria: Optional[TrojanCriteria] = None) -> Tuple[int, int]:
    """
    Count planetesimals in Jupiter's L4 and L5 Trojan regions.

    Args:
        massive_pos: (n_massive, 3) array of massive-body positions (AU).
        planetesimal_pos: (n_p, 3) array of planetesimal positions (AU).
        jupiter_index: index of Jupiter in massive_pos.
        criteria: Trojan detection criteria (defaults if None).

    Returns:
        (n_L4, n_L5)
    """
    if criteria is None:
        criteria = TrojanCriteria()

    massive_pos = np.asarray(massive_pos)
    planetesimal_pos = np.asarray(planetesimal_pos)
    if planetesimal_pos.size == 0:
        return 0, 0

    lambda_j, a_j = _jupiter_angle_and_a(massive_pos, jupiter_index)

    p_x = planetesimal_pos[:, 0]
    p_y = planetesimal_pos[:, 1]
    p_z = planetesimal_pos[:, 2]

    lam = np.arctan2(p_y, p_x)
    r = np.sqrt(p_x ** 2 + p_y ** 2 + p_z ** 2)

    dlam = np.arctan2(np.sin(lam - lambda_j), np.cos(lam - lambda_j))
    is_coorb = np.abs(r / a_j - 1) < criteria.coorbital_tolerance

    l4_min = np.deg2rad(criteria.l4_angle - criteria.sector_half_width)
    l4_max = np.deg2rad(criteria.l4_angle + criteria.sector_half_width)
    is_L4 = (dlam > l4_min) & (dlam < l4_max) & is_coorb

    l5_min = np.deg2rad(criteria.l5_angle - criteria.sector_half_width)
    l5_max = np.deg2rad(criteria.l5_angle + criteria.sector_half_width)
    is_L5 = (dlam > l5_min) & (dlam < l5_max) & is_coorb

    return int(is_L4.sum()), int(is_L5.sum())


def compute_orbital_elements(rel_pos, rel_vel, mu) -> Dict:
    """
    Compute orbital elements from a state vector relative to the primary.

    Args:
        rel_pos: position relative to the primary (AU), length-3.
        rel_vel: velocity relative to the primary (AU/yr), length-3.
        mu: G*(M_primary + m_body) in AU^3/yr^2.

    Returns:
        dict with a (AU), e, i (deg), E (AU^2/yr^2 specific energy).
    """
    rel_pos = np.asarray(rel_pos, dtype=float)
    rel_vel = np.asarray(rel_vel, dtype=float)

    r = np.linalg.norm(rel_pos)
    v = np.linalg.norm(rel_vel)

    E = 0.5 * v * v - mu / r
    a = -mu / (2 * E)

    h_vec = np.cross(rel_pos, rel_vel)
    h = np.linalg.norm(h_vec)

    e_sq = 1 + (2 * E * h * h) / (mu * mu)
    e = np.sqrt(max(e_sq, 0.0))

    i_rad = np.arccos(h_vec[2] / h) if h > 0 else 0.0

    return {'a': a, 'e': e, 'i': np.degrees(i_rad), 'E': E}


def compute_jupiter_orbit(massive_pos, massive_vel, massive_mass,
                          jupiter_index: int = 4) -> Dict:
    """Compute Jupiter's orbital elements relative to the Sun (index 0)."""
    massive_pos = np.asarray(massive_pos)
    massive_vel = np.asarray(massive_vel)
    rel_pos = massive_pos[jupiter_index] - massive_pos[0]
    rel_vel = massive_vel[jupiter_index] - massive_vel[0]
    mu = G * (massive_mass[0] + massive_mass[jupiter_index])
    return compute_orbital_elements(rel_pos, rel_vel, mu)


def classify_planetesimals(massive_pos,
                           planetesimal_pos,
                           jupiter_index: int = 4,
                           criteria: Optional[TrojanCriteria] = None) -> Dict[str, List[int]]:
    """
    Classify planetesimals by location relative to Jupiter.

    Returns a dict mapping region names to lists of particle indices:
    'L4', 'L5', 'inner', 'outer', 'coorbital_other'.
    """
    if criteria is None:
        criteria = TrojanCriteria()

    massive_pos = np.asarray(massive_pos)
    planetesimal_pos = np.asarray(planetesimal_pos)

    lambda_j, a_j = _jupiter_angle_and_a(massive_pos, jupiter_index)

    result = {'L4': [], 'L5': [], 'inner': [], 'outer': [], 'coorbital_other': []}
    if planetesimal_pos.size == 0:
        return result

    p_x = planetesimal_pos[:, 0]
    p_y = planetesimal_pos[:, 1]
    p_z = planetesimal_pos[:, 2]

    r = np.sqrt(p_x ** 2 + p_y ** 2 + p_z ** 2)
    lam = np.arctan2(p_y, p_x)
    dlam = np.arctan2(np.sin(lam - lambda_j), np.cos(lam - lambda_j))
    is_coorb = np.abs(r / a_j - 1) < criteria.coorbital_tolerance

    l4_min = np.deg2rad(criteria.l4_angle - criteria.sector_half_width)
    l4_max = np.deg2rad(criteria.l4_angle + criteria.sector_half_width)
    l5_min = np.deg2rad(criteria.l5_angle - criteria.sector_half_width)
    l5_max = np.deg2rad(criteria.l5_angle + criteria.sector_half_width)
    inner_edge = a_j * (1 - criteria.coorbital_tolerance)

    for idx in range(len(r)):
        if is_coorb[idx]:
            if l4_min < dlam[idx] < l4_max:
                result['L4'].append(idx)
            elif l5_min < dlam[idx] < l5_max:
                result['L5'].append(idx)
            else:
                result['coorbital_other'].append(idx)
        elif r[idx] < inner_edge:
            result['inner'].append(idx)
        else:
            result['outer'].append(idx)

    return result


def compute_l4_l5_ratio_slope(times: List[float],
                              l4_counts: List[int],
                              l5_counts: List[int]) -> Tuple[float, float]:
    """Linear slope (and intercept) of the L4/L5 ratio vs time."""
    times_array = np.asarray(times, dtype=float)
    l4_array = np.asarray(l4_counts, dtype=float)
    l5_array = np.asarray(l5_counts, dtype=float)

    ratio = l4_array / (l5_array + 1e-10)
    slope, intercept = np.polyfit(times_array, ratio, 1)
    return slope, intercept


def compute_population_summary(times: List[float],
                               l4_counts: List[int],
                               l5_counts: List[int]) -> Dict:
    """Summary statistics for Trojan population evolution."""
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
        'time_span': times[-1] - times[0],
    }
