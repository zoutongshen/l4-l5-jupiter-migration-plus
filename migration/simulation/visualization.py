"""
Visualization functions for Jupiter Trojan simulations.

All functions operate on NumPy position arrays (AU):
- massive_pos:      (n_massive, 3)
- planetesimal_pos: (n_p, 3)
plus a `names` list for the massive bodies. This matches the saved snapshot
format, so plotting works the same live or from a loaded pickle.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import List, Optional, Tuple

from .config import TrojanCriteria
from .analysis import compute_trojan_counts


PLANET_COLORS = {
    'Sun': 'yellow', 'Venus': 'orange', 'Earth': 'blue', 'Mars': 'red',
    'Jupiter': 'brown', 'Saturn': 'gold', 'Uranus': 'cyan', 'Neptune': 'darkblue',
}

PLANET_SIZES = {
    'Sun': 15, 'Venus': 6, 'Earth': 7, 'Mars': 5,
    'Jupiter': 12, 'Saturn': 10, 'Uranus': 8, 'Neptune': 8,
}


def plot_snapshot(massive_pos,
                  names,
                  planetesimal_pos,
                  time: float,
                  jupiter_index: int = 4,
                  show_trojan_zones: bool = True,
                  criteria: Optional[TrojanCriteria] = None,
                  ax: Optional[plt.Axes] = None,
                  xlim: Tuple[float, float] = (-8, 8),
                  ylim: Tuple[float, float] = (-8, 8)) -> plt.Figure:
    """Plot a snapshot of the solar system in the XY plane."""
    if criteria is None:
        criteria = TrojanCriteria()

    massive_pos = np.asarray(massive_pos)
    planetesimal_pos = np.asarray(planetesimal_pos)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure

    jx, jy = massive_pos[jupiter_index, 0], massive_pos[jupiter_index, 1]
    jupiter_r = np.sqrt(jx ** 2 + jy ** 2)
    jupiter_angle = np.arctan2(jy, jx)

    if show_trojan_zones:
        _draw_trojan_zones(ax, jupiter_r, jupiter_angle, criteria)

    ax.plot(0, 0, 'o', color='yellow', markersize=15,
            markeredgecolor='orange', markeredgewidth=2, label='Sun', zorder=10)

    for i in range(1, len(massive_pos)):
        name = names[i] if i < len(names) else 'Planet'
        ax.plot(massive_pos[i, 0], massive_pos[i, 1], 'o',
                color=PLANET_COLORS.get(name, 'gray'),
                markersize=PLANET_SIZES.get(name, 6), label=name, zorder=5)

    n_l4, n_l5 = compute_trojan_counts(massive_pos, planetesimal_pos,
                                       jupiter_index, criteria)

    if planetesimal_pos.size > 0:
        ax.plot(planetesimal_pos[:, 0], planetesimal_pos[:, 1], 'k.',
                markersize=1, alpha=0.3,
                label=f'Planetesimals (n={len(planetesimal_pos)})')

    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(jupiter_r * np.cos(theta), jupiter_r * np.sin(theta),
            'b--', alpha=0.3, linewidth=1)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('X (AU)', fontsize=12)
    ax.set_ylabel('Y (AU)', fontsize=12)
    ax.set_title(f't = {time:.1f} yr | L4={n_l4}, L5={n_l5}', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=8, ncol=2)

    return fig


def _draw_trojan_zones(ax, jupiter_r, jupiter_angle, criteria: TrojanCriteria):
    """Draw L4 and L5 Trojan zones on the axes."""
    sector_width = np.deg2rad(criteria.sector_half_width)
    r_inner = jupiter_r * (1 - criteria.coorbital_tolerance)
    r_outer = jupiter_r * (1 + criteria.coorbital_tolerance)

    for l_angle, color, label in [(criteria.l4_angle, 'blue', 'L4 zone'),
                                  (criteria.l5_angle, 'red', 'L5 zone')]:
        center = jupiter_angle + np.deg2rad(l_angle)
        theta = np.linspace(center - sector_width, center + sector_width, 50)
        x_inner, y_inner = r_inner * np.cos(theta), r_inner * np.sin(theta)
        x_outer, y_outer = r_outer * np.cos(theta), r_outer * np.sin(theta)
        ax.fill(np.concatenate([x_inner, x_outer[::-1]]),
                np.concatenate([y_inner, y_outer[::-1]]),
                color=color, alpha=0.15, label=label)


def plot_trojan_evolution(times: List[float],
                          l4_counts: List[int],
                          l5_counts: List[int],
                          title: str = "Trojan Population Evolution",
                          figsize: Tuple[int, int] = (14, 5)) -> plt.Figure:
    """Two-panel plot: L4/L5 counts vs time, and L4/L5 ratio with linear fit."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    times_array = np.asarray(times, dtype=float)
    l4_array = np.asarray(l4_counts, dtype=float)
    l5_array = np.asarray(l5_counts, dtype=float)

    ax1.plot(times_array, l4_array, 'b-', label='L4', linewidth=2)
    ax1.plot(times_array, l5_array, 'r-', label='L5', linewidth=2)
    ax1.set_xlabel('Time (yr)', fontsize=12)
    ax1.set_ylabel('Number of Trojans', fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    ratio = l4_array / (l5_array + 1e-10)
    slope, intercept = np.polyfit(times_array, ratio, 1)

    ax2.scatter(times_array, ratio, c='purple', alpha=0.5, s=20, label='L4/L5 data')
    ax2.plot(times_array, slope * times_array + intercept, 'g-', linewidth=2,
             label=f'Linear fit (slope={slope:.2e} yr⁻¹)')
    ax2.axhline(1.0, color='k', linestyle='--', alpha=0.5, label='Equal (L4=L5)')
    ax2.set_xlabel('Time (yr)', fontsize=12)
    ax2.set_ylabel('L4/L5 Ratio', fontsize=12)
    ax2.set_title('L4/L5 Ratio with Linear Fit', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def create_animation(times: List[float],
                     massive_pos_series,
                     names,
                     planetesimal_pos_series,
                     output_file: str = 'animation.gif',
                     fps: int = 15,
                     dpi: int = 80,
                     jupiter_index: int = 4) -> str:
    """
    Create an animated GIF from simulation snapshots.

    Args:
        times: (n_snap,) time values.
        massive_pos_series: (n_snap, n_massive, 3) positions (AU).
        names: list of massive-body names.
        planetesimal_pos_series: (n_snap, n_p, 3) positions (AU).
    """
    print(f"Creating animation with {len(times)} frames...")
    massive_pos_series = np.asarray(massive_pos_series)
    planetesimal_pos_series = np.asarray(planetesimal_pos_series)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    def update(frame):
        ax1.clear()
        ax2.clear()

        massive = massive_pos_series[frame]
        planetes = planetesimal_pos_series[frame]
        t = times[frame]

        ax1.plot(0, 0, 'yo', markersize=15, label='Sun')
        for i in range(1, len(massive)):
            name = names[i] if i < len(names) else 'Planet'
            ax1.plot(massive[i, 0], massive[i, 1], 'o',
                     color=PLANET_COLORS.get(name, 'gray'),
                     markersize=PLANET_SIZES.get(name, 6), label=name)

        if planetes.size > 0:
            ax1.plot(planetes[:, 0], planetes[:, 1], 'k.', markersize=0.5, alpha=0.3)

        ax1.set_xlim([-35, 35])
        ax1.set_ylim([-35, 35])
        ax1.set_xlabel('X (AU)', fontsize=12)
        ax1.set_ylabel('Y (AU)', fontsize=12)
        ax1.set_title(f'Solar System (t = {t:.1f} yr)', fontsize=14)
        ax1.grid(alpha=0.3)
        ax1.set_aspect('equal')
        ax1.legend(loc='upper right', fontsize=8, ncol=2)

        for r in [1, 5, 10, 20, 30]:
            ax1.add_patch(plt.Circle((0, 0), r, fill=False, color='gray',
                                     linestyle='--', alpha=0.2, linewidth=0.5))

        if planetes.size > 0:
            radii = np.linalg.norm(planetes, axis=1)
            ax2.hist(radii, bins=60, range=(0, 35), color='steelblue',
                     alpha=0.7, edgecolor='black')

        r_jup = np.linalg.norm(massive[jupiter_index])
        ax2.axvline(r_jup, color='brown', linestyle='--', linewidth=2, label='Jupiter')
        ax2.set_xlabel('Distance from Sun (AU)', fontsize=12)
        ax2.set_ylabel('Number of Planetesimals', fontsize=12)
        ax2.set_title('Radial Distribution', fontsize=14)
        ax2.set_xlim([0, 35])
        ax2.grid(alpha=0.3, axis='y')
        ax2.legend(fontsize=10)

        return ax1, ax2

    anim = FuncAnimation(fig, update, frames=len(times), interval=1000 / fps, blit=False)

    print(f"Saving to {output_file}...")
    anim.save(output_file, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print(f"Animation saved: {output_file} ({len(times)} frames, {len(times) / fps:.1f}s)")
    return output_file
