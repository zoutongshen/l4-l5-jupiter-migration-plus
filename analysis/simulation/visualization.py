"""
Visualization functions for Jupiter Trojan simulations.

- Snapshot plotting (XY plane, radial distribution)
- Population evolution plots
- Animation creation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from amuse.units import units
from typing import List, Optional, Tuple

from .config import TrojanCriteria
from .analysis import compute_trojan_counts


# Planet colors and sizes for visualization
PLANET_COLORS = {
    'Sun': 'yellow',
    'Venus': 'orange',
    'Earth': 'blue',
    'Mars': 'red',
    'Jupiter': 'brown',
    'Saturn': 'gold',
    'Uranus': 'cyan',
    'Neptune': 'darkblue'
}

PLANET_SIZES = {
    'Sun': 15,
    'Venus': 6,
    'Earth': 7,
    'Mars': 5,
    'Jupiter': 12,
    'Saturn': 10,
    'Uranus': 8,
    'Neptune': 8
}


def plot_snapshot(massive_bodies,
                  planetesimals,
                  time: float,
                  jupiter_index: int = 4,
                  show_trojan_zones: bool = True,
                  criteria: Optional[TrojanCriteria] = None,
                  ax: Optional[plt.Axes] = None,
                  xlim: Tuple[float, float] = (-8, 8),
                  ylim: Tuple[float, float] = (-8, 8)) -> plt.Figure:
    """
    Plot a snapshot of the solar system in the XY plane.

    Args:
        massive_bodies: Particles set with Sun and planets
        planetesimals: Particles set with test particles
        time: Current simulation time (years)
        jupiter_index: Index of Jupiter
        show_trojan_zones: Whether to highlight L4/L5 zones
        criteria: Trojan detection criteria
        ax: Matplotlib axes (creates new figure if None)
        xlim: X-axis limits
        ylim: Y-axis limits

    Returns:
        Matplotlib figure
    """
    if criteria is None:
        criteria = TrojanCriteria()

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure

    # Get Jupiter position for Trojan zone plotting
    jupiter = massive_bodies[jupiter_index]
    jupiter_x = jupiter.x.value_in(units.AU)
    jupiter_y = jupiter.y.value_in(units.AU)
    jupiter_r = np.sqrt(jupiter_x ** 2 + jupiter_y ** 2)
    jupiter_angle = np.arctan2(jupiter_y, jupiter_x)

    # Draw Trojan zones if requested
    if show_trojan_zones:
        _draw_trojan_zones(ax, jupiter_r, jupiter_angle, criteria)

    # Plot Sun
    ax.plot(0, 0, 'o', color='yellow', markersize=15,
            markeredgecolor='orange', markeredgewidth=2, label='Sun', zorder=10)

    # Plot planets
    for body in massive_bodies[1:]:
        name = getattr(body, 'name', 'Planet')
        x = body.x.value_in(units.AU)
        y = body.y.value_in(units.AU)
        color = PLANET_COLORS.get(name, 'gray')
        size = PLANET_SIZES.get(name, 6)
        ax.plot(x, y, 'o', color=color, markersize=size, label=name, zorder=5)

    # Plot planetesimals (classify by region)
    n_l4, n_l5 = compute_trojan_counts(massive_bodies, planetesimals,
                                        jupiter_index, criteria)

    # Get all planetesimal positions
    p_x = np.array([p.x.value_in(units.AU) for p in planetesimals])
    p_y = np.array([p.y.value_in(units.AU) for p in planetesimals])

    # Plot all planetesimals as small dots
    ax.plot(p_x, p_y, 'k.', markersize=1, alpha=0.3, label=f'Planetesimals (n={len(planetesimals)})')

    # Draw Jupiter's orbit
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(jupiter_r * np.cos(theta), jupiter_r * np.sin(theta),
            'b--', alpha=0.3, linewidth=1)

    # Formatting
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

    # L4 zone
    L4_angle = jupiter_angle + np.deg2rad(criteria.l4_angle)
    theta_L4 = np.linspace(L4_angle - sector_width, L4_angle + sector_width, 50)
    x_inner = r_inner * np.cos(theta_L4)
    y_inner = r_inner * np.sin(theta_L4)
    x_outer = r_outer * np.cos(theta_L4)
    y_outer = r_outer * np.sin(theta_L4)
    ax.fill(np.concatenate([x_inner, x_outer[::-1]]),
            np.concatenate([y_inner, y_outer[::-1]]),
            color='blue', alpha=0.15, label='L4 zone')

    # L5 zone
    L5_angle = jupiter_angle + np.deg2rad(criteria.l5_angle)
    theta_L5 = np.linspace(L5_angle - sector_width, L5_angle + sector_width, 50)
    x_inner = r_inner * np.cos(theta_L5)
    y_inner = r_inner * np.sin(theta_L5)
    x_outer = r_outer * np.cos(theta_L5)
    y_outer = r_outer * np.sin(theta_L5)
    ax.fill(np.concatenate([x_inner, x_outer[::-1]]),
            np.concatenate([y_inner, y_outer[::-1]]),
            color='red', alpha=0.15, label='L5 zone')


def plot_trojan_evolution(times: List[float],
                          l4_counts: List[int],
                          l5_counts: List[int],
                          title: str = "Trojan Population Evolution",
                          figsize: Tuple[int, int] = (14, 5)) -> plt.Figure:
    """
    Plot L4/L5 Trojan population evolution over time.

    Creates two panels:
    - Left: L4 and L5 counts vs time
    - Right: L4/L5 ratio with linear fit

    Args:
        times: List of time values (years)
        l4_counts: List of L4 counts
        l5_counts: List of L5 counts
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    times_array = np.array(times)
    l4_array = np.array(l4_counts)
    l5_array = np.array(l5_counts)

    # Left panel: L4 and L5 counts
    ax1.plot(times_array, l4_array, 'b-', label='L4', linewidth=2)
    ax1.plot(times_array, l5_array, 'r-', label='L5', linewidth=2)
    ax1.set_xlabel('Time (yr)', fontsize=12)
    ax1.set_ylabel('Number of Trojans', fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)

    # Right panel: L4/L5 ratio with fitted slope
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


def plot_radial_distribution(planetesimals,
                             jupiter_r: float,
                             time: float,
                             ax: Optional[plt.Axes] = None,
                             bins: int = 60,
                             range_au: Tuple[float, float] = (0, 35)) -> plt.Figure:
    """
    Plot radial distribution of planetesimals.

    Args:
        planetesimals: Particles set
        jupiter_r: Jupiter's current distance from Sun (AU)
        time: Current time (years)
        ax: Matplotlib axes
        bins: Number of histogram bins
        range_au: Radial range for histogram

    Returns:
        Matplotlib figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    radii = [p.position.length().value_in(units.AU) for p in planetesimals]

    ax.hist(radii, bins=bins, range=range_au, color='steelblue',
            alpha=0.7, edgecolor='black')
    ax.axvline(jupiter_r, color='brown', linestyle='--', linewidth=2, label='Jupiter')

    ax.set_xlabel('Distance from Sun (AU)', fontsize=12)
    ax.set_ylabel('Number of Planetesimals', fontsize=12)
    ax.set_title(f't = {time:.1f} yr | N = {len(planetesimals)}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')

    return fig


def create_animation(times: List[float],
                     snapshots_massive: List,
                     snapshots_planetesimals: List,
                     output_file: str = 'animation.gif',
                     fps: int = 15,
                     dpi: int = 80,
                     jupiter_index: int = 4) -> str:
    """
    Create animated GIF from simulation snapshots.

    Args:
        times: List of time values
        snapshots_massive: List of massive body snapshots
        snapshots_planetesimals: List of planetesimal snapshots
        output_file: Output GIF filename
        fps: Frames per second
        dpi: Resolution
        jupiter_index: Index of Jupiter

    Returns:
        Path to saved GIF
    """
    print(f"Creating animation with {len(times)} frames...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    def update(frame):
        ax1.clear()
        ax2.clear()

        massive = snapshots_massive[frame]
        planetes = snapshots_planetesimals[frame]
        t = times[frame]

        # Left panel: XY view
        ax1.plot(0, 0, 'yo', markersize=15, label='Sun')

        for body in massive[1:]:
            name = getattr(body, 'name', 'Planet')
            x = body.x.value_in(units.AU)
            y = body.y.value_in(units.AU)
            color = PLANET_COLORS.get(name, 'gray')
            size = PLANET_SIZES.get(name, 6)
            ax1.plot(x, y, 'o', color=color, markersize=size, label=name)

        p_x = [p.x.value_in(units.AU) for p in planetes]
        p_y = [p.y.value_in(units.AU) for p in planetes]
        ax1.plot(p_x, p_y, 'k.', markersize=0.5, alpha=0.3)

        ax1.set_xlim([-35, 35])
        ax1.set_ylim([-35, 35])
        ax1.set_xlabel('X (AU)', fontsize=12)
        ax1.set_ylabel('Y (AU)', fontsize=12)
        ax1.set_title(f'Solar System (t = {t:.1f} yr)', fontsize=14)
        ax1.grid(alpha=0.3)
        ax1.set_aspect('equal')
        ax1.legend(loc='upper right', fontsize=8, ncol=2)

        # Orbit circles
        for r in [1, 5, 10, 20, 30]:
            circle = plt.Circle((0, 0), r, fill=False, color='gray',
                                 linestyle='--', alpha=0.2, linewidth=0.5)
            ax1.add_patch(circle)

        # Right panel: Radial distribution
        radii = [p.position.length().value_in(units.AU) for p in planetes]
        ax2.hist(radii, bins=60, range=(0, 35), color='steelblue',
                 alpha=0.7, edgecolor='black')

        jupiter = massive[jupiter_index]
        r_jup = jupiter.position.length().value_in(units.AU)
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
    writer = PillowWriter(fps=fps)
    anim.save(output_file, writer=writer, dpi=dpi)

    plt.close(fig)
    print(f"Animation saved: {output_file} ({len(times)} frames, {len(times) / fps:.1f}s)")

    return output_file
