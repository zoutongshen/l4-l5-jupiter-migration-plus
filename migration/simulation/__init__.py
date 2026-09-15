"""
L4/L5 Jupiter Migration Simulation Module (REBOUND + REBOUNDx)

Shared code for all Jupiter Trojan simulations:
- Burn-in: relaxation of artificially injected particles (fixed Jupiter)
- Source tracking: pure-disk simulations to identify Trojan origins
- Migration: Jupiter migration simulations (inward/outward)

The engine is REBOUND (WHFast for the Sun + planets, massless test particles
for planetesimals) with REBOUNDx `modify_orbits_forces` providing Jupiter's
migration + eccentricity damping.

Usage:
    from simulation import SimulationConfig, run_simulation
"""

from .config import SimulationConfig, MigrationConfig, TrojanCriteria, DEFAULT_PLANET_DATA
from .bodies import (
    create_solar_system,
    add_planetesimals,
    add_trojan_disk,
    sim_from_state,
)
from .migration import attach_migration, MigrationDriver
from .analysis import (
    compute_trojan_counts,
    compute_orbital_elements,
    compute_population_summary,
    compute_l4_l5_ratio_slope,
    classify_planetesimals,
    compute_jupiter_orbit,
)
from .visualization import plot_snapshot, plot_trojan_evolution, create_animation
from .runner import run_simulation, load_checkpoint, save_checkpoint

__all__ = [
    # Config
    'SimulationConfig',
    'MigrationConfig',
    'TrojanCriteria',
    'DEFAULT_PLANET_DATA',
    # Bodies
    'create_solar_system',
    'add_planetesimals',
    'add_trojan_disk',
    'sim_from_state',
    # Migration
    'attach_migration',
    'MigrationDriver',
    # Analysis
    'compute_trojan_counts',
    'compute_orbital_elements',
    'compute_population_summary',
    'compute_l4_l5_ratio_slope',
    'classify_planetesimals',
    'compute_jupiter_orbit',
    # Visualization
    'plot_snapshot',
    'plot_trojan_evolution',
    'create_animation',
    # Runner
    'run_simulation',
    'load_checkpoint',
    'save_checkpoint',
]
