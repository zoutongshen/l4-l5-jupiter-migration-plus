"""
L4/L5 Jupiter Migration Simulation Module

This module provides shared code for all Jupiter Trojan simulations:
- Burn-in: Relaxation of artificially injected particles
- Source tracking: Pure disk simulations to identify Trojan origins
- Migration: Jupiter migration simulations (inward/outward)

Usage:
    from simulation import SimulationConfig, create_solar_system, run_simulation
"""

from .config import SimulationConfig, MigrationConfig
from .bodies import create_massive_bodies, create_planetesimals, create_trojan_disk
from .integrators import PlanetesimalIntegrator, ExternalGravityField, JupiterMigrationCode
from .analysis import compute_trojan_counts, compute_orbital_elements
from .visualization import plot_snapshot, plot_trojan_evolution, create_animation
from .runner import run_simulation, load_checkpoint, save_checkpoint

__all__ = [
    # Config
    'SimulationConfig',
    'MigrationConfig',
    # Bodies
    'create_massive_bodies',
    'create_planetesimals',
    'create_trojan_disk',
    # Integrators
    'PlanetesimalIntegrator',
    'ExternalGravityField',
    'JupiterMigrationCode',
    # Analysis
    'compute_trojan_counts',
    'compute_orbital_elements',
    # Visualization
    'plot_snapshot',
    'plot_trojan_evolution',
    'create_animation',
    # Runner
    'run_simulation',
    'load_checkpoint',
    'save_checkpoint',
]
