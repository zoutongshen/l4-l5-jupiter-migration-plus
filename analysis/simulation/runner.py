"""
Main simulation runner for Jupiter Trojan simulations.

Handles:
- Setting up integrators
- Running the simulation loop
- Checkpointing and resuming
- Saving results
"""

import pickle
from datetime import datetime
from typing import Optional, Dict, Tuple, List

from amuse.units import units, nbody_system
from amuse.community.huayno.interface import Huayno
from amuse.couple import bridge

from .config import SimulationConfig
from .bodies import create_massive_bodies, create_planetesimals
from .integrators import PlanetesimalIntegrator, ExternalGravityField, JupiterMigrationCode
from .analysis import compute_trojan_counts


def load_checkpoint(filename: str) -> Dict:
    """
    Load simulation state from a checkpoint file.

    Args:
        filename: Path to pickle file

    Returns:
        Dictionary containing simulation state
    """
    print(f"Loading checkpoint from {filename}...")
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print(f"  Time range: {data['times'][0]:.1f} - {data['times'][-1]:.1f} yr")
    print(f"  Snapshots: {len(data['times'])}")
    return data


def save_checkpoint(filename: str,
                    times: List[float],
                    snapshots_massive: List,
                    snapshots_planetesimals: List,
                    l4_counts: List[int],
                    l5_counts: List[int],
                    config: SimulationConfig) -> None:
    """
    Save simulation state to a checkpoint file.

    Args:
        filename: Output path
        times: List of time values
        snapshots_massive: List of massive body snapshots
        snapshots_planetesimals: List of planetesimal snapshots
        l4_counts: List of L4 counts
        l5_counts: List of L5 counts
        config: Simulation configuration
    """
    data = {
        'times': times,
        'snapshots_massive': snapshots_massive,
        'snapshots_planetesimals': snapshots_planetesimals,
        'l4_counts': l4_counts,
        'l5_counts': l5_counts,
        'config': {
            'name': config.name,
            'sim_type': config.sim_type,
            'end_time': config.end_time,
            'n_snapshots': config.n_snapshots,
            'n_planetesimals': config.n_planetesimals,
            'migration_enabled': config.migration.enabled,
            'migration_tau_a': config.migration.tau_a,
            'jupiter_initial_a': config.jupiter_initial_a,
        }
    }

    with open(filename, 'wb') as f:
        pickle.dump(data, f)

    print(f"Checkpoint saved to {filename}")


def setup_simulation(config: SimulationConfig,
                     massive_bodies=None,
                     planetesimals=None) -> Tuple:
    """
    Set up all simulation components.

    Args:
        config: Simulation configuration
        massive_bodies: Pre-existing massive bodies (for continuation)
        planetesimals: Pre-existing planetesimals (for continuation)

    Returns:
        Tuple of (massive_bodies, planetesimals, gravity_massive,
                  planetesimal_code, migration_bridge)
    """
    # Create or use provided bodies
    if massive_bodies is None:
        massive_bodies = create_massive_bodies(config)

    if planetesimals is None:
        planetesimals = create_planetesimals(config, massive_bodies)

    # Set up Huayno integrator for massive bodies
    converter = nbody_system.nbody_to_si(massive_bodies.total_mass(), 10 | units.AU)
    gravity_massive = Huayno(converter, number_of_workers=1, channel_type='sockets')
    gravity_massive.parameters.inttype_parameter = 13  # Symplectic
    gravity_massive.parameters.timestep_parameter = config.huayno_timestep
    gravity_massive.particles.add_particles(massive_bodies)

    # Set up planetesimal integrator
    gravity_field = ExternalGravityField(massive_bodies)
    planetesimal_code = PlanetesimalIntegrator(planetesimals, gravity_field)

    # Set up migration if enabled
    if config.migration.enabled:
        migration_code = JupiterMigrationCode(gravity_massive, massive_bodies, jupiter_index=4)
        migration_code.tau_a = config.migration.tau_a
        migration_code.tau_e = config.migration.tau_e

        # Create bridge for coupled evolution
        migration_bridge = bridge.Bridge(use_threading=False)
        migration_bridge.add_system(gravity_massive)
        migration_bridge.add_code(migration_code)
    else:
        migration_bridge = None

    return (massive_bodies, planetesimals, gravity_massive,
            planetesimal_code, gravity_field, migration_bridge)


def run_simulation(config: SimulationConfig,
                   verbose: bool = True) -> Dict:
    """
    Run a complete Jupiter Trojan simulation.

    Handles all three simulation types:
    - 'burnin': Relaxation with fixed Jupiter
    - 'source_tracking': Pure disk evolution
    - 'migration': Jupiter migration simulation

    Args:
        config: Simulation configuration
        verbose: Print progress updates

    Returns:
        Dictionary containing simulation results
    """
    # Load initial state if specified
    if config.input_file:
        input_data = load_checkpoint(config.input_file)
        massive_bodies = input_data['snapshots_massive'][-1].copy()
        planetesimals = input_data['snapshots_planetesimals'][-1].copy()
        t_initial = input_data['times'][-1]
        if verbose:
            print(f"Loaded initial state from t = {t_initial:.1f} yr")
    else:
        massive_bodies = None
        planetesimals = None
        t_initial = 0.0

    # Setup simulation components
    (massive_bodies, planetesimals, gravity_massive,
     planetesimal_code, gravity_field, migration_bridge) = setup_simulation(
        config, massive_bodies, planetesimals
    )

    # Simulation parameters
    end_time = config.end_time | units.yr
    dt = config.dt | units.yr
    n_snapshots = config.n_snapshots
    snapshot_interval = end_time / n_snapshots

    # Storage
    times = []
    snapshots_massive = []
    snapshots_planetesimals = []
    l4_counts = []
    l5_counts = []

    # Print simulation info
    if verbose:
        print("=" * 60)
        print(f"Running {config.sim_type} simulation: {config.name}")
        print("=" * 60)
        print(f"Duration: {config.end_time:.0f} years")
        print(f"Snapshots: {n_snapshots}")
        print(f"Massive bodies: {len(massive_bodies)}")
        print(f"Planetesimals: {len(planetesimals)}")
        if config.migration.enabled:
            print(f"Migration: tau_a = {config.migration.tau_a:.0f} yr")
        print("=" * 60)

    # Initialize timing
    time = 0 | units.yr
    next_snapshot = 0 | units.yr
    snapshot_count = 0
    start_time = datetime.now()

    # Save initial snapshot
    times.append(time.value_in(units.yr) + t_initial)
    snapshots_massive.append(massive_bodies.copy())
    snapshots_planetesimals.append(planetesimals.copy())
    n4, n5 = compute_trojan_counts(massive_bodies, planetesimals, criteria=config.trojan_criteria)
    l4_counts.append(n4)
    l5_counts.append(n5)

    if verbose:
        jupiter = massive_bodies[4]
        r_jup = jupiter.position.length().value_in(units.AU)
        print(f"Snapshot {snapshot_count}/{n_snapshots}: t={times[-1]:.1f} yr, "
              f"Jupiter r={r_jup:.3f} AU, L4={n4}, L5={n5}")

    snapshot_count += 1
    next_snapshot += snapshot_interval

    # Main simulation loop
    while time < end_time:
        # Evolve massive bodies
        if migration_bridge is not None:
            migration_bridge.evolve_model(time + dt)
        else:
            gravity_massive.evolve_model(time + dt)

        # Sync massive bodies
        channel_from_massive = gravity_massive.particles.new_channel_to(massive_bodies)
        channel_from_massive.copy()

        # Update gravity field reference
        gravity_field.massive_bodies = massive_bodies

        # Evolve planetesimals
        planetesimal_code.evolve_model(time + dt)

        time += dt

        # Save snapshot
        if time >= next_snapshot:
            planetesimal_code.sync_to_particles()

            times.append(time.value_in(units.yr) + t_initial)
            snapshots_massive.append(massive_bodies.copy())
            snapshots_planetesimals.append(planetesimals.copy())

            n4, n5 = compute_trojan_counts(massive_bodies, planetesimals,
                                           criteria=config.trojan_criteria)
            l4_counts.append(n4)
            l5_counts.append(n5)

            if verbose:
                jupiter = massive_bodies[4]
                r_jup = jupiter.position.length().value_in(units.AU)
                print(f"Snapshot {snapshot_count}/{n_snapshots}: t={times[-1]:.1f} yr, "
                      f"Jupiter r={r_jup:.3f} AU, L4={n4}, L5={n5}")

            snapshot_count += 1
            next_snapshot += snapshot_interval

            # Periodic checkpoint
            if snapshot_count % config.checkpoint_interval == 0:
                checkpoint_file = f"{config.name}_checkpoint.pkl"
                save_checkpoint(checkpoint_file, times, snapshots_massive,
                                snapshots_planetesimals, l4_counts, l5_counts, config)

    # Final sync
    planetesimal_code.sync_to_particles()

    # Stop integrator
    gravity_massive.stop()

    # Save final results
    save_checkpoint(config.output_file, times, snapshots_massive,
                    snapshots_planetesimals, l4_counts, l5_counts, config)

    # Print summary
    elapsed = (datetime.now() - start_time).total_seconds()
    if verbose:
        print("=" * 60)
        print(f"Simulation Complete!")
        print(f"  Time: {elapsed:.1f} seconds")
        print(f"  Snapshots: {len(times)}")
        print(f"  Final L4: {l4_counts[-1]}, L5: {l5_counts[-1]}")
        print(f"  Output: {config.output_file}")
        print("=" * 60)

    return {
        'times': times,
        'snapshots_massive': snapshots_massive,
        'snapshots_planetesimals': snapshots_planetesimals,
        'l4_counts': l4_counts,
        'l5_counts': l5_counts,
        'config': config,
        'elapsed_seconds': elapsed
    }
