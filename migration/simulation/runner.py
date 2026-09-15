"""
Main simulation runner for Jupiter Trojan simulations (REBOUND + REBOUNDx).

Handles:
- Building the simulation (fresh or continued from a checkpoint)
- Attaching migration physics
- The snapshot loop
- Checkpointing / resuming
- Saving results

Output / checkpoint format (pure NumPy, no REBOUND/AMUSE needed to load):
    {
        'times':            (n_snap,) float, years
        'massive_pos':      (n_snap, n_massive, 3) float, AU
        'massive_vel':      (n_snap, n_massive, 3) float, AU/yr
        'massive_mass':     (n_massive,) float, Msun
        'massive_names':    list[str]
        'planetesimal_pos': (n_snap, n_p, 3) float, AU
        'planetesimal_vel': (n_snap, n_p, 3) float, AU/yr
        'l4_counts':        (n_snap,) int
        'l5_counts':        (n_snap,) int
        'config':           dict (serialized SimulationConfig summary)
    }
"""

import os
import pickle
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import rebound

from .config import SimulationConfig
from .bodies import create_solar_system, add_planetesimals, sim_from_state
from .migration import attach_migration
from .analysis import compute_trojan_counts

# Planetesimals that wander past this heliocentric distance (AU) are treated as
# ejected/lost and removed from the integration. This prevents an unresolved
# WHFast close encounter from flinging a (massless) test particle to infinity,
# where the 0*inf term in the mass-weighted centre-of-mass would poison the
# whole simulation with NaN. 1000 AU is far beyond the disk (<=35 AU), so only
# genuinely unbound particles are removed.
ESCAPE_DISTANCE_AU = 1000.0

# Removed/escaped planetesimals are parked at this finite distance in the saved
# snapshots so arrays stay fixed-size and they never count as Trojans.
LOST_SENTINEL_AU = 1.0e8


def load_checkpoint(filename: str) -> Dict:
    """Load simulation state from a checkpoint/result file."""
    print(f"Loading checkpoint from {filename}...")
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print(f"  Time range: {data['times'][0]:.1f} - {data['times'][-1]:.1f} yr")
    print(f"  Snapshots: {len(data['times'])}")
    return data


def _config_summary(config: SimulationConfig) -> Dict:
    """Serializable summary of the config (for provenance in the output)."""
    return {
        'name': config.name,
        'sim_type': config.sim_type,
        'end_time': config.end_time,
        'dt': config.dt,
        'n_snapshots': config.n_snapshots,
        'n_planetesimals': config.n_planetesimals,
        'jupiter_initial_a': config.jupiter_initial_a,
        'planet_set': config.planet_set,
        'jupiter_index': config.jupiter_index,
        'massive_names': ['Sun'] + config.active_planet_names,
        'random_seed': config.random_seed,
        'migration_enabled': config.migration.enabled,
        'migration_tau_a': config.migration.tau_a,
        'migration_tau_e': config.migration.tau_e,
        'migration_direction': config.migration.direction,
    }


def save_checkpoint(filename, times, massive_pos, massive_vel, massive_mass,
                    massive_names, planetesimal_pos, planetesimal_vel,
                    l4_counts, l5_counts, config) -> None:
    """Write simulation state to a pickle file (new NumPy-array format)."""
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    data = {
        'times': np.asarray(times),
        'massive_pos': np.asarray(massive_pos),
        'massive_vel': np.asarray(massive_vel),
        'massive_mass': np.asarray(massive_mass),
        'massive_names': list(massive_names),
        'planetesimal_pos': np.asarray(planetesimal_pos),
        'planetesimal_vel': np.asarray(planetesimal_vel),
        'l4_counts': np.asarray(l4_counts, dtype=int),
        'l5_counts': np.asarray(l5_counts, dtype=int),
        'config': _config_summary(config),
    }
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    print(f"Checkpoint saved to {filename}")


def _build_simulation(config: SimulationConfig, verbose: bool):
    """Construct the simulation and return (sim, names, n_massive, t_initial)."""
    if config.input_file:
        data = load_checkpoint(config.input_file)
        names = data['massive_names']
        n_massive = len(names)
        sim = sim_from_state(
            data['massive_pos'][-1], data['massive_vel'][-1], data['massive_mass'],
            names, data['planetesimal_pos'][-1], data['planetesimal_vel'][-1], config,
        )
        t_initial = float(data['times'][-1])
        if verbose:
            print(f"Loaded initial state from t = {t_initial:.1f} yr "
                  f"({n_massive} massive, {sim.N - n_massive} planetesimals)")
    else:
        sim, names = create_solar_system(config)
        n_massive = len(names)
        add_planetesimals(sim, config, n_massive)
        t_initial = 0.0
    return sim, names, n_massive, t_initial


def _extract_state(sim, n_massive, n_p_orig):
    """Return (massive_pos, massive_vel, planetesimal_pos, planetesimal_vel).

    Planetesimal arrays are reconstructed at their ORIGINAL fixed size using
    each surviving particle's hash (slot = hash - 1). Removed/escaped slots are
    parked at LOST_SENTINEL_AU so the arrays stay aligned across snapshots and
    lost particles never register as Trojans.
    """
    n = sim.N
    pos = np.zeros((n, 3), dtype=np.float64)
    vel = np.zeros((n, 3), dtype=np.float64)
    hsh = np.zeros(n, dtype=np.uint32)
    sim.serialize_particle_data(xyz=pos, vxvyvz=vel, hash=hsh)

    ppos = np.full((n_p_orig, 3), LOST_SENTINEL_AU, dtype=np.float64)
    pvel = np.zeros((n_p_orig, 3), dtype=np.float64)
    if sim.N > n_massive:
        slots = hsh[n_massive:].astype(np.int64) - 1
        ppos[slots] = pos[n_massive:]
        pvel[slots] = vel[n_massive:]

    return pos[:n_massive], vel[:n_massive], ppos, pvel


def _integrate_with_escape(sim, target, n_massive):
    """Integrate to `target`, removing any planetesimal that escapes past
    ESCAPE_DISTANCE_AU. Returns the number of particles removed this call."""
    removed = 0
    while True:
        try:
            sim.integrate(target)
            return removed
        except rebound.Escape:
            # Remove every planetesimal currently beyond the escape distance.
            # Iterate over a snapshot of hashes (removal reindexes the list).
            d2_max = ESCAPE_DISTANCE_AU ** 2
            to_remove = []
            for p in sim.particles[n_massive:]:
                if not np.isfinite(p.x) or (p.x * p.x + p.y * p.y + p.z * p.z) > d2_max:
                    to_remove.append(int(p.hash.value))
            for h in to_remove:
                sim.remove(hash=h)
                removed += 1
            if not to_remove:
                # Escape flagged but nothing past the cutoff (shouldn't happen);
                # bail to avoid an infinite loop.
                return removed


def run_simulation(config: SimulationConfig, verbose: bool = True) -> Dict:
    """
    Run a complete Jupiter Trojan simulation (burnin / source_tracking / migration).

    Returns a dict with the recorded snapshot arrays and L4/L5 counts.
    """
    sim, names, n_massive, t_initial = _build_simulation(config, verbose)

    # Massive-body masses are constant; record once.
    massive_mass = np.array([sim.particles[i].m for i in range(n_massive)])

    jidx = config.jupiter_index
    n_p_orig = sim.N - n_massive

    # Remove runaway test particles before they overflow and NaN the COM.
    sim.exit_max_distance = ESCAPE_DISTANCE_AU
    n_lost = 0

    migration_driver = None
    if config.migration.enabled:
        # Jupiter-only migration driver (O(1) per nudge, independent of the
        # particle count -- see migration.py for why REBOUNDx is not used).
        migration_driver = attach_migration(sim, config, jupiter_index=jidx)

    end_time = config.end_time
    n_snapshots = config.n_snapshots
    snapshot_interval = end_time / n_snapshots

    times = []
    snaps_mpos, snaps_mvel = [], []
    snaps_ppos, snaps_pvel = [], []
    l4_counts, l5_counts = [], []

    if verbose:
        print("=" * 60)
        print(f"Running {config.sim_type} simulation: {config.name}")
        print("=" * 60)
        print(f"Duration: {end_time:.0f} years")
        print(f"Snapshots: {n_snapshots}")
        print(f"Massive bodies: {n_massive}")
        print(f"Planetesimals: {sim.N - n_massive}")
        print(f"WHFast dt: {sim.dt:.4f} yr")
        if config.migration.enabled:
            print(f"Migration: tau_a = {config.migration.tau_a:.0f} yr "
                  f"({config.migration.direction})")
        print("=" * 60)

    start_time = datetime.now()

    def record(snapshot_count):
        mpos, mvel, ppos, pvel = _extract_state(sim, n_massive, n_p_orig)
        times.append(sim.t + t_initial)
        snaps_mpos.append(mpos)
        snaps_mvel.append(mvel)
        snaps_ppos.append(ppos)
        snaps_pvel.append(pvel)
        n4, n5 = compute_trojan_counts(mpos, ppos, jupiter_index=jidx,
                                       criteria=config.trojan_criteria)
        l4_counts.append(n4)
        l5_counts.append(n5)
        if verbose:
            r_jup = np.linalg.norm(mpos[jidx])
            print(f"Snapshot {snapshot_count}/{n_snapshots}: t={times[-1]:.1f} yr, "
                  f"Jupiter r={r_jup:.3f} AU, L4={n4}, L5={n5}")
        return n4, n5

    # Initial snapshot
    record(0)

    checkpoint_file = f"results/pkl/checkpoints/{config.name}_checkpoint.pkl"

    def integrate_chunk(s, sub_target):
        return _integrate_with_escape(s, sub_target, n_massive)

    for snapshot_count in range(1, n_snapshots + 1):
        target = snapshot_count * snapshot_interval
        if migration_driver is not None:
            # Migrate Jupiter in fine sub-steps across this snapshot interval.
            n_lost += migration_driver.advance(target, integrate_fn=integrate_chunk)
        else:
            n_lost += _integrate_with_escape(sim, target, n_massive)
        record(snapshot_count)

        if snapshot_count % config.checkpoint_interval == 0:
            save_checkpoint(checkpoint_file, times, snaps_mpos, snaps_mvel,
                            massive_mass, names, snaps_ppos, snaps_pvel,
                            l4_counts, l5_counts, config)

    save_checkpoint(config.output_file, times, snaps_mpos, snaps_mvel,
                    massive_mass, names, snaps_ppos, snaps_pvel,
                    l4_counts, l5_counts, config)

    elapsed = (datetime.now() - start_time).total_seconds()
    if verbose:
        print("=" * 60)
        print("Simulation Complete!")
        print(f"  Time: {elapsed:.1f} seconds")
        print(f"  Snapshots: {len(times)}")
        print(f"  Escaped/removed planetesimals: {n_lost}/{n_p_orig}")
        print(f"  Final L4: {l4_counts[-1]}, L5: {l5_counts[-1]}")
        print(f"  Output: {config.output_file}")
        print("=" * 60)

    return {
        'times': np.asarray(times),
        'massive_pos': np.asarray(snaps_mpos),
        'massive_vel': np.asarray(snaps_mvel),
        'massive_mass': massive_mass,
        'massive_names': names,
        'planetesimal_pos': np.asarray(snaps_ppos),
        'planetesimal_vel': np.asarray(snaps_pvel),
        'l4_counts': np.asarray(l4_counts, dtype=int),
        'l5_counts': np.asarray(l5_counts, dtype=int),
        'config': config,
        'elapsed_seconds': elapsed,
    }
