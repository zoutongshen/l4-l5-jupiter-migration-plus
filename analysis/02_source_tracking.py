"""
Source Tracking Simulation (Pure Disk)
======================================

Purpose: track where Trojans originate by starting with a broad protoplanetary
disk and a fixed (non-migrating) Jupiter, then identifying which initial radial
regions supply particles to L4/L5.

1. Start with a broad disk (2-35 AU) of test particles
2. Evolve with Jupiter at a fixed position (no migration)
3. Identify which initial regions supply particles to L4/L5

Run from the analysis/ directory:
    conda activate trojan-rb
    python 02_source_tracking.py
"""

# %% [markdown]
# # Source Tracking Simulation (Pure Disk)
# Start broad, see which radii feed L4/L5.

# %%  Setup and imports
import os
import numpy as np
import matplotlib.pyplot as plt

from simulation import (
    SimulationConfig,
    run_simulation,
    plot_trojan_evolution,
    plot_snapshot,
    compute_population_summary,
    classify_planetesimals,
)

print('Simulation module loaded successfully!')

# %%  Configuration  (broader disk than burn-in)
DURATION_KYR = 100
N_PLANETESIMALS = 10000
DISK_A_MIN = 2.0
DISK_A_MAX = 35.0
N_SNAPSHOTS = 200

config = SimulationConfig.source_tracking(
    duration_kyr=DURATION_KYR,
    n_planetesimals=N_PLANETESIMALS,
    n_snapshots=N_SNAPSHOTS,
    disk_a_min=DISK_A_MIN,
    disk_a_max=DISK_A_MAX,
)

print(f'Configuration: {config.name}')
print(f'  Duration: {config.end_time:.0f} years')
print(f'  Disk: {DISK_A_MIN} - {DISK_A_MAX} AU')
print(f'  Planetesimals: {config.n_planetesimals}')
print(f'  Output: {config.output_file}')

# %%  Run simulation
results = run_simulation(config, verbose=True)

# %%  Standard analysis
times = results['times']
l4_counts = results['l4_counts']
l5_counts = results['l5_counts']
massive_pos = results['massive_pos']
massive_names = results['massive_names']
planetesimal_pos = results['planetesimal_pos']

os.makedirs('results/png', exist_ok=True)
fig = plot_trojan_evolution(times, l4_counts, l5_counts,
                            title=f'Trojan Evolution ({config.name})')
fig.savefig(f'results/png/{config.name}_evolution.png', dpi=150)

summary = compute_population_summary(times, l4_counts, l5_counts)
print('=' * 60)
print('POPULATION SUMMARY')
print('=' * 60)
print(f"Initial: L4={summary['initial_l4']}, L5={summary['initial_l5']}")
print(f"Final:   L4={summary['final_l4']}, L5={summary['final_l5']}")
print(f"L4/L5 ratio slope: {summary['ratio_slope']:.3e} yr^-1")
print('=' * 60)

# %% [markdown]
# ## Source region analysis
# Where did the final Trojans come from? Classify the final population, then
# look up each Trojan's *initial* heliocentric radius.

# %%  Classify final particles and trace initial radii
jidx = results['config'].jupiter_index
initial_pos = planetesimal_pos[0]
final_pos = planetesimal_pos[-1]

classification = classify_planetesimals(massive_pos[-1], final_pos, jupiter_index=jidx)
print('Final classification:')
print(f"  L4 Trojans: {len(classification['L4'])}")
print(f"  L5 Trojans: {len(classification['L5'])}")
print(f"  Other co-orbital: {len(classification['coorbital_other'])}")
print(f"  Inner region: {len(classification['inner'])}")
print(f"  Outer region: {len(classification['outer'])}")

initial_radii = np.linalg.norm(initial_pos, axis=1)
l4_initial_radii = initial_radii[classification['L4']]
l5_initial_radii = initial_radii[classification['L5']]

# %%  Plot source regions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].hist(initial_radii, bins=50, range=(0, 40), alpha=0.7, color='gray')
axes[0].axvline(5.2, color='brown', linestyle='--', linewidth=2, label='Jupiter')
axes[0].set_xlabel('Initial Distance (AU)'); axes[0].set_ylabel('Count')
axes[0].set_title('All Initial Particles'); axes[0].legend()

if len(l4_initial_radii) > 0:
    axes[1].hist(l4_initial_radii, bins=30, range=(0, 40), alpha=0.7, color='blue')
    axes[1].axvline(5.2, color='brown', linestyle='--', linewidth=2)
    axes[1].axvline(np.median(l4_initial_radii), color='blue', linestyle='-',
                    linewidth=2, label=f'Median: {np.median(l4_initial_radii):.1f} AU')
axes[1].set_xlabel('Initial Distance (AU)'); axes[1].set_ylabel('Count')
axes[1].set_title(f'L4 Trojan Sources (n={len(l4_initial_radii)})'); axes[1].legend()

if len(l5_initial_radii) > 0:
    axes[2].hist(l5_initial_radii, bins=30, range=(0, 40), alpha=0.7, color='red')
    axes[2].axvline(5.2, color='brown', linestyle='--', linewidth=2)
    axes[2].axvline(np.median(l5_initial_radii), color='red', linestyle='-',
                    linewidth=2, label=f'Median: {np.median(l5_initial_radii):.1f} AU')
axes[2].set_xlabel('Initial Distance (AU)'); axes[2].set_ylabel('Count')
axes[2].set_title(f'L5 Trojan Sources (n={len(l5_initial_radii)})'); axes[2].legend()

plt.tight_layout()
plt.savefig(f'results/png/{config.name}_sources.png', dpi=150)
print(f'Saved: results/png/{config.name}_sources.png')

# %%  Source statistics
print('=' * 60)
print('SOURCE REGION STATISTICS')
print('=' * 60)
for label, radii in [('L4', l4_initial_radii), ('L5', l5_initial_radii)]:
    if len(radii) > 0:
        print(f"{label} Trojans originated from:")
        print(f"  Range: {radii.min():.2f} - {radii.max():.2f} AU")
        print(f"  Median: {np.median(radii):.2f} AU")
        print(f"  Mean: {np.mean(radii):.2f} AU   Std: {np.std(radii):.2f} AU")
        print()
print('Use these source regions to guide particle injection for migration runs.')

# %%  Final snapshot
fig = plot_snapshot(massive_pos[-1], massive_names, final_pos, times[-1],
                    jupiter_index=jidx, show_trojan_zones=True, xlim=(-40, 40), ylim=(-40, 40))
fig.savefig(f'results/png/{config.name}_final.png', dpi=150)
print(f'Saved: results/png/{config.name}_final.png')
