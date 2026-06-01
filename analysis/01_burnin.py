"""
Burn-in Simulation
==================

Purpose: relax artificially injected particles with Jupiter at a fixed position
(no migration). Uses the shared `simulation` package (REBOUND + REBOUNDx).

Run from the analysis/ directory:
    conda activate trojan-rb
    python 01_burnin.py

The file uses `# %%` cell markers so it can also be run cell-by-cell in
VSCode / Cursor / the Jupyter interactive window.
"""

# %% [markdown]
# # Burn-in Simulation
# Relax injected particles with Jupiter fixed (no migration).

# %%  Setup and imports
import os

from simulation import (
    SimulationConfig,
    run_simulation,
    load_checkpoint,
    plot_trojan_evolution,
    plot_snapshot,
    create_animation,
    compute_population_summary,
)

print('Simulation module loaded successfully!')

# %%  Configuration
DURATION_KYR = 10          # thousands of years
JUPITER_A = 4.8            # Jupiter semi-major axis (AU)
N_PLANETESIMALS = 1000
INPUT_FILE = None          # e.g. 'results/pkl/jupiter_burnin_5kyr_5.2au.pkl' to continue
N_SNAPSHOTS = 200

config = SimulationConfig.burnin(
    duration_kyr=DURATION_KYR,
    jupiter_a=JUPITER_A,
    input_file=INPUT_FILE,
    n_planetesimals=N_PLANETESIMALS,
    n_snapshots=N_SNAPSHOTS,
    trojan_emphasis=True,
)

print(f'Configuration: {config.name}')
print(f'  Duration: {config.end_time:.0f} years')
print(f'  Jupiter at: {config.jupiter_initial_a} AU')
print(f'  Planetesimals: {config.n_planetesimals}')
print(f'  Output: {config.output_file}')

# %%  Run simulation
results = run_simulation(config, verbose=True)

# %%  Extract results
times = results['times']
l4_counts = results['l4_counts']
l5_counts = results['l5_counts']
massive_pos = results['massive_pos']
massive_names = results['massive_names']
planetesimal_pos = results['planetesimal_pos']

print(f'Loaded {len(times)} snapshots')
print(f'Time range: {times[0]:.1f} - {times[-1]:.1f} yr')

# %%  Plot Trojan population evolution
os.makedirs('results/png', exist_ok=True)

fig = plot_trojan_evolution(times, l4_counts, l5_counts,
                            title=f'Trojan Evolution ({config.name})')
fig.savefig(f'results/png/{config.name}_evolution.png', dpi=150)
print(f'Saved: results/png/{config.name}_evolution.png')

# %%  Population summary
summary = compute_population_summary(times, l4_counts, l5_counts)
print('=' * 60)
print('POPULATION SUMMARY')
print('=' * 60)
print(f"Initial: L4={summary['initial_l4']}, L5={summary['initial_l5']}, Total={summary['initial_total']}")
print(f"Final:   L4={summary['final_l4']}, L5={summary['final_l5']}, Total={summary['final_total']}")
print(f"Initial L4/L5 ratio: {summary['initial_ratio']:.3f}")
print(f"Final L4/L5 ratio:   {summary['final_ratio']:.3f}")
print(f"L4/L5 ratio slope:   {summary['ratio_slope']:.3e} yr^-1")
print(f"Retention: {summary['retention_percent']:.1f}%")
print('=' * 60)

# %%  Final snapshot
fig = plot_snapshot(massive_pos[-1], massive_names, planetesimal_pos[-1], times[-1],
                    show_trojan_zones=True)
fig.savefig(f'results/png/{config.name}_final.png', dpi=150)
print(f'Saved: results/png/{config.name}_final.png')

# %%  Animation (optional)
os.makedirs('results/gif', exist_ok=True)
gif_file = create_animation(
    times, results['massive_pos'], massive_names, results['planetesimal_pos'],
    output_file=f'results/gif/{config.name}.gif', fps=15, dpi=80,
)
print(f'Saved: {gif_file}')
