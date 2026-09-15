"""
Jupiter Migration Simulation
============================

Purpose: study how Jupiter's migration affects L4/L5 Trojan populations.

1. Load a burn-in state (relaxed particle distribution)
2. Apply migration physics to Jupiter (inward or outward)
3. Track L4/L5 population evolution
4. Compute ratio slopes for comparison (see slopes.md)

Sign convention: positive tau_a = inward migration (preserved from the
original AMUSE code; REBOUNDx's opposite convention is hidden inside
simulation/migration.py).

Run from the analysis/ directory:
    conda activate trojan-rb
    python 03_migration.py
"""

# %% [markdown]
# # Jupiter Migration Simulation
# Inward or outward migration of Jupiter, tracking L4/L5 asymmetry.

# %%  Setup and imports
import os
import numpy as np
import matplotlib.pyplot as plt

from simulation import (
    SimulationConfig,
    run_simulation,
    plot_trojan_evolution,
    plot_snapshot,
    create_animation,
    compute_population_summary,
    compute_jupiter_orbit,
)

print('Simulation module loaded successfully!')

# %%  Configuration
DIRECTION = 'outward'      # 'inward' or 'outward'
DURATION_KYR = 50
N_SNAPSHOTS = 200

if DIRECTION == 'inward':
    INITIAL_A = 5.6        # start further out
    TARGET_A = 5.2         # migrate inward to current position
else:                      # outward
    INITIAL_A = 4.8        # start further in
    TARGET_A = 5.2         # migrate outward to current position

# Burn-in state to continue from (generate with 01_burnin.py at the right a).
INPUT_FILE = f'results/pkl/jupiter_burnin_10kyr_{INITIAL_A}au.pkl'

if DIRECTION == 'inward':
    config = SimulationConfig.migration_inward(
        duration_kyr=DURATION_KYR, initial_a=INITIAL_A, target_a=TARGET_A,
        input_file=INPUT_FILE, n_snapshots=N_SNAPSHOTS)
else:
    config = SimulationConfig.migration_outward(
        duration_kyr=DURATION_KYR, initial_a=INITIAL_A, target_a=TARGET_A,
        input_file=INPUT_FILE, n_snapshots=N_SNAPSHOTS)

print(f'Configuration: {config.name}')
print(f'  Direction: {DIRECTION}')
print(f'  Duration: {config.end_time:.0f} years')
print(f'  Jupiter: {INITIAL_A} AU -> {TARGET_A} AU')
print(f'  Migration tau_a: {config.migration.tau_a:.0f} years')
print(f'  Input: {INPUT_FILE}')
print(f'  Output: {config.output_file}')

# %%  Run migration simulation
results = run_simulation(config, verbose=True)

# %%  Extract results
times = results['times']
l4_counts = results['l4_counts']
l5_counts = results['l5_counts']
massive_pos = results['massive_pos']
massive_vel = results['massive_vel']
massive_mass = results['massive_mass']
massive_names = results['massive_names']
planetesimal_pos = results['planetesimal_pos']

jidx = results['config'].jupiter_index   # 4 for all-8 planets, 1 for 'giants'

print(f'Loaded {len(times)} snapshots')
print(f'Time range: {times[0]:.1f} - {times[-1]:.1f} yr')

# %%  Track Jupiter's orbital evolution
jupiter_a, jupiter_e = [], []
for k in range(len(times)):
    orbit = compute_jupiter_orbit(massive_pos[k], massive_vel[k], massive_mass,
                                  jupiter_index=jidx)
    jupiter_a.append(orbit['a'])
    jupiter_e.append(orbit['e'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(times, jupiter_a, 'b-', linewidth=2)
ax1.axhline(INITIAL_A, color='green', linestyle='--', label=f'Initial ({INITIAL_A} AU)')
ax1.axhline(TARGET_A, color='red', linestyle='--', label=f'Target ({TARGET_A} AU)')
ax1.set_xlabel('Time (yr)'); ax1.set_ylabel('Semi-major axis (AU)')
ax1.set_title('Jupiter Migration'); ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(times, jupiter_e, 'r-', linewidth=2)
ax2.set_xlabel('Time (yr)'); ax2.set_ylabel('Eccentricity')
ax2.set_title('Jupiter Eccentricity'); ax2.grid(alpha=0.3)

plt.tight_layout()
os.makedirs('results/png', exist_ok=True)
plt.savefig(f'results/png/{config.name}_jupiter.png', dpi=150)
print(f'Jupiter: {jupiter_a[0]:.3f} AU -> {jupiter_a[-1]:.3f} AU')
print(f'Eccentricity: {jupiter_e[0]:.4f} -> {jupiter_e[-1]:.4f}')

# %%  Plot Trojan population evolution
fig = plot_trojan_evolution(times, l4_counts, l5_counts,
                            title=f'Trojan Evolution ({config.name})')
fig.savefig(f'results/png/{config.name}_evolution.png', dpi=150)
print(f'Saved: results/png/{config.name}_evolution.png')

# %%  Population summary
summary = compute_population_summary(times, l4_counts, l5_counts)
print('=' * 60)
print(f'MIGRATION SUMMARY: {DIRECTION.upper()}')
print('=' * 60)
print(f"Duration: {DURATION_KYR} kyr")
print(f"Jupiter: {INITIAL_A} -> {TARGET_A} AU\n")
print(f"Initial: L4={summary['initial_l4']}, L5={summary['initial_l5']}, Total={summary['initial_total']}")
print(f"Final:   L4={summary['final_l4']}, L5={summary['final_l5']}, Total={summary['final_total']}\n")
print(f"Initial L4/L5 ratio: {summary['initial_ratio']:.3f}")
print(f"Final L4/L5 ratio:   {summary['final_ratio']:.3f}")
print(f"L4/L5 ratio slope:   {summary['ratio_slope']:.3e} yr^-1\n")
print(f"Retention: {summary['retention_percent']:.1f}%")
print('=' * 60)

slope_label = f"{DURATION_KYR}kyr_{DIRECTION[:2]}"
print(f"\nFor slopes.md: {slope_label}: {summary['ratio_slope']:.3e} yr^-1")

# %% [markdown]
# ## Batch mode
# Define multiple scenarios and run them in a loop. Each `(direction, duration,
# initial_a, target_a, input_file)` continues from the named burn-in state.

# %%  Batch configuration (edit and run the next cell to execute)
SCENARIOS = [
    ('inward', 50, 5.6, 5.2, 'results/pkl/jupiter_burnin_10kyr_5.6au.pkl'),
    ('inward', 100, 5.6, 5.2, 'results/pkl/jupiter_burnin_10kyr_5.6au.pkl'),
    ('outward', 50, 4.8, 5.2, 'results/pkl/jupiter_burnin_10kyr_4.8au.pkl'),
    ('outward', 100, 4.8, 5.2, 'results/pkl/jupiter_burnin_10kyr_4.8au.pkl'),
]
print(f'Defined {len(SCENARIOS)} scenarios for batch processing')

# %%  Run all scenarios (uncomment to execute -- this takes a while!)
# all_results = []
# for direction, duration, initial_a, target_a, input_file in SCENARIOS:
#     if direction == 'inward':
#         cfg = SimulationConfig.migration_inward(
#             duration_kyr=duration, initial_a=initial_a, target_a=target_a,
#             input_file=input_file)
#     else:
#         cfg = SimulationConfig.migration_outward(
#             duration_kyr=duration, initial_a=initial_a, target_a=target_a,
#             input_file=input_file)
#     print(f'\n{"="*60}\nRunning: {cfg.name}\n{"="*60}')
#     result = run_simulation(cfg, verbose=True)
#     s = compute_population_summary(result['times'], result['l4_counts'], result['l5_counts'])
#     all_results.append((cfg.name, s['ratio_slope']))
# for name, slope in all_results:
#     print(f'{name}: {slope:.3e} yr^-1')

# %%  Final snapshot
fig = plot_snapshot(massive_pos[-1], massive_names, planetesimal_pos[-1], times[-1],
                    jupiter_index=jidx, show_trojan_zones=True)
fig.savefig(f'results/png/{config.name}_final.png', dpi=150)
print(f'Saved: results/png/{config.name}_final.png')

# %%  Animation (optional)
os.makedirs('results/gif', exist_ok=True)
gif_file = create_animation(
    times, results['massive_pos'], massive_names, results['planetesimal_pos'],
    output_file=f'results/gif/{config.name}.gif', fps=15, dpi=80)
print(f'Saved: {gif_file}')
