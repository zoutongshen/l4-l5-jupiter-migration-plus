"""
Parallel batch runner — benchmark REBOUND port against slopes.md
================================================================

Runs the independent migration scenarios concurrently across CPU cores
(each REBOUND/WHFast run is single-threaded, so N runs use N cores and finish
in roughly the wall-clock time of ONE run, up to the core count).

Tuned for an Apple M5 (4 performance cores). Default batch = the three 50 kyr
directions + a 2nd seed of the inward case, which all fit ~3.5 h wall-clock and
benchmark all three slopes.md directions at once.

USAGE (from the analysis/ directory):
    conda activate trojan-rb
    python run_batch.py

Override any setting via environment variables (these propagate to the worker
subprocesses, unlike editing module globals at runtime):
    BATCH_N_PARTICLES=10000 BATCH_DURATIONS=50 BATCH_SEEDS=42,1042 \
    BATCH_DIRECTIONS=inward,outward,insitu BATCH_WORKERS=4 python run_batch.py

Quick end-to-end smoke test (~seconds, tiny run, separate files):
    BATCH_N_PARTICLES=60 BATCH_BURNIN_KYR=1 BATCH_DURATIONS=1 python run_batch.py

NOTES
- 50 kyr / 10k particles / all-8 planets ~ 3.4 h each (single-threaded).
- 100 kyr ~ 6.8 h, 500 kyr ~ 34 h each: run those OVERNIGHT, not in a 4-h window.
- Consider pausing Dropbox sync while this runs (it writes large .pkl files into
  a Dropbox folder; sync churn competes for I/O).
- Outputs go to results/pkl/batch/ (gitignored). A summary table prints at the end
  and is also written to results/batch_summary.txt.
"""

# Keep each worker single-threaded so N processes don't oversubscribe cores.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# ----------------------------------------------------------------------------- #
# CONFIG  — defaults below; override via BATCH_* env vars (they reach the worker
# subprocesses, unlike editing these globals at runtime).
# ----------------------------------------------------------------------------- #
def _env_int(name, default):
    return int(os.environ.get(name, default))

def _env_ints(name, default):
    v = os.environ.get(name)
    return [int(x) for x in v.split(',')] if v else default

def _env_strs(name, default):
    v = os.environ.get(name)
    return [x.strip() for x in v.split(',')] if v else default

MAX_WORKERS = _env_int('BATCH_WORKERS', 4)          # = performance cores on the M5
N_PARTICLES = _env_int('BATCH_N_PARTICLES', 10000)  # production stats (~200 Trojans/cloud)
BURNIN_KYR = _env_int('BATCH_BURNIN_KYR', 10)       # relaxation before migration
SEEDS = _env_ints('BATCH_SEEDS', [42, 1042])        # spaced (config uses seed,+1,+2 internally)
DURATIONS_KYR = _env_ints('BATCH_DURATIONS', [50])  # 50~3.4h; 100~6.8h; 500~34h (overnight)

# Which directions to run. inward starts at 5.6, outward at 4.8, in-situ stays 5.2.
DIRECTIONS = _env_strs('BATCH_DIRECTIONS', ['inward', 'outward', 'insitu'])
START_A = {'inward': 5.6, 'outward': 4.8, 'insitu': 5.2}
TARGET_A = 5.2

# Only run the extra seed(s) for this/these direction(s) (the contested sign case);
# other directions use just SEEDS[0] to save cores. Set to all DIRECTIONS for full coverage.
MULTISEED_DIRECTIONS = _env_strs('BATCH_MULTISEED', ['inward'])

# Reuse burn-in files that already exist (skip recomputing them).
SKIP_EXISTING_BURNIN = os.environ.get('BATCH_SKIP_EXISTING', '1') == '1'

# slopes.md reference (yr^-1) for the printed comparison
SLOPES_MD = {
    ('inward', 50): -1.830e-07, ('inward', 100): 1.238e-06, ('inward', 500): 1.477e-07,
    ('outward', 50): 2.481e-07, ('outward', 100): 2.481e-07, ('outward', 500): 1.238e-06,
    ('insitu', 50): 2.279e-07, ('insitu', 100): 5.468e-07, ('insitu', 500): 7.152e-08,
}

BATCH_DIR = 'results/pkl/batch'


# ----------------------------------------------------------------------------- #
# Workers (module-level so they pickle for ProcessPoolExecutor)
# ----------------------------------------------------------------------------- #
def _burnin_path(a, seed):
    # Particle count is in the name so different-N runs never clobber each other.
    return f'{BATCH_DIR}/burnin_{a}au_seed{seed}_n{N_PARTICLES}.pkl'


def do_burnin(spec):
    """Relax a cloud at fixed Jupiter position `a` with the given seed."""
    from simulation import SimulationConfig, run_simulation
    a, seed = spec['a'], spec['seed']
    path = _burnin_path(a, seed)
    if SKIP_EXISTING_BURNIN and os.path.exists(path):
        return {'a': a, 'seed': seed, 'path': path, 'secs': 0.0, 'skipped': True}
    cfg = SimulationConfig.burnin(duration_kyr=BURNIN_KYR, jupiter_a=a,
                                  n_planetesimals=N_PARTICLES, random_seed=seed)
    cfg.name = f'burnin_{a}au_seed{seed}'
    cfg.output_file = path
    t0 = time.time()
    run_simulation(cfg, verbose=False)
    return {'a': a, 'seed': seed, 'path': path, 'secs': time.time() - t0, 'skipped': False}


def do_migration(spec):
    """Run one migration (or in-situ) continuing from a burn-in, return its slope."""
    from simulation import (SimulationConfig, run_simulation,
                            compute_population_summary, compute_jupiter_orbit)
    direction, dur, seed = spec['direction'], spec['dur'], spec['seed']
    a0 = START_A[direction]
    burnin = _burnin_path(a0, seed)

    if direction == 'inward':
        cfg = SimulationConfig.migration_inward(duration_kyr=dur, initial_a=a0,
                                                target_a=TARGET_A, input_file=burnin,
                                                n_planetesimals=N_PARTICLES, random_seed=seed)
    elif direction == 'outward':
        cfg = SimulationConfig.migration_outward(duration_kyr=dur, initial_a=a0,
                                                 target_a=TARGET_A, input_file=burnin,
                                                 n_planetesimals=N_PARTICLES, random_seed=seed)
    else:  # in-situ: no migration, Jupiter stays at 5.2
        cfg = SimulationConfig.burnin(duration_kyr=dur, jupiter_a=TARGET_A,
                                      input_file=burnin, n_planetesimals=N_PARTICLES,
                                      random_seed=seed)
    cfg.name = f'{direction}_{dur}kyr_seed{seed}'
    cfg.output_file = f'{BATCH_DIR}/{cfg.name}_n{N_PARTICLES}.pkl'

    t0 = time.time()
    res = run_simulation(cfg, verbose=False)
    secs = time.time() - t0

    s = compute_population_summary(res['times'], res['l4_counts'], res['l5_counts'])
    aN = compute_jupiter_orbit(res['massive_pos'][-1], res['massive_vel'][-1],
                               res['massive_mass'])['a']
    return {
        'direction': direction, 'dur': dur, 'seed': seed,
        'slope': float(s['ratio_slope']),
        'jupiter_a_final': float(aN),
        'final_l4': int(s['final_l4']), 'final_l5': int(s['final_l5']),
        'retention': float(s['retention_percent']),
        'finite': bool(np.all(np.isfinite(res['massive_pos']))),
        'secs': secs,
    }


# ----------------------------------------------------------------------------- #
def main():
    os.makedirs(BATCH_DIR, exist_ok=True)

    # Build the unique (position, seed) burn-ins needed.
    burnin_specs = {}
    migration_specs = []
    for direction in DIRECTIONS:
        seeds = SEEDS if direction in MULTISEED_DIRECTIONS else SEEDS[:1]
        for seed in seeds:
            a0 = START_A[direction]
            burnin_specs[(a0, seed)] = {'a': a0, 'seed': seed}
            for dur in DURATIONS_KYR:
                migration_specs.append({'direction': direction, 'dur': dur, 'seed': seed})

    print(f'Cores in use: {MAX_WORKERS} | particles: {N_PARTICLES} | '
          f'burn-ins: {len(burnin_specs)} | migrations: {len(migration_specs)}')
    print('Phase 1: burn-ins (parallel)...')
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for r in as_completed([ex.submit(do_burnin, s) for s in burnin_specs.values()]):
            d = r.result()
            tag = 'reused (existing)' if d.get('skipped') else f"{d['secs']:.0f}s"
            print(f"  burn-in {d['a']}au seed{d['seed']}: {tag}")

    print('Phase 2: migrations (parallel)...')
    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for r in as_completed([ex.submit(do_migration, s) for s in migration_specs]):
            d = r.result()
            results.append(d)
            print(f"  done {d['direction']}_{d['dur']}kyr seed{d['seed']}: "
                  f"slope={d['slope']:.3e}  finite={d['finite']}  ({d['secs']:.0f}s)")

    # Summary table vs slopes.md
    results.sort(key=lambda x: (x['direction'], x['dur'], x['seed']))
    lines = []
    lines.append('=' * 84)
    lines.append(f"{'scenario':<26}{'seed':>6}{'slope (yr^-1)':>16}{'slopes.md':>14}"
                 f"{'a_final':>9}{'L4/L5':>9}")
    lines.append('-' * 84)
    for d in results:
        ref = SLOPES_MD.get((d['direction'], d['dur']))
        ref_s = f'{ref:.3e}' if ref is not None else '   --'
        same_oom = (ref is not None and ref != 0 and
                    0.2 < abs(d['slope'] / ref) < 5.0)
        flag = ' OOM~' if same_oom else ''
        lines.append(f"{d['direction']+'_'+str(d['dur'])+'kyr':<26}{d['seed']:>6}"
                     f"{d['slope']:>16.3e}{ref_s:>14}{d['jupiter_a_final']:>9.3f}"
                     f"{str(d['final_l4'])+'/'+str(d['final_l5']):>9}{flag}")
    lines.append('=' * 84)
    lines.append(f"total wall-clock: {(time.time()-t_start)/3600:.2f} h")
    summary = '\n'.join(lines)
    print('\n' + summary)
    with open('results/batch_summary.txt', 'w') as f:
        f.write(summary + '\n')
    with open('results/batch_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('\nSaved: results/batch_summary.txt, results/batch_results.json')


if __name__ == '__main__':
    main()
