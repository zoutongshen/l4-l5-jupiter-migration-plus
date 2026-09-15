# migration — parked Jupiter-migration simulations (frozen)

This directory was `analysis/` until 2026-09-15. It holds the REBOUND port (commits
2027074 and 29137d5) of the original AMUSE code, which simulated Trojan capture and loss
while Jupiter migrates. The project has since moved to present-day integrations of the
real catalogue (`campaign/`). **This code is kept for reference and is not edited.**

## Contents

| Path | What |
|---|---|
| `simulation/` | Package: config, bodies, runner, analysis, migration driver, visualization |
| `01_burnin.py`, `02_source_tracking.py`, `03_migration.py`, `run_batch.py`, `plot_batch.py`, `Binary_detector.py`, `generate_3d_gif.py` | Driver scripts (REBOUND) |
| `generate_resonant_chain.py`, `resonance_chain_planetary_system.py` | Course tools that need AMUSE; they do not run in `trojan-rb` |
| `tests/test_physics.py` | 8 tests; they passed on 2026-09-15 before the move |
| `results/png/`, `results/gif/` | Figures from the migration runs |
| `results/batch_results.json`, `batch_summary.txt` | Timings and summaries of the June 2026 batch |
| `results/pkl` | **Symlink** to `/Volumes/Crucial-X10-Pro/trojans/migration_era/pkl`. That directory has its own README listing which pickles are AMUSE-era and unloadable |
| `slopes.md` | Retired. Its numbers and labels are unreliable; kept for provenance |
| `_archive/` | The original notebooks |

## Running it

From this directory, with the `trojan-rb` env active:
`python 01_burnin.py`, or `pytest tests`.

- The script docstrings still say "run from `analysis/`". That now means `migration/`.
- Outputs go to `results/…` relative to this directory, so pickles land on the SSD
  through the symlink.

## Known caveats

- `jupiter_index` defaults to 4 in `simulation/analysis.py`, `visualization.py` and
  `migration.py`. With `planet_set='giants'`, index 4 is Neptune; it is only correct
  for the default all-planets set.
- `sim.integrate()` uses `exact_finish_time=1`, which shortens the last WHFast step
  before every snapshot.
- Loss times are not recorded, only the snapshot at which a particle disappears.
