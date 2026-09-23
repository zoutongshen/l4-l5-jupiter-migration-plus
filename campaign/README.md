# campaign — present-day integrations of the Jupiter Trojans

Stages 0–3 of `docs/MSc_PROJECT_PLAN.md`, set up at the 2026-09-15 meeting.

The first run is a 10 Myr pilot:
- **Particles:** the whole catalogue, with no clones.
- **Massive bodies:** the Sun with Mercury–Mars folded in; Jupiter, Saturn, Uranus and
  Neptune kept separate.
- **Integrator:** WHFast at dt = 1% of Jupiter's period.

## Layout

| Path | What | Rule |
|---|---|---|
| `trojans/` | Library: `paths`, constants, fetch, catalogue, initial conditions, integration, diagnostics | Computes and returns; never chooses where to write |
| `scripts/` | One CLI per step, prefixed by stage (`s0_`, `s1_`, `s3_`, `s4_`) | The only code that writes files, and only to paths from `trojans.paths` |
| `configs/` | One TOML per run | Frozen once that run has started |
| `tests/` | pytest | Writes only to `tmp_path`. `test_layout.py` enforces the repo structure |
| `slurm/` | ALICE job templates wrapping the same scripts | |
| `reports/<run_id>/` | Small tracked outputs: figures, summary CSVs, notes | Nothing heavy |

Heavy outputs go under the data root, `data/` → `/Volumes/Crucial-X10-Pro/trojans`, or
`$TROJAN_DATA` on ALICE:
- `raw/<source>/<date>/`
- `catalogue/<id>/`
- `initial/<id>/`
- `runs/<run_id>/`

See `data/README.md` for details.

## Setup

From the repo root:

```
mamba env update -n trojan-rb -f environment.yml
conda run -n trojan-rb pip install --no-deps -e campaign
conda run -n trojan-rb pytest campaign/tests
```

## Running the pipeline

Run every command from the repo root in the `trojan-rb` env. `D` is the fetch date,
for example `2026-09-15`.

| Step | Command | Writes |
|---|---|---|
| 1.1 fetch | `python campaign/scripts/s1_fetch.py --date D` | `data/raw/{sbdb,mpc,v24}/D/` |
| 1.1 Horizons | `python campaign/scripts/s1_fetch_horizons.py --date D` | `data/raw/horizons/D/` |
| 1.2–1.3 catalogue | `python campaign/scripts/s1_build_catalogue.py --date D` | `data/catalogue/D/`, `reports/catalogue_D/` |
| 0.3 initial conditions | `python campaign/scripts/s3_build_initial.py --catalogue-id D` | `data/initial/D_giants/` |
| 0.4 Horizons check | `python campaign/scripts/s0_horizons_check.py --catalogue-id D` | `data/raw/horizons_check/`, `reports/horizons_check_*/` |
| 0.6 cost | `python campaign/scripts/s0_benchmark.py cost` | `data/runs/benchmark_cost/`, `reports/benchmark/` |
| 0.6 accuracy runs | `python campaign/scripts/s3_launch_local.py benchmark_k1 --workers 1` (also `k4`, `k8`, `halfdt`) | `data/runs/benchmark_*/` |
| 0.6 compare | `python campaign/scripts/s0_benchmark.py compare` | `reports/benchmark/` |
| 3 run | `python campaign/scripts/s3_launch_local.py smoke` (or `pilot_10myr`) | `data/runs/<run_id>/` |
| 3 merge | `python campaign/scripts/s3_merge.py --run-id <run_id>` | `data/runs/<run_id>/merged/` |
| 3 report | `python campaign/scripts/s3_report.py --run-id <run_id>` | `reports/<run_id>/` |
| 4 mass ladder | `python campaign/scripts/s4_run_ladder.py --workers 8` | `data/runs/mass_ladder/` |
| 4 ladder report | `python campaign/scripts/s4_ladder_report.py --run-id mass_ladder` | `reports/mass_ladder/` |

Stage 4 is a side experiment, not part of the 4.5 Gyr campaign: it asks where the
massless-test-particle assumption breaks (`docs/MASS_THRESHOLD.md`). It does not use
the chunk pipeline — one massive Trojan per simulation, IAS15 rather than WHFast,
because WHFast loses `|dE/E| ~ 9e-2` at 24 M_Jup and misplaces the threshold. It
reuses `trojans.initial.build_simulation`'s `trojan_mass` argument and the
classification helpers in `trojans.integrate`, and nothing else.

Rerunning any step into an existing directory needs `--resume`. For a launched run,
`--resume` continues partial chunks from their last checkpoint and refuses a changed
config. On ALICE, use `s3_launch_local.py <config> --prepare-only`, then
`slurm/run_chunk_array.sh`.
