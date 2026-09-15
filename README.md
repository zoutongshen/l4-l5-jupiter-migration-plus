# L4-L5-Jupiter-Migration-Plus

MSc thesis project (Leiden, supervisor Simon Portegies Zwart) on the number asymmetry
between Jupiter's leading (L4) and trailing (L5) Trojan swarms.

## Where things live

```
.
├── README.md          this map
├── environment.yml    conda env `trojan-rb` (never install into base)
├── docs/              proposal, working plan, literature reviews
├── literature/        papers by topic, plus READING_LIST.md
├── migration/         parked Jupiter-migration code and results; frozen, see migration/README.md
├── campaign/          present-day integrations of the real Trojan catalogue (in progress)
└── data -> /Volumes/Crucial-X10-Pro/trojans    all heavy data, on the external SSD (not in git)
```

| Directory | Contents |
|---|---|
| `docs/` | `MSc_PROPOSAL.md`, `MSc_PROJECT_PLAN.md` (the plan of record), `ASYMMETRY_REVIEW.md`, `VOKROUHLICKY_DISISTO_REVIEW.md` |
| `literature/` | `00_reviews/` … `06_methods_missions/`, `99_course_background/` |
| `migration/` | REBOUND port of the migration simulations. Kept for reference and not developed further. Its pickles are on the SSD |
| `campaign/` | Library, scripts, configs, tests and small reports for the present-day runs |
| `data/` | Symlink to the SSD. See `data/README.md` there for its layout |

Paths quoted inside any document are relative to the repo root.

## Output rules

1. **Heavy outputs go to `data/` (the SSD), never into the repo.** The only simulation
   products tracked in git are small figures and summary tables under
   `campaign/reports/<run_id>/`.
2. **One module builds paths:** `campaign/trojans/paths.py`. Nothing writes relative to
   the current working directory.
3. **Nothing is overwritten silently.** A run directory is re-entered only with
   `--resume`.
4. **Every output directory records its provenance:** git commit, REBOUND version, host,
   command.
5. **No pickles for new work.** Tables are parquet, arrays are HDF5, checkpoints are
   REBOUND binary.
6. **New top-level entries** mean updating this map and the layout test,
   `campaign/tests/test_layout.py`, in the same commit.

If the SSD isn't mounted, campaign scripts stop with an error rather than writing
somewhere else.
