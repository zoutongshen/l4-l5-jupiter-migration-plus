# Run plan: capture while Jupiter grows and migrates (stage 8)

Drafted 2026-11-03, after Simon agreed to migration arms. Full plan with the
literature table: `~/.claude/plans/vivid-drifting-hummingbird.md`. Code:
`campaign/trojans/history.py`, `campaign/scripts/s8_*.py`, configs
`campaign/configs/mig_*.toml`.

**In one line.** Start from the same kind of mirror-symmetric disc as the
no-migration control (`RUN_disc_capture_control.md`), and let Jupiter grow *and*
migrate. Then ask where the resulting L4/L5 difference sits in D, I and e. The
target is the observed cold-L4 excess: N4/N5 = 2.0 below I_p = 15° and 1.0 above.

---

## 1. Why

- Loss today is symmetric (pilot), and capture without migration is symmetric
  (disc_grow: cold bin 1.02 ± 0.05). Something mirror-odd must have acted.
  Migration is the natural candidate: Pirani 2019a finds the sign follows the
  direction of migration.
- **Pirani never splits N4/N5 by inclination or eccentricity.** Her pre-stirred
  discs change e and i together and give a weaker ratio (1.15–1.4) than her flat
  disc (1.6–1.8). So the new measurement is how the migration asymmetry depends
  on a particle's starting I and e. A uniform twin disc gives that once, and it
  can be reweighted to any physical disc afterwards.

## 2. The model

- **Bodies:** folded Sun + Jupiter, circular and planar; WHFast, dt = 1% of P_J
  at the smallest a Jupiter reaches.
- **Jupiter's history** is a pure function of time (`JupiterHistory`). After every
  check (8 steps) `HistoryForcing` puts Jupiter at a(t) on the circular orbit for
  m(t). This makes resumes bitwise exact (tested).
  - Migration laws: `const_nu` (default; a^(3/2) linear in time), `linear`,
    `exponential`.
  - Growth laws: `exponential` (as stage 6), `linear`.
- **Rate variable:** ν = |ȧ| / (q Ω a). Pirani's asymmetry window is
  0.17 < ν < 1.45 (v_lim; v_crit after Ogilvie & Lubow 2006). `const_nu` keeps ν
  fixed while the mass is fixed. Growth lowers ν at the same ȧ.
- **Trojan:** a tadpole of one swarm in every 10 kyr window of the last 1 Myr.
  This is the same classifier as stage 6 and the catalogue work. Pirani's
  ±2.5 R_H count is reported next to it, for comparison only.
- **Snapshots** (new; `snapshots.h5` per chunk):
  - per particle: a, e, I, σ and heliocentric x, y, z;
  - Jupiter's a, m and λ;
  - cadence: every `fine_every_yr` until the forcing ends, then every window (10 kyr);
  - each flush is read back, because of the SSD corruption.

## 3. Phase 1: validation (run first)

| Arm | Config | Setup | Expect |
|---|---|---|---|
| V1 | `mig_v1_insitu.toml` | no migration; 10 M⊕ → 1 M_J over 0.1 Myr; disc_grow's disc (same seed); 1.2 Myr | reproduces disc_grow (1.009 ± 0.034), ideally particle by particle |
| V2 | `mig_v2_pirani.toml` | Pirani "×0.25" track: 8.8 → 5.4 au over 0.69 Myr; 0.1 M⊕ → 1 M_J (exponential) over the same time; flat disc (e < 0.01, I < 0.01°) at 5.0–9.4 au; +1.2 Myr fixed | N4/N5 > 1; Pirani's count ≈ 1.6 ± 0.4 |
| V3 | `mig_v3_outward.toml` | V2 reversed: 5.4 → 8.8 au | N4/N5 < 1 (Pirani ≈ 0.6) |

**Caveat on V2.** Pirani's growth track is only a figure. Here ν starts at 6.9
(above v_crit) and falls as Jupiter grows. It is inside the window only during
about 19–46% of the migration (a ≈ 8.2–7.2 au, m ≈ 0.5–4 M⊕). If V2 comes out
symmetric, the first suspect is the growth law, not the code.

**Smoke test (done 2026-11-03):**
- 500 particles, 6.0 → 5.2 au, ν₀ = 0.5, 50 kyr.
- Runs end to end, and the report and GIF work.
- Jupiter in the snapshots matches a(t) to 7e-14; the mass matches exactly.
- 63 tests pass (2 layout failures were already there).

## 4. Phase 2: science scan (after Phase 1 passes)

Warm twin disc (e uniform 0–0.3, I uniform 0–40°), from a_end − 0.4 to
a_start + 0.6 au. One factor at a time around a nominal:
- **Nominal:** inward 8.2 → 5.2 au, ν₀ = 0.5 at 10 M⊕ (≈ 80 kyr), growth during
  the migration, t_grow = 0.1 Myr.
- **Rate:** ν₀ ∈ {0.1, 0.3, 0.5, 1.0}.
- **Excursion:** Δa ∈ {1, 3, 6} au.
- **Growth timing:** during the migration, or after it (expected to lose the
  asymmetry, as Pirani's Neptune case does).
- **t_grow:** {0.1, 0.5} Myr.
- **Outward** at the nominal ν.

Main output: N4/N5 against starting I0 and e0 (`binned.csv`, `binned_i0_e0.csv`),
and against final I, D and e.

## 5. Commands

Launch from your own terminal (macOS throttles background shells), with the SSD
mounted:

```bash
conda activate trojan-rb
cd "<repo>/campaign"
python scripts/s8_run_migration.py configs/mig_v1_insitu.toml 2>&1 | tee ../data/runs/mig_v1_insitu.log
python scripts/s8_run_migration.py configs/mig_v2_pirani.toml --workers 4 2>&1 | tee ../data/runs/mig_v2_pirani.log &
python scripts/s8_run_migration.py configs/mig_v3_outward.toml --workers 4 2>&1 | tee ../data/runs/mig_v3_outward.log
python scripts/s8_migration_report.py --run-id mig_v1_insitu      # likewise v2, v3
python scripts/s8_capture_gif.py --run-id mig_v2_pirani
```

Any interrupted arm continues with `--resume`.

**Cost (estimate):**
- V1 ≈ 6 CPU-h, the same as disc_grow.
- V2 and V3 ≈ 6–12 CPU-h each: 20k particles over 1.89 Myr, many removed early.
- In total about 4 h of wall time on 8 cores.
- Snapshots ≈ 0.1–0.3 GB per arm.

## 6. Results

### Phase 1 (finished 2026-11-04 00:31)

| Arm | N4/N5 (tadpole classifier) | Pirani ±2.5 R_H count | Trojans / disc | CPU-h |
|---|---|---|---|---|
| V1 in situ | 1796/1780 = 1.009 ± 0.034 | — | 18% | 4.70 |
| V2 inward 8.8 → 5.4 | 205/182 = **1.13 ± 0.11** | 1.13 (same set) | 1.9% | 14.6 |
| V3 outward 5.4 → 8.8 | 95/124 = **0.77 ± 0.10** | 0.77 | 1.1% | 13.0 |

- **V1** reproduces disc_grow particle by particle (20000/20000 fates identical).
- **V2 and V3 have the sign Pirani predicts**: L4 for inward, L5 for outward.
  The V2 − V3 difference is 0.36 ± 0.16 (2.3σ). The amplitudes are weaker than
  Pirani's ×0.25 run (≈ 1.6 and ≈ 0.6).
- **When the asymmetry forms** (co-orbital σ > 0 : σ < 0, `coorbital.csv`):
  it is set in the first ~100–200 kyr, while Jupiter is 0.1–1 M⊕ and ν = 7 → 0.7.
  - V2: 1.40 at 50 kyr, then 1.14 from 200 kyr on.
  - V3: 0.68 from 100 kyr on.

  After Jupiter passes ~1 M⊕, ν < 0.17, nothing new forms, and the ratio is
  only frozen in.
- **Where the Trojans come from:** all of them start in Jupiter's *initial*
  co-orbital zone.
  - V2: 204 + 182 from a0 = 8.6–9.0 au, and 1 from 8.2–8.6.
  - V3: 217 of 219 from 5.0–5.8 au.

  Nothing is picked up along the 3.4 au path. So with this growth law the
  migration does not sweep material into the swarms: it carries the seed's
  starting co-orbitals and biases their sides early.
- Both swarms end cold (I < 0.1°, e < 0.02). The flat disc has nothing to say
  about I, as planned.

**Interpretation.** The code passes the sign test, and the migration
asymmetry is real but weak here. That is because the exponential growth law
puts the in-window ν phase at a sub-Earth-mass seed. A larger effect needs ν
in the window while Jupiter is massive enough to hold tadpoles. This argues
for Phase 2's design: ν₀ is defined at 10 M⊕ and the seed starts as a 10 M⊕
core, so the asymmetry is made where it can be kept. Pirani's exact
amplitude depends on her figure-only growth track and is not pursued further.

Minor: `summary.json` `max_coorbital_side_ratio` only tracks the maximum, so it
reads 1.0 for V3. Read `coorbital.csv` instead.
