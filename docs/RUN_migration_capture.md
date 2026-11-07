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

### Phase 1b: code checks and Pirani's own track (2026-11-04/05)

**Check 1: tadpole count = Pirani count in V2/V3.** Physical, not a bug. In
a flat disc after 1.2 Myr, the particles with their whole orbit inside ±2.5 R_H
are exactly the tadpoles; the 265 others with |a − a_J| < 2.5 R_H are too eccentric.

**Check 2: migration at a fixed mass** (ν = 0.5 the whole way, 7.2 ↔ 5.2 au, flat
disc, 10k particles; `mig_c2_*`):

| Mass | Inward | Outward | Twins one-sided: L4+none vs L5+none (in / out) |
|---|---|---|---|
| 10 M⊕ | 0.99 ± 0.13 | 0.98 ± 0.14 | 39/36, 24/28 |
| 1 M_J | **1.17 ± 0.08** | **0.83 ± 0.06** | **146/80**, **81/154** |

- At 1 M_J, migration alone gives the right sign. The mirror-twin pairs make
  this a paired test: inward kills the L5 twin about twice as often as the L4 twin,
  and outward does the reverse.
- At 10 M⊕ nothing survives as an asymmetry. Not yet checked why; the likely
  reason is that tadpoles at 10 M⊕ re-mix over the following Myr.

**Check 3: Pirani's ×0.25 track itself**, digitized from her Figs. 3 and 13
(`scripts/s8_pirani_track.py` → `configs/tracks/pirani_x025*.csv`, new
`migration_law = "table"`; 3 new tests, 66 pass). ν stays at 0.35 → 0.17 while
the core grows to 20 M⊕, against 7 → 0.06 in V2.

| Arm | N4/N5 | Pirani |
|---|---|---|
| V2b inward (`mig_v2b_pirani_track`) | **1.27 ± 0.14** (179/141) | 1.60 ± 0.44 |
| V3b outward, same clock, drift reversed | **0.82 ± 0.13** (78/95) | ≈ 0.6 (different set-up: 5 → 18 au) |

- The asymmetry now grows steadily while ν is in the window. Co-orbital side
  ratio: 1.07 at 20 kyr, 1.20 at 50 kyr, 1.27–1.29 at 100–200 kyr. It is then
  frozen in as ν drops below 0.17.
- All Trojans again come from Jupiter's starting zone, as in her Fig. 10.
- **Conclusion.** The code is right, and the shortfall against Pirani in V2
  was mostly the growth law. On her track we agree within her error
  (0.7σ). Her ×0.25 run sits at the edge of the window (her ×0.15 gives 1.23),
  so the exact value is sensitive to the details. Still flat Trojans:
  inclination needs Phase 2.

### Block B: all four giants (2026-11-05)

Giants on physical tracks (`scripts/s8_tracks.py`; ν = ν_I/(1+(m/M_gap)^p),
calibrated on Pirani's ×0.25 track: ν_I = 0.225, M_gap = 25 M⊕, p = 3), forced
circular while their history runs, then free N-body.

| Arm | Set-up | N4/N5 | Note |
|---|---|---|---|
| B1 `mig_g1_insitu` | J grows in situ at 5.4; S/U/N fixed at 8.6/16/18.5 (P_S/P_J = 2.03); warm disc | 614/644 = **0.95 ± 0.05** | symmetric in every D/I/e bin; but cold Trojans destroyed: survival relative to V1 is 0.02 at i0 < 10°, 0.37 at 15–20°, 0.78 at 30–40° |
| B2 `mig_g2_pirani` | Pirani ×0.25 track for J (ends 5.59); generated S/U/N (S ends 8.6); flat disc | **0/0** | asymmetry forms exactly as in V2b (co-orbital side ratio 1.25 from 100 to 600 kyr), then every co-orbital is lost between 600 and 700 kyr |
| B3 `mig_g3_pirani_out` | B2 reversed: J 5.59 → 8.76 (outward), S/U/N reversed and scaled (S 13.95 → 18.36; P_S/P_J 3.9 → 3.0, never crosses the 2:1); flat disc | 80/96 = **0.83 ± 0.13** | as V3b (0.82 ± 0.13): side ratio set to 0.83 by ~110 kyr, then frozen; 177 co-orbitals, no loss after the forcing ends |
| B1b `mig_g1b_insitu_32` | as B1, giants in a compact chain (Roig & Nesvorný 2015 positions, scaled: S 7.355, U 9.981, N 15.874; NB P_S/P_J = 1.59, near 8:5, free and non-resonant, so not a true 3:2 lock) | 9/8 | **0.5% of V1's Trojans survive** (17 of 3576), at all i0: as RN15 (10000 → 44) |
| B1c `mig_g1c_insitu_wide` | as B1, today's orbits scaled to J at 5.4 (S 9.899, U 19.918, N 31.211) | 1744/1705 = **1.02 ± 0.03** | survival relative to V1 0.91/0.91/0.97/1.03/0.95/1.02 for i0 0–5/5–10/10–15/15–20/20–30/30–40°; symmetric in every bin |

- **B2 diagnosis.** Saturn's track crosses P_S/P_J = 2.00 at 600 kyr (S at 8.87 au,
  78 M⊕), converging on Jupiter. The co-orbital count drops 315 → 56 by 620 kyr and
  → 0 by 940 kyr. This is the known destabilisation of Trojans by the J–S 1:2
  (Morbidelli+2005; Marzari & Scholl 2007; Robutel & Bodossian 2009). Before the
  crossing, the giants change nothing: formation matches V2b (1.27).
- **Consequence.** The end configuration and the *path* of the J–S period ratio
  decide survival. The C0 config as first written (S ending at 8.6, starting
  250 kyr later) would cross the 2:1 too, so it was stopped before it started.
  Co-eval tracks (S starting with J) ending in the 3:2 region never cross it:
  S_end 7.355 peaks at P_S/P_J = 1.98 (≈ 300 kyr, M_S ≈ 45 M⊕), S_end 7.0 at 1.88.
- Real convergent migration with gas damping would *capture* Saturn into the 2:1
  rather than let it pass (Masset & Snellgrove 2001; Pierens & Nelson 2008); the
  forced circular tracks cannot represent that.

**Choice of C0's end configuration (2026-11-05 20:55).** Survival of the
gas-era Trojans depends entirely on where Saturn is: near or across the 2:1
(B1, B2) or in the compact 3:2 chain (B1b) almost nothing survives; in the wide
configuration (B1c) survival is flat in i0 and symmetric. So C0 ends in the wide
configuration, with Saturn forming together with Jupiter (`tracks/c0_saturn.csv`,
13.80 → 9.90 au; P_S/P_J ≥ 2.02 throughout, ≥ 2.3 once M_S > 20 M⊕). C0 then
measures the formation asymmetry against I0 and e0 without a configuration-
dependent loss on top. How a compact or resonant phase (and the instability)
reshapes it is a stage-2 question (`PIRANI_MODEL_STAGES.md`: s4, s5, C).

### C0: warm twin disc, all giants, wide end configuration (2026-11-05 22:28)

`mig_c0_nominal`: generated tracks (J 8.64 → 5.40, co-eval S 13.80 → 9.90), warm disc
(e0 ≤ 0.3, i0 ≤ 40°, 5.0–9.25 au), 20k particles, 12.9 CPU-h.

- **Overall N4/N5 = 194/202 = 0.96 ± 0.10.** 396 Trojans (2.0% of the disc); 309 of
  them start at a0 = 8.48–8.86 au, i.e. Jupiter's starting zone again.
- By i0: 1.18 ± 0.34 (0–5°), 1.24 ± 0.31 (5–10°), 0.93 ± 0.21 (10–15°), 1.00 ± 0.27,
  0.69 ± 0.14 (20–30°), 1.14 ± 0.34 (30–40°). By e0: no trend.
- Cold *and* circular starts lean to L4: e0 < 0.1 & i0 < 10°: 27/16 = 1.69 ± 0.53;
  e0 < 0.1 & i0 < 5°: 18/8. Everything else ≈ 1. Suggestive (≈ 1.3σ), too few particles.
- The co-orbital side ratio stays at 1.00 ± 0.06 throughout (V2b, flat: 1.27): in a
  warm disc most "co-orbitals" are eccentric passers, and the mechanism acts only on
  the cold, near-circular minority. Consistent with Pirani 2019b (pre-stirred discs
  weaker: 1.15).
- **Next (C0b):** concentrate the disc where Trojans come from (a0 8.2–9.0 au) and on
  cold starts (e0 ≤ 0.1, i0 ≤ 15°) to resolve the i0 dependence: ~10× the particles per
  cold bin at the same cost.

### C0b: cold starts in Jupiter's starting zone (2026-11-06 00:11)

`mig_c0b_cold_zone`: C0 with a0 8.2–9.0 au, e0 ≤ 0.1, i0 ≤ 15°; 3312 Trojans.

| i0 (°) | 0–2.5 | 2.5–5 | 5–7.5 | 7.5–10 | 10–12.5 | 12.5–15 |
|---|---|---|---|---|---|---|
| N4/N5 | 1.09 ± 0.10 | 1.06 ± 0.09 | 1.01 ± 0.08 | 1.02 ± 0.09 | 1.21 ± 0.10 | 1.14 ± 0.10 |

| e0 | 0–0.025 | 0.025–0.05 | 0.05–0.075 | 0.075–0.1 |
|---|---|---|---|---|
| N4/N5 | **1.25 ± 0.10** | 1.15 ± 0.08 | 0.97 ± 0.07 | 1.05 ± 0.07 |

- **Overall 1.08 ± 0.04.** The migration asymmetry depends on e0, not on i0: it needs
  near-circular starts (e0 ≲ 0.05) and is flat in i0 up to 15°. C0's cold lean was noise.
- Trojans keep their inclination (median final I 8.1° vs i0 7.4°).
- Side ratio among co-orbitals builds 1.0 → 1.14 between 140 and 240 kyr (Jupiter 8 → 36 M⊕), then settles at 1.07–1.09.
- **Implication for the cold-L4 excess:** with these tracks, gas-era migration makes
  an e-dependent, I-independent excess of ≤ 1.25, not the observed I-dependent 2.0.
  An I-dependence would have to come from a later stage (e.g. a symmetric, hotter
  captured component diluting the primordial one) or from a disc whose e and I
  are correlated (cold and circular together).

### C1: twice the migration rate, cold zone (2026-11-06 02:06)

`mig_c1_nu045_cold`: ν_I = 0.45 (C0b: 0.225); J 11.36 → 5.40, co-eval S 18.45 → 11.50
(never reaches P_S/P_J = 2); ice giants as C0; disc a0 10.95–11.75, e0 ≤ 0.1, i0 ≤ 15°.
4401 Trojans.

| | all | e0 < 0.025 | 0.025–0.05 | 0.05–0.075 | 0.075–0.1 |
|---|---|---|---|---|---|
| C0b (ν_I 0.225) | 1.08 ± 0.04 | 1.25 ± 0.10 | 1.15 ± 0.08 | 0.97 ± 0.07 | 1.05 ± 0.07 |
| **C1 (ν_I 0.45)** | **1.25 ± 0.04** | **1.66 ± 0.12** | 1.30 ± 0.08 | 1.11 ± 0.06 | 1.08 ± 0.06 |

- By i0 (0–15° in 2.5° bins): 1.39, 1.14, 1.31, 1.19, 1.18, 1.29 (± 0.1): still flat in i0.
- Co-orbital side ratio peaks at 1.41 (190 kyr, Jupiter 18 M⊕), settles at 1.25 once
  Jupiter passes ~40 M⊕, and is then frozen. Only 2% of co-orbitals are lost after
  the gas phase (4478 → 4403), so the different Saturn end does not matter.
- **Reading.** A faster rate raises the excess, and for near-circular starts reaches
  Pirani's saturated range (1.6–1.8). The dependence is on e0, never on i0, so the
  mechanism produces an e-sorted, not an I-sorted, L4 excess.

### C2: four times the rate, cold zone (2026-11-06 03:59)

`mig_c2_nu09_cold`: ν_I = 0.9; J 16.00 → 5.40; co-eval S 24.25 → 11.50 (same end as C1);
disc a0 15.17–16.65, e0 ≤ 0.1, i0 ≤ 15°. 2967 Trojans.

**Rate series (cold zone, all four giants):**

| ν_I | all | e0 < 0.025 | 0.025–0.05 | 0.05–0.075 | 0.075–0.1 | i0 trend |
|---|---|---|---|---|---|---|
| 0.225 (C0b) | 1.08 ± 0.04 | 1.25 ± 0.10 | 1.15 ± 0.08 | 0.97 ± 0.07 | 1.05 ± 0.07 | flat |
| 0.45 (C1) | 1.25 ± 0.04 | 1.66 ± 0.12 | 1.30 ± 0.08 | 1.11 ± 0.06 | 1.08 ± 0.06 | flat |
| 0.9 (C2) | **1.37 ± 0.05** | **1.81 ± 0.14** | 1.43 ± 0.10 | 1.18 ± 0.09 | 1.18 ± 0.09 | flat (1.28–1.52) |

- The excess grows with the rate but more slowly each doubling (e0 < 0.025: +0.41,
  then +0.15): it is saturating, at about Pirani's 1.6–1.8 (her Table 3).
- Co-orbital side ratio peaks at 1.85 (180 kyr, Jupiter 15 M⊕), relaxes to 1.37 as
  Jupiter grows to ~50 M⊕, then frozen; 3% loss after the gas phase. The diverging
  2:1 crossing at 144 kyr (M_S 8 M⊕) leaves no mark.
- **Still flat in i0 at every rate.** The gas-era migration excess is sorted by e0;
  at best ≈ 1.8 for e0 < 0.025 and ≈ 1.4 overall for e0 ≤ 0.1. It does not by itself
  produce an inclination-dependent ratio.

### D1: C2 rate, near-circular starts at inclinations up to 40° (2026-11-06 19:07)

`mig_d1_c2_hot`: C2 tracks (ν_I 0.9, J 16.00 → 5.40, S 24.25 → 11.50, ice giants as C0);
disc a0 15.17–16.65, **e0 ≤ 0.05, i0 0–40° uniform**, 15000 pairs. 3206 Trojans.
Jupiter replay matches snapshots to 1.3e-9 au.

| i0 (°) | 0–5 | 5–10 | 10–15 | 15–20 | 20–25 | 25–30 | 30–35 | 35–40 |
|---|---|---|---|---|---|---|---|---|
| N4/N5 | 1.28 ± 0.12 | 1.52 ± 0.13 | 1.36 ± 0.12 | 1.75 ± 0.16 | 1.82 ± 0.20 | 2.13 ± 0.26 | 1.99 ± 0.29 | 2.16 ± 0.40 |
| capture eff. | 0.134 | 0.162 | 0.152 | 0.131 | 0.099 | 0.082 | 0.057 | 0.036 |

- **Overall 1.61 ± 0.06.** By e0: below 15° the e0 < 0.025 half gives 1.47–1.99 and the
  0.025–0.05 half 1.08–1.29 (as C2); above 20° both halves are ≈ 2.
- **The excess rises with i0**, from ≈ 1.3 at i0 < 5° to ≈ 2.1 above 25°. Capture falls
  with i0 on both sides, faster for L5. In final inclination: 1.28 ± 0.09 (I < 10°),
  1.4–1.6 (10–20°), 1.74 ± 0.14 (20–30°), 2.1 ± 0.2 (> 30°).
- Captured Trojans are pumped in I at high i0 (median 32° → 45° for i0 30–40°);
  L5 Trojans have smaller libration amplitudes than L4 (D 15–29° vs 20–31°).
- **Implication:** gas-era migration makes a HOT-L4 excess, the opposite of the observed
  one (≈ 2 below I_p 15°, ≈ 1 above). On these tracks, migration cannot be the source
  of the cold excess unless the hot population is replaced or diluted later by a
  symmetric component (e.g. capture at the instability, Nesvorný+2013 style).

### D2: C2 rate with an earlier gap, M_gap 12.5 M⊕ (2026-11-06 21:10)

`mig_d2_mgap12`: ν_I 0.9, M_gap 12.5 (C2: 25); J 10.35 → 5.40, S 15.95 → 11.50; disc
a0 9.81–10.76, e0 ≤ 0.1, i0 ≤ 15°. 3044 Trojans. Replay matches to 1.3e-9 au.

| | all | e0 < 0.025 | 0.025–0.05 | 0.05–0.075 | 0.075–0.1 | i0 0–5 / 5–10 / 10–15 |
|---|---|---|---|---|---|---|
| C2 (M_gap 25) | 1.37 ± 0.05 | 1.81 ± 0.14 | 1.43 ± 0.10 | 1.18 ± 0.09 | 1.18 ± 0.09 | 1.40 / 1.29 / 1.44 |
| **D2 (M_gap 12.5)** | **1.19 ± 0.04** | **1.38 ± 0.11** | 1.12 ± 0.08 | 1.04 ± 0.07 | 1.29 ± 0.10 | 1.16 / 1.10 / 1.34 |

- Halving M_gap lowers the excess from C2 to about C1's level, although ν_I is the same:
  the co-orbital side ratio still peaks (1.64 at 158 kyr) but Jupiter slows before it
  is heavy enough to freeze that ratio in. The excess is set by how fast Jupiter
  migrates while it is ≈ 10–30 M⊕, not by ν_I alone.

### D3: base rate with a late gap, M_gap 50 M⊕ (2026-11-06 23:15)

`mig_d3_mgap50_nu0225`: ν_I 0.225, M_gap 50 (C0b: 25); J 12.99 → 5.40, S 22.73 → 11.50;
disc a0 12.31–13.51, e0 ≤ 0.1, i0 ≤ 15°. 3283 Trojans. Replay matches to 1.3e-9 au.

**Five runs, cold zone (e0 ≤ 0.1, i0 ≤ 15°), ordered by ν at 20 M⊕:**

| run | ν_I | M_gap | ν(10 / 20 / 30 M⊕) | all | e0 < 0.025 | peak side ratio |
|---|---|---|---|---|---|---|
| C0b | 0.225 | 25 | 0.21 / 0.15 / 0.08 | 1.08 ± 0.04 | 1.25 ± 0.10 | 1.18 |
| D2 | 0.9 | 12.5 | 0.60 / 0.18 / 0.06 | 1.19 ± 0.04 | 1.38 ± 0.11 | 1.64 |
| **D3** | 0.225 | 50 | 0.22 / 0.21 / 0.19 | **1.19 ± 0.04** | **1.39 ± 0.11** | 1.27 |
| C1 | 0.45 | 25 | 0.42 / 0.30 / 0.17 | 1.25 ± 0.04 | 1.66 ± 0.12 | 1.46 |
| C2 | 0.9 | 25 | 0.84 / 0.60 / 0.33 | 1.37 ± 0.05 | 1.81 ± 0.14 | 1.85 |

- D3 sits between C0b and C1. D2 and D3 reach the same final excess along very
  different ν(m): D2 is fast while light and slow by 30 M⊕, D3 is slow throughout but
  stays slow past 30 M⊕. Neither ν_I, ν(10 M⊕) nor ν(30 M⊕) orders the five runs; ν near
  20 M⊕ does (monotonic in both columns). A fast early phase builds a large side ratio
  (D2 peaks at 1.64) that relaxes if Jupiter slows before ≈ 20 M⊕.
- By i0 (0–5 / 5–10 / 10–15): 1.17 / 1.35 / 1.06; no trend within 15°.
- Caveat: five runs, ±0.1 errors; "ν at ≈ 20 M⊕" is an ordering, not a fit.

### D4: C2 continued to 10 Myr after the gas phase (2026-11-07 07:49)

`mig_d4_c2_long`: C2 exactly (same seed and particles), classified in the last 1 Myr of
10 Myr after the forcing ends; giants free (J 5.40, S 11.50, U 19.9, N 31.2).
Replay matches to 1.3e-9 au over 10.7 Myr. 58.5 CPU-h.

| | all | e0 < 0.025 | 0.025–0.05 | 0.05–0.075 | 0.075–0.1 | i0 0–5 / 5–10 / 10–15 |
|---|---|---|---|---|---|---|
| C2 (1.2 Myr) | 1.37 ± 0.05 | 1.81 ± 0.14 | 1.43 ± 0.10 | 1.18 ± 0.09 | 1.18 ± 0.09 | 1.40 / 1.29 / 1.44 |
| **D4 (10 Myr)** | **1.38 ± 0.05** | **1.82 ± 0.15** | 1.42 ± 0.10 | 1.20 ± 0.09 | 1.19 ± 0.09 | 1.41 / 1.30 / 1.45 |

- Particle by particle: of C2's 1716 L4 and 1251 L5 Trojans, exactly 42 of each are
  lost by 10 Myr (survival 97.6% vs 96.6%); no particle changes swarm, none is gained.
- The migration excess is frozen once the gas phase ends; with the free wide
  configuration, 10 Myr of erosion is symmetric and small (as the no-migration control
  and the 10 Myr pilot).

### D5, D6: inclinations to 40° at the C1 and C0b rates (2026-11-07 12:32)

D1's set-up (e0 ≤ 0.05, i0 0–40° uniform, 15000 pairs) on the C1 tracks
(`mig_d5_c1_hot`, a0 10.95–11.75) and the C0b tracks (`mig_d6_c0b_hot`, a0 8.2–9.0).
4741 and 3782 Trojans. Replays match to 1.3e-9 au.

**N4/N5 by i0, three rates:**

| i0 (°) | 0–5 | 5–10 | 10–15 | 15–20 | 20–25 | 25–30 | 30–35 | 35–40 | all |
|---|---|---|---|---|---|---|---|---|---|
| D6 (ν_I 0.225) | 1.23 | 1.00 | 1.14 | 1.23 | 1.20 | 1.18 | 1.12 | 0.88 | 1.13 ± 0.04 |
| D5 (ν_I 0.45) | 1.15 | 1.32 | 1.23 | 1.43 | 1.32 | 1.23 | 1.20 | 0.97 | 1.25 ± 0.04 |
| D1 (ν_I 0.9) | 1.28 | 1.52 | 1.36 | 1.75 | 1.82 | 2.13 | 1.99 | 2.16 | 1.61 ± 0.06 |

(per-bin errors ≈ 0.08–0.12 for D5/D6, 0.12–0.40 for D1)

| e0 / i0 | D6 0–15 | D6 15–40 | D5 0–15 | D5 15–40 | D1 0–15 | D1 15–40 |
|---|---|---|---|---|---|---|
| < 0.025 | 1.16 ± 0.09 | 1.24 ± 0.08 | 1.40 ± 0.09 | 1.45 ± 0.08 | 1.75 ± 0.13 | 2.12 ± 0.17 |
| 0.025–0.05 | 1.07 ± 0.07 | 1.07 ± 0.06 | 1.11 ± 0.06 | 1.11 ± 0.06 | 1.17 ± 0.08 | 1.72 ± 0.13 |

(D5's 0.025–0.05 halves have identical counts, 654/590 each; checked: a coincidence,
thresholds of 14° and 16° give 612/540 | 696/640 and 684/614 | 624/566.)

- **At ν_I ≤ 0.45 the excess is flat in i0 out to 35°** (perhaps falling to ≈ 0.9–1.0 at
  35–40°). It rises with i0 only at the fastest rate (D1), where high-i L5 capture fails.
  In final I, D6 gives 1.14 / 1.15 / 1.07 and D5 1.24 / 1.32 / 1.17 for I < 15 / 15–30 / > 30°.
- Capture efficiency falls with i0 at every rate (D6: 0.14 → 0.08; D5: 0.19 → 0.07).
- **Summary of Block D for the cold-L4 question:** gas-era migration on physical tracks
  makes an excess sorted by e0 (circular starts) and set by the rate near 20 M⊕; it
  is either flat in I (ν_I ≤ 0.45) or larger at high I (ν_I 0.9), and it is frozen for at
  least 10 Myr afterwards (D4). It never makes the observed cold-only excess (≈ 2 below
  I_p 15°, ≈ 1 above). That needs another ingredient: a later symmetric hot component, an
  e–I correlated planetesimal disc, or an I-dependent loss after the gas phase.

### D7: full (e0, i0) map at the C2 rate (2026-11-07 14:52)

`mig_d7_c2_warm`: C2 tracks and zone (a0 15.17–16.65), **e0 ≤ 0.3, i0 ≤ 40°**, 20000 pairs.
2174 Trojans; overall 1.81 ± 0.08. Replay matches to 1.3e-9 au.

N4/N5 (capture efficiency) by e0 and i0:

| e0 \ i0 | 0–10° | 10–20° | 20–30° | 30–40° |
|---|---|---|---|---|
| 0–0.05 | 1.73 ± 0.22 (0.160) | 1.57 ± 0.22 (0.118) | 2.08 ± 0.35 (0.093) | 2.41 ± 0.61 (0.045) |
| 0.05–0.1 | 1.29 ± 0.16 (0.152) | 1.53 ± 0.20 (0.143) | 2.15 ± 0.45 (0.061) | 2.28 ± 0.64 (0.035) |
| 0.1–0.15 | 1.44 ± 0.22 (0.111) | 2.76 ± 0.60 (0.076) | 3.07 ± 0.91 (0.036) | 2.8 ± 1.5 (0.011) |
| 0.15–0.2 | 1.61 ± 0.32 (0.062) | 3.3 ± 1.0 (0.040) | 1.13 ± 0.32 (0.030) | – |
| 0.2–0.3 | 2.88 ± 0.66 (0.029) | 3.0 ± 0.9 (0.017) | 2.3 ± 1.0 (0.007) | – |

- At e0 ≤ 0.1 and i0 < 15° it matches C2 (dip at e0 0.05–0.1). Beyond that the
  excess **grows with both e0 and i0**: wherever capture is inefficient, L5 capture fails first.
- Reweighted to discs (`s8_reweight.py`, N4/N5 in final-I bins 0–10/10–20/20–30/>30°):
  every model, independent or cold-circular + hot-eccentric mixtures, gives a ratio
  that RISES with I (e.g. mix_50: 1.66 / 1.77 / 2.23 / 2.18; observed 2.15 / 1.38 / 0.93 / 1.12).
  At the C2 rate an e–I correlated disc does not rescue the cold-only pattern.
