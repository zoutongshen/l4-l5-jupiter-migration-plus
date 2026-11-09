# Stage 8 summary: can gas-era migration make the cold-L4 excess? (2026-11-05 to 11-07)

**Target.** The observed asymmetry is an extra cold L4 component. N4/N5 by proper I
(complete sample, families removed):

| I_p | 0–10° | 10–20° | 20–30° | > 30° |
|---|---|---|---|---|
| observed N4/N5 | 2.15 ± 0.14 | 1.38 ± 0.09 | 0.93 ± 0.08 | 1.12 ± 0.21 |

**Short answer.** Gas-era migration on physical tracks gives a real L4 excess (up to about 1.8),
but it is sorted by the starting **eccentricity**, not by inclination. At no rate, gap mass
or disc distribution does it reproduce the cold-only pattern above.

Full log with every table: `docs/RUN_migration_capture.md` §6. Commits fae78ae … (D10/D11 latest).

---

## Model

- REBOUND WHFast, dt = 1% of P_J; the inner planets folded into the Sun.
- **All four giants are separate bodies.** Each grows and migrates on its own track:
  - pebble core, then slow envelope contraction, then runaway gas accretion, over 0.69 Myr;
  - ν = ν_I / (1 + (m/M_gap)³), so the migration rate is type I while light and falls off once a gap opens;
  - calibrated on Pirani+2019's ×0.25 track: ν_I = 0.225, M_gap = 25 M⊕;
  - each giant is held on a circular orbit until its track ends, then moves freely.
- **Mirror-twin disc.** Every particle has a mirror partner, so any L4/L5 difference comes from the forcing, not from the initial conditions.
- **Trojan definition:** tadpole in every 10 kyr window of the last 1 Myr. Results are binned in e0, i0 and final I.
- **End configuration:** J 5.4, S 11.5, U 19.9, N 31.2 au, i.e. a wide J–S pair just before the instability. Jupiter's later inward step to 5.2 au is not modelled.
- **Checks:**
  - a no-migration control is symmetric;
  - reversing the migration flips the sign (B3: 0.83);
  - replaying the giants alone matches Jupiter to 1e-9 au in every run.

## What the J–S configuration does (Block B)

| configuration | effect on Trojans | literature |
|---|---|---|
| Saturn crosses the 2:1 while converging (Pirani's own track + Saturn) | **all destroyed** (315 → 0 within 20 kyr of the crossing) | Morbidelli+2005, Marzari & Scholl 2007 |
| Compact chain (Saturn near 8:5) | 0.5% survive | Roig & Nesvorný 2015 |
| Wide (P_S/P_J ≈ 3) | 100% survive (1.02 ± 0.03) | — |

All the science runs therefore use the wide end configuration. Caveat: with forced
circular orbits the giants cannot lock into resonance. That is the likely reason we
lose everything at the 2:1, whereas Pirani 2019a keeps Trojans in a 1:2 lock.

## Main results (Blocks C and D, all four giants, cold-to-warm twin discs)

**1. The excess needs near-circular starts and grows with rate, then saturates.**
Cold zone (e0 ≤ 0.1, i0 ≤ 15°):

| ν_I (rate) | all | e0 < 0.025 | e0 0.075–0.1 |
|---|---|---|---|
| 0.225 (C0b) | 1.08 ± 0.04 | 1.25 ± 0.10 | 1.05 ± 0.07 |
| 0.45 (C1) | 1.25 ± 0.04 | 1.66 ± 0.12 | 1.08 ± 0.06 |
| 0.9 (C2) | 1.37 ± 0.05 | **1.81 ± 0.14** | 1.18 ± 0.09 |

The saturation level of about 1.8 matches Pirani's 1.6–1.8.

**2. What sets it is Jupiter's speed when the gap brake starts to bite.**
- Halving the gap mass at ν_I 0.9 drops the excess to 1.19 (D2 vs C2's 1.37).
- Doubling it at ν_I 0.225 raises it to 1.19 (D3 vs C0b's 1.08).
- The co-orbital L4/L5-side imbalance builds from about 3 M⊕, peaks, then partly relaxes. The peak
  sits at m ≈ 0.6–0.7 M_gap, where ν has fallen to 0.6–0.8 ν_I: 9 M⊕ (D2), 16–23 M⊕ (M_gap 25), 33 M⊕ (D3).
- E1 and E1b move the pebble isolation mass from 20 to 10 and to 30 M⊕. The peak stays at 16–17 M⊕
  (ν ≈ 0.70 ν_I) although it moves in time from 158 to 230 kyr. So the brake sets the peak, not the
  growth law or the clock. The "≈ 20 M⊕" seen earlier is just 0.7 × M_gap for M_gap = 25.
- The peak mass is independent of e0. The fraction of the peak that survives falls with e0
  (C2: 60% at e0 < 0.025, 25% at 0.05–0.1), which is where the e0-sorting comes from.

**3. The excess is frozen once the gas phase ends.** C2 run on to 10 Myr with free giants (D4): 1.38 ± 0.05. Exactly 42 L4 and 42 L5 Trojans were lost, and none switched swarm.
The warm maps behave the same way (D10, D11 = D7, D8 to 10 Myr): 2–4% lost, the same in both swarms
and in every I bin, with no swaps. N4/N5 by final I is unchanged to within 0.04
(D10: 1.50 / 1.64 / 2.20 / 2.32; D11: 1.26 / 1.24 / 1.16 / 1.01).

**4. The inclination dependence is wrong at every rate.** Near-circular or full-warm starts, i0 0–40°, N4/N5 by i0:

| ν_I | 0–10° | 10–20° | 20–30° | 30–40° | trend |
|---|---|---|---|---|---|
| 0.225 (D6, e0 ≤ 0.05) | 1.10 | 1.18 | 1.19 | 0.99 | flat |
| 0.45 (D5, e0 ≤ 0.05) | 1.23 | 1.32 | 1.28 | 1.10 | slightly falling |
| 0.45 (D8, e0 ≤ 0.3) | 1.28 | 1.17 | 1.26 | **0.80** | slightly falling |
| 0.65 (D9, e0 ≤ 0.1) | 1.18 | 1.35 | 1.45 | 1.18 | flat to rising |
| 0.9 (D1, e0 ≤ 0.05) | 1.41 | 1.53 | 1.95 | **2.05** | rising |
| 0.9 (D7, e0 ≤ 0.3) | 1.60 | 1.89 | 2.05 | **2.37** | rising |

(errors ±0.06–0.10 below 30°, ±0.08–0.40 above)

At ν_I 0.45 the ratio falls with inclination, the observed direction, but only to about a quarter of the observed size.
At ν_I 0.9 the excess sits mostly at **high** inclination, because capture into L5 fails first wherever capture is rare (high e0 or high i0).

**5. A disc where e and I are correlated doesn't rescue it.** `scripts/s8_reweight.py` reweights the full (e0, i0) maps (D7, D8) to several discs:
- independent Rayleigh distributions in e and sin I;
- a mix of cold-circular and hot-eccentric bodies.

Every model either stays at about 1.2 at all I or rises with I. Example in final-I bins, mix_50 at ν_I 0.9: 1.66 / 1.77 / 2.23 / 2.18.

## What this means

- Migration during the gas era is a working source of an **L4 excess** (sign set by
  inward migration, size up to about 1.8, fixed afterwards). On its own, though, it gives an
  **e-sorted or hot** excess, not the observed **cold-only** one.
- The observed I-sorting therefore needs something else. Candidates:
  1. a later process that depends on inclination: instability-era capture or loss (e.g. Nesvorný+2013 jump capture adds a symmetric hot population and dilutes the hot excess);
  2. physics this model leaves out: giants that can lock into resonance and keep some eccentricity, gas drag on planetesimals, a damped rather than forced migration.

## Limitations (exploration level; to discuss with Simon)

- **Forced giants.** The giants are held on circular orbits along their tracks: no resonance capture, e ≈ 0.
- **Sharp gas cut-off.** The gas disc vanishes instantly at 0.69 Myr, and it has no effect on the particles.
- **Assumed tracks.** The ice-giant tracks are assumed; Pirani's rate runs carry only J + S.
- **End configuration.** The wide configuration was chosen so that Trojans survive, not because it is the most realistic.
- **Disc.** Starting elements are on a uniform grid, so other distributions are applied by reweighting afterwards.
- **Time covered.** 1.2 Myr after the gas phase, 10 Myr in D4, D10 and D11. No 4.5 Gyr erosion.
- **No instability.** No instability step (Block E, deferred).

## Run list

| id | run | key change | Trojans | N4/N5 |
|---|---|---|---|---|
| B2 | mig_g2_pirani | Pirani track + Saturn crosses the 2:1 | 0 | – |
| B3 | mig_g3_pirani_out | B2 reversed | 176 | 0.83 ± 0.13 |
| C0 | mig_c0_nominal | warm disc, ν_I 0.225 | 396 | 0.96 ± 0.10 |
| C0b | mig_c0b_cold_zone | cold zone, ν_I 0.225 | 3312 | 1.08 ± 0.04 |
| C1 | mig_c1_nu045_cold | ν_I 0.45 | 4401 | 1.25 ± 0.04 |
| C2 | mig_c2_nu09_cold | ν_I 0.9 | 2967 | 1.37 ± 0.05 |
| D1 | mig_d1_c2_hot | ν_I 0.9, i0 to 40° | 3206 | 1.61 ± 0.06 |
| D2 | mig_d2_mgap12 | ν_I 0.9, M_gap 12.5 | 3044 | 1.19 ± 0.04 |
| D3 | mig_d3_mgap50_nu0225 | ν_I 0.225, M_gap 50 | 3283 | 1.19 ± 0.04 |
| D4 | mig_d4_c2_long | C2 to 10 Myr | 2883 | 1.38 ± 0.05 |
| D5 | mig_d5_c1_hot | ν_I 0.45, i0 to 40° | 4741 | 1.25 ± 0.04 |
| D6 | mig_d6_c0b_hot | ν_I 0.225, i0 to 40° | 3782 | 1.13 ± 0.04 |
| D7 | mig_d7_c2_warm | ν_I 0.9, e0 to 0.3, i0 to 40° | 2174 | 1.81 ± 0.08 |
| D8 | mig_d8_c1_warm | ν_I 0.45, e0 to 0.3, i0 to 40° | 3992 | 1.19 ± 0.04 |
| D9 | mig_d9_nu065 | ν_I 0.65, e0 to 0.1, i0 to 40° | 4530 | 1.29 ± 0.04 |
| D10 | mig_d10_d7_long | D7 to 10 Myr | 2099 | 1.81 ± 0.08 |
| D11 | mig_d11_d8_long | D8 to 10 Myr | 3925 | 1.18 ± 0.04 |
| E1 | mig_e1_miso10 | C2 with M_iso 10 | 2616 | 1.43 ± 0.06 |
| E1b | mig_e1_miso30 | C2 with M_iso 30 | 3143 | 1.43 ± 0.05 |

GIFs (Trojans with Saturn, Uranus and Neptune drawn in): `campaign/reports/<run>/capture.gif`.
