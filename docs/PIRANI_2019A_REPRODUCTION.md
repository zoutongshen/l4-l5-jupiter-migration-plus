# Reproducing Pirani et al. 2019a (stage 9)

This is the single reference for the reproduction: what she did, what we built, the growth tracks, the checks, every assumption, and what is still open.

- **Paper:** Pirani, Johansen, Bitsch, Mustill & Turrini 2019, A&A 623, A169 (arXiv:1902.04591), in `literature/02_mechanisms/`.
- **Why:** Simon asked on 2026-11-10 for her paper to be reproduced exactly before we extend it.
- **Plan:** `~/.claude/plans/vivid-drifting-hummingbird.md`. Progress is tracked in TickTick, under Trojans → Meeting.
- **Compute:** haagsebeek.strw.leidenuniv.nl (`ssh haagsebeek`, about 24 cores, data on `/data2/zshen`).

---

## 1. What she ran

### 1.1 Set-up (§3, App. A–B)

| item | value | where |
|---|---|---|
| integrator | parallel MERCURY, hybrid symplectic, **dt = 50 d** | §3 |
| disc | **4–26 au: 44 rings of 0.5 au × 1000 = 44,000 massless particles**. Equal numbers per ring give Σ ∝ r⁻¹. e ∈ [0, 0.01], i ∈ [0°, 0.01°]. No twins | §3, Fig. 2 |
| repeats | **10 per scenario**, with a, e and i re-randomised. Quoted value = mean of the 10; ± = their unbiased standard deviation (not the error of the mean); the ratio error is propagated | §3, §4.1.1 |
| time | t = 0 → 5 Myr. Seeds of ~1e-2 M⊕ are present from t = 0 and start to grow and migrate at **J 2.31, S 2.56, U ~2.70, N ~2.70 Myr**. The gas is gone at **3 Myr** (Ṁ ≤ 2e-9 M☉/yr) | §3.1, eq. 1 |
| gas disc | **Ida, Guillot & Morbidelli 2016, α = 1e-3** (BLJ15 had 0.0054), with Ṁ(t) from Hartmann+1998 | §3.1 |
| growth tracks | "generated similarly to Bitsch, Lambrechts & Johansen 2015 (BLJ15)", with the BLJ15 pebble flux corrected (BLJ15 corrigendum, A&A 609, C2). Gas accretion and migration follow BLJ15's prescriptions | §1, §3.1 |
| planets | a follows the track. e and i are free, damped by "tidal gas drag" (Cresswell & Nelson 2008 fits, t_wave of Tanaka & Ward 2004), **only below 30 M⊕** and only while the gas is present | App. B |
| particles | aerodynamic gas drag: Adachi+1976 orbit-averaged ȧ, ė, i̇; r = 50 km, ρ = 1 g/cm³, C_D = 0.44; until 3 Myr | App. A |
| Trojans | perihelion and aphelion both within ±2.5 R_H of a_J: tadpoles, horseshoes and quasi-satellites, with "high-e interlopers" excluded. Counted at 5 Myr | §4.1.1 |

### 1.2 Scenarios and what each produced

| scenario | planets | repeats | her results |
|---|---|---|---|
| nominal | J, S, U, N from 18, 21, 23, 24 au. They end in a compact chain, J near 5.0 au, **J and S locked in the 1:2 resonance** | 10 | Table 2 (5–200 Myr), Figs. 4–7, 9–11, 14; Hildas in Table 4, Figs. 15, 17; belt implantation §4.4 |
| in situ | J and S only, at 5.4 and 8.6 au, same growth clock | 10 | Table 3 first row (0.94 ± 0.35), Table 4, Fig. 16 |
| rate scan | J and S only, at ×0.15 / 0.25 / 0.5 / 0.75 / 1.0 / 1.25 / 1.5 / 2.0; starts in Table 1 | 10 each | Table 3, Fig. 8, Table 5, Fig. 18 |
| outward | Jupiter seed at 5 au, migrating out to 18 au | 1 | Fig. 12 (N4/N5 ≈ 0.6) |
| §5 instability | after 3 Myr: a(t) = a0 + Δa[1 − exp(−t/τ)], with Δa_J = −0.2, Δa_S = +0.9, τ = 0.5 Myr or 5 kyr, and matching e(t), i(t) laws (eqs. 2–4); plus single jumps of Jupiter by 0.16 and 0.23 au. **Narrow discs** (around Jupiter's core for Trojans; 4–9 au for Hildas). Run to 600 Myr and 1 Gyr | — | §5, Figs. 19–21 (40% / 20% of Trojans survive the jumps) |

### 1.3 Gaps in the paper (each needs an assumption here; to ask her)
- **G1:** her growth tracks are only figures (m vs a), with no time axis. Saturn, Uranus and Neptune have no clock at all. → §3.
- **G2:** the paper gives no Saturn track for the rate scan, only its starting radii (Table 1). → §3.4.
- **G3:** the disc extent for the fast rates is not stated. At ×1.25–2, J and S start at 22–35.5 au, beyond the 26 au disc, yet these runs capture the most Trojans.
- **G4:** how a(t) is imposed. Her Jupiter ends at ~5.0 au, and she notes the end point differs between rates. That points to an imposed ȧ (a force), not an imposed a. → §2.
- **G5:** particle removal rules are not stated. We use MERCURY's defaults: collision with a planet or the Sun, ejection at 100 au.
- **G6:** the "high-e interloper" cut and the Hilda definition are not quantified.
- **G7:** which planets were in the outward run is not stated.

---

## 2. The N-body physics we built (committed c0444de)

| module | what | checks (`tests/test_pirani.py`) |
|---|---|---|
| `trojans/gasdisc.py` | Ida+2016 eqs. 9–13: h = max(h_vis, h_irr), Σ = Ṁ/(3πα h² r² Ω), ρ_mid, ∂lnP/∂lnr, η | Σ within 5% of Ida eqs. 12–13; Ṁ(3 Myr) = 2.05e-9 |
| `trojans/pirani_forces.py` | giant forces as a REBOUND callback on the giants only: migration a_m = v ȧ/(2a) from the track; CN08 e/i damping below 30 M⊕; mass m(t) from the track | the planet follows its track to 0.2%; e and i damp at the B.1/B.2 rates; no damping above 30 M⊕ |
| `trojans/pirani_forces.py` | Adachi particle drag, applied every 100 yr to the orbital elements | reproduces A.5–A.7 to 1e-9 |
| `trojans/integrate.py` | MERCURIUS option; impact removal (Sun radius 0.005 au, planet radii at 1.33 g/cm³); `collision` removal reason | a resume reproduces a straight run bit for bit with everything on |
| `trojans/disc.py` | `RingDisc`: 44 × 1000, no twins | ring counts and ranges |

**Decisions inside the physics:**
- **Accretion-rate law:** her eq. 1 has no time offset and diverges at t = 0. We use BLJ15's (t + 10⁵ yr) form, which differs from hers by < 7% after 2 Myr.
- **Inclination damping:** Cresswell & Nelson's printed acceleration, −v_z/t_i, damps i at only half the rate (measured in a test). Pirani's B.2 and Tanaka & Ward say di/dt = −i/t_i, so we use −2v_z/t_i, as REBOUNDx does.
- **Impacts:** REBOUND's own collision search is O(N²) with test particles (measured 100× slower). Impacts are therefore checked at the runner's checks, every 8 steps.
- **Integrator cost:** MERCURIUS costs 0.47 CPU-s per particle-Myr, against 0.31 for WHFast (bench, 5000 particles).

**α = 1e-3 everywhere:**
- the disc: h_vis, Σ, ρ, η;
- the track model: pebble flux, Stokes number, pebble scale height, the gap criterion's Reynolds number, the type-II viscosity, the PBK11 viscosity;
- the N-body damping and drag, which read the same disc.

The track model refuses any other α (`test_alpha_is_pirani_value_everywhere`), and `reports/pirani_tracks/summary.json` records `"alpha": 0.001`.

---

## 3. Her growth tracks

### 3.1 Her curves, exactly (`scripts/s9_digitize_pirani.py` → `configs/tracks/pirani19a/fig*.csv`)
**How they were extracted.** Figs. 1, 3 and 13 are vector graphics in the arXiv PDF, so the coordinates of every curve were read from the PDF with PyMuPDF (installed in `trojan-rb`). This is not pixel digitising. The page-to-data maps are anchored on the tick labels; Fig. 13's y-axis is mapped from its label centres, because the frame edge sits at 10^−1.3.

**What was extracted.**
- **Fig. 1:** J, S, U and N, a vs m, with a flag for solid (pebble) vs dashed (gas) phase.
- **Fig. 3:** Jupiter at 8 rates.
- **Fig. 13:** Jupiter's |ȧ|(m) at nominal, ×0.25, "×0.15" and ×2. Each curve has only 42 vertices, one per ~0.07 dex.

**Checks on the extraction.** The starting radii read from the figures match Table 1. Fig. 1's Jupiter and Fig. 3's ×1.00 are the same curve.

### 3.2 The rate scan is one track, stretched
- **Paths:** for every Fig. 3 curve, a_k(m) − a_end,k = k·(a_nom(m) − a_end,nom). The fitted k is constant to 1–2% along the whole curve.
- **Speeds:** every Fig. 13 curve is exactly k × the nominal (ratios 0.126, 0.251 and 1.997, constant in m). Her §3.3 says the same: the rate runs are "a contraction or expansion of the nominal model growth tracks", with the same m(t).
- **Her "×0.15" is k = 0.125.** The rate ratio is 0.126, and Fig. 8's Δa = 1.7 au = 0.125 × 13.45 au.
- **End points** (from Fig. 3): 5.43, 5.44, 5.07, 5.00, 5.07, 5.34, 5.64 and 5.68 au, for k = 0.125 to 2.

### 3.3 The growth model (`trojans/pirani_tracks.py`): BLJ15's recipe on the Ida+2016 disc

| piece | prescription |
|---|---|
| pebble accretion | Ida+2016 eqs. 59–61 (χ = 1, η′ = η, ζ = 1/(1 + τ²)). Flux = **FLUX_FACTOR** × Ida eq. 54. Stokes number **TAU_S** (fixed; Ida eq. 47 is available as an option) |
| isolation | BLJ15 eq. 16, 20 (h/0.05)³ M⊕, × **ISO_FACTOR**. 10% of the accreted mass is envelope before isolation |
| contraction | Piso & Youdin 2014 (BLJ15 eq. 17; f = 0.2, κ_env = 0.05 cm²/g, ρ_c = 5.5 g/cm³) while M_env < M_core |
| runaway | Machida+2010 (BLJ15 eqs. 18–19), the smaller branch, **capped at 0.8 Ṁ*** |
| type I | Paardekooper+2011 total torque (Lindblad + saturated corotation; Bell & Lin ice opacity, 0.5% dust), × TORQUE_SCALE. Only used in the forward check |
| gap / type II | Crida+2006 P, Crida & Morbidelli 2007 f(P); P < 1 → τ_II = τ_ν max(1, M/(4πΣr²)) (BLJ15 eqs. 21–23) |

**Calibrations.** Pirani gives none of these numbers.

| | value | fitted to |
|---|---|---|
| TAU_S | **0.08** | the planets' positions in her Fig. 4 snapshots (§3.5). BLJ15's range is 0.05–0.5; Ida eq. 47 gives ~0.14, which puts Saturn 32 kyr early |
| FLUX_FACTOR | **4.3** | Jupiter's 306 M⊕ at 3 Myr with TAU_S. BLJ15's own flux is "corrected" without a number |
| ISO_FACTOR | **1.066** | where her dashed (gas) lines start, for J and S together (her 20.6 and 22.7 M⊕) |
| Jupiter start (her "t ~ 2.31 Myr") | **2.298 Myr** | her Fig. 4 snapshots at 2.5 and 2.6 Myr, while the other giants are still seeds; her 2.3 Myr panel already shows Jupiter 0.04 au inside its start |
| U, N start times (her "~2.70 Myr") | **2.708, 2.719 Myr** | their final masses (18.1, 14.4 M⊕) |
| TORQUE_SCALE | 0.6 | forward check only (§3.5) |

### 3.4 How the N-body tracks are built (`scripts/s9_pirani_tracks.py`)
**Principle:** take from her what she shows, model only what she doesn't.

**Path a(m): her Fig. 1 curve.**
- **Smoothing:** medians in log-m bins (0.25 dex below 2 M⊕, 0.08 dex above), joined by a monotone spline. Her plotted vertices step by 0.01–0.03 au where a barely changes, and the N-body forcing differentiates a(t). The largest deviation from her curve is 0.19 au (J) and 0.21 au (S), at the isolation corner, and ≤ 0.012 au for U and N.
- **Jupiter below 3 M⊕:** a(m) is built from her Fig. 13 rate along the clock, because her path is too flat there to differentiate. That puts Jupiter's seed at **18.22 au**, against her 18.40 (×2.00 then starts at 31.96 vs Table 1 32.3).

**Clock m(t).**
- **Jupiter:**
  - 0.018 → 3 M⊕: the growth model, from 2.298 Myr.
  - **3 → 100 M⊕: her own clock**, dm/dt = |ȧ|_Fig13 / |da/dm|_Fig1 (see §3.6). It is 222 kyr long; the model alone takes 224.
  - Above 100 M⊕: the growth model, stretched by 1.005 so that it ends at 3.0 Myr, with 317 M⊕.
- **Saturn, Uranus, Neptune:** the growth model along their Fig. 1 paths, from 2.56 / 2.708 / 2.719 Myr.

**Format.** Each table starts at t = 0 with the seed parked at its starting a, then has 3000 rows to 3 Myr. Columns: t_yr, a_au, m_earth.

| file (`configs/tracks/pirani19a/`) | what | start → end |
|---|---|---|
| `track_nominal_jupiter.csv` | nominal Jupiter | 18.22 → 5.08 au, 0.018 → 317 M⊕ |
| `track_nominal_saturn.csv` | nominal Saturn | 21.27 → 8.02 au, → 113 M⊕ |
| `track_nominal_uranus.csv`, `_neptune.csv` | nominal ice giants | 22.50 → 16.01 au, → 18.1 M⊕; 24.00 → 18.57 au, → 14.4 M⊕ |
| `track_rate_x{0.15,0.25,0.50,0.75,1.00,1.25,1.50,2.00}_jupiter.csv` | a_end,k + k(a_nom − a_end,nom), same m(t); ends from Fig. 3 | e.g. ×0.15: 7.08 → 5.43; ×2.00: 31.96 → 5.68 |
| `track_rate_x…_saturn.csv` | same rule on nominal Saturn; end chosen so the start equals Table 1 (G2) | 10.45 … 35.5 au starts, exactly Table 1 |
| `track_insitu_{jupiter,saturn}.csv` | 5.4 and 8.6 au, nominal clocks | — |
| `track_outward_jupiter.csv` | nominal path mirrored, a = 5.08 + 18.22 − a_nom | 5.08 → 18.22 au |

### 3.5 Verification (`reports/pirani_tracks/`: `verification.png`, `snapshots.png`, `wobble.png`, `summary.json`)

**Planet positions in her Fig. 4** (= Fig. 5 to 0.05 au). These come from one of her nominal N-body simulations, read from the raster panels (±0.1 au). They're a direct check of the clocks, because her figures of the tracks have no time axis.

| t (Myr) | J: hers / ours | S: hers / ours | U: hers / ours | N: hers / ours |
|---|---|---|---|---|
| 2.3 | 18.36 / 18.22 | 21.22 / 21.26 | 22.48 / 22.50 | 24.00 / 24.00 |
| 2.5 | 13.97 / 13.94 | 21.22 / 21.26 | 22.48 / 22.50 | 24.00 / 24.00 |
| 2.6 | 6.99 / 6.99 | 21.22 / 21.24 | 22.48 / 22.50 | 24.00 / 24.00 |
| 2.7 | 6.27 / 6.30 | 20.50 / 20.63 | 22.57 / 22.50 | 24.00 / 24.00 |
| 2.8 | 5.82 / 5.86 | 16.84 / 16.30 | 22.30 / 22.40 | 23.82 / 23.91 |
| 2.9 | 5.37 / 5.42 | 9.51 / 9.62 | 20.60 / 21.03 | 23.01 / 22.89 |
| 3.0 | 4.93 / 5.08 | 7.88 / 8.02 | 15.48 / 16.01 | 19.25 / 18.57 |

**Timing offsets** (ours later +):
- Jupiter: 0 / 0 / +7 / +9 / +11 kyr at 2.5–2.9 Myr.
- Saturn: +6 / −8 / +3 kyr at 2.7–2.9 Myr.

**Residuals:**
- **All four planets end inside or outside their own plotted tracks in her N-body.** J and S end 0.15 au inside, and U and N are 0.5 and 0.7 au off in opposite directions. Her planets are free bodies under an imposed migration, so these are planet–planet interactions (G4). Our N-body has the same set-up, so the pilot should show them too.
- **Uranus at 2.9 Myr** is 0.43 au further out than hers, i.e. 13 kyr late. That's a small timing offset in the ice-giant clock.

| check | ours | hers | status |
|---|---|---|---|
| Jupiter mass at 3 Myr | 317 M⊕ | 306 | +3.6%, from the earlier start fitted to the snapshots (1% in Hill radius) |
| **Saturn mass at 3 Myr** | **113.2 M⊕** | **113** | **not fitted** (the flux is fixed by Jupiter) |
| gas phase starts, J / S | 20.1 / 22.4 M⊕ | 20.6 / 22.7 | ISO_FACTOR shared by both |
| Jupiter clock, 3–100 M⊕: growth model alone vs her Figs. 1 + 13 | 224 kyr | 222 | **1%**; with τ from Ida eq. 47 it was 253 kyr |
| her type-I speed vs Tanaka+2002 in the Ida disc, 0.2–8 M⊕ | — | 0.42–0.48×, constant | consistent with a partly positive corotation torque |
| her type-I speed vs PBK11 | — | 0.59–0.84×, rising with m | PBK11 gets the mass trend, not the normalisation |
| forward model (migration integrated, PBK11 × 0.6): final masses | J 316, S 110, U 17.3, N 13.9 M⊕ | 306, 113, 18.1, 14.4 | **growth verified** |
| forward model: final a | J 4.2, S 6.5, U 18.3, N 20.7 au | 5.1, 8.0, 16.0, 18.6 | migration physics only approximate → we use her paths |
| track speeds vs Fig. 13 at 4 rates | equal from 0.2 to 100 M⊕ | — | by construction (§3.4) |
| track starts vs Table 1 | within 1% (J ×2.00: 32.04 vs 32.3) | — | ✓ |

### 3.6 The wobble in our first Fig. 13 reproduction, and why the clock is now hers

**What it was.** With the growth model's clock along her path, the tracks' |ȧ| wobbled between 20 and 40 M⊕: a spike to ~130 au/Myr at 20.5 M⊕, a dip to ~48 at 23, and a dip to ~40 at 35. Her Fig. 13 is smooth, at ~95–100 au/Myr. See `wobble.png`.

**Why.** ȧ = (da/dm)·(dm/dt).
- **Her path (Fig. 1) has two sharp kinks:**
  - at 20.5 M⊕, where her gas phase starts, |da/dm| drops from 0.29 to 0.05 au/M⊕;
  - at ~37 M⊕, where runaway starts (M_env = M_core), it steps down again.
- **What smooth speed would need.** For her speed to stay smooth through the first kink, her growth must jump to ~1800 M⊕/Myr just after isolation. That is **~3× BLJ15's cap of 0.8 Ṁ* (≈ 660 M⊕/Myr)**.
- **Our clock.** It obeys the cap, so Jupiter sits on her flat path segments too long, and the speed dips. Our gas onset is also slightly early (20.1 vs 20.5 M⊕), which gives the spike just before.

**Why her Fig. 13 can't show it.** Her Fig. 13 has a vertex only every ~17% in mass, at 20 and 24 M⊕, so it would smooth over any real dip. Either her gas onset is not capped like BLJ15's, or her real rate also dipped. Her figures don't distinguish the two.

**Why it matters.** This is the window where the stage-8 runs built their L4/L5 imbalance (peak at ~16–20 M⊕). Crossing 15–45 M⊕, Jupiter moves 5.6 au, and the first model clock (τ from Ida eq. 47) took 75 kyr there, against the 64 kyr her figures imply. With τ = 0.08 the model's total over 3–100 M⊕ agrees to 1%, but the kinks remain, so the window clock is still used.

**What we did.** For Jupiter between 3 and 100 M⊕ the clock is the one her two figures imply. The tracks then reproduce both her Fig. 1 path and her Fig. 13 rate. The cost is a growth rate briefly above BLJ15's cap, which is consistent with her own figures. Saturn has no Fig. 13, so its track keeps the model clock, and the same kind of wobble exists in Saturn's ȧ around its kinks (22.7 M⊕). Saturn's speed doesn't drive Jupiter's Trojans.

---

## 4. Status and next steps
1. ✓ Physics modules and tests (c0444de). ✓ Tracks and verification (a86d7a3 and the commit with this document).
2. ✓ Runner and report: `s9_run_pirani.py` and `s9_pirani_report.py`, with per-scenario configs and repeats 0–9 (bac466a). The report gives: Trojans by q, Q within 2.5 R_H; N(t) and N4/N5(t) (Fig. 7); co-orbital sides (Fig. 11); Trojan origin (Fig. 10); and a Hilda *region* count (G6). Statistics as hers: mean and unbiased std of the repeats. Still to add: belt implantation, e/i histograms (Fig. 9), and a Hilda libration test.
3. ✓ Haagsebeek: bare repo plus clone, `trojan-rb` env and data, all on `/data2/zshen`.
4. ✓ Test 1 (planets only, 0–5 Myr): J–S lock into the 1:2. ✓ P0 benchmark (C force kernel).
   ✓ **Test 3, 2026-11-10** (`reports/pir_test3_tenth/`): the nominal scenario with 1/10 of her disc (100 per ring, 4,400 particles), 0–5 Myr, 24 workers, 1.4 h wall. Her full-disc numbers ×1/10 against ours:

   | quantity | hers ×1/10 | test 3 |
   |---|---|---|
   | Trojans before migration (1.5–2.29 Myr) | ~6 | 15 |
   | peak (2.3–3 Myr) | ~40 | 40 (at 2.71 Myr) |
   | Trojans at 5 Myr | ~27 | 24 |
   | N4/N5 at 5 Myr | 1.8 | 3.8 ± 1.9 (19/5; Poisson-limited) |
   | co-orbital sides at 3.4 Myr, L4/L5 (Fig. 11) | 27/15 | 26/14 |
   | Trojan origin (Fig. 10) | peak at ~18 au | all 24 from 18.08–18.40 au (the seed's ring) |
   | Hildas at 5 Myr (Table 4) | ~170 | 73 (region count, G6) |
   | Jupiter at 5 Myr | ~5.0 au | 5.06 au, 316.7 M⊕ |
   | particles alive at 5 Myr | – | 2,822 of 4,400 |

   - The peak, the end count and the Fig. 11 split all match.
   - Our pre-migration count is 2.5× hers. Before 2.31 Myr the seed is 0.01 M⊕ (2.5 R_H ≈ 0.1 au), so this count is pure geometry: particles whose q and Q happen to fall in the window. It depends on her ring set-up and seed mass, so it is worth asking her.
   - Our Hilda count is a region count, not a libration test. Definition still open (G6).
   - **Cost:** 34.7 CPU-h, i.e. 5.68 CPU-s per particle-Myr over the whole run. That is 4× the quiet-phase benchmark, because encounters during migration cost more.
5. ✓ **P1 pilot, 2026-11-10/11** (`reports/pir_p1_nominal_r0/`). Her full nominal set-up, repeat 0: 44,000 particles, 0–5 Myr, 24 workers.
   - Cost: 8 h wall and 189 CPU-h, i.e. 3.09 CPU-s per particle-Myr. That is half of test 3's rate per particle; the full disc parallelises better.

   | quantity | hers (nominal) | P1 |
   |---|---|---|
   | Trojans before migration | ~60 | 98 |
   | peak | ~400 | 356 (at 2.70 Myr) |
   | Trojans at 5 Myr | ~270 | 226 |
   | **N4 / N5 at 5 Myr** | ~174 / ~96 (1.8) | **189 / 37 (5.1 ± 0.9)** |
   | co-orbital sides at 3.4 Myr, L4/L5 (Fig. 11) | 270/150 | 265/115 |
   | Trojan origin | peak at ~18 au | 206 of 226 from the seed's ring (18.0–18.5 au) |
   | Jupiter at 5 Myr; J–S | 5.0 au, 1:2 | 5.06 au, 317 M⊕; 1:2 (test 1) |
   | Hildas | ~1700 | 829, a region count only (G6). Before Jupiter arrives this is ~825 disc particles near a_32 |

   - **What matches:** the total counts (within ~15%), the Fig. 11 co-orbital split, and N4 itself.
   - **What doesn't: N5 is 2.6× too low**, so N4/N5 is 5.1 against her 1.8. This is >3σ for one run, and test 3 (an independent seed) also gave 3.8. So it is systematic, not noise.
   - **Where the missing L5 Trojans are.** L5-side co-orbitals do exist (58 at 5 Myr). But they are hot: median e is 0.08 on L5 against 0.045 on L4, the upper quartile is 0.17, and inclinations are 0.3° against 0.007°. So many fail the q, Q ≤ 2.5 R_H test. L4 holds a cold population carried in from the seed's ring, with e ≈ 0.01 and i ≈ 0.01° through migration. L5 is heated during capture: at 2.6–3.0 Myr its median e is 0.11–0.19 and its inclination 3–9°, against 0.013–0.05 and ~0.01° on L4.
   - **Candidate causes**, to check against her set-up (cheap 1/10-disc tests, ~1.5 h each):
     1. The seed is 0.01 M⊕ for 2.3 Myr, so its co-orbital "Trojans" are just disc particles (our pre-migration 98 against her 60). What she did with the seed and its ring before 2.31 Myr matters.
     2. The L5 heating by Saturn. Our Saturn trails Jupiter closely through 2.56–3 Myr; its start and clock (G2) are ours, not hers.
     3. U/N drift off their tracks (as in her Fig. 4); test with J+S only.
     4. Her "high-e interlopers excluded" cut and an unstated inclination treatment (G6). These only remove Trojans, so they can't raise N5, but they change what N_tot means.
6. ☐ P2 (nominal and in situ, ×10), P3 (rate scan and outward), P4 (nominal to 200 Myr), P5 (§5 instability).
   - Measured cost (P1): ~190 CPU-h, i.e. 8 h on 24 workers, per full run. P2 + P3 (101 runs) is therefore ~19,000 CPU-h, about 5 weeks on haagsebeek alone. In situ and rate runs carry only J and S, so they may come in cheaper.
   - To bring this down: fewer particles per ring for the rate scan, or more machines.

**Ask Pirani** (email still to send): her track files (G1, G2), the disc extent at ×1.25–2 (G3), the outward run's planets (G7), the interloper e-cut and Hilda definition (G6), and whether her gas onset was capped (§3.6).

### 4.1 Diagnosis of the L5 deficit (2026-11-11/12)
All runs below use only the Trojan source rings at her density (1000 per 0.5 au), so counts are ~7% below a full disc.

**Rate test, her stated disc (e <= 0.01 uniform), J + S, 0-5 Myr** (`reports/pir_src/`), against her Table 3:

| run | ours N_tot | N4/N5 | hers N_tot | N4/N5 |
|---|---|---|---|---|
| in situ | 55 | 26/29 = 0.90 ± 0.24 | 51 ± 8 | 0.94 ± 0.35 |
| x0.25 | 109 | 65/44 = 1.48 ± 0.29 | 106 ± 11 | 1.60 ± 0.44 |
| x0.5 | 125 | 90/35 = 2.57 ± 0.51 | 165 ± 11 | 1.63 ± 0.29 |
| x1.0 | 188 | 159/29 = 5.48 ± 1.11 | 240 ± 9 | 1.80 ± 0.26 |

- In situ and x0.25 reproduce her numbers. The deficit appears only for fast migration.
- L4 matches her at every rate: implied hers 102 and 154 for x0.5 and x1.0, against our 90 and 159.
- The missing Trojans are all on L5. Ours gives 35 and 29 there; her numbers imply 63 and 86.
- x1.0 with J + S alone gives the same ratio as the full nominal (P1): Uranus and Neptune play no role.

**What has been ruled out** (capture tests, Jupiter alone, rings 17-19.5 au, 0-2.6 Myr; `reports/pir_capture*`, `pir_convergence`):

| change | L5 at 2.6 Myr | verdict |
|---|---|---|
| base | 57 | – |
| particle gas drag off | 75 | no effect: drag at 18 au moves a by 0.001 au/Myr and damps e over 30-300 Myr |
| planet e/i damping off | 57 | no effect: Jupiter starts circular, so there is nothing to damp |
| Jupiter on her own Figs. 1+13 clock from the seed (seed 18.40 au) | 47 | no effect |
| dt 25 d | 67 | converged |
| MERCURIUS switchover 6 R_H | 49 | converged |
| particle e <= 0.005 | 53 | colder disc: L5 lower still |
| **particle e <= 0.02** | **102** | **the only lever found** |
| her Fig. 7 | 159 | – |

- e <= 0.02 with the full nominal system to 5 Myr (two seeds) gives N4/N5 = 2.88 and 2.90, against 6.0 at e <= 0.01.
- The L5 losses at 2.34-2.46 Myr are dynamical. L5 Trojans swap side or leave the co-orbital region; their e stays ~0.01.
- Retention from 2.40 to 2.60 Myr: L4 80%; L5 28% at e <= 0.01 and 48% at e <= 0.02.
- This is the migration-driven asymmetry mechanism of her sect. 4.2. Our result has its expected rate dependence: it grows with rate, as in stage 8. Hers stays at 1.6-1.8 for every rate from x0.25 up.

**Open:**
- Her particle e distribution. The paper says "randomised in [0, 0.01]", but our runs need an effectively hotter disc.
- How her MERCURY applied the migration (gap G4). Nothing in our set-up that is converged and checkable weakens the trailing-side loss at fast rates.

## 5. Regenerate
```
conda run -n trojan-rb python campaign/scripts/s9_digitize_pirani.py     # her curves from the PDF
conda run -n trojan-rb python campaign/scripts/s9_pirani_tracks.py       # tracks + reports/pirani_tracks/
conda run -n trojan-rb python -m pytest -q campaign/tests/test_pirani.py
```
