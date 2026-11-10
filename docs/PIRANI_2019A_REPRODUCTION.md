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
| pebble accretion | Ida+2016 eqs. 59–61 (χ = 1, η′ = η, ζ = 1/(1 + τ²)). Flux = **FLUX_FACTOR** × Ida eq. 54. Stokes number from Ida eq. 47 |
| isolation | BLJ15 eq. 16, 20 (h/0.05)³ M⊕, × **ISO_FACTOR**. 10% of the accreted mass is envelope before isolation |
| contraction | Piso & Youdin 2014 (BLJ15 eq. 17; f = 0.2, κ_env = 0.05 cm²/g, ρ_c = 5.5 g/cm³) while M_env < M_core |
| runaway | Machida+2010 (BLJ15 eqs. 18–19), the smaller branch, **capped at 0.8 Ṁ*** |
| type I | Paardekooper+2011 total torque (Lindblad + saturated corotation; Bell & Lin ice opacity, 0.5% dust), × TORQUE_SCALE. Only used in the forward check |
| gap / type II | Crida+2006 P, Crida & Morbidelli 2007 f(P); P < 1 → τ_II = τ_ν max(1, M/(4πΣr²)) (BLJ15 eqs. 21–23) |

**Calibrations.** Pirani gives none of these numbers.

| | value | fitted to |
|---|---|---|
| FLUX_FACTOR | **8** | Jupiter's clock (her Figs. 1 + 13) and its 306 M⊕ at 3 Myr. BLJ15's own flux is "corrected" without a number |
| ISO_FACTOR | **1.066** | where her dashed (gas) lines start, for J and S together (her 20.6 and 22.7 M⊕) |
| U, N start times (her "~2.70 Myr") | **2.721, 2.739 Myr** | their final masses (18.1, 14.4 M⊕) |
| TORQUE_SCALE | 0.6 | forward check only (§3.5) |

### 3.4 How the N-body tracks are built (`scripts/s9_pirani_tracks.py`)
**Principle:** take from her what she shows, model only what she doesn't.

**Path a(m): her Fig. 1 curve.**
- **Smoothing:** medians in log-m bins (0.25 dex below 2 M⊕, 0.08 dex above), joined by a monotone spline. Her plotted vertices step by 0.01–0.03 au where a barely changes, and the N-body forcing differentiates a(t). The largest deviation from her curve is 0.19 au (J) and 0.21 au (S), at the isolation corner, and ≤ 0.012 au for U and N.
- **Jupiter below 3 M⊕:** a(m) is built from her Fig. 13 rate along the clock, because her path is too flat there to differentiate. That puts Jupiter's seed at **18.26 au**, against her 18.40.

**Clock m(t).**
- **Jupiter:**
  - 0.018 → 3 M⊕: the growth model.
  - **3 → 100 M⊕: her own clock**, dm/dt = |ȧ|_Fig13 / |da/dm|_Fig1 (see §3.6).
  - Above 100 M⊕: the growth model, stretched by 1.088 so that it ends at 3.0 Myr with 305 M⊕.
- **Saturn, Uranus, Neptune:** the growth model along their Fig. 1 paths, from 2.56 / 2.721 / 2.739 Myr.

**Format.** Each table starts at t = 0 with the seed parked at its starting a, then has 3000 rows to 3 Myr. Columns: t_yr, a_au, m_earth.

| file (`configs/tracks/pirani19a/`) | what | start → end |
|---|---|---|
| `track_nominal_jupiter.csv` | nominal Jupiter | 18.26 → 5.08 au, 0.018 → 305 M⊕ |
| `track_nominal_saturn.csv` | nominal Saturn | 21.27 → 8.02 au, → 116 M⊕ |
| `track_nominal_uranus.csv`, `_neptune.csv` | nominal ice giants | 22.50 → 16.01 au, → 18.1 M⊕; 24.00 → 18.57 au, → 14.4 M⊕ |
| `track_rate_x{0.15,0.25,0.50,0.75,1.00,1.25,1.50,2.00}_jupiter.csv` | a_end,k + k(a_nom − a_end,nom), same m(t); ends from Fig. 3 | e.g. ×0.15: 7.08 → 5.43; ×2.00: 32.04 → 5.68 |
| `track_rate_x…_saturn.csv` | same rule on nominal Saturn; end chosen so the start equals Table 1 (G2) | 10.45 … 35.5 au starts, exactly Table 1 |
| `track_insitu_{jupiter,saturn}.csv` | 5.4 and 8.6 au, nominal clocks | — |
| `track_outward_jupiter.csv` | nominal path mirrored, a = 5.08 + 18.26 − a_nom | 5.08 → 18.26 au |

### 3.5 Verification (`reports/pirani_tracks/verification.png`, `wobble.png`, `summary.json`)

| check | ours | hers | status |
|---|---|---|---|
| Jupiter mass at 3 Myr | 305 M⊕ | 306 | fitted |
| **Saturn mass at 3 Myr** | **116 M⊕** | **113** | **predicted (3%)** |
| gas phase starts, J / S | 20.1 / 22.4 M⊕ | 20.6 / 22.7 | ISO_FACTOR shared by both |
| growth model's own clock vs hers, relative to t(5 M⊕), at 1 / 2 / 10 / 20 / 30 / 50 / 100 M⊕ | +9 / +3 / +1 / +9 / +11 / +17 / +32 kyr | — | the model is 12% slower than her figures imply over 3–100 M⊕ (253 vs 222 kyr), hence §3.4 |
| her type-I speed vs Tanaka+2002 in the Ida disc, 0.2–8 M⊕ | — | 0.42–0.48×, constant | consistent with a partly positive corotation torque |
| her type-I speed vs PBK11 | — | 0.59–0.84×, rising with m | PBK11 gets the mass trend, not the normalisation |
| forward model (migration integrated): final masses | J 304, S 113, U 17.7, N 14.1 M⊕ | 306, 113, 18.1, 14.4 | **growth verified** |
| forward model: final a | J 3.8, S 6.0, U 17.4, N 20.0 au | 5.1, 8.0, 16.0, 18.6 | migration physics only approximate → we use her paths |
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

**Why it matters.** This is the window where the stage-8 runs built their L4/L5 imbalance (peak at ~16–20 M⊕). Crossing 15–45 M⊕, Jupiter moves 5.6 au, and the model clock took 75 kyr against the 64 kyr her figures imply.

**What we did.** For Jupiter between 3 and 100 M⊕ the clock is the one her two figures imply. The tracks then reproduce both her Fig. 1 path and her Fig. 13 rate. The cost is a growth rate briefly above BLJ15's cap, which is consistent with her own figures. Saturn has no Fig. 13, so its track keeps the model clock, and the same kind of wobble exists in Saturn's ȧ around its kinks (22.7 M⊕). Saturn's speed doesn't drive Jupiter's Trojans.

---

## 4. Status and next steps
1. ✓ Physics modules and tests (c0444de). ✓ Tracks and verification (a86d7a3 and the commit with this document).
2. ☐ Runner and report: `s9_run_pirani.py` and `s9_pirani_report.py`, with per-scenario configs and repeats 0–9. The report gives her quantities: Trojans by q, Q within 2.5 R_H at 5 Myr; N(t) and N4/N5(t) (Figs. 7, 11); Hildas (Tables 4, 5); belt implantation; e and i histograms (Fig. 9); Trojan origin (Fig. 10). Statistics as hers: mean and unbiased std of the 10.
3. ☐ Haagsebeek: clone, `trojan-rb` env, data on `/data2/zshen`.
4. ☐ P0 benchmark, then the **P1 pilot**: one nominal run to 5 Myr, compared with Figs. 7, 10, 11 and Table 2. Expected: ~60 Trojans before migration, a peak of ~400, ~270 at 5 Myr, N4/N5 ≈ 1.8; horseshoe sides 50/50 → 270/150; J–S end in the 1:2.
5. ☐ P2 (nominal and in situ, ×10), P3 (rate scan and outward), P4 (nominal to 200 Myr), P5 (§5 instability).
   - Estimated cost: 11,000–22,000 CPU-h for P2 + P3, i.e. 3–6 weeks on haagsebeek's ~24 cores.

**Ask Pirani** (email still to send): her track files (G1, G2), the disc extent at ×1.25–2 (G3), the outward run's planets (G7), the interloper e-cut and Hilda definition (G6), and whether her gas onset was capped (§3.6).

## 5. Regenerate
```
conda run -n trojan-rb python campaign/scripts/s9_digitize_pirani.py     # her curves from the PDF
conda run -n trojan-rb python campaign/scripts/s9_pirani_tracks.py       # tracks + reports/pirani_tracks/
conda run -n trojan-rb python -m pytest -q campaign/tests/test_pirani.py
```
