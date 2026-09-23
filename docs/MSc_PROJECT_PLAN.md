# MSc Research Project — Working Plan

**Status:** Stages 0–3 under way (first 10 Myr pilot), set at the 2026-09-15 meeting. Rewritten 2026-09-12.
The research question is still being settled; the setup decisions below are fixed.
**Length:** 1 year. **Supervisor:** Simon Portegies Zwart.
**Repo:** `L4-L5-Jupiter-Migration-Plus`. **Compute:** ALICE (Leiden HPC).

Background, with every number sourced:

- `docs/ASYMMETRY_REVIEW.md` — the field, the models, the research gaps G1–G6.
- `docs/VOKROUHLICKY_DISISTO_REVIEW.md` — the two anchor papers, side by side.
- `literature/READING_LIST.md`.

---

## 0. The project in one paragraph

There are ≈ 1.43 times as many Trojans leading Jupiter (L4) as trailing it (L5), and
nobody knows why. Part of the answer could be that the two swarms erode at different
rates today: Di Sisto et al. integrated the known Trojans for 4.5 Gyr and found L5
losing slightly more. But their sample mixed collisional families in with everything
else, was not corrected for survey bias, and their 2019 rerun's L4/L5 difference is
not statistically significant (0.9σ). Meanwhile Vokrouhlický et al. 2024 published the
first debiased, family-free description of both swarms, and found the only significant
orbital difference between them is in inclination.

This project integrates every catalogued Trojan for 4.5 Gyr, each object once (no
clones) and labelled by swarm and family. It measures whether the
swarms really do erode differently, whether families explain it, and whether the
inclination difference explains it.

Every outcome is a result. At worst: "present-day erosion is symmetric to within X%,
so the asymmetry is entirely primordial."

---

## 1. Research questions

| | Question | Gap |
|---|---|---|
| **Q1** | Do today's L4 and L5 swarms erode at different rates over 4.5 Gyr, with honest errors? | G4 |
| **Q2** | Do collisional families account for any difference? | G1 |
| **Q3** | Does L5's hotter inclination distribution predict its higher escape? | G1 |
| **Q4** | Why are L4 families sharp and L5 families diffuse? | G2 |
| **Q5** | Are the swarms static or dynamic — who stays, who leaves, how? | Simon's ask |

By-products: an L4/L5 dynamical symmetry test over Gyr (G3); a bound on
back-extrapolating today's escape rate to primordial times (G6).

---

## 2. What is simulated

### 2.1 Setup

| | |
|---|---|
| Code | REBOUND, WHFast |
| Massive bodies | **Decided 2026-09-15:** Sun with the Mercury, Venus, Earth–Moon and Mars masses folded in; Jupiter, Saturn, Uranus and Neptune kept as separate bodies with their system (planet + moons) masses, DE440 |
| Test particles | **Massless. They feel the Sun and planets, never each other.** So particles of different kinds cannot interfere, and chunking them across jobs is bookkeeping, not physics. *Justified 2026-09-23: the co-orbital threshold sits at 40.9 M_Jup (Routh) / 24.0 M_Jup (Liberato & Winter 2020), ten orders of magnitude above the largest real Trojan — see `docs/MASS_THRESHOLD.md` and `campaign/reports/mass_ladder/`* |
| Start | Present day. Planets from JPL Horizons at the catalogue epoch; Trojans from their catalogued orbits at the same epoch |
| End | +4.5 Gyr forward, or when the particle escapes |
| Timestep | **Decided 2026-09-15:** 1% of Jupiter's orbital period, ≈ 0.1186 yr. Stage 0.6 measures its cost and checks it against dt/2 |
| Not included | Collisions, Yarkovsky, gas, migration, planetary mass growth. Nothing but gravity |

### 2.2 Two runs, and nothing else

They are separate campaigns with separate purposes. **No artificial particles are
seeded into the main run.**

| | **Main run** (Stages 3–5) | **Family run** (Stage 6) |
|---|---|---|
| Particles | Every catalogued Jupiter Trojan at its real orbit, once: the union of the V24, JPL (TJN) and MPC definitions | Synthetic fragments thrown from each real family's location at 10–100 m/s |
| Count | ≈ 16,500 | ≈ 10,000 |
| Duration | 4.5 Gyr | 1–4.5 Gyr |
| Real or artificial | Real orbits only | Artificial, but modelling a real event (the family-forming collision) |
| Answers | Q1, Q2, Q3, Q5 | Q4 |

Only Trojans are simulated as particles — not the main belt, not anything else.

### 2.3 No clones (decided 2026-09-15)

Each object is integrated once. A single 4.5 Gyr trajectory in a chaotic system is one
draw, so escape fractions carry binomial errors over objects, and those errors include
each object's chaotic scatter; per-object escape probabilities are not measured.
At the meeting, cloning was judged to cost more in complexity than it returns.

### 2.4 What is recorded per particle

Escape time and cause (both Hill-radius criteria, Stage 0.5); proper elements and
libration state sampled through the run.

### 2.5 Orbital coordinates

Vokrouhlický et al. 2024's three quasi-proper amplitudes, because that is the system
his debiased distributions are published in: **A** (inclination), **B** (eccentricity),
**C** (libration amplitude). All three follow in closed form from osculating elements
plus Jupiter's (his Eqs. 1–3).

### 2.6 Survey bias

- **H ≤ 13.6 is taken as complete** (meeting 2026-09-15, slightly inside V24's
  H ≈ 13.7–13.8 limit): the catalogue contains essentially every Trojan that bright, so
  this subset *is* the true population and needs no correction. All population numbers
  come from it. Flagged `complete` in the master table.
- **Fainter objects are incomplete.** They are simulated, because they add statistics
  on the dynamics, but never used for population numbers.
- **Vokrouhlický's debiased model** (his Eqs. 6–8, Table 3) is needed only if a result
  must extend below H ≈ 13.6, as weights w_i = n(A_i, B_i, C_i) / n_obs(A_i, B_i, C_i).

### 2.7 Which question uses which particles

**Every listed object is used throughout**, one trajectory each; the magnitude cut
below is the `complete` flag, H ≤ 13.6.

| | Objects | Magnitude cut | Swarms | Output |
|---|---|---|---|---|
| **Q1** | Background only (family members excluded) | **H ≤ 13.6** — the complete sample, so no debiasing is needed | L4 and L5 **separately** | Escape fraction per swarm, with errors |
| **Q2** | Family members **and** background | (i) H ≤ 13.6, to compare with Q1; (ii) none | L4 and L5 separately | (i) **Contamination:** swarm escape fraction with families included vs excluded. (ii) **Check:** family vs background escape at matched (A, B, C) — should agree, since dynamics depends only on the orbit; a mismatch means the cells are too coarse. Both repeated at several membership cutoffs |
| **G3** | Background | none | L4 and L5 **separately**, then compared | Is P_esc(A, B, C) the same function in both swarms? |
| **Q3** | Background | none for the map; H ≤ 13.6 for the distributions it is applied to | **Pooled** into one map, then applied to each swarm's distribution | Predicted L4/L5 difference, compared with Q1 |
| **Q4** | Family members, **plus the separate family run** (§2.2) | none | L4 and L5 separately | Spread in proper elements vs time |
| **Q5** | Everything | none | L4 and L5 separately | Residence times, escape-time distributions, state sequence before escape |
| **G6** | Everything | none | L4 and L5 separately | Escape rate vs time |

**Why Q3 pools the swarms — this is the crux.** Building a map from L4 objects and
applying it to the L4 distribution just returns Q1; likewise for L5. The test has to be:
build **one** map from both swarms together, then apply that single map to L4's
distribution and to L5's distribution separately, and see whether the predicted
difference matches the measured one.

- Prediction matches Q1 → the orbital-distribution difference (mainly inclination)
  explains the erosion difference.
- Prediction fails → L4 and L5 objects at the *same* (A, B, C) do not escape alike, so
  either the dynamics is not symmetric (G3 catches this) or (A, B, C) is an incomplete
  description of what drives escape.

Pooling is only legitimate if G3 passes, so **G3 runs first**.

**Why families are separated at all.** Not because they move differently — gravity sees
only the orbit, so a family member and a background object on the same orbit share an
escape probability. They are separated because they are not independent samples: one
collision put hundreds of fragments on nearly one orbit, so in a swarm average that
orbit gets hundreds of votes where it is really one parent body. The 1.43 ratio counts
the background for the same reason. Adding family members back into Q3's map is a
robustness check, not the headline.

---

## 3. Stages

Each stage has numbered steps, a deliverable, and a completion test. Weeks run from
project start; stages overlap.

### Stage 0 — Set-up (weeks 1–3)

**Goal:** code that integrates real Trojans correctly and cheaply.

1. **Environment.** Project conda env on ALICE (never base) with REBOUND. Confirm
   allocation and queue limits.
2. ~~Fix giants-only mode.~~ **Moot (2026-09-15):** the migration-era code is frozen in
   `migration/`; the present-day code is new in `campaign/` and finds bodies by name.
3. **Real planets at the real epoch.** `create_solar_system` currently randomises
   planet angles (`bodies.py:89–99`). Replace with JPL Horizons states at the
   catalogue epoch. Build and cache the initial simulation locally — ALICE compute
   nodes have no network.
4. **Real Trojans in,** from JPL SBDB elements at the same epoch (Horizons states for
   objects whose orbit epoch differs).
   *Check:* propagate 20 known Trojans ±400 yr and compare with Horizons (Horizons'
   small-body ephemerides do not reach +1,000 yr comfortably).
5. **Escape bookkeeping.** `runner.py` currently removes particles past a distance and
   records no escape time. Record per particle: time of leaving the libration zone;
   first encounter within 1 and 2.1 Hill radii of any planet; which planet. Keep
   distance-based removal as a backstop.
6. **Benchmark.** Planet set and timestep are fixed (§2.1), so this measures rather than
   chooses: CPU per particle-Myr; the sampling interval for encounter checks; dt vs
   dt/2 over 1 Myr for 1,000 real Trojans.

**Deliverable:** working giants mode, validated real-epoch initialisation, cost table.
**Done when:** the 1,000-yr propagation matches Horizons and cost is measured, not
extrapolated.

### Stage 1 — Data and the family cross-match (weeks 2–5)

**Goal:** the master table, and the cheapest decisive test in the project.

1. **Catalogue.** JPL SBDB, with Vokrouhlický's Trojan selection: 4.95 < a < 5.45 au,
   e < 0.6, I < 50°, 20° < |σ| < 130°, applied at the common epoch. Compared with JPL's
   TJN class and the MPC Jupiter Trojan list; the run integrates the union of all three.
   The first 1 Myr of the pilot is the cleaning integration.
2. **Vokrouhlický's catalogue** from https://sirrah.troja.mff.cuni.cz/~mira/tmp/trojans/
   (mirrored 2026-09-15 to `data/raw/v24/`): `greeks-hcm_20230405_GS` (L4 proper
   elements) and `trojans-hcm_20230405_GS` (L5 proper elements — the "trojans" directory
   is the L5 camp only), `trojans_L4_families_2023Apr`, `trojans_L5_families_2023Apr`
   (membership).
3. **Master table,** one row per object: designation, swarm, H, `complete` flag
   (H ≤ 13.6), family or `background` (one column per membership cutoff, Stage 1.6),
   proper elements, (A, B, C), and a `disisto2013` flag. Membership is per object,
   from clustering, so objects inside a family's region can still be background.
   Objects discovered after April 2023 carry no family label in Vokrouhlický's files;
   almost all are fainter than H = 13.6, so Q1 is unaffected, and they are labelled
   with our own proper elements after Stage 2.
4. **Reconstruct Di Sisto's sample.** He used no private catalogue: his input was the
   MPC's *numbered* Jupiter Trojans as of March 2013, i.e. today's catalogue filtered
   by date and orbit quality. Since asteroid numbers are issued sequentially, the
   filter is a number threshold. Checked 2026-09-12 against JPL SBDB: of 16,391
   Trojans (10,633 numbered), **the 2,972nd numbered one is (329008)** — so
   `number ≤ 329008` reproduces his count exactly.
   The swarm split comes out 1,974 L4 / 998 L5 against his 1,975 / 997 (checked
   2026-09-14). Still to confirm from the MPC numbering circulars that (329008) dates
   from March 2013.

   **Where his bias sits** (JPL magnitudes, checked 2026-09-14):

   | H | His sample (L4 / L5, ratio) | Today's catalogue (L4 / L5, ratio) |
   |---|---|---|
   | ≤ 12 | 535 (313 / 222) 1.41 | 535 (313 / 222) 1.41 |
   | 12–13.8 | 2,115 (1,393 / 722) 1.93 | 3,218 (1,935 / 1,283) 1.51 |
   | > 13.8 | 322 (268 / 54) 4.96 | 12,638 (8,167 / 4,471) 1.83 |

   His H ≤ 12 part was already complete and matches the debiased 1.43. In 12–13.8 he
   had 72% of the L4 objects but only 56% of the L5 ones. Stage 5.1 splits his escape
   fractions at H = 12 to see whether his L4/L5 escape difference lives in the biased
   part.
5. **The cross-match,** on three samples, answering two different questions:
   - *Is the published result contaminated?* Family share per swarm in the
     `disisto2013` subset — a claim about the literature.
   - *Must we remove families from our own analysis?* Family share per swarm in the
     complete modern sample, and in the whole catalogue.

   Also: where does each family sit relative to Vokrouhlický's stability boundary —
   core or edge?
6. **Interlopers.** Membership from clustering is uncertain, and Vokrouhlický reports
   L4 families as clean but L5 families as having "possibly a substantial population
   of interlopers" — so family removal is least reliable in the swarm where it matters
   most. Build membership at several cutoff velocities (he used 20–60 m/s) and carry
   the spread as a systematic through every family-dependent result.
7. **The bound.** What family share, and what family-vs-background escape difference,
   would reproduce Di Sisto's L4/L5 gap? Is that plausible?

**Where each orbital variable comes from** — all per-object and measured, none
statistical:

| Variable | Source |
|---|---|
| Osculating a, e, i, Ω, ω, M | MPCORB / JPL SBDB |
| Orbit covariance | JPL SBDB (`sbdb.api?cov=mat`), for numbered objects |
| A, B, C | closed form from osculating elements + Jupiter's |
| Proper elements, incl. libration amplitude | not published by MPC or JPL: Vokrouhlický's catalogue (to April 2023) or our own short integration (Stage 2.3) |
| Family membership | not published by MPC or JPL either. Vokrouhlický's is the current and most complete identification (9 L4 + 4 L5, on his own proper elements); the PDS release (Nesvorný et al. 2015) and Brož & Rozehnal (2011) predate the recent discovery flood and are cross-checks, not substitutes |

**Deliverable:** the master table, plus a two-page note — "how much of Di Sisto's
sample was families?" — the first thing to show Simon.
**Done when:** family counts per swarm, with uncertainties, for all three samples.

### Stage 2 — Diagnostics and validation (weeks 4–10)

**Goal:** the measurement code the repo lacks, proven against published values.

1. **Resonant angle and libration state.** σ = λ − λJ; classify each particle over a
   running window as tadpole L4 / tadpole L5 / horseshoe / quasi-satellite /
   non-co-orbital; measure libration amplitude.
2. **(A, B, C)** from osculating elements — closed form.
3. **Proper elements** over a running window: the Milani (1993) synthetic method as
   used in Vokrouhlický's pipeline, giving Δa_P, e_P, sin I_P. Save ~every 1 Myr.
4. **Validation A.** Proper elements for 500 catalogue Trojans against Vokrouhlický's;
   agreement must be small compared with family sizes.
5. **Validation B.** Reproduce his stability zone (his Figs. 12–14): 100 Myr, cells in
   (C, B) at 3–4 values of A, both swarms.
6. **Tests.** A pytest per diagnostic against analytic cases: a particle at L4, a known
   libration amplitude.

**Deliverable:** a co-orbital diagnostics module, plus validation plots.
**Done when:** Validations A and B pass. **Do not launch Stage 4 before this.**

### Stage 3 — Pilot (weeks 10–13)

**Goal:** the particle set and configuration, proven end to end.

1. ~~Clones~~ and 2. ~~Clone budget~~: **removed at the 2026-09-15 meeting** (§2.3).
   The old design is in git history (`git show 86dc33e:docs/MSc_PROJECT_PLAN.md`).
3. **Pilot.** The whole catalogue for 10 Myr at the §2.1 planet set and timestep, on
   the laptop (`campaign/configs/pilot_10myr.toml`). Measure wall time, memory, output
   volume; test escape bookkeeping, diagnostics, and checkpoint/restart.
4. **Freeze and pre-register.** Fix particle set, planet set, timestep, output cadence.
   Write down the Stage 5 analyses before seeing results.

**Deliverable:** frozen particle set and configuration; pilot report.
**Done when:** projected cost fits the allocation.

### Stage 4 — The 4.5 Gyr campaign (weeks 13–22, mostly waiting)

1. **Job array.** Chunk the particles, each chunk carrying the planets; test particles
   don't interact, so this scales perfectly. Pass 2 reuses the same machinery.
2. **Output per particle:** escape time and cause; proper elements and (A, B, C)
   ~every 1 Myr; libration-state summary.
3. **Checkpoints** every ~100 Myr, restart via `sim_from_state`.
4. **Interim looks** at 0.5 and 1 Gyr, for sanity only — the analysis is
   pre-registered.
5. **While waiting:** prepare Stage 6's synthetic families; read Holt et al. 2020a/b,
   Tsiganis et al. 2005, Robutel & Gabern 2006 (Part I), Freistetter 2006, Hou et al.
   2014; draft introduction and methods.

**Deliverable:** the per-particle outcome table.
**Done when:** every chunk reaches 4.5 Gyr or empties.

### Stage 5 — Erosion analysis (weeks 20–32)

1. **Benchmark against the only published numbers.** Our particle set contains Di
   Sisto's objects as a subset (Stage 1.4), so this costs nothing, and it is the
   *only* external check on the quantity we actually measure — Stage 2 validates
   proper elements and the 100 Myr stability zone, neither of which tests 4.5 Gyr
   escape.

   - **Compare:** escape fraction per swarm, `disisto2013` subset, using his escape
     criterion — 1 Hill radius for 2014, 2.1 for 2019 (we record both).
   - **Against:** 23.2 / 28.3% (2014) and 23.6 / 25.1% (2019).
   - **Split at H = 12** (Stage 1.4): does his L4/L5 escape difference live in the
     complete bright part or the biased faint part?
   - **"Agreement" means within ~2–3%,** not exact: his counts carry ± 1.0% (L4) and
     ± 1.4% (L5) binomial error plus the chaotic scatter of one realisation per
     object. Without clones, our numbers carry the same kind of scatter.
   - **A real mismatch** (say 15% or 35%) means a systematic difference — planet set,
     timestep, integrator, escape bookkeeping — and must be diagnosed before any of
     our numbers are trusted. That diagnosis is gap G4, so it is a result either way.
2. **Q1.** Complete (H ≤ 13.6), non-family objects: per-swarm escape fractions with
   binomial errors. Is L5 higher, and by how much?
3. **Q2.** (i) Swarm escape fractions from the complete sample with families included
   vs excluded — the contamination effect. (ii) Family vs background escape at matched
   (A, B, C), per family — expected to agree; a mismatch means the cells are too coarse.
   Repeat both at each membership cutoff from Stage 1.6 and quote the spread — the
   dominant systematic on every family result, and worse for L5.
4. **P_esc(A, B, C)** from all real objects, both swarms pooled. Check that L4 and L5
   objects at the same position escape at the same rate — the symmetry test, G3.
5. **Q3.** Apply P_esc to each swarm's complete-sample background distribution and compare the predicted
   L4/L5 difference with the measured one from step 2. Then P_esc against A at fixed
   (B, C): does escape rise with inclination, as Vokrouhlický's Fig. 14 suggests?
   - Prediction matches measurement → the inclination difference explains the erosion
     difference.
   - Difference vanishes once families are removed → it was a family effect.
   - Prediction fails → (A, B, C) is too coarse to capture what drives escape, itself
     a finding about the parametrisation the field now uses.
6. **Primordial ratio.** The erosion amplifier with its error, and what it implies for
   the primordial ratio. Escape rate vs time bounds the back-extrapolation error (G6).
7. **Q5.** Fraction of each swarm surviving 4.5 Gyr with unchanged proper elements;
   escape-time distributions; state sequence before escape (tadpole → horseshoe →
   encounter).

**Deliverable:** the core results chapter, and a draft paper on family-resolved
present-day erosion of the Jupiter Trojans.
**Done when:** each step has a number with an error bar.

### Stage 6 — Families over Gyr (weeks 28–40)

**Goal:** Q4.

1. **Observed families.** How fast each real family's spread in proper elements grows,
   L4 vs L5, at matched (A, B, C).
2. **Synthetic families.** At each family's location, eject fragments at 10–100 m/s
   isotropically and integrate for Gyr — a small run. Measure spread vs time and
   ejection speed.
3. **Compare** with the observed shapes (Vokrouhlický's Figs. 30–31). Which of his
   three candidates reproduces "L4 sharp, L5 diffuse": ejection speed,
   location-dependent diffusion, or age?
4. **Ages,** as a by-product: the time to reach each observed spread.

**Deliverable:** the families chapter.
**Done when:** the three explanations are ranked on quantitative evidence.

### Stage 7 — Write-up (weeks 40–52)

1. Thesis chapters: introduction; methods and validation (Stages 0–2); campaign
   (3–4); erosion (5); families (6).
2. Submit the Stage 5 paper.
3. Release code and data: master table, per-particle outcomes, diagnostics module.

---

## 4. Timeline

| Month | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 Set-up | ■ | | | | | | | | | | | |
| 1 Data + cross-match | ■ | ■ | | | | | | | | | | |
| 2 Diagnostics + validation | | ■ | ■ | | | | | | | | | |
| 3 Pilot | | | ■ | | | | | | | | | |
| 4 Campaign | | | | ■ | ■ | | | | | | | |
| 5 Erosion analysis | | | | | ■ | ■ | ■ | ■ | | | | |
| 6 Families | | | | | | | ■ | ■ | ■ | | | |
| 7 Write-up | | | | | | | | | | ■ | ■ | ■ |

About six weeks of slack in months 9–12.

---

## 5. Decision points

| When | Question | If no |
|---|---|---|
| End of Stage 1 | Do family shares differ between swarms by more than a few percent? | Q2 weakens; Q1 and Q3 carry the project |
| End of Stage 2 | Do our proper elements match Vokrouhlický's? | Don't launch. Fix, or bin by his catalogue values at t = 0 and drop Stage 6.1 |
| End of Stage 3 | Does the campaign fit the allocation? | Complete sample only; longer sampling interval |
| Stage 5.1 | Do our escape fractions match the published ones within ~2–3%? | Diagnose the systematic (planet set, timestep, escape criterion) before interpreting our own numbers |

---

## 6. Compute budget

- **Rate:** ≈ 6.4 CPU-years per 10⁴ particles for 4.5 Gyr with all planets at
  dt ≈ 0.031 yr, extrapolated from `migration/results/batch_results.json`. Stage 0.6
  replaces this with a measured number.
- **Particles:** ≈ 16,500, no clones (§2.3).
- **Cost:** to be replaced by the Stage 0.6 measurement. The old extrapolation, scaled
  to giants-only at dt ≈ 0.12 yr and 16,500 particles, is of order 1 CPU-year — an
  upper bound, since escaped particles stop costing.
- **Storage:** 10 kyr window summaries for 16,500 particles are ≈ 0.8 GB per 10 Myr;
  a 4.5 Gyr run needs a coarser cadence (e.g. 1 Myr windows, ≈ 4 GB).

---

## 7. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Cost higher than estimated | Complete sample only; ALICE job arrays |
| Vokrouhlický's catalogue disappears | Mirror it now (Stage 1.2) |
| Proper-element validation fails | Bin by his catalogue values at t = 0; replace Stage 6.1 with libration-amplitude and e/i dispersions |
| L5 family membership too interloper-ridden | Quote results across cutoffs; fall back to tight-cutoff core vs everything else, which brackets the answer |
| Orbital space too thinly sampled where it matters | Report as large error bars. Only if a region proves genuinely empty of catalogued objects does seeding synthetic particles there become worth its cost — decide after Stage 5, never before |
| Everything null | Still results: "erosion is symmetric to X%", "families don't bias it", "the primordial ratio equals today's 1.43 ± 0.05" |
| Small family members feel Yarkovsky | Out of scope for a gravity-only model; state as a limitation wherever results lean on km-sized objects |
| Simon wants migration back | The parked plan and its code are in git: `git show 7306ac0:MSc_PROJECT_PLAN.md` |

---

## 8. For Simon

1. **Binaries (his 2026-08-23 ask).** The migration-era argument — binaries stripped
   during horseshoe passages near Jupiter — doesn't carry over, since present-day
   tadpole orbits never approach Jupiter. Options: drop it; use this project's record
   of how closely each escaping Trojan passes Jupiter; or a separate observational
   comparison of binary fractions per swarm, which has never been measured.
2. **Static vs dynamic** is Q5, and costs nothing extra. Confirm this is what he meant.
3. **Registration form** was signed with the migration title. Does it need amending?
4. ~~**Planet set.**~~ **Answered 2026-09-15:** inner planets folded into the Sun, the
   four giants separate, dt = 1% of Jupiter's period (§2.1).
5. **Scope:** collisions and Yarkovsky out — confirm.
6. **Pre-registered analysis** (Stage 3.4): does he want to see it before the campaign?

---

## 9. Changelog

Bookkeeping starts when the project does. Planning-stage deliberation is in git
history, not here.

- **2026-09-12** — Plan written.
- **2026-09-15** — Meeting with Simon; research question still open, but a first run
  agreed: Stages 0–3, **no clones**, whole catalogue with H ≤ 13.6 as the complete
  flag, V24's Trojan definition compared with the JPL and MPC labels, inner planets
  folded into the Sun, giants separate, dt = 1% P_J, 10 Myr pilot. Repo restructured
  (`docs/`, `migration/`, `campaign/`, data on the external SSD).
