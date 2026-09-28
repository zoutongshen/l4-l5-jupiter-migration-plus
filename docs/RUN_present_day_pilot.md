# Run note — present-day pilot, 10 Myr

Prepared 2026-09-28. Branch `present-day-pilot`, commit `b380f9e`.
Machine-generated results: `campaign/reports/pilot_10myr/`.

**In one line.** The whole known Trojan catalogue integrated forward 10 Myr under the
four giants. The headline is methodological: **the raw L4/L5 escape comparison is an
observational artefact**, and it inverts once poorly-determined orbits are cut.

---

## 1. Question

Does present-day dynamical erosion change the L4/L5 number ratio, and by how much? This
is the null test that Di Sisto, Ramos & Beaugé (2014) ran to 4.5 Gyr and found could not
produce the observed asymmetry. This pilot is a 10 Myr slice of that, run to establish
cost, the pipeline, and the diagnostics before committing to the full span.

**This is not a 4.5 Gyr erosion rate.** Escape fractions here are percent-level and
cannot be extrapolated linearly — the unstable population empties first.

---

## 2. Initial conditions

| | |
|---|---|
| Epoch t₀ | JD 2461200.5 TDB |
| Frame | heliocentric ecliptic J2000; converted to barycentric before integrating |
| Catalogue | 16,479 candidates → **16,445 integrated** (L4 10,446 / L5 5,999) |
| Source | JPL SBDB + MPC + Vokrouhlický et al. (2024), union of the three definitions |
| Planet states | JPL Horizons at t₀, giant-planet **system barycentres** |
| Clones | **None.** One trajectory per real object |

Selection follows V24 §2: `4.95 < a < 5.45 au`, `e < 0.6`, `I < 50°`, `20° < |σ| < 130°`.
An object enters the run if any of the three catalogues calls it a Trojan
(`v24_selected | jpl_tjn | mpc_trojan`). Swarm assignment is `σ > 0 → L4`.

No clones is a deliberate choice: errors are binomial over real objects, and each
trajectory's chaotic scatter is inside that error rather than being averaged away by
clone ensembles. It also means every number here refers to objects that exist.

---

## 3. N-body integration — specifics

This section is the one to read closely.

### 3.1 Code and integrator

| | |
|---|---|
| Code | REBOUND 4.6.0 (REBOUNDx present in the env but **unused** by this campaign) |
| Integrator | **WHFast**, Wisdom–Holman symplectic map, Jacobi coordinates |
| `safe_mode` | 1 (synchronise every step; no unsynchronised drift) |
| Symplectic corrector | 0 (none) |
| Timestep | **dt = 0.118609362512 yr**, exactly 1% of Jupiter's Keplerian period |

WHFast is the correct choice *here* and only here: every Trojan is a massless test
particle, so there is no mass-ratio problem, and the symplectic map buys 4.5 Gyr
affordability. (The mass ladder run cannot use it — see `RUN_mass_ladder.md` §4.)

dt = 1% P_J was fixed at the 2026-09-15 meeting. Stage 0.6 benchmarked it against dt/2
and against sampling cadences k = 1, 4, 8; all agree on encounters and final
classification, so k = 8 is safe and costs 8× less than k = 1.

### 3.2 Massive bodies — five, not nine

Mercury through Mars are **folded into the Sun**; the four giants are separate, each
carrying its planet-plus-moons system mass.

| body | mass (M☉) | a at t₀ (au) | R_Hill (au) |
|---|---|---|---|
| Sun + Mercury…Mars | 1.0000059769986 | — | — |
| Jupiter system | 9.54791910e−04 | 5.202509 | 0.355201 |
| Saturn system | 2.85885670e−04 | 9.540790 | 0.435787 |
| Uranus system | 4.36624961e−05 | 19.271275 | 0.470513 |
| Neptune system | 5.15138377e−05 | 30.084864 | 0.776154 |

Masses from JPL DE440 (Park et al. 2021). Hill radii use the circular approximation
`a (m/3M)^(1/3)` and are computed once at t₀.

### 3.3 Units and G

Simulation units are **au, Julian year, M☉**. G is *derived from DE440's GM☉* rather than
taken from CODATA:

```
G = GM_sun · yr² / au³ = 39.4769264212 au³ yr⁻² M☉⁻¹
```

This matters for reproducibility: it makes heliocentric elements from JPL convert exactly,
with no offset between the catalogue's orbit and the one the integrator reconstructs.

### 3.4 Test-particle treatment

```
sim.N_active          = 5      # Sun + four giants
sim.testparticle_type = 0      # Trojans feel the planets; no back-reaction, no mutual gravity
```

Trojans are **strictly massless**. They do not perturb the planets and do not feel each
other. The consequence used throughout the pipeline: particles cannot interfere, so
splitting 16,445 of them across 8 independent processes is bookkeeping, not physics.

*(That assumption has since been measured rather than assumed — see
`RUN_mass_ladder.md` §6 and `docs/MASS_THRESHOLD.md`. It holds with ~9 orders of margin
for single-body dynamics.)*

### 3.5 Sampling, diagnostics and removal

Every **8 steps** (≈ 0.949 yr) the runner reads all states and records, per Trojan:

- first entry within **2.1 R_Hill** of any giant — Di Sisto+ (2019) encounter criterion;
- first entry within **1.0 R_Hill** — Di Sisto+ (2014) criterion, at which the particle
  is **removed**;
- removal for `r > 100 au` or a non-finite state;
- first crossing of σ = 180° (a wrap) and of σ = 0° (a sign change), where
  σ = λ − λ_J.

Into 10 kyr windows it accumulates per-Trojan means of `a, e, I` and the V24 quasi-proper
amplitudes `A` (inclination), `B` (eccentricity), `C` (libration), plus σ min/max and its
circular mean, and a libration class. Classification is by which separatrix was crossed:

| crossed 180° | crossed 0° | class |
|---|---|---|
| no | no | tadpole (L4 if σ̄ > 0, else L5) |
| yes | no | horseshoe |
| no | yes | quasi-satellite |
| yes | yes | circulating |

1,000 windows over the run. State is checkpointed every 10 windows (100 kyr); restart is
**bit-for-bit identical** to an uninterrupted run, which is asserted in the test suite.

### 3.6 Achieved accuracy and cost

| | |
|---|---|
| Integrated span | 10,000,192.57 yr |
| Total CPU | **152.6 CPU-h** (549,230 s) |
| Per object | 3.34 CPU-s per Trojan-Myr |
| Wall | ≈ 20 h, 8 chunks concurrent on a laptop |
| Naive 4.5 Gyr projection | **7.8 CPU-yr** for this population, no removals |

---

## 4. Results

### 4.1 The headline is a data-quality finding

**Read the orbit-quality row, not the raw one.** Objects with JPL condition code U ≥ 5
escape at percent level, and L4 carries roughly twice L5's share of them — mostly recent
discoveries on short arcs. The raw sample therefore shows an L4 excess that is an
observational artefact, not dynamics.

| sample | swarm | n | removed (1 R_H) | fraction |
|---|---|---|---|---|
| all integrated | L4 | 10,446 | 302 | 0.0289 |
| all integrated | L5 | 5,999 | 74 | 0.0123 |
| V24-selected | L4 | 10,362 | 220 | 0.0212 |
| V24-selected | L5 | 5,974 | 50 | 0.0084 |
| **V24-selected, well determined (U ≤ 4)** | **L4** | **8,815** | **20** | **0.0023** |
| **V24-selected, well determined (U ≤ 4)** | **L5** | **5,528** | **21** | **0.0038** |
| complete (H ≤ 13.6), V24-selected | L4 | 1,836 | 2 | 0.0011 |
| complete (H ≤ 13.6), V24-selected | L5 | 1,213 | 3 | 0.0025 |

The cut **inverts the comparison**: raw, L4 escapes 2.3× more than L5; cleaned, L5 escapes
1.67× more than L4 — the direction Di Sisto+ (2014) report.

### 4.2 The asymmetry amplifier

`S4/S5 = (1 − f4)/(1 − f5)` — the factor by which erosion multiplies N4/N5.

| sample | escape ratio L5/L4 | amplifier S4/S5 | n escapers |
|---|---|---|---|
| all integrated | 0.428 ± 0.055 | 0.9831 ± 0.0022 | 380 |
| V24-selected | 0.389 ± 0.061 | 0.9867 ± 0.0019 | 273 |
| **V24-selected, U ≤ 4** | **1.674 ± 0.522** | **1.0015 ± 0.0010** | **41** |
| complete (H ≤ 13.6) | 2.270 ± 2.071 | 1.0014 ± 0.0016 | 5 |
| Di Sisto 2013 numbered | 5.928 ± 6.841 | 1.0025 ± 0.0018 | 4 |

For reference over 4.5 Gyr: Di Sisto+ (2014) 1.071 ± 0.025, Di Sisto+ (2019) 1.020 ± 0.023.

**Interpretation.** On the clean sample the amplifier is 1.0015 ± 0.0010 — consistent with
unity, i.e. **no detectable asymmetry generated in 10 Myr**, and consistent in sign and
scale with the 4.5 Gyr literature once the span is accounted for. The L5/L4 escape ratio
of 1.67 is **not significant** (41 escapers total).

### 4.3 Survival

16,445 → 16,065 alive; 380 removed, essentially all by the 1 R_Hill criterion (encounter
and removal counts differ by ≤ 3 across every sample, so objects that reach 2.1 R_H
almost always go on to be removed).

---

## 5. Do collisional families change this?

Families are not independent trials — a family is a set of fragments from one parent, so
*N* members are closer to one correlated unit than to *N* draws — and the two swarms carry
them in very different proportion. Checked on the same V24 + U ≤ 4 sample, using V24's own
family assignments.

| swarm | group | n | escapers | fraction |
|---|---|---|---|---|
| L4 | **family members** | 1,485 | **0** | 0 |
| L4 | background | 7,330 | 20 | 0.002729 ± 0.000609 |
| L5 | **family members** | 392 | **0** | 0 |
| L5 | background | 5,136 | 21 | 0.004089 ± 0.000890 |

**Not one family member escaped** — 1,877 objects across 13 families, zero removals, against
**5.65 expected** at each swarm's own background rate. Poisson P(0 | 5.65) = 0.0035; Fisher
exact p = 0.0047.

| variant | n L4 | n L5 | escapers | escape ratio L5/L4 | amplifier S4/S5 |
|---|---|---|---|---|---|
| with families (as §4.2) | 8,815 | 5,528 | 20 / 21 | 1.674 ± 0.522 | 1.0015 ± 0.0010 |
| **families removed** | 7,330 | 5,136 | 20 / 21 | 1.499 ± 0.467 | 1.0014 ± 0.0011 |

**The conclusion is unchanged.** The amplifier moves 1.0015 ± 0.0010 → 1.0014 ± 0.0011,
consistent with unity either way; the L5/L4 ratio stays consistent with unity either way.

**But families are not neutral.** L4 is 16.8% family members, L5 only 7.1%. Since no family
member escapes, including them dilutes L4's escape fraction more than L5's, inflating the
apparent L5 excess from 1.50 to 1.67. This bias runs **opposite** to §4.1's — short-arc
orbits inflate L4's escapes, families suppress them. The raw comparison is sensitive to
sample composition in at least two independent ways.

**Caveat — partly true by construction.** V24 identify families as clusters in
proper-element space, which requires proper elements stable enough to cluster. An object
whose elements diffuse cannot be assigned to a family at all, so family membership is
stability-selected *before* any integration. The 0/1,877 is real for this sample but is not
purely a dynamical result.

**Effective sample size.** Counting each family as one correlated unit: L4 8,815 → 7,339
(−16.7%), L5 5,528 → 5,140 (−7.0%). The binomial errors in §4.2 are therefore mildly
optimistic, and asymmetrically so — not enough to change a result already consistent with
unity.

Source: `campaign/reports/pilot_10myr/explore_families.csv` (sample `clean`), produced by
`campaign/scripts/s3_explore.py`.

---

## 6. Status and what it does not say

- **Timespan.** 10 Myr, not 4.5 Gyr. The unstable population empties first, so these
  fractions are not a rate.
- **The clean sample is small.** 41 escapers carry the headline number. The 4.5 Gyr run
  is what makes it a measurement.
- **The orbit-quality cut is now a required step**, not an option — it changes the sign of
  the raw result. Worth deciding whether U ≤ 4 or the H ≤ 13.6 completeness cut is the
  primary sample for the thesis.
- **Not included:** collisions, Yarkovsky, gas, migration, planetary mass growth. Gravity
  only.

## Reproduce

```
python campaign/scripts/s3_launch_local.py pilot_10myr --prepare-only
python campaign/scripts/s3_launch_local.py pilot_10myr --resume --workers 8
python campaign/scripts/s3_merge.py  --run-id pilot_10myr
python campaign/scripts/s3_report.py --run-id pilot_10myr
```

Config `campaign/configs/pilot_10myr.toml` is frozen; the launcher refuses to resume
against a changed copy.
