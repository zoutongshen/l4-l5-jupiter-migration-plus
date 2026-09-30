# What the catalogue means by "Trojan", "U" and "L4/L5"

Written 2026-09-30. It prepares the U ladder and the KS tests planned after the
2026-09-29 meeting. The numbers are for the catalogue built on 2026-09-15
(`data/catalogue/2026-09-15/master_table.parquet`) and the 16,445 objects in the
`pilot_10myr` run.

**Short answer.**
- **"Trojan" means three different things:**
  - V24 uses an element-and-angle box.
  - JPL uses a looser box with no angle cut.
  - MPC publishes no criterion at all.
- **U says how well the *orbit fit* is known.** It says nothing about whether the
  object is dynamically stable. At U ≥ 7 the fit uncertainty in a is as large as
  the tadpole itself.
- **L4/L5 labels agree across all three sources** for every V24-selected object.

---

## 1. "Trojan": three definitions

| Label | Definition | Checks libration? | Source |
|---|---|---|---|
| **V24** | At one common epoch: 4.95 < a < 5.45 au, e < 0.6, I < 50°, 20° < \|σ\| < 130°, where σ = λ − λ_J | Only indirectly, through the σ window. V24 then integrates for 1 Myr and drops the objects that stop librating (26 L4 and 12 L5, "mostly Jupiter-family comets masquerading as Trojans") | Vokrouhlický et al. 2024, §2; `docs/VOKROUHLICKY_DISISTO_REVIEW.md` §2.1 |
| **JPL class TJN** | "Asteroids trapped in Jupiter's L4/L5 Lagrange points (4.6 au < a < 5.5 au; e < 0.3)" | No. It is an a/e box. It has no σ or inclination cut and doesn't test libration | [SBDB filter docs](https://ssd-api.jpl.nasa.gov/doc/sbdb_filter.html) |
| **MPC list** | Not stated. Neither the [list](https://www.minorplanetcenter.net/iau/lists/JupiterTrojans.html) nor its [symbol page](https://www.minorplanetcenter.net/iau/lists/StatsExplanation.html) gives a selection rule. The only definition given is for the Ln column: "L4 is 60° preceding the planet and L5 is 60° following" | Unknown | MPC pages, fetched 2026-09-30; our copy of the list is from 2026-09-15 and was last updated 2026-05-30 |

**How our code applies them** (`campaign/trojans/catalogue.py`):
- The candidate set is the union of three things:
  - the SBDB superset of V24's box;
  - everything JPL labels TJN;
  - MPC-listed objects that SBDB knows under another designation.
- `v24_selected` applies V24's four cuts to Horizons states at the common epoch t0 = JD 2461200.5.
- **We do not apply V24's 1 Myr cleaning.** The pilot's libration table (`campaign/reports/pilot_10myr/report.md`) is our version of that step.

**How the three labels overlap** (in-run objects; `reports/catalogue_2026-09-15/label_comparison.csv`):

| Region | L4 | L5 | Median H | What it is |
|---|---|---|---|---|
| V24+JPL+MPC | 9,789 | 5,936 | 14.6 / 14.5 | The core: every definition agrees |
| V24+JPL (not on the MPC list) | 547 | 17 | 15.9 | Mostly recent discoveries: 506 of the 564 have U = 7–9. The MPC list, last updated 2026-05-30, has not caught up with them. Why they are so L4-heavy isn't checked; recent observing geometry favouring L4 is a guess |
| JPL only | 60 | 20 | 15.6 / 15.2 | Inside JPL's a/e box but outside V24's. Median a = 4.91 au, σ spread over the whole circle, 75% have U = 9. Mostly not Trojans. In the pilot all 80 were removed within 1 Myr |
| All other combinations | 70 | 40 | 15–17 | Small edge cases (V24 only, V24+MPC, JPL+MPC, MPC only, none) |

So the V24 selection (10,362 L4 and 5,974 L5) is the right base sample. JPL's label adds
non-librators. MPC lists only 29 in-run objects that V24's box misses.

## 2. U: the orbit-uncertainty parameter

### 2.1 Definition

JPL's `condition_code` is documented as the "MPC 'U' parameter: orbit uncertainty
estimate 0-9, with 0 being good, and 9 being highly uncertain"
([SBDB filter docs](https://ssd-api.jpl.nasa.gov/doc/sbdb_filter.html)). JPL computes
it from its own orbit fit. The MPC's definition
([UValue.html](https://www.minorplanetcenter.net/iau/info/UValue.html)) is:

```
RUNOFF = (dT * e + 10/P * dP) * ko/P * 3600 * 3        [arcsec per decade]
CONS   = ln(648000) / 9
U      = INT( ln(RUNOFF) / CONS ) + 1,   clamped to 0 <= U <= 9
```

Here dT is the uncertainty in the perihelion time (days), dP the uncertainty in the
period (days), P the period in years, and ko the Gaussian constant in degrees per day.
**RUNOFF is the 3σ drift in the object's position along its orbit after 10 years.**
Each step in U is a factor of 648000^(1/9) = **4.42** in runoff; 648,000″ is 180°,
which caps U at 9.

The MPC itself warns that U "should not be used as a predictor for the uncertainty in
the future motion" of close-approaching objects. **U describes the fit. It says
nothing about dynamics.**

### 2.2 What U means for a Trojan

For Trojans e is small, so the dP term dominates. With dP/P = 1.5 σ_a/a and
10/P_J ≈ 0.84 orbits per decade, the formula converts to a 1σ uncertainty in the
semi-major axis:

σ_a ≈ a · RUNOFF[deg] / (3 × 0.843 × 360 × 1.5) ≈ 0.0038 au × RUNOFF[deg]

This conversion is ours, not the MPC's, and it ignores the dT·e term. The result is
the upper edge of each U class:

| U | Runoff ≤ | σ_a ≤ (au) | σ_a as a fraction of the tadpole width |
|---|---|---|---|
| 0 | 1″ | 1 × 10⁻⁶ | negligible |
| 1 | 4.4″ | 5 × 10⁻⁶ | negligible |
| 2 | 20″ | 2 × 10⁻⁵ | negligible |
| 3 | 87″ | 9 × 10⁻⁵ | < 0.1% |
| 4 | 6.4′ | 4 × 10⁻⁴ | ≈ 0.3% |
| 5 | 28′ | 1.8 × 10⁻³ | ≈ 1–3% |
| 6 | 2.1° | 7.9 × 10⁻³ | ≈ 5–10% |
| 7 | 9.2° | 0.035 | ≈ 20–50% |
| 8 | 41° | 0.155 | ≈ 100% |
| 9 | > 41° | > 0.155 | orbit essentially unknown |

"Tadpole width" is the proper Δa of a Trojan, ≈ 0.07–0.19 au (C = 1.58 to 2.0;
`VOKROUHLICKY_DISISTO_REVIEW.md` §2.2).

**What this predicts for the ladder.**
- For U ≤ 5, the starting condition is the real Trojan to within a few percent of its
  libration width.
- At U = 6 it is marginal.
- At U ≥ 7 the fitted a, and hence the libration amplitude we integrate, is
  effectively a random draw across the tadpole. Such objects will escape at a rate
  set by the fit error, not by their dynamics.
- So on physical grounds the cut should fall between U ≤ 4 and U ≤ 6, and U ≥ 7 is
  certainly unusable. Step 2 tests this against the escape data.

### 2.3 What U tracks in practice

This is the V24-selected sample:

| U | n | Median H | Median arc (days) | Median year of first observation | Fraction numbered |
|---|---|---|---|---|---|
| 0 | 12,522 | 14.3 | 7,183 | 2006 | 0.85 |
| 1 | 799 | 15.1 | 3,576 | 2015 | 0.04 |
| 2 | 420 | 15.2 | 2,388 | 2019 | 0 |
| 3 | 350 | 15.2 | 1,632 | 2020 | 0 |
| 4 | 252 | 15.4 | 1,245 | 2020 | 0 |
| 5 | 111 | 15.3 | 69 | 2021 | 0 |
| 6 | 179 | 15.5 | 39 | 2021 | 0 |
| 7 | 223 | 15.9 | 14 | 2024 | 0 |
| 8 | 383 | 16.0 | 10 | 2025 | 0 |
| 9 | 1,097 | 16.0 | 4 | 2015 | 0 |

- **U is really a proxy for arc length.** Objects with U ≤ 4 are multi-opposition orbits,
  with median arcs over 3 years.
- **There is a sharp break between U = 4 and U = 5.** The median arc falls from 1,245
  days to 69, meaning single-apparition orbits. **U ≤ 4 means "observed at more than
  one opposition"**, which is a more natural reason for the cut than "half and half".
- **U = 9 is a separate group.** These are old single-night or few-night detections
  (median first seen in 2015, arc 4 days) that were never recovered.
- **U correlates with H only because faint objects get short arcs.** In the complete
  sample (H ≤ 13.6) only one object in 3,049 has U ≥ 5. So the complete sample is
  untouched by the orbit-quality problem.
- **The two swarms carry different shares of bad orbits.** U ≥ 5 is 14.9% of L4 but
  7.5% of L5.
  This is why the pilot's raw whole-catalogue comparison came out inverted
  (`campaign/reports/pilot_10myr/report.md`).

## 3. L4 / L5

| Source | Rule |
|---|---|
| Ours (`swarm_from_sigma`) | sign of σ = λ − λ_J at t0: σ > 0 is L4 (leading), σ < 0 is L5 (trailing) |
| V24 | Separate catalogues: "greeks" = L4, "trojans" = L5 (`greeks-hcm_…`, `trojans-hcm_…`) |
| MPC | The Ln column; "L4 is 60° preceding the planet and L5 is 60° following" |

**Across all 16,336 V24-selected objects the three labels never disagree:**
- 12,373 objects are in V24's proper catalogue, and all of them match;
- 15,733 are on the MPC list, and all of them match.

The sign of σ is unambiguous inside V24's 20° < |σ| < 130° window. Outside the V24
selection, MPC and our label disagree for two objects (`label_comparison.csv`).
**The L4/L5 split is not a source of uncertainty.**

## 4. Consequences for the next steps

1. **The base sample is V24-selected.** JPL-only objects are mostly not Trojans.
2. **The U ≤ 4 cut has a physical meaning:** multi-opposition orbits, with σ_a well
   under 1% of the tadpole width. The error-propagation table predicts that the
   escape rate stays flat up to U ≈ 5 and jumps from U ≈ 7. Step 2 checks this.
3. **The complete sample (H ≤ 13.6) is where to run the KS tests on e, I and D**
   (Step 3). There U is irrelevant (every orbit is good) and discovery bias is
   smallest.

## 5. What the U ladder shows (Step 2, `campaign/scripts/s3_uladder.py`)

Figures: `campaign/reports/pilot_10myr/fig_grand.png` and `fig_uladder.png`; numbers in
`uladder.csv`.

- **The asymmetry doesn't depend on where we cut, up to U ≤ 7.** The escape ratio L5/L4
  for the cut U ≤ k is 1.6–2.1 for every k from 0 to 7 (1.67 ± 0.52 at k = 4). It
  falls to 1.25 at k = 8 and inverts to 0.39 at k = 9. So U ≤ 4 is a conservative,
  safe cut; U ≤ 5, 6 or 7 would report the same asymmetry.
- **Orbit error only takes over at U ≥ 8.** Taking each U on its own, U = 1–7 escape
  at ≈ 1% in 10 Myr, U = 8 at ≈ 3.5% and U = 9 at ≈ 20%. At fixed size
  (14 < H < 15.5) U = 9 still escapes at 23%, so this is the orbit fit, not size. It
  matches §2.2, where σ_a reaches the whole tadpole at U = 8.
- **Unexpected: U = 1–4 escape ≈ 5× more often than U = 0**, at fixed H as well
  (0.39% vs 0.11%). Yet their σ_a is < 0.3% of the tadpole, so orbit error can't be
  the reason. What differs is *which* orbits they are. At 14 < H < 15.5 the U = 1–4
  objects have median I = 17° against 11° for U = 0, and three times as many sit in
  the unstable edge (C > 2: 0.26% vs 0.08%). The likely reading is discovery order:
  surveys found the cold, stable Trojans first, and the hotter, marginal ones later,
  so they still have short arcs. This is a hypothesis; it hasn't been tested.
- **Consequence for Step 3.** Inclination distributions below completeness are shaped
  by discovery order. The KS tests should be quoted on the complete sample.
