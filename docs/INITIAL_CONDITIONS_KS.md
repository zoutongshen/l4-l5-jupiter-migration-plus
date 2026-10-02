# Do L4 and L5 start differently, and does that make L5 erode faster?

Written 2026-09-30. This is Step 3 of the plan that followed the 2026-09-29 meeting.
- Script: `campaign/scripts/s3_distributions.py`.
- Figures: `campaign/reports/pilot_10myr/fig_ecdf.png`, `fig_composition.png`.
- Numbers: `ks_tests.csv`, `ic_reweighting.csv`, `escape_split.csv`.

**Short answer.**
- **Inclination differs strongly between the swarms; D doesn't; e differs only barely.**
- **The pilot's escape asymmetry (1.67) was not an effect of the swarms' starting
  orbits.** It came from comet-like interlopers that V24's element box admits and that
  V24's own 1 Myr cleaning would remove. For objects every catalogue calls a Trojan,
  L4 and L5 erode at the same rate: 8 of 8,803 against 5 of 5,512, a ratio of
  1.00 ± 0.57.
- **So present-day erosion produces no asymmetry at all.** That strengthens Simon's
  reading that the asymmetry, in number and in inclination, was set at capture.

---

## 1. What "cumulative is better for KS" means

The KS statistic *is* the largest vertical gap between the two empirical cumulative
distributions (ECDFs). Plotting the ECDFs therefore shows the test directly, with no
binning. Histograms hide that gap behind bin choices.

Anderson–Darling (k-sample) is the companion test. It weighs the tails more heavily,
and the tails are where escapers sit. Its p-value comes from 999 permutations, so it
bottoms out at 10⁻³.

## 2. The distributions (`fig_ecdf.png`)

**The proper elements are from V24.** Proper values average out the short-period
oscillations of the osculating ones, so they are the right quantities for comparing
populations.

**The three samples:**
- well determined (U ≤ 4);
- complete (H ≤ 13.6), where orbits are good and discovery bias is least;
- complete with V24 families removed.

**Medians and tests:**

| Variable | Sample | Median L4 | Median L5 | KS D | p(KS) | p(AD) |
|---|---|---|---|---|---|---|
| **proper I** | complete | 10.8° | 15.7° | **0.211** | 6 × 10⁻²⁹ | < 10⁻³ |
| | complete, no families | 11.1° | 15.1° | 0.174 | 2 × 10⁻¹⁸ | < 10⁻³ |
| | U ≤ 4 | 9.2° | 15.8° | 0.260 | 10⁻¹⁶⁴ | < 10⁻³ |
| **proper e** | complete | 0.048 | 0.051 | 0.050 | 0.049 | 0.039 |
| | complete, no families | 0.048 | 0.053 | 0.074 | 0.001 | < 10⁻³ |
| **D** | complete | 15.1° | 14.5° | 0.044 | 0.11 | 0.10 |
| | complete, no families | 14.7° | 14.4° | 0.036 | 0.33 | 0.16 |
| | U ≤ 4 | 15.4° | 14.2° | 0.062 | 8 × 10⁻¹² | 0.001 |

- **I: L5 is hotter.**
  - The difference is large and robust in every sample.
  - Our complete-sample KS D = 0.211 matches V24's debiased 0.221 (their H < 13.5 fit).
  - Removing families shrinks it a little (to 0.174) but leaves it very significant.
    In L4 the cold Eurybates family pulls the distribution down.
  - The U ≤ 4 sample exaggerates it (0.260). Below completeness, the cold L4 objects
    were found first (see the U-ladder section of `CATALOGUE_DEFINITIONS.md`).
- **e: almost homogeneous.**
  - The background difference is statistically significant but physically tiny: a
    median of 0.048 against 0.053, with KS D = 0.07.
  - In the full complete sample it is marginal (p ≈ 0.05).
- **D: no difference in the complete sample** (p = 0.11 and 0.33).
  - The U ≤ 4 sample shows a small one (D = 0.06). That is the same discovery-bias
    effect, since faint objects enter that sample.
  - **The libration amplitude is not a distinguishing feature of the two swarms.**

## 3. Does L5's starting distribution explain its extra escapes? (`fig_composition.png`)

**Method: indirect standardisation.**
- Cut (e, D, I) into strata.
- Take the pooled L4+L5 escape fraction in each stratum.
- Apply it to each swarm's own counts. That gives the escapes each swarm *would* have
  if both escaped identically at fixed starting orbit.
- The intervals come from a bootstrap over objects.

| Strata | Expected L5/L4 | Observed |
|---|---|---|
| e, D | 1.68 | 1.67 |
| I | 0.93 | 1.67 |
| e, D, I | 1.76 | 1.67 |

- **The starting (e, D) mix reproduces the whole observed ratio.**
- **Inclination explains none of it.** By I alone, L5 would escape slightly *less*.

**What the "(e, D) mix" actually is.** Escape in 10 Myr is almost deterministic from
where an object starts. The extreme-e tail leaves within a few kyr; the rest leaves
very rarely. Looking at who sits in that tail:

| Group (U ≤ 4) | L4 escaped | L5 escaped | Median e |
|---|---|---|---|
| Trojan by V24 and JPL/MPC | 8 / 8,803 | 5 / 5,512 | 0.069 |
| V24 box only (interloper) | 12 / 12 | 16 / 16 | 0.41–0.47 |

**About the 28 "V24 box only" objects:**
- They pass V24's loose cut (e < 0.6), but JPL doesn't label them TJN (their e is above
  JPL's 0.3 limit) and the MPC doesn't list them.
- Their Tisserand parameters (T_J = 2.3–2.8) are in the Jupiter-family-comet range.
- Every one escaped, most within a few kyr.
- V24's own 1 Myr cleaning removes these, and we don't apply it
  (`CATALOGUE_DEFINITIONS.md` §1).

**Because L5 is the smaller swarm, a similar number of interlopers (16 against 12)
makes a larger fraction of it.** That produced the whole of the pilot's 1.67.
Removing them gives **1.00 ± 0.57**. Applying V24's cleaning (dropping objects removed
within 1 Myr) gives 1.20 ± 0.91.

**Correction to earlier numbers.** The pilot report's headline escape ratio (1.67 ± 0.52;
`reports/pilot_10myr/report.md`) includes these interlopers. The U ladder
(`CATALOGUE_DEFINITIONS.md` §5) has been redrawn without them. Every cut up to U ≤ 7
now gives a ratio consistent with 1.

**What "interloper" means here.** It is an operational definition, based on the
catalogue labels, not on dynamics. An interloper is an object that passes V24's
element box at the epoch but is neither JPL class TJN nor on the MPC Trojan list
(`label_region == "V24"`). In this sample all 28 have e = 0.31–0.60 and
T_J = 2.25–2.82. 18 of them have U = 0, so this is not an orbit-quality problem, and
no U cut removes them. V24's dynamical definition (a 1 Myr libration check) removes
the same objects and gives 1.20 ± 0.91.

## 4. What this means for the capture question

1. **Present-day erosion is symmetric** within the pilot's statistics (13 real escapers).
   With a symmetric rate it can't turn a symmetric population into N4/N5 ≈ 1.6. The
   asymmetry, and the inclination difference, must have been in place before the
   present configuration. That points to capture or the early dynamical history.
2. **The observable a capture theory has to explain is the I distribution** (KS D ≈ 0.2,
   L5 hotter by about 5°), together with N4/N5. It is not D, and e only marginally.
3. **The escape-rate statistics are thin.** 13 escapers can't exclude a real asymmetry
   of tens of percent. Longer runs would tighten this; that is gap G6 in the plan.

**Still open.** Should the run's base sample follow V24's full definition, cleaning
included? Or should it be the V24 ∩ JPL/MPC intersection? Either removes the
interlopers. I'd suggest the intersection, because it is catalogue-based and doesn't
depend on our own integration.

**Added 2026-10-02.** Split by inclination, the complete sample's number excess sits
entirely below I ≈ 15–20°. The ratio N4/N5 is 2.01 ± 0.10 below 15° and 0.98 ± 0.06
above it (families removed; `ratio_by_inclination.csv`). The inclination difference and
the number asymmetry are therefore one feature: an extra cold L4 component. See
`docs/TIME_SYMMETRY_HYPOTHESIS.md` §6.
