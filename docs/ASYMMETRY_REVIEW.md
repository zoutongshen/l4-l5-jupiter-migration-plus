# The L4/L5 Jupiter Trojan Asymmetry: Models, Di Sisto, and Where the Gaps Are

**Drafted 2026-09-03**, after the supervisor meeting of 2026-09-02 redirected the
project away from Jupiter migration and toward the asymmetry itself, with Di Sisto
et al. (2014) named as the paper to build on.

Scope: **dynamical explanations only**, whether simulated or analytic. Observational
bias and collisional grinding are excluded as candidate *mechanisms*, but retained
where they act as constraints a dynamical model must respect.

Companion documents: `docs/VOKROUHLICKY_DISISTO_REVIEW.md` (the two anchor papers read in
full, side by side), `docs/MSc_PROJECT_PLAN.md` (the migration project, now parked as a
fallback) and `literature/READING_LIST.md`.

---

## 1. What actually has to be explained

### 1.1 The number ratio

| Source | N4/N5 | Sample |
|---|---|---|
| Szabó et al. 2007 | 1.6 ± 0.1 | SDSS MOC3, complete to H ≈ 12.3 |
| Nakamura & Yoshida 2008 | 1.85 ± 0.42 | Subaru, D > 2 km |
| Grav et al. 2011 | 1.4 ± 0.2 | WISE, D > 10 km |
| Grav et al. 2012 | 1.34 | 55 L4 / 41 L5, D > 50 km |
| Nesvorný et al. 2013 §3.4 | 1.3 ± 0.1 | H < 12, families removed |
| Yoshida & Terai 2017 / Uehata et al. 2022 | 1.40 ± 0.15 | Subaru HSC, D > 2 km |
| **Vokrouhlický et al. 2024 §5.2** | **f45 = 1.43 ± 0.05**; f45 = 1 excluded at **7.9σ** | debiased CSS background, families removed |
| Bottke et al. 2023 §2.2 | 9 vs 9 | D > 100 km; consistent with 1.43 (binomial p = 0.30) |
| Vokrouhlický et al. 2024 Fig. 15 | 46 vs 26 (1.77) | H < 10, complete MPC counts, families included |

Adopt **1.43 ± 0.05 at H ≈ 13.8** (Vokrouhlický et al. 2024, Fig. 25 caption). It is
the only bias-corrected, family-subtracted determination. The paper's bookkeeping is
inconsistent: the abstract gives 1.45 ± 0.05 "for H < 15", Conclusion 6 gives
1.43 ± 0.05 "at magnitude H = 15", and the introduction's naive MPC estimate is
1.45 ± 0.05 at H ≤ 13.8. Cite the Fig. 25 value with its magnitude limit.

### 1.2 Is the ratio size-dependent? Not demonstrably

An earlier version of this section claimed the asymmetry switches on only below
~50 km, and called that the largest unclaimed hole in the field. Checked against the
figure, the data do not support it.

- **Vokrouhlický et al. 2024 Fig. 25** (PDF p. 31) plots the cumulative background
  ratio f45(<H). The model median is nearly flat: ≈ 1.3 at H ≈ 9 and 1.43 at H ≈ 13.8.
  The caption's "the largest Trojans, below magnitude ≃ 9 − 9.5, exhibit symmetry"
  means only that the 1σ band is wide enough to include 1 there. It includes 1.43
  just as well, and the caption adds that "the statistical robustness increases with
  increasing H".
- **Bottke et al. 2023's 9 vs 9 at D > 100 km** would turn up 30% of the time if the
  true ratio were 1.43.
- **The brightest well-counted bin leans the other way.** Fig. 15 gives 46 L4 vs 26 L5
  objects with H < 10 in the complete MPC catalogue. A symmetric population produces
  that or more only 1.2% of the time; a ratio of 1.43 does so 23% of the time. The
  count includes family members; removing the six Table 1 family largest remnants
  brighter than H = 10 (four in L4, two in L5) still leaves 42 vs 24.
- **More data won't sharpen the bright end.** Every large Trojan is already known, so
  it is small-number limited for good.

So the size-independent ratio that every dynamical mechanism predicts is *consistent*
with the data. Di Sisto et al. 2019 (§2.1) assume exactly that when scaling escape
rates to km-sized bodies. The one genuinely size-dependent L4/L5 feature in this
literature is in inclination, not number, and it is the families (§3.4). The
unexplained ΔH ≈ 0.4 mag offset (§1.3) remains, but that is an albedo/shape question,
not a number-ratio one (G5).

### 1.3 Which L4-vs-L5 observables actually differ

Vokrouhlický et al. 2024 fit bias-corrected background distributions (families
removed) in three orbital parameters — A (inclination), B (eccentricity), and
C (semimajor-axis offset, equivalently the **libration amplitude**) — and compare
the swarms with a KS test.

| | KS distance D | p (same parent) |
|---|---|---|
| **A — inclination** | **0.221** | **≤ 10⁻²⁰** |
| B — eccentricity | 0.051 | 0.08 |
| **C — libration amplitude** | **0.043** | **0.17** |

At a stricter H < 12 cut: D = 0.193 / 0.093 / 0.084, p = 2.5 × 10⁻⁴ / 0.20 / 0.35.
Their conclusion, verbatim:

> "the L4 and L5 distributions in eccentricity and semimajor axis (or the libration
> amplitude) appear to be compatible with one another, while the distribution in
> inclination still appears to be statistically inconsistent. **Here again we leave a
> more thorough analysis and a possible explanation for future studies.**"

The full observable inventory:

| Observable | Status | Number |
|---|---|---|
| Number ratio | Measured, unexplained | 1.43 ± 0.05 at H ≈ 13.8; no demonstrated size dependence (§1.2) |
| **Inclination distribution** | **Different, unexplained** | KS D = 0.221; background medians ≈ 11.5° vs 17.5° (read off Vokrouhlický Fig. 26). Raw means including families: 10.4° vs 14.2° (Di Sisto 2014) |
| Eccentricity distribution | No difference | p = 0.08 |
| **Libration amplitude** | **No difference** | p = 0.17 |
| Magnitude-distribution slopes | No difference | equivalent to H ≈ 14 |
| Taxonomy (D > 50 km) | No difference | D-type 84% vs 71–80% |
| Albedo | No significant difference | 0.07 ± 0.03 both |
| ΔH offset, differential distributions | **Different, unexplained** | L4 brighter by ≈ 0.4 mag beyond H ≈ 11 |
| Mean elongation b/a | **Different, contested** | 0.77 ± 0.02 vs 0.86 ± 0.02 (McNeill et al. 2021) |
| Colour | **Contradictory** | KS p = 0.01 (McNeill) vs "vanishes when split by inclination" (Szabó 2007; Roig et al. 2008) |
| **Family morphology** | **Different, explicitly unexplained** | 9 families at L4 vs 4 at L5; L4 sharp, L5 diffuse |
| Rotation, slow-rotator, binary fraction per swarm | **Never measured** | ~15% slow rotators overall, no L4/L5 split |

Two caveats worth carrying. Vokrouhlický's own Conclusion 7 hedges the inclination
result — the L5 CSS data are shallower by ≈ 0.35–0.40 mag because Jupiter's mean
anomaly moved during phase II, so he calls the inclination asymmetry "an interesting
hypothesis that needs to be verified." And Bottke et al. 2023 record that "the
discrepancy between McNeill et al. (2021) and Schemel and Brown (2021) has yet to be
resolved."

---

## 2. The dynamical models

### 2.1 Comparison

| Model | Era | Integrator / dt | Massive bodies | N particles | Best ratio | Sign robust? |
|---|---|---|---|---|---|---|
| Pirani et al. 2019a | gas disc | MERCURY hybrid / 50 d | 4 giants, growing + migrating | 44,000 × 10 runs | 1.80 ± 0.26 | **Yes** |
| Pirani et al. 2019b | gas disc + 4.5 Gyr | MERCURY hybrid / 140 d | Jup (+ Sat) + embryos | 10⁵ | 1.60 | **No** |
| Nesvorný et al. 2013 | GP instability | SyMBA replay + swift_rmvs3 / 0.25 yr | 5 planets | 1.25 × 10⁸ cloned | 1.3 ± 0.5 | **No** |
| Li et al. 2023a | GP instability | swift_rmvs3 / 0.5 yr | Sun + Jupiter only | 500/swarm | 1.50 → 1.68 | Yes |
| Li et al. 2023b (FFP, indirect) | any | RKF7-8 + swift / 0.05 yr | Sun, Jup, FFP | 500/swarm | 1.65 → 1.81 | Yes |
| Li et al. 2023b (FFP, direct) | any | RKF7-8 + swift / 0.05 yr | Sun, Jup, FFP | 500/swarm | **1.86** | Yes, by construction |
| Di Sisto et al. 2014 | Gyr | EVORB / 0.2 yr | Sun + 4 giants | 2972 real + 2 × 18,200 | ×1.07 ± 0.03 amplifier | Yes, 3.0σ |
| Di Sisto et al. 2019 | Gyr | EVORB / 7.3 d | Sun + Venus–Neptune | 2972 real | ×1.02 ± 0.02 amplifier | **No** (0.9σ) |

The Di Sisto amplifier errors are binomial, computed here from the papers' escape
counts (§3.3); neither paper quotes one.

### 2.2 What each one claims

**Pirani et al. 2019a — inward gas-disc migration of a growing Jupiter.** Relative
drift between the migrating planet and co-orbital particles widens the L4-side path
of the horseshoe orbit and narrows the L5-side, so particles spend more *time*
leading. Jupiter's mass growth then pinches horseshoes into tadpoles, freezing the
excess in. Note this is a **residence-time** asymmetry, not a capture-rate asymmetry,
and it is reversible until the growth step: "when the migration stops the excess
simply disappears because the horseshoe orbits are back symmetric."

The causal test is the cleanest in the literature: outward migration gives
N4/N5 ≈ 0.6, in-situ growth gives 0.94 ± 0.35, and removing gas drag changes nothing
— so relative drift, not drag, is the cause.

*Admitted failures:* completely flat inclinations; captured Trojan mass 10⁻¹–10⁻² M⊕
against an observed ~10⁻⁵ M⊕; requires Δa_J > 3.5 au inward, which Deienno et al.
(2022) argue over-implants the main belt.

**Pirani et al. 2019b — the fixes break the mechanism.** Of three ways to obtain the
observed inclinations, secular evolution of a flat population fails outright;
embryos embedded in the swarms stir correctly but cannot be removed in 4.5 Gyr and
can **reverse** the ratio to 0.18; and a pre-stirred disc works but drops the
asymmetry from 1.60 to **1.11–1.15**, recovering 1.3–1.4 only by starting Jupiter's
seed at 23–29.5 au. Her own diagnosis: "The low ratio is due to less efficiency in
the mechanism generating the asymmetry in the pre-stirred case."

**Nesvorný et al. 2013 — jump capture plus ice-giant traversal.** The asymmetry is a
separate, later step from the capture: the ejected ice giant traverses one swarm and
scatters it. Their own verdict: "**the statistics is not good enough to rule out
f45 = 1 at more than 1σ**", and two of three cases favour L5. Notably, Nesvorný's own
2018 ARA&A review drops the asymmetry claim entirely.

**Li et al. 2023a — sub-critical outward jump.** A purely survival-side mechanism:
outward drift contracts the L4 island (amplitudes shrink, objects stabilise) and
expands L5 (amplitudes grow, cross L3, escape). Control model Δa_J = 0.5 au,
τ = 1000 yr → ȧ_J = 1.5 × 10⁻⁴ au/yr, R45 = 1.50 at 1 Myr rising to 1.68 at 1 Gyr.

**Li et al. 2023b — a free-floating planet.** Two variants. *Indirect:* the FFP
accelerates Jupiter above the critical rate ȧ_crit = 3.8 × 10⁻³ au/yr, temporarily
destroying the L4 island. *Direct:* the FFP passes through the L5 swarm — all 500 L4
survive, 269/500 L5 survive, R45 = 1.86. **The direct case is the only model in the
set that reproduces the ratio and the observed libration amplitudes at once**,
because it alone is seeded with Δσ₀ = 0–40°.

### 2.3 Two problems that cut across all of them

**Amplitude versus ratio.** Li reaches 1.6 only with final resonant amplitudes
≳ 15–20°, and ≳ 35° in the fast cases, against observed amplitudes concentrated
below 30°. The proposed fix — collisional damping, Marzari & Scholl (1998b, 2000) —
**has never been simulated for this problem by anyone.**

**Comparability.** Pirani counts as Trojans all particles with aphelion and
perihelion within ±2.5 Jupiter Hill radii, which admits horseshoes and
quasi-satellites; Li and Di Sisto use tadpole criteria. The published ratios are not
directly comparable without recomputing on a common membership rule.

---

## 3. Di Sisto et al. (2014) and (2019) in detail

Both PDFs in `literature/` are arXiv preprints (arXiv:1604.05331v1 and
arXiv:1811.00352v1). Cite Icarus 243, 287–295 (2014) and Di Sisto, Ramos & Gallardo
(2019), Icarus.

### 3.1 What was integrated

**Real Trojans.** All 2972 numbered Jupiter Trojans as of March 2013 — **1975 L4 and
997 L5**, ratio 1.98 raw. Deliberately not magnitude-limited: "we will consider the
complete (numbered) population regardless of the absolute magnitude." Nothing is
debiased; the catalogue source, the element epoch, and the L4/L5 membership rule are
all unstated.

**Fictitious Trojans (2014 only).** A regular grid of **18,200 massless particles
around each swarm**: a = 5.01–5.40 au (Δ0.01, 40 values), e = 0–0.34 (Δ0.01, 35),
i = 0–60° (Δ5°, 13) — 40 × 35 × 13 = 18,200 exactly. All angles are **fixed** at
ϖ = ϖ_J ± 60°, Ω = Ω_J, M = M_J.

That last point matters more than it looks. With every angle pinned at the Lagrange
configuration, **libration amplitude is not a sampled dimension** — it is a derived
quantity set by the (a − a_J) offset and e. The grid is effectively the
zero-amplitude corner of the space, and the L4 and L5 grids are exact mirror images.
So the "no difference between the swarms" result is a null test of the *dynamics*,
along one slice, not a test of the populations.

### 3.2 Method

EVORB (Fernández et al. 2002), Sun plus the four giant planets with the inner
planets' masses folded into the Sun, integrated **4.5 Gyr forward** from the present
configuration. Escape ≡ first approach within **1 Hill radius** of any giant planet,
at which point the particle is deleted. No Yarkovsky, no collisions, no gas, no
migration — and no explicit statement excluding them; the justification is a single
citation to Robutel et al. (2005) that Trojan instability "appears purely
gravitational." Each real Trojan is integrated **once**.

The 0.2 yr timestep is stated only for a short pre-integration, never for the
production runs.

### 3.3 Results, and the revision

| | 2014 | 2019 |
|---|---|---|
| L4 escape fraction, 4.5 Gyr | 458/1975 = 23.2% (paper: "~23%") | 466/1975 = 23.6% (paper: ± 0.2%) |
| L5 escape fraction, 4.5 Gyr | **282/997 = 28.3%** | **250/997 = 25.1%** (paper: ± 0.2%) |
| Binomial error, L4 / L5 (derived) | ± 1.0% / ± 1.4% | ± 1.0% / ± 1.4% |
| L5 − L4 significance (derived) | **z = 3.0**, p = 0.002 | **z = 0.9**, p = 0.37 |
| Survival-rate fit, L4 | −3.9929e−11 ± 2.45e−13 yr⁻¹ (t > 2 Gyr) | 7.0398e−11 ± 8e−14 (full span) |
| Survival-rate fit, L5 | −5.11663e−11 ± 4.14e−13 yr⁻¹ (t > 2 Gyr) | 7.5590e−11 ± 1.3e−13 (full span) |
| Primordial N0(L4)/N0(L5) | **1.3 ± 0.2** | **1.373 ± 0.204** |
| Share of unbiased asymmetry from Gyr evolution | **~25%** | **~7%** |
| Planets | Sun + 4 giants | Sun + Venus–Neptune |
| Timestep | 0.2 yr | 7.3 d |
| Escape threshold | 1 R_Hill, deleted | 2.1 R_Hill, evolution continued |

Fictitious populations, 2014 only: sf_L4 = −1.10675e−11 ± 8.819e−14,
sf_L5 = −1.07656e−11 ± 9.248e−14 yr⁻¹ — "virtually identical". **This was never
redone with the 2019 setup**, so the symmetry claim still rests entirely on 2014's
numerics.

The same group, on the same input orbits, revised its headline by a factor of ~3.5
and never analysed why. The 2019 paper says only "in agreement with the results of
D14, although the difference is smaller." Under counting statistics the two
amplifiers, 1.071 ± 0.025 and 1.020 ± 0.023, differ by 1.5σ. For two runs of the *same*
objects that is an upper bound on the noise, so the revision is partly noise and
possibly partly systematic (G4). The 2019 "± 0.2%" comes from repeated 80% subsamples
of the same trajectories. It is not a counting error, and with binomial errors the
2019 L4/L5 difference is not significant.

### 3.4 The argument, and where it fails

Di Sisto's reasoning is a four-link chain:

1. Fictitious particles show L4 ≡ L5 dynamically. **Demonstrated.**
2. Real L5 escapes faster than real L4. **Demonstrated in 2014 (3.0σ); not
   significant in 2019 (0.9σ)** (§3.3).
3. Therefore the difference lies in the initial *distributions*. **Asserted.**
4. Therefore the relevant distributional difference is *inclination* (⟨i⟩ = 10.4° vs
   14.2°). **Implied, never stated.**

Link 3 is stated outright (2014 §3.3, arXiv p. 18): "Since the dynamics appear equal,
then the difference must lie in the distribution of the Trojans around each Lagrange
point." Link 4 is not. The next sentence turns straight to the inclination difference
("As we mentioned in Sect (2.1), the inclinations of the observed Trojans in L4 show an
excess…"), and the scenarios offered next (Nesvorný's ice giant, differential
collisional evolution) explain where that difference came from, not how it would drive
escape. Quote the passage when citing it; don't paraphrase it as a causal claim.

Link 4 also sits uneasily with his own data. 2014 §2.2 (arXiv p. 9), on the escaped
(red) versus stable (black) asteroids in Figs. 1 and 3:

> "While there appears to be little correlation between both sub-sets in the (i, a)
> plane, the (e, a) and (e, D) planes show that the escaped asteroids are preferably
> those with larger libration amplitude or with larger orbital eccentricity."

This is a visual impression of scatter plots, with no statistical test, and it concerns
which *individual* asteroids escape, not the L4/L5 asymmetry. It is weak evidence either
way, and the paper contains a point on each side:

- *Against an inclination effect:* Fig. 7 (§2.2). The time-weighted inclination
  distributions keep their shape over 4.5 Gyr: "the dynamical evolution of Trojans
  doesn't change their original inclination properties."
- *For an inclination effect:* §2.1 (arXiv p. 6) cites Tsiganis et al. (2005): "The
  'effective' stability region shrinks with increasing orbital inclination."

The paper reports a and e as essentially identical between the swarms: ⟨a⟩ = 5.2062 vs
5.2068 au, ⟨e⟩ = 0.072 vs 0.074. So the variables his plots associate with escape (D
and e) do not obviously differ between the swarms, and the one that does differ (i) has
an effect on escape that nobody has measured.

**The inclination gap Di Sisto invokes has two parts, and only one of them is
families.** His ⟨i⟩ values come from all 2972 numbered Trojans, families included, and
he notes they agree with Slyusarev (2013). Vokrouhlický et al. 2024 §5.2 says such
earlier inclination anomalies "are readily explained by the population of the principal
Trojan families."

- **Families explain the *low-inclination L4 excess among small bodies*.** Eurybates
  (875 members, sin I_P = 0.129, I ≈ 7.4°, all at H ≥ 9.82) and Arkesilaos (235, I ≈
  8.9°) sit exactly there. Di Sisto's own size split (§2.1) says the L4 excess at
  i ≲ 10° "is maintained for the smaller Trojans but it is not noticed for the greater
  ones."
- **Families don't explain *L5 being hotter at all sizes*,** which Di Sisto also
  reports. That part survives family removal and debiasing, and the gap grows.
  Vokrouhlický's background medians are A ≈ 0.20 vs ≈ 0.30 (read off Fig. 26), i.e.
  ≈ 11.5° vs ≈ 17.5°, and the KS distance barely changes at H < 12 (0.221 → 0.193).

Di Sisto's §2.3 reviews the family literature and computes proper elements, but he
never separates family members from the background in his escape statistics (G1).

**Don't confuse this with the 2019 inclination argument.** Di Sisto et al. 2019 §4.2
says "the only variable that affects the result of an encounter is the different mean
initial inclination." That is an Öpik-theory argument about what happens *after*
escape (L4 escapees collide with Jupiter about three times as often as L5 escapees),
not about the escape rate. The 2019 Fig. 2 repeats the red/black scatter plots, but
the text says nothing about inclination and escape.

---

## 4. Candidate research gaps

### G1 — Di Sisto's explanation has never been tested against Vokrouhlický 2024

**Recommended headline.** Put the two papers side by side, which nobody has done:

| | Associated with escape? (Di Sisto 2014 §2.2, by eye) | Differs between swarms? (Vokrouhlický 2024, KS) |
|---|---|---|
| Libration amplitude | Yes: escapers "preferably those with larger libration amplitude" | Not detected (p = 0.17) |
| Eccentricity | Yes: "…or with larger orbital eccentricity" | Not detected (p = 0.08) |
| Inclination | "little correlation" in the (i, a) plane | **Yes** (p ≤ 10⁻²⁰) |

**As far as anyone has checked, the variables associated with escape do not differ
between the swarms, and the variable that differs has no demonstrated effect on
escape.** This is an untested mismatch, not a refutation. Every entry in the left
column is a visual reading of scatter plots, and both "not detected" entries are null
results (§3.4). Di Sisto's link 3 ("the difference must lie in the distribution of the
Trojans around each Lagrange point") is therefore unsupported: nobody has shown which
distributional difference would produce 23.6% vs 25.1%, and at 0.9σ nobody has shown
that difference is real (§3.3). Both papers are in
`literature/`; neither cites the other on this point, because Vokrouhlický's
measurement postdates both Di Sisto papers.

*The test*, in three steps, cheapest first.

1. **Is there anything to explain?** The 2019 difference is 0.9σ (§3.3). First
   establish the present-day escape asymmetry with a real error bar (G4), mainly by
   cloning the marginal objects; more real objects help less than they seem.
2. **Is it families?** Cross-match the integrated sample against Vokrouhlický's public
   family catalogue (https://sirrah.troja.mff.cuni.cz/~mira/tmp/trojans/) and report
   escape fractions for background and family members separately.
   - Why it matters: families can bias the escape fraction either way. Fragments are kicked mostly in libration
     amplitude (V24: families are "most extended in the daP coordinate"), so a family has a
     stable core and a halo reaching towards the unstable edge. An old family's halo has
     already leaked away, leaving mostly stable members that *dilute* the escape fraction. A
     young family (V24 leaves open whether Hektor's dense core is a recent cratering event)
     has been filtered for less time than the background and can *leak faster*. Ages are
     unknown, so the sign is too. Sharp L4 families diluting and diffuse L5 families leaking
     would both push L5's escape above L4's, so the morphology difference (G2) and the escape
     difference could be one phenomenon.
   - Scale: if ≈ 6% of the 2019 L4 sample were effectively stable family members and
     both backgrounds eroded identically, L4 would show exactly 23.6% against L5's
     25.1%. Eurybates alone is ≈ 11% of today's L4 catalogue.
   - The cross-match alone, with no integration, takes an afternoon and bounds the
     effect.
3. **Is it inclination?** Measure escape probability as a function of (A, B, C) on a
   grid that samples the angles and libration amplitude, which Di Sisto's does not.
   Propagate Vokrouhlický's **debiased, family-free** distributions through it and
   predict the escape asymmetry.
   - Vokrouhlický's own Fig. 14 motivates this. The stable zone shrinks with
     inclination: its low-B edge falls from C ≈ 2.0 at A ≤ 0.25 to C ≈ 1.76 at
     A = 0.65. So inclination can drive escape *through* the (D, e) boundary while
     showing "little correlation" in an (i, a) scatter plot.
   - The integration is gravity-only with massless particles, so size cannot enter the
     map; it enters only through the distributions. Propagate them H-binned as well as
     whole to get the size dependence of the erosion contribution with no extra
     integration.

- If step 3 reproduces the asymmetry found in step 1, link 3 is proven and the
  operative variable identified.
- If it doesn't, link 3 fails, and the Gyr contribution is *unexplained* rather than
  merely small.
- If step 1 finds no asymmetry, Gyr erosion contributes nothing and the whole
  asymmetry is primordial, which is a clean result in itself.

Every outcome is publishable. Vokrouhlický explicitly leaves the door open: "we
leave a more thorough analysis and a possible explanation for future studies."

*Why it fits.* Non-migration; squarely on the asymmetry; built on the paper Simon
named; and it needs the resonant-angle and libration-amplitude machinery from the
parked migration project — the one component of that work that transfers intact.

### G2 — Why are the L5 families diffuse and the L4 families sharp?

Vokrouhlický et al. 2024 identify **9 statistically robust families at L4 and 4 at
L5**, and flag the difference as an open problem in his own words:

> "The morphology of the L4 and L5 families is surprisingly different: the L4 families
> are sharp and highly concentrated clusters, with possibly very low populations of
> interlopers, while the L5 families are diffuse and broad clans, with possibly a
> substantial population of interlopers. **The origin of this difference is unknown**,
> but possibly bears important clues about the past collisional evolution of these
> populations as well as their origin."

And the candidate causes, also his: "the families in L5, while also existing, seem to
be blurred by either higher ejection velocities, dynamical diffusion at their
particular location or some other effect." He tests none of the three.

Critically, this is **in tension with his own 100 Myr stability experiment** (1,040
cells in the C–B plane, 28 orbits per cell spread over 7 values of A), which found
"no statistically meaningful differences, implying that the orbital stability of the
L4 and L5 regions is the same" (a comparison of 50%-stability curves by eye; no
statistic is given). If dynamical diffusion is the cause, it must
therefore be Gyr-scale and conditioned on where the families actually sit — precisely
the Di Sisto regime, and invisible at 100 Myr.

*The test.* Seed synthetic families at the real L4 and L5 family locations in
(A, B, C) and measure differential spreading over Gyr. Highest scientific upside;
shares all of G1's machinery. G1 step 2 gives a first empirical look for free.

### G3 — Reconcile the disagreement about present-day symmetry

Three papers say the swarms are dynamically equivalent today: Di Sisto et al. 2014
(fictitious grid), Vokrouhlický et al. 2024 (100 Myr clones), Holt et al. 2020. Two
say otherwise: **Freistetter 2006**, in which Saturn's perturbation makes L4
intrinsically more stable — cited approvingly by Li et al. 2023a as "further
enhancing the L4/L5 number asymmetry" — and **Hou, Scheeres & Liu 2014**, in which a
temporary asymmetry arises from identical initial conditions in both regions.

Nobody has reconciled these. Weight of evidence favours symmetry, so this is a check
to run rather than the headline — but note both large-grid symmetry tests have blind
spots: Di Sisto's samples a single angular configuration, Vokrouhlický's runs only
100 Myr.

### G4 — Is present-day erosion asymmetric at all?

This is now the first question, not the safety net. Counting statistics alone (§3.3)
show:

- **2014:** 23.2% vs 28.3%, z = 3.0. **2019:** 23.6% vs 25.1%, z = 0.9, not
  significant.
- **Counting noise could account for the whole "factor of 3.5 revision", but maybe not
  all of it.** The two amplifiers, 1.071 ± 0.025 and 1.020 ± 0.023, differ by 1.5σ
  under binomial errors. For two runs of the same objects, that is an upper bound on
  the run-to-run noise: a change of setup re-randomizes only the marginal escapers,
  while deep-stable and fast-unstable objects behave the same way in both runs. Part
  of the shift may therefore be systematic (planet set, timestep). Clones measure the
  split.
- **The escape threshold is ruled out as a cause.** 2019's 2.1 R_Hill criterion is
  strictly more inclusive than 2014's 1 R_Hill (a trajectory that reaches 1 R_Hill has
  already passed 2.1), so it cannot explain L5's drop from 282 to 250 escapers. That
  leaves the planet set, the timestep, and noise.
- **2019's "± 0.2%" is not a counting error.** It comes from repeated 80% subsamples of
  the same trajectories; the binomial errors are ± 1.0% (L4) and ± 1.4% (L5).
- **2019's linear escape rates don't match its own fractions.** 7.04 and 7.56 × 10⁻¹¹
  yr⁻¹ integrate to 31.7% and 34.0% over 4.5 Gyr, not 23.6% and 25.1%, and the
  normalization is never stated.

*The test.* Integrate the complete modern sample (H ≲ 13.8) for 4.5 Gyr and clone its
marginal objects. Two design points:

- **Clones matter more than extra objects.** The complete background is only
  ≈ 2,040 L4 + ≈ 1,410 L5 (Vokrouhlický's introduction), barely more than Di Sisto's
  2,972. Going fainter adds objects but brings the survey bias back.
- **Clones give each object's escape probability.** That removes the chaotic noise
  from "how will *these* swarms erode" and separates noise from systematics in the
  2014 → 2019 shift.

Whether the *distributions* differ in how easily objects escape is a separate question.
It is limited by the finite number of real objects, and is answered without that limit
by the synthetic grid of G1 step 3. Report escape fractions with honest errors,
separately for background and family members (G1 step 2). Either answer is a result: an escape asymmetry at a stated
significance, or none, in which case the Gyr contribution is zero.

This matters downstream: Li et al. 2023a builds a ×1.1 Gyr correction on the *2014*
number.

### G5 — Size dependence: dropped as framed

The premise ("D > 100 km is 9:9, f45 → 1 below H ≈ 9–9.5") does not survive the
figure (§1.2). The ratio is consistent with being size-independent, which is what
every dynamical model predicts, so there is no size dependence of the number ratio to
explain.

Two observational loose ends remain, both outside this project's dynamical scope:

- **Vokrouhlický's ΔH ≈ 0.4 mag offset** between the differential distributions beyond
  H ≈ 11. With equal size distributions, an albedo explanation needs L4 *brighter*:
  p_L4/p_L5 = 10^(0.4 × 0.4) ≈ 1.45, roughly 0.07 (L4) vs 0.05 (L5).
  - Vokrouhlický et al. 2024 (§5.2) print the reverse, 0.05 (L4) vs 0.07 (L5), and
    their formula "1 − 10^(−0.4ΔH) ≃ 0.44" evaluates to 0.31. An earlier version of
    this section copied their numbers.
  - The shape alternative has the correct sign: McNeill et al. (2021) find L4 more
    elongated, hence brighter on average; Schemel & Brown (2021) dispute it.
- **The L5 "vacancy" at H ≈ 9.5–10** (Vokrouhlický Fig. 15: 46 vs 26 at H < 10).
  Probably small numbers; an afternoon's statistics check.

### G6 — Back-extrapolation

Both Di Sisto papers convert present-day escape rates into a primordial ratio by
assuming the rate extrapolates backwards, while the 2014 paper simultaneously states
the early rate "was much larger during the early stages of the Solar System, and we
are currently just detecting the tail of the distribution." The paper flags the
problem — "this assumption is not obvious, and must be considered with care" — and
never resolves it. Quantifying the error changes the primordial ratio that every
origin model is required to hit.

---

## 5. Feasibility — read before choosing

Measured from `migration/results/batch_results.json`: **≈ 45–49 s of CPU per kyr** at
N = 10⁴ particles with `planet_set='all'` (8 massive bodies, dt ≈ 0.031 yr, clamped by
Venus; single-threaded workers).

- **Extrapolated, 4.5 Gyr ≈ 6.4 CPU-years per 10⁴ particles.** That is ≈ 2 CPU-years
  for Di Sisto's 2,972 and ≈ 8 for the ≈ 12,400-object modern catalogue. An earlier
  version of this section said 6,400 CPU-years, a factor-1000 slip.
- **This matches Di Sisto et al. 2019's own experience:** 156 jobs of ~20 Trojans at a
  7.3 d step, "several computers … for several months".
- **A Di Sisto-scale 4.5 Gyr run is feasible** in weeks on a small cluster, and roughly
  ten times cheaper with the levers below.
- The extrapolation comes from 50 kyr migration runs; confirm it with a Trojan-only
  benchmark.

Three levers, all already present in the code:

- **`planet_set='giants'`** — drops the four terrestrials and lets the timestep rise
  from 0.031 to 0.1 yr. Saturn must stay: its secular resonances drive Trojan
  instability (Marzari & Scholl 2002). `docs/MSc_PROJECT_PLAN.md` §4 records that dropping
  the terrestrials "doesn't work" as an untested open item — resolve that first.
- **`add_trojan_disk`** (`migration/simulation/bodies.py:232`) — seeds L4 and L5
  directly instead of the structured disk, where ~85% of particles are never
  co-orbital. Written, exported, and never called by any script.
- **Snapshot volume** — each 50 kyr / 10⁴-particle run is 92 MB in the
  all-particles-all-snapshots format.

For reference: Di Sisto et al. 2019 used 7.3 d (0.02 yr) with Venus–Neptune; Di Sisto
et al. 2014 states 0.2 yr only for a pre-integration; Vokrouhlický et al. 2024 used
swift_rmvs4 at 0.25 yr with the giants only, ~8× the current timestep.

**Reusable as-is:** the REBOUND/WHFast setup and unit convention
(`bodies.py:25–58`), the timestep clamp, `add_trojan_disk`, the escape-detection and
hash-slot/sentinel scheme (`runner.py:127–173`), the NumPy-pickle snapshot format and
`sim_from_state` resume path, `run_batch.py`'s process pool, and the pytest physics
harness.

**Does not exist anywhere in the repository** (verified by grep): the resonant angle
σ = λ − λ_J, libration amplitude, proper elements, per-particle orbital elements, and
per-particle escape times. `compute_trojan_counts` uses the instantaneous heliocentric
*radius* and longitude, so it miscounts librators at the ends of their swing. **The
resonant-angle classifier and the libration amplitude are the first thing to build
for either G1 or G2.**

---

## 6. Literature still needed

Cited by papers in `literature/`, but absent:

| Priority | Work | Why |
|---|---|---|
| 1 | **Holt et al. 2020**, MNRAS 495, 4085 (and the second Holt 2020 paper) | The direct post-2014 successor; Bottke's authority for "no particular difference between the two regions" |
| 2 | **Freistetter 2006**, A&A 453, 353 | The direct counter-claim to Di Sisto |
| 3 | **Hou, Scheeres & Liu 2014**, CeMDA 119, 119 | Temporary asymmetry from identical ICs |
| 4 | **Robutel & Gabern 2006 Part I**, MNRAS 372, 1463 | Diffusion map for the present-day configuration; the folder has only Part II |
| 5 | **Tsiganis et al. 2005**, CeMDA 92, 71 | Lyapunov maps on (D, e) grids — the closest precedent for an amplitude-resolved erosion map |
| 6 | **Marzari, Tricarico & Scholl 2003**, Icarus 162, 453 | The libration-amplitude distribution |
| 7 | **Levison, Shoemaker & Shoemaker 1997**, Nature 385, 42 | The founding Gyr-scale Trojan integration |
| 8 | **Milani 1993**, CeMDA 57, 59 | The definition of D to adopt for catalogue comparability |
| 9 | **de Elía & Brunini 2007**, A&A 475, 375 | Collisional removal ≈ 3× the dynamical rate; L4 only |
| 10 | **dell'Oro et al. 1998**, A&A 339, 272 | The only source of separate L4/L5 collision probabilities |
| 11 | **Roig et al. 2008**, A&A 483, 911 | The counterweight to McNeill on colour |
| 12 | **Schemel & Brown 2021**, PSJ 2, 40 | Directly contradicts McNeill et al. 2021 |

### Data-hygiene fixes

- `literature/01_observations/Uehata+2022_L4_vs_L5_physical_properties_ATLAS.pdf` is
  **not** Uehata et al. 2022. It is **McNeill et al. 2021, PSJ 2, 6**. The real
  Uehata et al. 2022 (AJ 163, 213) is absent.
- `literature/03_coorbital_theory/RobutelGabern2009_Resonant_structure_of_Jupiter_Trojans_II.pdf`
  is **Robutel & Bodossian 2009**, MNRAS 399, 69.
- `READING_LIST.md`'s Tier-1 Di Sisto entry quotes only the 2014 numbers and should
  note the 2019 revision.
- Nesvorný et al. 2013 §3.4 has an L4/L5 typo: "279 L4 Trojans" should read L5.

---

## 7. Recommendation

Take **G1** to Simon as the headline, with **G4's question as its first step** and
**G2** as the extension. Reading both anchor papers in full
(`docs/VOKROUHLICKY_DISISTO_REVIEW.md`) changed the order of work:

1. **G1 step 2, the family cross-match** (days, no integration): how much of Di Sisto's
   sample is family members, per swarm.
2. **G4, is the escape asymmetry real?** One 4.5 Gyr campaign on the complete modern
   sample, family members flagged and marginal objects cloned. Feasible (§5).
3. **G1 step 3, the inclination-resolved escape map**, propagated through
   Vokrouhlický's debiased, family-free distributions. The same integrations serve.
4. **G2, family morphology**, from the flagged family members of step 2 plus synthetic
   families.

All four are non-migration, build on the paper Simon named, address questions the
original authors left open, and reuse the resonant-angle and libration-amplitude
machinery, the one piece of the parked migration project that survives the reframe.
Each has a publishable answer either way. G4 in particular either measures an escape
asymmetry at a stated significance or shows the Gyr contribution is zero.

**In practice these share one integration** of the catalogued Trojans plus clones:
survey bias changes which Trojans are catalogued, not how they move, so it is
handled by weighting objects with Vokrouhlický's debiased density rather than by
simulating extra particles. See `docs/MSc_PROJECT_PLAN.md` §2.
