# Vokrouhlický et al. 2024 and Di Sisto et al. 2014/2019, Side by Side

**Drafted 2026-09-10.** Companion to `docs/ASYMMETRY_REVIEW.md`, whose research gaps (G1–G6)
this document re-ranks in §6. All three papers were read in full:

- **V24.** Vokrouhlický, Nesvorný, Brož, Bottke, Deienno, Fuls & Shelly, *Orbital and
  absolute magnitude distribution of Jupiter Trojans*, arXiv:2401.15537v1 (submitted to
  AJ). `literature/01_observations/Vokrouhlicky+2024_Orbital_and_Absolute_Magnitude_Distribution_of_JTs.pdf`
- **D14.** Di Sisto, Ramos & Beaugé, *Giga-year evolution of Jupiter Trojans and the
  asymmetry problem*, Icarus 243, 287 (2014); arXiv:1604.05331v1.
  `literature/02_mechanisms/DiSisto+2014_Gigayear_evolution_of_JTs_and_the_asymmetry_problem.pdf`
- **D19.** Di Sisto, Ramos & Gallardo, *The dynamical evolution of escaped Jupiter Trojan
  asteroids, link to other minor body populations*, Icarus (2019); arXiv:1811.00352v1.
  `literature/02_mechanisms/DiSisto+2019_Dynamical_evolution_of_escaped_Jupiter_Trojans.pdf`

Section and page numbers refer to the arXiv versions. Numbers marked *(derived)* were
computed here from the papers' own data; everything else is quoted or read directly.

---

## 0. The status quo in eight lines

1. **The number asymmetry is real.** f45 = N4/N5 = 1.43 ± 0.05 for the debiased,
   family-free background at H ≈ 13.8; symmetry excluded at 7.9σ (V24).
2. **No size dependence of that ratio has been demonstrated.** V24's model median is
   nearly flat (≈ 1.3 at H ≈ 9 to 1.43 at H ≈ 13.8). The bright end is unconstrained,
   not symmetric, and the brightest well-counted bin (H < 10: 46 vs 26) leans the other
   way.
3. **The inclination asymmetry is also real.** After family removal and debiasing, L5
   is hotter: KS D = 0.221, medians ≈ 11.5° vs ≈ 17.5° (read off V24 Fig. 26). The
   *low-inclination L4 excess among small bodies* that older work saw is the Eurybates
   and Arkesilaos families.
4. **Eccentricity and libration amplitude show no detected difference** (p = 0.08,
   0.17).
5. **The dynamics are symmetric.** Identical stability zones over 100 Myr (V24);
   identical 4.5 Gyr erosion of identical initial conditions (D14's fictitious grid).
6. **Whether today's real populations erode differently is open.** D14: L5 loses 28.3%
   vs L4 23.2% (3.0σ). D19: 25.1% vs 23.6% (0.9σ, not significant) *(derived)*.
7. **Nobody has integrated the population whose asymmetry needs explaining.** D14/D19
   integrated the 2013 numbered list: undebiased, families included, raw ratio 1.98.
8. **Gyr erosion multiplies the ratio by 1.02–1.07 (± 0.025)** — between none and about
   a fifth of the excess over unity, if today's rates extrapolate backwards (never
   checked). The rest is primordial, early, or due to something neither paper models.

---

## 1. The three papers at a glance

| | V24 | D14 | D19 |
|---|---|---|---|
| Question | What are the complete (debiased) orbital and magnitude distributions of each swarm? | Does present-day chaotic diffusion change N4/N5? | Where do escaping Trojans go? |
| Kind | Observations + survey-bias model; one 100 Myr stability experiment | N-body, 4.5 Gyr | N-body, 4.5 Gyr |
| Sample | MPC Feb 2023: 7,903 L4 / 4,177 L5; CSS G96 detections 2013–2022 | 2,972 numbered (Mar 2013): 1,975 L4 / 997 L5 | Same 2,972 |
| Families | Identified (HCM on new proper elements) and fitted separately | In the sample, never separated | Same |
| Debiased? | Yes: detection probability over 223,865 survey frames | No | No |
| Dynamics | swift_rmvs4, dt = 0.25 yr, Sun + 4 giants (terrestrial masses folded in), 100 Myr | EVORB, Sun + 4 giants (inner masses in Sun); dt stated only for a pre-integration (0.2 yr) | EVORB, Sun + Venus–Neptune, dt = 7.3 d |
| Loss criterion | Leaves the libration zone | Encounter within 1 R_Hill → deleted | First encounter within 2.1 R_Hill = escape; then followed to collision or r > 1000 au |
| Orbital variables | Quasi-proper A (≈ sin I), B (≈ e), C (libration integral) | Osculating at a representative plane; D = libration semi-amplitude | Osculating |
| Headline | f45 = 1.43 ± 0.05; L5 hotter; 9 L4 vs 4 L5 families | Escape L4 23% / L5 28.3%; primordial N4/N5 = 1.3 ± 0.2 | Escape L4 23.6% / L5 25.1%; primordial 1.373 ± 0.204 |

---

## 2. Vokrouhlický et al. 2024: the present-day population

### 2.1 Data

- **Catalogue.** MPCORB of 22 Feb 2023, osculating elements at MJD 60,000. Selection:
  4.95 < a < 5.45 au, e < 0.6, I < 50°, 20° < |σ| < 130° with σ = λ − λJ. This gives
  7,929 L4 and 4,189 L5 candidates. A 1 Myr integration discards 26 and 12 (mostly
  Jupiter-family comets masquerading as Trojans), leaving **7,903 L4 and 4,177 L5**
  (raw ratio 1.89). The April 2023 proper-element catalogue keeps short arcs and has
  8,144 and 4,252.
- **Survey.** Catalina Sky Survey station G96 (Mt Lemmon, 1.5 m), January 2013 – June
  2022, split at the May 2016 camera upgrade.
  - Phase I: 61,585 frames; 4,551 L4 and 2,460 L5 objects detected.
  - Phase II: 162,280 frames, 4× the field of view; 6,307 L4 and 3,041 L5.
  - CSS detected 75–85% of all known Trojans.
- **Completeness.** H ≈ 13.7–13.8. No Trojan brighter than H ≈ 14.2 has been found in
  five years; the largest discovered in the last decade are 2013 BD8 (L4, H = 13.7)
  and 2013 SP62 (L5, H = 13.6).
- **Biases modelled.** Sky coverage (galactic plane, Jupiter's 11.9-yr cycle),
  photometric depth (50% efficiency at m ≈ 20–21), and tracklet failure for slow
  movers (w below 0.04 deg/day in phase I, 0.08 in phase II). Phase function H–G with
  G = 0.09 (0.15 for the error runs).
- **A bias that matters for L4 vs L5.** During phase II Jupiter moved from mean anomaly
  ≈ 160° to ≈ 310°, so L4 sat nearer Earth and the survey reached it ≈ 0.35–0.40 mag
  deeper than L5. Phase I pointed near the ecliptic and missed high-inclination
  orbits, which hurts L5 more because L5 is hotter.

### 2.2 The orbital variables

V24 does not fit osculating elements. It uses three quasi-proper amplitudes, each the
radius of a circle around Jupiter's forced value:

| Variable | Definition | Meaning | Conversion |
|---|---|---|---|
| **A** | ζ = sin I e^{iΩ} = ζJ + A e^{iφ} | ≈ proper sin I relative to Jupiter's orbit | i ≈ arcsin A: 0.1 → 5.7°, 0.2 → 11.5°, 0.3 → 17.5°, 0.5 → 30°, 0.7 → 44° |
| **B** | z = e e^{i(ϖ−ϖJ)} = eJ e^{±iπ/3} + B e^{iψ} | ≈ proper e | — |
| **C** | (da/aJ)² = (8µ/3)[C − f(σ)], f(σ) = (1 + 4\|sin σ/2\|³)/(2\|sin σ/2\|) | Libration integral: size of the tadpole loop. C = 1.5 is the Lagrange point; tadpoles have 1.5 < C < 2.5 | Proper Δa ≈ aJ √(8µ(C − 1.5)/3): C = 1.58 → 0.074 au, C = 2.0 → 0.186 au *(derived)* |

C comes from the zero-e, zero-i averaged problem (Sicardy & Dubois 2003; Morais 1999,
2001). It plays the role of Milani's (1993) libration amplitude D used by Di Sisto;
they measure the same thing but are not numerically identical.

The model assumes the background factorizes, dN(A, B, C) = dN(A) dN(B) dN(C), each of
the form x^α1 exp[−½(x/s)^α2] (for C: exp[−½((C − 1.5)/sC)^γ]), restricted to the
numerically stable zone. Families are superimposed as narrow functions at fixed
locations. Magnitude and orbit are assumed independent, on the argument that within a
swarm "everything hits everything else".

### 2.3 The resonant structure: where Trojans can live

**The experiment.** 1,040 cells in the (C, B) plane (ΔB = 0.01, ΔC = 0.015), 28 orbits
per cell spread over 7 fixed A values, random angles. Sun + 4 giants, swift_rmvs4,
dt = 0.25 yr, 100 Myr forward from MJD 60,000, for both swarms.

**What it shows** (Figs. 12–14):

- **The stable zone is a triangle in (C, B).** Higher eccentricity requires a smaller
  libration amplitude, the Rabe (1965, 1967) relation, which keeps orbits away from
  Jupiter. At B ≈ 0 the zone reaches C ≈ 1.95–2.05; at B ≈ 0.2 it closes near
  C = 1.5.
- **It widens slightly, then shrinks, with inclination.** Reading Fig. 14: the low-B
  edge sits at C ≈ 1.95–2.05 for A ≤ 0.25, ≈ 1.85 for A = 0.35–0.45, and ≈ 1.76 for
  A = 0.65, and the zone narrows at every B.
- **Inclination is capped at ≈ 43°** (A ≈ 0.7). V24 attributes the cap to secular
  resonances (s7, s8, and combinations with the great-inequality frequency; Marzari
  et al. 2003; Robutel & Gabern 2006). D19 adds ν16, which raises inclinations past
  20° on 10⁸ yr (Marzari & Scholl).
- **The bins are too coarse** to resolve the fine secondary and secular structure of
  Robutel & Gabern (2006).
- **The swarms fill the zone "to full capacity".** A few observed Trojans sit in
  unstable bins; these are today's escapers.
- **L4 and L5 limits coincide at every A** (Fig. 14's mirror plot). No statistic is
  given; it is a comparison of 50%-stability curves by eye.
- V24 notes that a Gyr integration would shrink the zone further.

### 2.4 The background population

**Size.** Debiased background to H = 14.5 (Table 3): **N4 = 3,951 ± 44, N5 = 2,664 ± 39**,
a ratio of 1.48 *(derived)*. Rescaled to H = 13.5 for the KS tests: 1,587 and 1,083.

**Magnitude distribution.** Cumulative slopes on six spline segments (Table 3):

| H segment | 7–8.5 | 8.5–10 | 10–12 | 12–13 | 13–14 | 14–15 |
|---|---|---|---|---|---|---|
| L4 | 1.24 ± 0.45 | 0.66 ± 0.09 | 0.43 ± 0.02 | 0.50 ± 0.01 | 0.45 ± 0.01 | 0.39 ± 0.06 |
| L5 | 1.40 ± 0.39 | 0.64 ± 0.09 | 0.43 ± 0.03 | 0.47 ± 0.02 | 0.47 ± 0.01 | 0.34 ± 0.01 |

- A steep leg (≈ 0.91) to H ≈ 8.5–9, then ≈ 0.45–0.5, shallowing to **0.38 ± 0.01 at
  H = 15**. The H < 15.5 extension (L4 only) gives 0.30, which V24 flags as unreliable.
- A shallow local maximum of the slope at H ≈ 12–13. Splines beat a broken power law
  decisively (evidence ratio 5 × 10⁻³ for L4 and 10⁻⁵ for L5 against the broken law).
- **L4 and L5 slopes agree to H ≈ 14**, except between H = 9 and 10, where L4 is
  steeper. V24 calls this "formally at the origin" of the asymmetry in their model.
- **Beyond H ≈ 11 the differential distributions differ.** L4 is shifted ≈ 0.4 mag
  towards smaller H (brighter).

**The asymmetry factor** (Fig. 25; 10,000 MultiNest posterior solutions):

- At H ≈ 13.8, **f45 = 1.43 ± 0.05**, and f45 = 1 is excluded at 7.9σ.
- Below H ≈ 9–9.5 the 1σ band includes 1 — and also 1.43. The model median there is
  ≈ 1.3.
- The complete MPC counts show an "anomalous vacancy" of L5 at H ≈ 9.5–10: **46 L4
  vs 26 L5 objects with H < 10** (Fig. 15). The L5 paucity "continues beyond the 11
  magnitude".

**Orbital distributions of the background** (Fig. 26):

| | KS D, H < 13.5 | p | KS D, H < 12 | p | Median, H < 13.5 (read off Fig. 26) |
|---|---|---|---|---|---|
| A (inclination) | 0.221 | ≤ 10⁻²⁰ | 0.193 | 2.5 × 10⁻⁴ | L4 ≈ 0.20 (≈ 11.5°); L5 ≈ 0.30 (≈ 17.5°) |
| B (eccentricity) | 0.051 | 0.08 | 0.093 | 0.20 | both ≈ 0.055 |
| C (libration) | 0.043 | 0.17 | 0.084 | 0.35 | both ≈ 1.58 (Δa_P ≈ 0.074 au) |

The KS tests compare the best-fit *model* CDFs, using the model population sizes as
N; they are not tests on the detected objects. V24 itself calls the B and C agreement
"not entirely robust". At H < 12 the L4 inclination median rises to ≈ 0.23 (≈ 13°)
while L5 stays at ≈ 0.30.

**Albedo and shape.** WISE finds no significant L4/L5 albedo difference (Grav et al.
2011, 2012), though albedos spread out below ≈ 30 km. V24 offers two explanations for
the ΔH ≈ 0.4 offset: an albedo difference, or small L4 Trojans being more irregular
because they are more collisionally evolved (O'Brien & Morbidelli 2008; McNeill et al.
2021). Lucy-target work suggests WISE albedos may be systematically too high.

### 2.5 The families: complete census

Families were found by hierarchical clustering (HCM, d3 metric) on V24's new synthetic
proper elements, computed with the Milani (1993) method via the Holt et al. (2020a)
pipeline. The catalogue is public: https://sirrah.troja.mff.cuni.cz/~mira/tmp/trojans/.

The table merges V24 Table 4 (all 13 families) with Table 1 (the 8 fitted in the
debiasing model). I_P is derived from sin I_P.

| Swarm | Family | v_cut (m/s) | N_mem | D_LR (km) | D_LF (km) | D_parent (km) | H_LR | sin I_P (I_P) | e_P | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| L4 | (3548) **Eurybates** | 40 | **875** | 68 | 53 | 140 | 9.82 | 0.129 (7.4°) | 0.044 | Largest family; ≈ 11% of the L4 catalogue; sharp |
| L4 | (20961) **Arkesilaos** | 40 | 235 | 23 | 21 | 110 | 11.97 | 0.155 (8.9°) | 0.029 | Steep SFD; may rival Eurybates at H ≈ 16.5–17. Vinogradova (2020) ties it to (2148) Epeios |
| L4 | (624) **Hektor** | 40 | 118 | 230 | 21 | 250 | 7.45 | 0.326 (19.0°) | 0.054 | Dense core of very small fragments; D-type; Hektor has a satellite. Alternatively only a tight cluster at 20 m/s |
| L4 | (2797) **Teucer** | 60 | 86 | 113 | 17 | 130 | 8.87 | 0.358 (21.0°) | 0.073 | Found by Milani (1993), Vinogradova (2015) |
| L4 | (9799) **Thronium** (ex-1996 RJ) | 20 | 69 | 72 | 22 | 120 | 9.69 | 0.526 (31.7°) | 0.049 | High inclination, easy to see against a thin background |
| L4 | (4489) Dracius* | 60 | 24 | 95 | 15 | — | — | ≈ 0.355 (20.8°) | ≈ 0.029 | New |
| L4 | (24534) 2001 CX27* | 60 | 45 | 29 | 15 | — | — | ≈ 0.219 (12.7°) | ≈ 0.100 | New |
| L4 | (38599) 1999 XC210* | 60 | 22 | 20 | 12 | — | — | ≈ 0.243 (14.1°) | ≈ 0.039 | New; compact |
| L4 | (432306) 2009 SQ357* | 40 | 24 | 7.3 | 6.8 | — | — | ≈ 0.231 (13.4°) | ≈ 0.005 | New; small bodies only. (4035) Thestor and (6545) Leitus nearby but offset in Δa_P |
| L5 | (1867) **Deiphobus*** | 60 | 233 | 131 | 24 | 180 | 8.45 | 0.475 (28.4°) | 0.029 | Split from the old "Ennomos family": higher i, smaller libration amplitude |
| L5 | (4709) **Ennomos** | 50 | 88 | 80 | 20 | 110 | 8.68 | 0.459 (27.3°) | 0.032 | Lower i, larger e and libration amplitude; merges with Deiphobus above 60 m/s |
| L5 | (247341) 2001 UV209 | 55 | 46 | 18 | 16 | 90? | 12.62 | 0.415 (24.5°) | 0.041 | Vinogradova (2020) ties it to (37519) Amphios |
| L5 | (261781) 2006 BG132* | 60 | 31 | 14 | 13 | — | — | ≈ 0.316 (18.4°) | ≈ 0.081 | New; near the (1172) Aneas cluster but offset in e_P |

\* New in V24. Their sin I_P and e_P (marked ≈) are the largest remnant's position read off Figs. 30–31; Δa_P isn't plotted.

**What the census says.**

- **Counts: 9 families at L4, 4 at L5.**
- **Share of the population.** Families are 10–15% of each swarm at H ≈ 14.5, and the
  share rises towards fainter magnitudes because family size distributions are steeper
  than the background's. Cumulative slopes: Eurybates ≈ 0.74, Arkesilaos ≈ 0.94,
  Ennomos/Deiphobus ≈ 0.74, against ≈ 0.46 for the background.
  - L4: Eurybates is ≈ 11% of the catalogue; with Hektor and Arkesilaos, ≈ 15%.
    Eurybates + Arkesilaos are ≈ 12% at H = 14.5 and could reach 25–30% at
    H ≈ 16.5–17 if the power law holds.
  - L5: Ennomos/Deiphobus is ≈ 10% at H ≈ 14.25.
- **Fitted populations** (H < 14.5): Eurybates 385 ± 20, Deiphobus 210 ± 10. Smaller
  families are poorly constrained.
- **Where they sit.** The two most populous L4 families sit at **7–9°**; the other
  seven L4 families at 13–32°. The three sizeable L5 families sit at **24–28°**, and
  the small new one (2006 BG132, 31 members) at ≈ 18°. So in raw data the families
  *exaggerate* L5's hotter inclinations, and Eurybates and Arkesilaos are exactly the
  small-body, low-i L4 excess Di Sisto describes (§4.2b).
- **Morphology.** L4 families are "sharp and highly concentrated clusters, with possibly
  very low populations of interlopers"; L5 families are "diffuse and broad clans, with
  possibly a substantial population of interlopers". V24: "The origin of this
  difference is unknown." Three candidate causes are named and none tested: higher
  ejection velocities, dynamical diffusion at their location, "or some other effect".
- **Consequence for the fit.** Diffuse, overlapping L5 families (the Ennomos and
  Deiphobus masks overlap) are harder to separate from the background. This is one
  reason the L5 solution is less reliable than L4's.

### 2.6 V24's own caveats

- The L5 solution is unreliable beyond H ≈ 14–14.5; its break beyond H ≈ 14 "should
  not be trusted".
- **Conclusion 7 downgrades the inclination asymmetry** to "an interesting hypothesis
  that needs to be verified", because of the L5 modelling problems.
- Rotation-phase bias at the faint end (small, elongated Trojans detected near
  lightcurve maximum) is ignored.
- Within each distribution the parameters are strongly correlated (sA with α2, and so
  on). Don't compare individual Table 3 values between swarms: L5's sA = 0.335 against
  L4's 0.053 looks enormous but trades off against α2 = 3.0 vs 1.0.
- L5 observing conditions improved from the 2023 season; V24 proposes revisiting with
  CSS, Rubin and NEO Surveyor.

### 2.7 Inconsistencies inside V24

- **f45 bookkeeping.**
  - Abstract: 1.45 ± 0.05 "for H < 15".
  - Introduction (naive MPC count minus families: N4 ≈ 2,040, N5 ≈ 1,410): 1.45 ± 0.05
    at H ≤ 13.8.
  - Fig. 25 caption: 1.43 ± 0.05 at H ≈ 13.8.
  - Conclusion 6: 1.43 ± 0.05 "at magnitude H = 15".

  Cite the Fig. 25 value with its H ≈ 13.8 limit.
- **The albedo example has the wrong sign.** L4's distribution is shifted to *smaller*
  H, i.e. brighter. With equal size distributions that requires a *higher* L4 albedo:
  p_L4/p_L5 = 10^(0.4 × 0.4) ≈ 1.45, roughly 0.07 (L4) vs 0.05 (L5). V24 (§5.2) prints
  0.05 (L4) and 0.07 (L5). Its formula "1 − 10^(−0.4ΔH) ≃ 0.44" also evaluates to
  0.31; 0.44 is 10^(0.4ΔH) − 1. The shape explanation in the next paragraph (L4 more
  irregular, "slightly brighter") has the correct sign.
- **"Symmetric" at the bright end** only means the 1σ band includes 1 there. The
  caption itself adds that "statistical robustness increases with increasing H".

---

## 3. Di Sisto et al. 2014 and 2019: integrating today's Trojans for 4.5 Gyr

**In one paragraph.** Take every numbered Trojan of 2013 and integrate it for 4.5 Gyr
under the Sun and the giant planets (with or without the terrestrials). About a quarter
of each swarm leaves, almost always through a close encounter with Jupiter. The escape
rate falls over the first ~2 Gyr, then stays roughly constant without levelling off. L5
loses a slightly larger fraction: 5 points more in 2014, 1.5 points in 2019. The
escapers are those at the edge of the stable zone, with large libration amplitude or
eccentricity, and the inclination distributions keep their shape. After escape, a
Trojan lives a median ~250 kyr on comet- and Centaur-like orbits before being ejected
(≈ 95%) or colliding, and escapees are a negligible source for other populations. A
fictitious population with mirror-image L4 and L5 initial conditions erodes
identically.

### 3.1 What was integrated

| | D14 | D19 |
|---|---|---|
| Real Trojans | All 2,972 numbered as of March 2013: 1,975 L4 + 997 L5 (raw ratio 1.98). "Regardless of the absolute magnitude"; not debiased; families included | Same |
| Fictitious | 18,200 per swarm: a 5.01–5.40 au (Δ0.01), e 0–0.34 (Δ0.01), i 0–60° (Δ5°); every angle pinned: ϖ = ϖJ ± 60°, Ω = ΩJ, M = MJ | None |
| Planets | Sun + Jupiter–Neptune; inner planets' masses added to the Sun | Sun + Venus–Neptune |
| Integrator | EVORB (Fernández et al. 2002) | EVORB |
| Timestep | 0.2 yr stated for the pre-integration to the representative plane; not stated for the 4.5 Gyr runs | 7.3 d (≈ 1/30 of Venus's period) |
| Escape | Encounter within 1 R_Hill of a planet → particle removed | First encounter within 2.1 R_Hill = escape time; then followed to collision or r > 1000 au |
| Elements | Osculating, recorded where M − MJ = 0, ϖ − ϖJ = ±60°, Ω − ΩJ = 0; i relative to the outer Solar System's Laplace plane | Osculating |
| Compute | Not stated | 156 jobs of ~20 Trojans (108 L4, 48 L5), "several computers … for several months" |
| Repeats | Each real Trojan once | Each once; error bars from repeated 80% subsamples |

### 3.2 What the 4.5 Gyr integrations show

**Real Trojans, D14.**

- **Survivors: 1,517 of 1,975 L4 and 715 of 997 L5**, so escape is 23.2% (L4) and
  28.3% (L5).
- Escape goes through Jupiter encounters in 96% (L4) and 99% (L5) of cases; the rest
  are Saturn.
- The escape rate falls with time, is nearly constant after 2 Gyr, and shows "no
  evidence of a leveling off". Late-time (t > 2 Gyr) survival slopes: −3.99 × 10⁻¹¹
  yr⁻¹ (L4), −5.12 × 10⁻¹¹ yr⁻¹ (L5).
- N4/N5 in the numbered sample rises from 1.97 today to ≈ 2.13 after 4.5 Gyr (Fig. 5).
- **Who escapes:** "preferably those with larger libration amplitude or with larger
  orbital eccentricity", with "little correlation" in the (i, a) plane. Both are
  readings of scatter plots (Figs. 1, 3).
- The stable Trojans, and the stable ones with H ≤ 12, show "no distinction between
  the two swarms" in (D, e). Few Trojans have small D and small |e − eJ|: most
  oscillate substantially in both degrees of freedom.
- **Permanence maps** (Fig. 6): identical in (a, e). In (a, i) they differ only through
  individual objects. (83983) 2002 GE39 stays at i ≈ 54° in L4 for the full 4.5 Gyr;
  (19844) and (12999), at 40–50° in L5, escape near the end.
- **Inclination distributions keep their shape** (Fig. 7): "the inclination distribution
  of Trojans must be primordial or at least a result of an initial collisional
  evolution".

**Fictitious Trojans, D14.**

- Half are lost within a few thousand years. By 4.5 Gyr, 81.4% of L4 is gone, and L5
  is "almost identical". Losses: Jupiter 81%, Saturn 19%.
- Late survival slopes: −1.107 × 10⁻¹¹ (L4) vs −1.077 × 10⁻¹¹ (L5), "virtually
  identical". Permanence maps show no difference "even on Gyr timescales".
- With every angle pinned to the Lagrange geometry, the grid samples one angular
  configuration: libration amplitude is not an independent dimension
  (`docs/ASYMMETRY_REVIEW.md` §3.1).

**Real Trojans, D19.**

- **Escape: 466 of 1,975 (23.6%) L4; 250 of 997 (25.1%) L5.** First encounters are
  almost all with Jupiter; one L4 Trojan meets Mars first, and 5 L4 and 3 L5 meet
  Saturn first.
- **Before escape:** "a slow diffusion among resonances". Secular and secondary
  resonances slowly raise e and change i until planetary encounters start (as in
  Robutel & Gabern 2006). The jump in a comes after the encounter.
- Linear escape-rate fits: aL4 = 7.04 × 10⁻¹¹, aL5 = 7.56 × 10⁻¹¹ yr⁻¹. These do not
  match the escape fractions; see §3.4.

**After escape, D19.**

- **Fates.** L5: 2 hit the Sun, 1 hits Jupiter, 247 are ejected. L4: 2 Sun, 2 Saturn,
  12 Jupiter, 450 ejected. L4 escapees collide proportionally **3× more often**. Öpik
  theory traces this to their lower mean inclination (10.7° vs 14.5°): lower encounter
  speed, hence a larger impact parameter.
- **Encounters.** Every escapee meets Jupiter; 95–97% meet Saturn. L5 escapees reach
  the inner planets more often (Venus 13% vs 6%, Earth 19% vs 10%); L4 escapees spend
  more of their encounters with Uranus and Neptune (6–8% vs 2–3%).
- **Where they go.** L5 escapees favour the inner Solar System: near the 1:1, the Hildas
  and the outer main belt. L4 escapees spread across the whole outer region, including
  a Kozai island at a = 40–100 au, i = 30–60°, inside Neptune's 1:5 and 1:6 resonances.
- **Temporary resonance captures** (number of 10³-yr orbital states, L4 / L5):

  | Resonance | a (au) | L4 | L5 |
  |---|---|---|---|
  | 5:2 | 2.82 | 1,543 | 144 |
  | 9:4 | 3.03 | 464 | 796 |
  | 7:4 | 3.58 | 358 | 787 |
  | 3:2 (Hildas) | 3.97 | 510 | 983 |
  | 4:3 | 4.29 | 936 | 340 |
  | **1:1 Jupiter** | 5.20 | **6,467** | **2,535** |
  | 2:3 | 6.82 | 516 | 413 |
  | **1:1 Saturn** | 9.55 | **1,594** | **852** |

- **Lifetimes after escape.** Mean 3.5 Myr (L4) vs 0.7 Myr (L5), but the L4 mean is
  dominated by five escapees that live > 100 Myr in the Centaur and TNO regions.
  Median ≈ 250–264 kyr for both.
- **Regions reached.** JFC zone 97% (both); Centaur zone 90% / 84%; TNO zone (a > 30 au)
  78% / 76%.
- **Scaled to the real swarms.** D19 adopts the Wong & Brown (2015) magnitude
  distribution and assumes N4 = 1.4 N5 *independent of size*, giving ~265,000 L4 and
  ~190,000 L5 Trojans with D > 1 km. That yields **18 L4 and 14 L5 Trojans with D > 1
  km leaving the swarms per Myr**.
  - Contribution to JFCs, Centaurs and TNOs: negligible (≈ 1 of ≈ 450 JFCs with
    R > 1 km).
  - Dynamical-only estimates: 2 D > 1 km Trojans in NEO orbits, 1 in an Encke-type
    orbit, and one crossing Jupiter's Roche limit every 1.4 Myr.
  - The conclusions multiply these by ≈ 4 to add collisional ejection, which de Elía &
    Brunini (2007) put at ≈ 3× the dynamical rate for small L4 bodies.

### 3.3 The primordial-ratio argument

Both papers assume that today's survival fractions also describe the past 4.5 Gyr run
backwards, and divide the observed ratio by the ratio of survival fractions:

  N0(L4)/N0(L5) = [Ns(L4)/Ns(L5)] / [S(L4)/S(L5)]

- **D14:** S4/S5 = 0.768/0.717 = 1.07. With Grav et al. (2011)'s 1.4 ± 0.2 this gives
  **1.3 ± 0.2**; erosion "contributed to ∼ 25% of the present unbiased asymmetry".
- **D19:** S4/S5 = 0.764/0.749 = 1.02, giving **1.373 ± 0.204**, "∼ 7%".
- D14 itself warns that the early rate "was much larger during the early stages of the
  Solar System, and we are currently just detecting the tail", and that the backward
  assumption "is not obvious, and must be considered with care".
- **The quoted ± 0.2 comes entirely from Grav et al. 2011.** The escape statistics
  contribute no error term.

With binomial errors on the escape counts *(derived)*:

| | Escape L4 | Escape L5 | L5 − L4 | Amplifier S4/S5 | Primordial ratio using V24's 1.43 ± 0.05 |
|---|---|---|---|---|---|
| D14 | 23.2 ± 1.0% | 28.3 ± 1.4% | z = 3.0 (p = 0.002) | 1.071 ± 0.025 | 1.34 ± 0.06 |
| D19 | 23.6 ± 1.0% | 25.1 ± 1.4% | z = 0.9 (p = 0.37) | 1.020 ± 0.023 | 1.40 ± 0.06 |

### 3.4 Inconsistencies inside Di Sisto

1. **D14's abstract and conclusion disagree.** The abstract says the asymmetry "must be
   primarily primordial". The conclusion says the original ratio "must have been much
   closer to unity, indicating that perhaps both Lagrange points contained originally
   similar number of asteroids". Its own Eq. 3 gives 1.3 ± 0.2, not much closer to
   unity than 1.4.
2. **D19's error bars are not counting errors.** D19 quotes 23.6 ± 0.2% and
   25.1 ± 0.2% from repeated 80% subsamples of the same trajectories. The binomial
   errors on 466/1975 and 250/997 are ± 1.0% and ± 1.4%; even the subsample spread
   should be ≈ 0.5–0.7%. With counting errors, D19's L4/L5 difference is 0.9σ. D14
   called its own difference "statistically significant" without a test; that claim
   holds up at 3.0σ.
3. **D19's escape rates don't match its escape fractions.** aL4 and aL5 integrate to
   31.7% and 34.0% over 4.5 Gyr, not 23.6% and 25.1%, and the paper never says what
   the fit is normalized to or over what span. D14's late-time slopes are smaller than
   its average rates, as expected if early escape is faster; D19's are larger.
4. **D19's median lifetimes are swapped.** §4.2 gives 264 kyr for L5 and 249 kyr for
   L4; the abstract and conclusions give 264 kyr for L4 and 249 kyr for L5.
5. **D19 says only L4 escapees reach the trans-Neptunian zone** (abstract, conclusions),
   but its Table 4 has 76% of L5 escapees reaching a > 30 au. The claim probably means
   the long-lived Kozai island, which only L4 escapees visit.
6. **D19's subsample errors are erratic.** The subsample error on aL4 (± 1 × 10⁻¹²) is
   12× its fit error (± 8 × 10⁻¹⁴); the one on aL5 (± 1 × 10⁻¹³) is smaller than its
   fit error (± 1.3 × 10⁻¹³).
7. **D14's timestep** is stated only for the pre-integration.
8. **D14 does not identify families.** §2.3 reviews Beaugé & Roig (2001) and Brož &
   Rozehnal (2011) ("the only robust family appears to be the inner core of the
   Menelaus family … whose largest member is Eurybates"; in L5 "no significant
   agglomeration") and computes proper elements. Its introduction describes L5 as
   having "small (albeit compact) agglomerations", and misprints Eurybates as
   "Eurypides". Family members are never separated in the escape statistics.

---

## 4. Side by side

### 4.1 Where they agree

- **The dynamics are symmetric.** V24's stability limits coincide over 100 Myr at 7
  inclinations; D14's fictitious grids erode identically over 4.5 Gyr. D14 also cites
  Érdi 1996, Marzari et al. 2002 and Robutel & Gabern 2006.
- **The same boundary physics.** D14's escapers have large D and large e; V24's stable
  zone is a triangle in which large B forces small C.
- **Mean a and e are equal between the swarms.** D14: 5.2062 vs 5.2068 au, 0.072 vs
  0.074. V24: B and C distributions indistinguishable.
- **L5 is hotter in inclination**: D14's raw means, V24's debiased background.
- **The instability is gravitational**, running through secular and secondary
  resonances to Jupiter encounters (D14 citing Robutel et al. 2005; D19's pre-escape
  diffusion; V24's s7, s8 and great-inequality terms).

### 4.2 Where they conflict, or talk past each other

**(a) They describe different populations.** D14/D19 integrate the raw 2013 numbered
list: undebiased, families included, raw ratio 1.98. V24 describes the debiased,
family-free background, ratio 1.43. Di Sisto applies a survival ratio measured on the
first population to an observed ratio (Grav's 1.4) that describes something like the
second. **Nobody has integrated the population V24 describes.**

**(b) The inclination difference has two parts, and each paper sees a different one.**

- D14 finds ⟨i⟩ = 10.4° (L4) vs 14.2° (L5), raw, families included. Split at
  d = 30 km: small 9.6° vs 13.2°, large 15° vs 18°. The L4 excess at i ≲ 10° "is
  maintained for the smaller Trojans but it is not noticed for the greater ones",
  while L5's high-i bias holds at both sizes.
- V24 says the earlier inclination anomalies "are readily explained by the population
  of the principal Trojan families". The two biggest L4 families sit at 7–9° with
  steep size distributions — that is D14's small-body, low-i L4 excess.
- But V24's family-free, debiased background is *still* hotter at L5, with a bigger gap
  than D14's raw means: medians ≈ 11.5° vs ≈ 17.5°. Its KS distance barely changes at
  H < 12.

So: **the low-i L4 excess among small bodies is families; L5 being hotter at all sizes
is real.** D14's own size split already contains this decomposition, and the second
part is what needs explaining.

**(c) Does inclination drive escape?** D14 sees "little correlation" by eye in the
(i, a) plane. V24's Fig. 14 shows the stable zone shrinking with A (the low-B edge
falls from C ≈ 2.0 at A ≤ 0.25 to ≈ 1.76 at A = 0.65), and Tsiganis et al. (2005)
report the "effective" stability region shrinking with inclination. These are
compatible. If inclination acts by moving the boundary in (D, e), escape looks like a
D-and-e effect and the signal in any (i, a) scatter plot is diluted. A 1D projection
cannot tell "no inclination effect" from "an inclination effect acting through the
boundary". The joint escape probability P(A, B, C) can.

**(d) The family census changed completely.** D14 worked with one robust family (the
Eurybates core) and nothing in L5. V24 has 9 + 4 families, the L5 ones diffuse. D14/D19
count every family member as an ordinary Trojan. That matters because families can bias the escape fraction either way. Fragments are kicked mostly in libration
amplitude (V24: families are "most extended in the daP coordinate"), so a family has a
stable core and a halo reaching towards the unstable edge. An old family's halo has
already leaked away, leaving mostly stable members that *dilute* the escape fraction. A
young family (V24 leaves open whether Hektor's dense core is a recent cratering event)
has been filtered for less time than the background and can *leak faster*. Ages are
unknown, so the sign is too. Sharp L4 families diluting and diffuse L5 families leaking
would both push L5's escape above L4's, so the morphology difference (G2) and the escape
difference could be one phenomenon.
For scale: if ≈ 6% of D19's L4 sample were effectively stable family members, and the
L4 and L5 backgrounds eroded identically at 25.1%, L4 would show exactly 23.6% — the
whole difference *(derived)*. Eurybates alone is ≈ 11% of today's L4 catalogue. Its
share of the brighter 2013 numbered sample was smaller, by an amount nobody has
measured.

**(e) The two symmetry tests have different blind spots.** V24: 100 Myr, coarse bins,
the whole stable zone weighted cell by cell. D14's fictitious grid: 4.5 Gyr, but one
angular configuration. Neither samples the real population's distribution under Gyr
erosion. D14/D19's real-Trojan runs do, but without debiasing or family removal.

**(f) Significance.** D14's escape difference is 3.0σ; D19's is 0.9σ. Under binomial
errors the two amplifiers (1.071 ± 0.025 vs 1.020 ± 0.023) differ by 1.5σ. **The
2014 → 2019 "revision" is consistent with counting noise.** But for two runs of the
*same* objects the binomial error is an upper bound: a change of setup re-randomizes
only the marginal escapers, while deep-stable and fast-unstable objects behave the same
in both. Part of the shift may be systematic (planet set, timestep). Clones would
measure the split.

The binomial error is the right one for Di Sisto's own inference: whether the two
*distributions* differ in how easily objects escape (link 3). For the narrower question
of how *these particular* objects erode, only chaotic noise counts.

The escape threshold is ruled out as a cause: D19's 2.1 R_Hill criterion is strictly
more inclusive than D14's 1 R_Hill (any trajectory reaching 1 R_Hill passed 2.1 R_Hill
first), so it cannot explain L5's drop from 282 to 250 escapers.

**(g) The observed ratio.** Both Di Sisto papers divide Grav et al. (2011)'s 1.4 ± 0.2.
V24's 1.43 ± 0.05 is four times tighter, so the error on the primordial ratio is now
dominated by the escape statistics (± 0.025 on the amplifier), not the observations.

**(h) Size.** When scaling to km-sized bodies, D19 assumes the asymmetry is
size-independent (§2.1); V24's Fig. 25 is consistent with that. The only size-dependent
L4/L5 feature in either paper is D14's small-body low-i excess, which is the families.

---

## 5. What is established and what isn't

| Claim | Status | Evidence |
|---|---|---|
| f45 ≈ 1.43 for the family-free background at H ≈ 13.8 | **Established** | V24, 7.9σ |
| f45 depends on size | **Not established** | V24 Fig. 25 median flat; bright end unconstrained; H < 10 gives 46 vs 26 |
| The L5 background is hotter in inclination | **Strong, hedged by the authors** | KS D = 0.221 (0.193 at H < 12); V24 Conclusion 7 |
| The low-i L4 excess among small bodies | **Explained: families** | Eurybates and Arkesilaos at 7–9° |
| e and libration amplitude equal between swarms | **No difference detected** | p = 0.08, 0.17 (null results) |
| L4 and L5 dynamics identical | **Strong** | V24 100 Myr; D14 fictitious 4.5 Gyr; earlier literature |
| Today's populations erode differently | **Open** | D14 3.0σ; D19 0.9σ; families never separated |
| Gyr erosion supplies ≤ ~20% of the excess over unity | **Likely** | Amplifier 1.02–1.07 ± 0.025; back-extrapolation untested |
| L4 families sharp, L5 families diffuse | **Observed, unexplained** | V24 Conclusion 2 |
| ΔH ≈ 0.4 mag offset beyond H ≈ 11 | **Observed, unexplained** | V24 Fig. 25, left |
| L5 vacancy at H ≈ 9.5–10 | **Observed, possibly small numbers** | V24 Fig. 15 |

---

## 6. Research gaps worth our time

Ranked by payoff × tractability within twelve months, using the G labels of
`docs/ASYMMETRY_REVIEW.md` §4. **Feasibility, corrected** (`docs/ASYMMETRY_REVIEW.md` §5): a
4.5 Gyr integration of 10⁴ particles costs ≈ 6 CPU-years in the current all-planets
setup, roughly ten times less with giants only and a longer step. Di Sisto's own
2,972-particle runs took "several computers … for several months". Gyr integrations of
today's catalogue are feasible on a small cluster.

### 1st — G4: is present-day erosion asymmetric at all?

D19's difference is 0.9σ, D14's is 3.0σ, and the two are 1.5σ apart. Establish whether
the effect exists before explaining it.

- **Done here:** the counting statistics above.
- **Next:** integrate the complete modern sample (H ≲ 13.8) and clone its marginal
  objects: the ~25% that escape, and their neighbours in (A, B, C).
  - **Clones matter more than extra objects.** The complete background is only ≈ 2,040
    L4 + ≈ 1,410 L5, barely more than D14/D19's 2,972. The ≈ 12,400-object catalogue
    is larger only because it reaches fainter, incomplete magnitudes, which brings the
    survey bias back.
  - **Clones give each object's escape probability.** That removes the chaotic noise
    and splits the 2014 → 2019 shift into noise and systematics.
  - **The distribution question goes to the grid.** Whether the two *distributions*
    differ in how easily objects escape is limited by the finite number of real
    objects; the synthetic grid of G1 step 3 has no such limit.
- **Output:** the first escape asymmetry with an honest error bar. Both answers are
  results: an asymmetry at a stated significance, or none, in which case Gyr erosion
  contributes nothing.

### 2nd — G1, step 2: are the escape statistics family-contaminated?

No paper tests this.

- **An afternoon, no integration:** cross-match the 2,972 Trojans numbered by March
  2013, and any sample you integrate, against V24's public family catalogue. Count
  family members per swarm. If the family shares differ between the swarms by several
  percent, the published escape difference is suspect, whichever way the families bias
  it (§4.2d).
- **In the G4 integration:** report escape fractions for background and families
  separately.
- **Free by-product:** each family's spreading over Gyr, which is G2's data.

### 3rd — G1, step 3: the inclination-resolved escape map

Measure P_esc(A, B, C) over Gyr on a grid that samples the angles and the libration
amplitude, unlike D14's pinned grid. Propagate V24's debiased, family-free
distributions through it, both whole and H-binned, and predict the erosion amplifier.
Compare with G4.

- Settles (c): whether inclination drives escape through the stability boundary.
- Settles whether a hotter L5 background alone predicts more L5 escape.
- The particles are massless, so size enters only through the distributions. The
  H-binned propagation gives the size dependence of the erosion at no extra cost.

### 4th — G2: family morphology

Seed synthetic families at the real L4 and L5 locations (§2.5) and measure spreading
over Gyr. V24's 100 Myr run is too short to see diffusion; G1 step 2 gives the first
empirical look. Highest scientific upside.

### Lower priority

- **G6, back-extrapolation.** Still unresolved; the same integrations give the time
  dependence of the escape rate, which bounds the error of running it backwards.
- **G3, the symmetry disagreement** (Freistetter 2006; Hou et al. 2014). Worth a check;
  both papers here support symmetry.
- **G5, now observational only.** The ΔH ≈ 0.4 offset: an albedo explanation needs L4
  brighter (≈ 0.07 vs 0.05), the reverse of V24's printed example. The L5 vacancy at
  H ≈ 9.5–10 is an afternoon's statistics check.

### Dropped

- **"Explain the size dependence of the number ratio."** There is none in the data
  (`docs/ASYMMETRY_REVIEW.md` §1.2).

### Suggested order

The family cross-match (days) → the G4 campaign, with families flagged → the G1 escape
map, reusing the same integrations → G2 from the flagged families plus synthetic
families. The resonant-angle and libration-amplitude diagnostic
(`docs/ASYMMETRY_REVIEW.md` §5) is needed from the second step on.

**In practice these share one integration** of the catalogued Trojans plus clones:
survey bias changes which Trojans are catalogued, not how they move, so it is
handled by weighting objects with Vokrouhlický's debiased density rather than by
simulating extra particles. See `docs/MSc_PROJECT_PLAN.md` §2.
