# What does "Jupiter Trojan" mean, operationally, in each local paper?

Source: every PDF in `literature/` except `99_course_background`. Text came from `pdftotext` (layout and raw) and was kept in a temporary folder only. Page numbers below are **PDF page numbers** (arXiv/preprint pagination), not journal pagination.

Note on terms: no local paper uses "Greek camp / Trojan camp" operationally. Only Pirani+2019a mentions it, as naming: "the population that librates around L4 ... called the 'Greek camp'; ... L5 ... the 'Trojan camp'" (p2). Everywhere else, L4 = leading = σ ≈ +60°, and L5 = trailing = σ ≈ −60°.

## 0. Library hygiene findings (read first)

1. **`01_observations/Uehata+2022_L4_vs_L5_physical_properties_ATLAS.pdf` is NOT Uehata+2022.** It is **McNeill, Erasmus, Trilling, Emery et al. (2021), "Comparison of the physical properties of the L4 and L5 Trojan asteroids from ATLAS data"** (arXiv:2101.04602, PSJ 2021). The real Uehata, Terai, Ohtsuki & Yoshida 2022 (AJ 163, 213) is a Subaru/HSC paper that gives N_L4/N_L5 ≃ 1.40 ± 0.15 for D > 2 km (cited by Yoshida+2026, p1). It is **not in the library**.
2. **`03_coorbital_theory/RobutelGabern2009_Resonant_structure_of_Jupiter_Trojans_II.pdf` is Robutel & Bodossian (2009)**, MNRAS ("The resonant structure of Jupiter's trojan asteroids-II", arXiv:0809.3526). Paper I is Robutel & Gabern 2006. It is not local.
3. The local Holt+2020 is **Holt+2020b** (MNRAS 495, 4085). Vokrouhlický+2024 takes its proper-element method from **Holt+2020a** (MNRAS 499, 3630, Appendix A), which is not local.
4. Lykawka+2009, Chen+2016 and Parker 2015 are about **Neptune** Trojans. Their definitions do not apply directly to Jupiter Trojans (JTs), but they matter as definitional practice.

---

## 1. Observational / catalogue papers

### Vokrouhlický, Nesvorný, Brož, Bottke, Deienno, Fuls & Shelly 2024, AJ 167, 138 (arXiv:2401.15537) [V24]
- **Context:** an observational catalogue selection, then a 1-Myr dynamical cleaning, then CSS (G96) debiasing in quasi-proper elements. It also builds a new synthetic proper-element catalogue (Appendix A).
- **Source catalogue:** "we downloaded the orbital catalog MPCORB.DAT from the Minor Planet Center website (the February 22, 2023 version)" (p3). The osculating elements are at the common epoch MJD 60,000. About 2% of roughly 1,245,000 orbits were rejected as poorly determined single-opposition orbits given at other epochs.
- **Selection box (osculating):** "(i) semimajor axis a in between 4.95 au and 5.45 au, (ii) eccentricity smaller than 0.6, (iii) inclination smaller than 50°, and (iv) absolute value of the resonant angle σ larger than 20° and smaller than 130°" (p3). σ = λ − λ_J (defined on p9, Eq. 4).
  - Result: 7,929 orbits about L4 and 4,189 about L5.
  - **L4/L5 rule:** the sign of σ is implicit (|σ| window on each side). It is never stated in words.
- **Libration test (1 Myr):** "numerically propagate all selected orbits, together with all planets and the Sun, for a 1 Myr timespan. ... 26 and 12 candidate Trojans near L4 and L5, respectively, escaped from the libration region about their Lagrange center, were discarded from the simulation by striking Jupiter, or were ejected" (p3).
  - Survivors: "7,903 bodies near L4 and 4,177 bodies near L5 ... with their σ librating for the entire timespan of the integration" (p3).
  - This uses **nominal orbits only (no clones)**, and "additional losses might occur if the integration was extended" (p3).
  - Observed all-H ratio: 7903/4177 = **1.892**.
- **Quasi-proper elements used for debiasing (not true proper elements):**
  - A, the forced-subtracted sin I amplitude: ζ = sin I e^{iΩ} = ζ_J + A e^{iφ} (Eq. 1, p9).
  - B, the forced-subtracted eccentricity: z = e e^{i(ϖ−ϖ_J)} = e_J e^{±iπ/3} + B e^{iψ} (Eq. 2, p9).
  - C, the integral of the Sicardy–Dubois/Morais zero-order libration: (da/a_J)² = (8µ/3)[C − f(σ)], f(σ) = (1+4|sin σ/2|³)/(2|sin σ/2|) (Eqs. 3–4, p9).
  - "for tadpole orbits C_min = 1.5 and C_max = 2.5" (p9).
  - The parameters "are basically osculating at an epoch close to the decade of CSS observations" (p9).
  - "C crudely represents the proper semimajor axis da_P, such that da_P ≃ a_J √(8µΔC/3)" (p9).
- **Stable domain "D" (this is the "D" in V24; it is not a libration amplitude):**
  - **Grid:** the (C, B) plane was cut into ΔB = 0.01 × ΔC = 0.015 cells (1040 cells), with 28 orbits per cell (7 fixed A values, random angles). These were integrated for **100 Myr** with swift_rmvs4 at dt = 0.25 yr, with terrestrial-planet masses added to the Sun.
  - **Escape:** orbits leaving their libration zone were "eliminated" (pp13–14).
  - **Stability limit:** "the red dashed line connecting the first bins with at 50% partial instability for the given B value" (Fig. 12 caption, p12).
  - **Debiasing domain:** the debiasing integrates "over a stable domain D of the orbital parameter space (dD = dA dB dC)" (Eq. 31, p22).
  - **L4/L5 symmetry:** "There are no statistically meaningful differences, implying that the orbital stability of the L4 and L5 regions is the same" (p14).
  - **Caveat:** "the integrated timespan of 100 Myr is rather short, and extending it to Gyr would lead to additional collapse of the stability zone" (p14).
- **Proper-element catalogue (Appendix A, p38):**
  - **Method:** "closely follows the synthetic theory of Milani (1993) (see also technical steps outlined in the Appendix A of Holt et al. (2020a))".
  - **Input:** the "MPC catalog of minor planet orbits as of April 15, 2023 containing about 1,275,000 entries, and selected Jupiter Trojans using the same steps as described in Sec. 2". This gave 8,144 L4 and 4,252 L5 orbits. Short-arc orbits were kept this time, except arcs of "less than a few days".
  - **Not used for debiasing:** "In spite of having this new set of proper elements available to us, we opted to continue using the simplified" A, B, C (p14).
  - **Hosting:** the catalogue is at sirrah.troja.mff.cuni.cz/~mira/tmp/trojans/.
- **Families removed:** 9 at L4 and 4 at L5, identified by clustering in proper-element space. "f45" is always quoted for the **background (non-family) population**.
- **Quoted N4/N5 (be careful, there are three different numbers):**
  - Intro preamble, from **raw MPC counts, families removed, H ≤ Hc = 13.8**: "N4 ≃ 2040 and N5 ≃ 1410 ... f45 = 1.45 ± 0.05" (p2), with naive √N errors.
  - Abstract: "ratio 1.45 ± 0.05 for H < 15" (p1). This is the **bias-corrected background model**.
  - Fig. 25 caption: "At the approximate observational completeness limit, H ≃ 13.8, f45 = 1.43 ± 0.05 (σ solution), with f45 = 1 excluded at ≃ 7.9σ level" (p31).
  - Conclusions: "At magnitude H = 15 we find f45 = N4(<H)/N5(<H) = 1.43 ± 0.05" (pp34–35).
  - So the abstract (1.45, H < 15) and the conclusions (1.43, H = 15) disagree slightly. The debiased model normalisation is N_back = 3951 ± 44 (L4) and 2664 ± 39 (L5) at H ≤ 14.5 (Table 3, p25), which gives a ratio of **1.483**. V25 re-quotes this as 3951 ± 44 / 2664 ± 36.
  - At H = 13.5 (rescaled): N4 = 1587, N5 = 1083 (p32), a ratio of 1.465.
  - For the largest bodies, "below magnitude ≃ 9 − 9.5, exhibit symmetry" (p31).
- **Horseshoe / quasi-satellite:** excluded by construction, through the |σ| ∈ (20°, 130°) window and the 1-Myr libration requirement. C ∈ (1.5, 2.5) is the tadpole range.
- **Definitions cited:** Milani 1993; Beaugé & Roig 2001; Holt+2020a App. A; Brož & Rozehnal 2011; Rozehnal+2016; Sicardy & Dubois 2003; Yoder+1983; Morais 1999, 2001; Bien & Schubart 1987 (B as proper e); Levison+1997; Robutel & Gabern 2006; Vinogradova 2015, 2019.

### Szabó, Ivezić, Jurić & Lupton 2007, MNRAS 377, 1393 (arXiv:astro-ph/0703026)
- **Context:** SDSS MOC3 observations. Candidates are selected **kinematically, not by orbit**.
- **"Known Trojans" training set:** objects matched to Bowell's ASTORB, "here defined as objects with semimajor axis in the range 5.0–5.4 AU" (p4). The 482 observations of 313 KTs are defined by "5.0 AU < a < 5.4 AU and e < 0.2" (p5).
- **Candidates:** cuts in angular velocity (v, v_λ) as a function of the angle from opposition φ, for −25° < φ < 25° (Eqs. 1–2, p5). This gives 1187 detections of about 860 objects. Completeness of the kinematic selection is 89%, and contamination "about 3%" (pp2, 5).
- **L4/L5 rule:** sky position in "a coordinate system center[ed] on Jupiter and aligned with its orbit": L4 at λ_Jup = +60°, L5 at −60° (Fig. 3, p5). The counting is by sign of λ_Jup: "1.9 times more objects with λ_Jup > 0 than with λ_Jup < 0" (raw, p7).
- **No libration test, no proper elements, no dynamics.**
- **Ratio:**
  - Assuming symmetry of the λ–β distribution: **1.8 ± 0.2**.
  - Fitting double 2-D Gaussians whose shape comes from 1178 ASTORB Trojans (σ_β = 9°, σ_λ = 14°): "**1.6 ± 0.1**" (p7). The abstract says this holds "independently of limiting object's size".
  - The sample is "complete ... to H = 13.8" (p2).
  - Other papers turn this into an "H < 13.8 kinematically selected sample" (Pirani+2019a) or "H ≃ 12 complete" (N13, Di Sisto 2014). **Szabó's sample is a sky-plane/kinematic sample, not a dynamically confirmed one.**
- **Definitions cited:** Jewitt, Trujillo & Luu 2000 (JTL) for the velocity limits; Ivezić+2001 (I01).

### Grav, Mainzer, Bauer, Masiero & Nugent 2012, ApJ 759, 49 (arXiv:1209.1549)
- **Context:** WISE/NEOWISE thermal fits for 478 JTs; taxonomy.
- **No explicit Trojan or L4/L5 definition in this paper.** It says only "Almost 5000 objects have been found with orbits that have been identified to be in the 1:1 mean-motion resonance with Jupiter" (p3). Sample membership is inherited from Grav+2011b, which is **not local**.
- **Size cut:** "we will assume that the sample of large Jovian Trojans (diameters larger than 50km) is complete. This limit corresponds to H ∼ 10.0 for an object with 7% visible albedo" (p8).
- **Ratio:** "This yields 55 objects in the leading (L4) cloud and 41 objects in the trailing cloud" (p20). That is "**L4/L5 = 1.34**" for D > 50 km (p21; 55/41 = 1.341). The paper also restates Grav+2011 as "1.4 ± 0.2 times more objects for all size seen by WISE" (p21).

### "Uehata+2022" file, actually McNeill+2021 (PSJ; arXiv:2101.04602)
- **Context:** ATLAS sparse photometry for shapes, phase curves and colours.
- **Definition:** none of its own. L4/L5 membership is taken straight from the MPC list: "As of January, 2020, the number of known Trojans listed by the Minor Planet Center is 7673 ... 4952 orbit as part of the L4 cloud and 2721 orbit in the L5" (p1). That gives MPC ratio 1.82.
- **Photometric quality cuts only:** ≥100 o-band and ≥20 c-band observations, minimum phase angle ≤ 5°, and phase-angle coverage ≥ 6°. This leaves 209 L4 and 133 L5 (p3).

### Yoshida, Terai & Ohtsuki 2026 (arXiv:2603.16485; journal not given in PDF)
- **Context:** Subaru Suprime-Cam two-band survey of **L4 only** (9.2 deg²).
- **Definition is kinematic:** fields within 20° in ecliptic longitude of L4 and within 10° of opposition (p3).
  - Candidates are moving objects "within the estimated range of apparent sky motion" from a Monte Carlo of known-JT orbits: about −28″ to −17″/hr in longitude and ±16″/hr in latitude (p4). "Ultimately, 120 objects were identified as JT asteroids" (p4).
  - **No orbits, no libration.**
  - Heliocentric distance is estimated from sky motion "assuming a non-eccentric orbit" (p5).
- **Unbiased subsample:** R < 5.6 au and H_g < 16.9, giving 44 objects (p5).
- **Ratio:** quoted only from others: "N_L4/N_L5 ≃ 1.40 ± 0.15 for JTs with D > 2 km (Yoshida & Terai 2017; Uehata et al. 2022), which is consistent with ... H < 15, 1.45 ± 0.05, by Vokrouhlický et al. (2024)" (p1). This **treats a size-limited sky-survey ratio as equivalent to V24's magnitude-limited, family-subtracted, debiased background ratio**.

### Greenstreet, Gladman & Jurić 2023/24, ApJL (arXiv:2309.06609) "Jupiter's Metastable Companions"
- **Context:** the most careful dynamical classification in the library.
- **Sample:**
  - Query of the JPL Horizons SBDB for "4.5≤a≤5.9 au (semimajor axis a within twice Jupiter's Hill sphere radius)", with a-sigma defined and data-arc > 30 d. Cometary designations were removed by hand.
  - "In August 2022 this resulted in a sample set containing 11,581 known objects" (p2).
  - **1000 clones per object** (best fit plus 999 drawn from the JPL covariance via SBDynT), integrated **0.5 Myr** with SWIFT-RMVS4 (Venus–Neptune). Particles are removed within 1000 R_J (≃ 0.48 au, "about 2 Hill spheres"), or at r < 0.4 au or r > 19 au (pp2, 4).
- **Resonance test (App. A.2, p9):** "a 1 kyr running window ... a particle's average semimajor axis must remain within 0.2 au of a_J and no individual semimajor axis value may differ by no more than 0.65 au from a_J within the 1 kyr window, if the object is to be considered resonant during that 1 kyr time period."
- **Classes:**
  - "**Trojans**": "≥95% of the 1000 clones remain in the 1:1 jovian resonance for 0.5 Myr". These "are deemed long-term stable" (p5).
  - "**Transients**": ≥95% of clones resonant for ≥1 kyr but then leave.
  - "**Non-Resonant**": <5% of clones resonant for any 1 kyr.
  - "**Insecure**": 5–95%.
- **Island / L4–L5 rule:** "ϕ11 = λ − λJ ... L4 or L5 Trojans (ϕ11 remains in the leading or trailing hemisphere, respectively, during a 1 kyr running period), Horseshoes (ϕ11 crosses the direction 180° away from Jupiter at any time during a window interval), or Quasi-satellites (all other objects since these must cross ... at ϕ11 = 0°)" (p9). The authors admit misclassification of large-amplitude librators affects "<10% of cases" before manual adjustment.
- **Counts:** "7482 L4 Trojans and 3941 L5 Trojans (11,423 Trojans in total)" (p9), a ratio of **1.898**, all H. There are also 27 metastable co-orbitals: 12 L4, 4 L5, 2 horseshoe, 8 QS and 1 retrograde.
- **Key finding about catalogue definitions:** "The dashed rectangle shows the a, e region that JPL Horizons and the Minor Planet Center ... currently define as the 'jovian Trojan' parameter space; the 14 non-resonant objects in this box ... are not Trojans, however, given that ≥95% of their 1000 numerically-integrated clones are ejected from the resonance in as little as tens or hundreds of years" (Fig. 2 caption, p4).
  - The paper **does not state the box numerically**. From outside knowledge, the JPL SBDB class "TJN" is 4.6 < a < 5.5 au, e < 0.3, which is exactly Holt+2020's query box. Verify this.
- **Classification is at the current epoch:** "We base our classifications on the start of the integrations (i.e., the current time)" (p5).
- **Definitions cited:** Greenstreet+2020; Alexandersen+2013; Alexandersen+2021; DiRuzza+2023; Wajer & Królikowska 2012.

### Parker 2015, Icarus (arXiv:1409.6735). Neptune Trojans, plus a JT reference sample
- **JT comparison sample:** "a nearly complete, moderately bright sample of Jupiter Trojans (H ≥ 11) [sic; the text says H ≤ 11], drawn from the Minor Planet Center (oscillating [sic] inclination and eccentricity) and AstDys-2 and the PETrA Project (libration amplitude)" (Fig. 2 caption, p4). The text on p5 says "H ≤ 11 sample of observed osculating inclination, libration amplitude, and osculating eccentricity". **The caption and text conflict on the H cut.**
- **Libration amplitude (NT):**
  - 100 clones each, integrated 1 Myr.
  - "Half-peak RMS libration" L11 = √2 × RMS(φ − ⟨φ⟩), described as "always smaller than the peak-to-peak definition L = ½[max(φ) − min(φ)]" (p2). **Amplitude conventions differ between papers by up to a factor of ~2.**
- **Capture criterion (simulations):**
  - Down-selection at 5τ_a: "semi-major axis within one Neptune Hill radius".
  - Then 100 Myr, then a 100 kyr high-cadence run: "if they were librating around the L4 or L5 points of Neptune with libration amplitudes less than 35° and had eccentricities less than 0.12, they were identified as stable Neptune Trojans" (p12).
- **Result:** "no captured Trojan population was found to be significantly asymmetric between the L4 and L5 swarms" (p12).

---

## 2. Long-term stability / escape papers (present-day sample integrated forward)

### Holt, Nesvorný, Horner, King, Marschall, Kamrowski, Carter, Brookshaw & Tylor 2020b, MNRAS 495, 4085 (arXiv:2005.03635)
- **Initial query:** "An initial dataset was obtained from the JPL Small-Body Database ... selecting all objects with orbital semi-major axes between 4.6 au and 5.5 au and an orbital eccentricity less than 0.3. This process yielded an initial selection of 7202 objects, obtained on 17th April, 2018" (p4).
  - No i or σ cut.
  - HORIZONS ephemerides at 2000-Jan-01. A footnote gives HORIZONS access on 13 Nov 2019, and Fig. 1 says "as of 19th Aug. 2019". **There are three different dates in one paper.**
- **"Stable" filter:** "we then filtered our sample to discard temporarily captured objects by limited selection to those objects present in the AstDys proper element database (Knežević & Milani 2017). Since objects in this list require the completion of simulations spanning 1 × 10^6 years to generate the proper elements ... this set can be considered initially stable objects" (p4).
  - This leaves 5553 "nominally 'stable'" Trojans (4780 numbered and 773 multi-opposition).
  - The **libration test is outsourced to AstDyS** synthetic proper elements: ≥1 Myr, with numbered and multi-opposition objects only.
- **Clones:** "Eight clones ... at the vertices of a cuboid in x-y-z space" at ±1σ positional uncertainty, so 9 particles per object. Integrated 4.5 Gyr with REBOUND WHFAST, dt = 0.3954 yr = P_J/30 (pp4–5).
- **Escape definition:** "once the test particle obtains an osculating semi-major axis of less than 4.6 au or greater than 5.5 au" (p5). This is a **semimajor-axis box only, with no libration check**, so horseshoe or switch-to-other-island would count as "Trojan".
- **L4/L5 rule:** **not stated**. Only "leading (L4) and trailing (L5)" (p1).
- **Counts:** Table 2 (p6) gives L4 3634 and L5 1919, a ratio of 1.894.
  - Escape (pool) is 23.19% at L4 and 24.89% at L5.
  - The **all-9-clones-escape** subset is only 5.01% and 5.04%. So the escape fraction depends strongly on the clone criterion.
  - Text: "considering the number of known objects (L4 1.89 larger)" and "L4 1.56 larger" by volume (p5).
  - Implied primordial ratio is 1.84 ± 0.003 (p6).
  - Albedos are 0.076 (L4) and 0.071 (L5) for converting H to D.
- **Definitions cited:** Knežević & Milani 2017 (AstDyS); Milani & Knežević 1992; Šidlichovský & Nesvorný 1996; Beaugé 2001; Brož & Rozehnal 2011; Levison+1997; Tsiganis+2005a.

### Di Sisto, Ramos & Beaugé 2014, Icarus 243, 287 (arXiv:1604.05331)
- **Sample:** "As of March 2013, there were 2972 numbered Jupiter Trojans ... Of these, 1975 (over 66%) display tadpole orbits around L4, while 997 are associated to L5" (p4).
  - **The source catalogue is never named.**
  - **No element cuts are given.**
  - **No H cut:** "we will consider the complete (numbered) population regardless of the absolute magnitude" (p4).
  - The ratio is 1975/997 = **1.98**. The text says "current value of ∼1.97" (p10).
  - Intro: "As of March 2012, a total of 5179 members have been cataloged, including numbered and multi-oppositional asteroids. Of these, 3394 are located around L4, while only 1785 inhabit the tadpole region around L5" (p2).
- **L4/L5 rule:** by plotting convention in the "representative plane" M − M_J = 0, ϖ − ϖ_J = ±60°, Ω − Ω_J = 0. "The upper half of the plots (positive values of e and i) corresponds to L4" (p4). The elements are osculating at that plane crossing.
- **Escape (4.5 Gyr, EVORB, Sun plus 4 giants, dt = 0.2 yr):** "Each integration was stopped if the particle suffered a close encounter with a planet (i.e. mutual distance smaller than the corresponding Hill radius)" (p8). This is a **1 R_Hill encounter criterion**.
  - Escapes: 23% at L4 and 28.3% at L5.
  - The ratio rises from ∼1.97 to ∼2.13 in 4.5 Gyr.
- **Libration amplitude D:** "the semi-amplitude of libration" (p5, after Tsiganis+2005). Proper elements come from the semi-analytical model of Beaugé & Roig 2001 (p13).
- **Fictitious grid:**
  - 5.01 ≤ a ≤ 5.4, 0 ≤ e ≤ 0.34, 0 ≤ i ≤ 60°.
  - ϖ = ϖ_J ± 60°, Ω = Ω_J, M = M_J.
  - 18,200 per point (p15).
- **No clones.**
- **Definitions cited:** Beaugé & Roig 2001; Brož & Rozehnal 2011; Milani 1993; Levison+1997; Tsiganis+2005; Szabó+2007 for "complete up to H = 12" (this is a misattribution of Szabó's H = 13.8 completeness, or an older convention).

### Di Sisto, Ramos & Gallardo 2019, Icarus 319, 828 (arXiv:1811.00352)
- **Sample:** "all numbered Jupiter Trojan asteroids as of March 2013. Thereby, a numerical integration of 1975 L4 Trojans and 997 L5 Trojans" (p7). This is the **same sample as DS14**. **No source catalogue, element cuts or L4/L5 rule are given.**
- **Escape:**
  - Setup: Sun plus Venus–Neptune, EVORB, dt = 7.3 d, 4.5 Gyr.
  - Removal: on collision or r > 1000 au.
  - "encounters at less than 2.1 Hill's Radius with the planets were registered ... If a Trojan has an encounter with a planet, usually Jupiter, the time of the first encounter is considered the 'escape time'" (p8).
  - **This is a different escape criterion from DS14 (1 R_H, giants only), and gives different escape fractions from the same sample:** 466/1975 = 23.6% at L4 and 250/997 = 25.1% at L5. DS14 gave 23% and 28.3%.
- **Ratio used:** adopts **Grav+2011's 1.4** as the "unbiased" ratio: "we will consider that the number of L4 Trojans is 1.4 the number of L5 Trojans, and that this asymmetry does not depend on the size" (p6). Primordial is N0(L4)/N0(L5) = 1.373 ± 0.204 (p10) or 1.376 ± 0.204 (p15). **The two pages disagree in the third decimal.**
- **Definitions cited:** Tsiganis+2005 (effective stability); Levison+1997; Di Sisto+2014.

### Hellmich, Mottola, Hahn, Kührt & de Niem 2019, A&A (arXiv:1909.08584)
- **Sample:** "numbered and multi-opposition Trojan orbits, consisting of 3642 L4 and 1925 L5 objects", with covariance from **AstDyS**. Clones were drawn to equalise the swarms: 13–14 per L4 object and 25–26 per L5 object, 98,304 particles in total (p2). The ratio is 1.892.
- Intro (MPC): "As of August 2019, the Minor Planet Center lists 4603 numbered, multi-, and single-opposition Trojans in the leading (around L4) but only 2476 in the trailing cloud" (p1), a ratio of 1.859.
- **L4/L5 rule:** "λ − λ_J = ±π/3 + D cos θ + O(D²) ... For L4 Trojans, the difference in mean longitude is positive, while it is negative for L5 Trojans" (p2, after Tsiganis+2005).
- **Libration amplitude:** filtered a(t); "the oscillation amplitude in semimajor axis (d) as the mean of the maximum excursion ... in 2 kyr intervals over the 200 kyr of integration". Converted with d = √(3µ) a_J D ≈ 0.2783 D (Erdi 1988), and checked against AstDyS to within 0.61° (p4).
- **Escape:** "heliocentric distance became smaller than 3 au, exceeded 8 au, or when the particles approached Jupiter closer than twice the planet's Hill radius" (p5).

### Robutel & Bodossian 2009, MNRAS 399, 69 (file mislabelled "RobutelGabern2009")
- **Theory and numerics.** Fictitious **L4 only**:
  - Grid a ∈ [5.2035, 5.4030] × e ∈ [0.05, 0.30] (8000 particles).
  - "σ = λ − λ1 = π/3, σ_g = ϖ − ϖ1 = π/3, Ω = Ω1 and I = I1 + I*" with I* = 2°, 20°, 30° (p5).
  - Integrated 2 × 5 Myr (SABA4).
- **Survival:** "bodies which are not ejected from the co-orbital region before the end of the integration" (p5). **"Co-orbital region" is not quantified.** Stability is measured by the frequency-map diffusion index log10|(ν(1) − ν(2))/ν(1)|.

---

## 3. Capture / migration simulations (the "model" side of any N4/N5 comparison)

### Nesvorný, Vokrouhlický & Morbidelli 2013, ApJ 768, 45 (N13 / NVM13)
- **Generic description (not an operational cut):** "a ≃ a_Jup, eccentricities e ≲ 0.15, inclinations i ≲ 35°, and Δλ = λ − λ_Jup ∼ ±60° ... full libration amplitude up to D ≃ 70°" (pp1–2).
- **Capture pipeline:**
  - **Phase 1:** 10 Myr through the instability, with clones at r < 8 au (N_clo = 19).
  - **Phase 2:** "we identified Trojan 'candidates' by selecting particles with orbits near Jupiter's L4 and L5 at the end of Phase 1. This selection was very liberal" (p6). **No numbers are given.** Phase 2 runs 100 Myr with residual migration.
  - **Phase 3:** "an additional numerical integration over 4 Gyr ... the long-term stability requirements shaved off about 50% of Trojans ... The removed particles typically had D > 60°, large e and/or large i" (p6). **The survival criterion for Phase 3 is not given.**
  - The resulting "stable Trojans" are those that "survived to the end of Phase 3" (p7).
- **D definition:** "The full libration amplitude D corresponds to the angular distance between extremes of λ − λ_Jup during libration" (Fig. 7 caption, p23). This is a **full (peak-to-peak) amplitude**, unlike DS14's semi-amplitude. Proper elements follow the Brož & Rozehnal 2011 method. "We found no significant difference between the orbital distributions of L4 and L5 Trojans, and clumped these distributions together."
- **Observed ratio:**
  - From MPC data (date not given): "the number of the known L4 Trojans with H < 12 is 361, if 9 known Eurybates family members with H < 12 are removed (Brož & Rozehnal 2011). For comparison, there are 279 L4 [sic, = L5] Trojans with H < 12. This indicates that f45 = 1.3 ± 0.1" (p11). 361/279 = 1.294.
  - H < 12 is described as "complete sample according to Szabó et al. 2007".
  - L4/L5 membership is taken from the MPC list as-is.
- **Model ratios:** f45 = 1.3 ± 0.5 (Case 1), f54 = 1.4 ± 0.4 (Case 2), f54 = 1.8 ± 0.9 (Case 3) (p11). These are from 30, 41 and similar numbers of stable captures.

### Roig & Nesvorný 2015, AJ 150, 186 (arXiv:1509.06105)
- **Context:** asteroids in the jumping-Jupiter model. Trojans are a side population.
- **"Trojan swarms (TS) set":** "5.2 ≤ a ≤ 5.7 au, e < 0.2, I < 40°" with "initial angles M, ω, Ω ... randomly distributed between 0 and 360°" (p7). **This is an a-e-i box with random longitudes, so no tadpole/L4/L5 placement at all.** a is shifted outward because Jupiter starts at about 5.5 au.
- **Elimination:** within planet radius, q < 1 au, or a > 100 au (p7).
- **Survival "in the TS region" is never defined.** "All of them have been scattered out of the 3:2 and 1:1 mean motion resonances" (p23). How resonance membership was tested is not stated.
- **No L4/L5 distinction and no ratio.**

### Pirani, Johansen, Bitsch, Mustill & Turrini 2019a, A&A 623, A169
- **Simulation Trojan definition (the key one):** "We consider as Trojans all those particles that have aphelion and perihelion within ±2.5 Jupiter Hill radii from the planet semimajor axis. This way we include almost all the main coorbital resonances: tadpole orbits (Trojans, that librate around ±60°), horseshoe orbits (objects that librate around 180° encompassing L4 and L5) and, occasionally, quasi-satellite orbits ..., excluding interlopers with high eccentricity" (p8).
  - This is a **single-epoch geometric (q, Q) cut**, with **no libration test**.
  - It **explicitly includes horseshoes and quasi-satellites**.
  - **No L4/L5 assignment rule is given**, even though N_L4/N_L5 is reported. It is unclear how horseshoe particles are split.
- **Snapshots:** results are "Mean quantities of the Trojans at t = 5 Myr" (Table 3, p9). N_L4/N_L5 = 1.80 ± 0.26 for the nominal rate, with Ntot = 240 ± 9 over 10 runs. The in-situ case gives 0.94 ± 0.35.
- **Observed ratios quoted:**
  - Grav+2011: 1.4 ± 0.2, D > 10 km.
  - Szabó+2007: 1.6 ± 0.1, "kinematically selected sample ... H < 13.8".
  - Nakamura & Yoshida 2008: 1.8 ± 0.57, D > 2 km.
  - Vinogradova & Chernetenko 2015: about 2.0, H < 19.
  - Their own count: "We also analysed the Trojans in the MPC [list, footnote: minorplanetcenter.net/iau/lists/t_jupitertrojans.html] with H<12 ... 330 L4 objects and 229 L5 objects. As in Nesvorný et al. (2013), we excluded 9 Eurybates family members from L4 finding an asymmetry ratio of about 1.4 ± 0.1" (p2). 321/229 = 1.40.
  - MPC total "updated on the 30th of May 2018, is 71801" [sic, 7180] (p2).
  - **Note:** N13 counted 361 L4 (after removing Eurybates) and 279 L5 with H < 12 in about 2012, while Pirani counted 330 L4 (before removal) and 229 L5 with H < 12 in 2018. **The H < 12 counts went down over time**, presumably because MPC H values were revised. So "H < 12 complete" is not a stable sample definition.

### Pirani, Johansen & Mustill 2019b, A&A 631, A89
- **No Trojan definition in this paper.** It implicitly inherits 2019a.
- The only operational statement is about timing: "The starting inclination distribution of the Trojans is evaluated at 5 Myr, that is 2 Myr after Jupiter reaches its current semimajor axis. This is because we want to avoid to count eccentric interlopers stirred by the migration as Trojans" (p3).
- **Ratios:** initial N_L4/N_L5 is 1.60 (J0, 2600 Trojans) and 1.48 (JS0, 2208) (Table 1, p3). The paper then evolves 4.5 Gyr with massive embryos: J250 goes 1.60 → 0.81 at 100 Myr → 0.40 at 500 Myr (Table 3, p6).
- The "current observed inclination distribution" histogram has **no stated source or cuts**.

### Li, Xia, Yoshida, Georgakarakos & Li 2023a, A&A 669, A68 (arXiv:2211.13877)
- **Observed sample:**
  - "As of 2022 January 26, 11552 Jupiter Trojans have been registered in the Minor Planet Center" (footnote: minorplanetcenter.net/iau/lists/JupiterTrojans.html). "the overall leading-to-trailing number ratio is 1.84" (p3).
  - 10854 multi-opposition objects. "even if observations were made at only a single opposition, the obtained Lagrangian point classification (i.e. near L4 or L5) and absolute magnitude are quite confidential [sic, = reliable]" (p3).
  - **L4/L5 assignment is the MPC list's.**
  - Resonant amplitudes come "through numerical integration", with **duration and method unspecified**.
  - H_complete = 12.3. N4/N5 = 0.92 for H < 9 (24 objects).
  - **Adopted target:** "we will consider N4/N5 = 1.6 as the" unbiased value, from Szabó 2007 (p3).
- **Simulation:**
  - σ = λ − λ_J (Eq. 3, p4).
  - **Pre-runs:** 2000 yr with Jupiter fixed. "the 2000 yr evolution is long enough to identify the tadpole motion" (p5).
  - **Initial conditions:** a = 5.2 au, e ∈ [0, 0.3], i = 0.01 [units unclear], σ0 = 60° + Δσ0 (L4) or −60° − Δσ0 (L5), keeping "only the rather stably librating particles with Δσ0 ≤ 80°". The standard set is Δσ0 = 30°–80°, with 500 per swarm.
  - **Count:** after 1 Myr, the "survived" N4 and N5 are counted (Table 1, p6). **The criterion for "survived" (still librating? not ejected? which island?) is not stated.** A particle switching L5 → horseshoe is shown (Fig. 3b) but no rule is given for counting it.
  - **Result:** R45 = 1.50 (Δσ0 = 30–80°), rising to 1.68 at 1 Gyr. The paper says "the final value of R45 could be simply increased by a factor of 1.1" (p7).

### Li, Xia, Georgakarakos & Yoshida 2023b, A&A (arXiv:2304.13598)
- Same test-Trojan set-up as 2023a: "σ = (Ω + ω + M) − λ_J ... For the L4 and L5 swarms, we set σ0 to be 60° + Δσ0 and −60° − Δσ0 respectively ... Nc ... 500. More details can be found in Sect. 3.1 in Li et al. (2023)" (p5).
- After 1 Myr (swift_rmvs3, dt = 0.05 yr), "the numbers of surviving L4 (N4) and L5 (N5) Trojans" are reported. **"Surviving" is again undefined.** The "observed resonant amplitude distribution" is used qualitatively ("ΔΣ0 of 0–40° according to the observed Jupiter Trojans", p10) with no stated source.
- **Target:** "R45 ∼ 1.6" throughout.

### Horner & Evans 2006, MNRAS 367, L20 (arXiv:astro-ph/0511791)
- **Capture criterion:** "the data were searched for objects whose semimajor axis stayed within one Hill radius of that of Jupiter, Saturn, Uranus and Neptune ... These provisional candidates were then examined in the co-rotating frame ... To ensure that an object was indeed captured, a minimum number of 800 orbital periods was required ... in the case of Jupiter, the object had to show Trojan-like or satellite-like behaviour for at least 9.5 kyr" (p2).
  - The co-rotating-frame classification is **by visual inspection**.
  - Output was every 100 yr.
- **Horseshoes counted as Trojans:** Table 1's "Jupiter Trojans 10" includes horseshoes. Table 2 splits Jupiter into L4 2, L5 5 and horseshoe 3 (p3). "Objects that displayed two periods of Trojan behaviour in different regions gave a score of 0.5 in each region occupied" (Table 2 caption).

### Lykawka, Horner, Jones & Mukai 2009, MNRAS 398, 1715 (Neptune Trojans)
- **Known NTs:** 100 clones (3σ in a, e) integrated 10 Myr. "average libration amplitude, A, is calculated by sequentially measuring the individual maximum and minimum displacement values for each clone ... then averaging". The centre of libration is "60±5°, 300±5° and 180±5° for L4, L5 and horseshoe orbits, respectively" (p6). The code is RESTICK (Lykawka & Mukai 2007b).
- **Simulations:** "we used the RESTICK package to ... detect the Trojans present at the end of the simulation" (p9). This is an **end-of-run snapshot via RESTICK**, whose window and thresholds are not given here.
- **Three categories counted separately:** N_H, N_L4 and N_L5 (Table 4). Amplitudes reach "~60-70° for L4/L5 orbits (in the limit of leaving the tadpole orbit) and 150-170° for horseshoe orbits" (p10).
- **Horseshoes are "Trojans" here:** "a relatively large number of Trojans surviving on horseshoe orbits" (p10).

### Chen, Ma & Zheng 2016, MNRAS (arXiv:1602.04303) (Neptune Trojans)
- **Pre-filter:** 10^5 yr without migration; 10,000 "stable particles" per L-point are kept (p5).
- **Count after migration (10^7 yr):** "remaining NTs (objects cor[ot]ating with Neptune) by counting the number of particles with |a − a_N| < 2 AU and Δλ ∈ (0°, 120°) ∪ (240°, 360°)" (p5). This is an **instantaneous snapshot**. The L4/L5 split is by Δλ sector, so large-amplitude tadpoles and horseshoes in that phase range are counted.
- **Second "stable" filter:** "Objects are classified as stable using the criterion by Zhou et al. (2011)": three stable i windows (0–12°, 22–36°, 51–59°) with e upper limits of 0.1, 0.12 and 0.04 (p5).
- **f45 is reported for both "remaining" and "stable".** "The large value ... for the with-damping value of ... f45, is transient as the excess objects are primarily unstable" (p5). **This is a direct demonstration that the counting definition changes f45.**

### Nesvorný 2018, ARA&A 56, 137
- Generic definition copied from N13: "a ≃ a5 = 5.2 au, e < 0.15, i < 40°, and δλ = λ − λ5 ∼ ±60° ... full libration amplitude, D, up to D ≃ 70°" (p18). (N13 has i ≲ 35°; this has i < 40°.)
- The comparison sample is "known Jupiter Trojans with absolute magnitude H < 12 (black line; the H cut is done to limit the effect of the observational bias)" (Fig. 8, p19).

### Nesvorný, Vokrouhlický, Bottke, Noll & Levison 2018, Nature Astronomy (Patroclus–Menoetius)
- Generic description: "semimajor axes a ≃ 5.2 au, eccentricities e < 0.15, and inclinations i < 40°" (p1). "The planetesimals that evolved onto JT orbits were selected for further use" (p2). **No operational criterion**; it inherits N13.

### Montesinos et al. 2020, A&A (dust trapping; not asteroids)
- Trapping is a **snapshot box**: "a particle will be in tadpole orbit around a Lagrangian point if it belongs to the cylindrical areas ... delimited by the position of the Lagrangian points at ±π/3 adding (or subtracting) an angle of ±π/8 rad in the azimuthal direction, and ... r_p ± R_Hill" (p4).

---

## 4. Reviews

### Bottke, Marschall, Nesvorný & Vokrouhlický 2023, Space Sci. Rev. (preprint 19 Oct 2023)
- **Definition (descriptive only):** "populations that orbit around the L4 and L5 Lagrange points of Jupiter ... contained within Jupiter's 1:1 mean motion resonance" (p3). "eccentricities ... e < 0.15, values required to remain within the stable L4 and L5 zones" (p4).
- **Fig. 1 sample:** "Proper orbital elements of Jovian Trojans. The left two panels show the orbits of 4607 objects at the L4 libration zone, while the right two panels show the orbits of 2721 objects at the L5 libration zone (see Holt et al. 2020 ...). The proper orbital elements ... were calculated using the methods of Milani (1993)" (p26–27). The ratio is 1.69. Note that 2721 equals McNeill's MPC Jan-2020 L5 count. **The source and date of the catalogue are not stated.**
- **Ratio stance:**
  - "The population of D > 100 km bodies ... contains 9 objects in both L4 and L5" (p4).
  - "If we examine D > 10 km bodies, the ratio of L4 to L5 objects moves to 1.4 ± 0.2 (Grav et al. 2011) ... consistent with both populations being statistically comparable" (p4).
  - The review **does not cite V24's 1.43–1.45**, although V24's authors are co-authors.

### Emery, Marzari, Morbidelli, French & Grav 2015, Asteroids IV, 203–220
- "Trojans share Jupiter's orbit around the Sun, residing the L4 and L5 stable Lagrange regions" (p2). **No operational definition.**
- The size distribution uses "absolute magnitudes from the Minor Planet Center catalog" (p4) and the "Minor Planet Center online list of Jupiter Trojans" (via Wong+2014, H < 12.3; p5).
- It speaks of "the ~30% asymmetry" (p16) and of N13 asymmetries "at the 30-80% level (not necessarily in favor of L4)" (p17).

### Morbidelli+2021 (asteroid belt review), Raymond & Morbidelli 2014, and the 04_migration_physics papers
- They only mention Trojans; **none define them**. Morbidelli+2021 calls them "Jupiter's 1:1 resonant populations (the so-called Trojans)" (p15).

### Vokrouhlický+2025 (Hildas), AJ
- **No JT definition of its own.** It re-quotes the V24 background at H ≤ 14.5: "3951 ± 44 in the L4 swarm and (ii) 2664 ± 36 in the L5 swarm" (p22). V24's Table 3 has ±39 for L5.

### Leleu+2019, Lhotka & Celletti 2015, Sicardy & Dubois 2003
- These are analytical theory papers.
- "Tadpole vs horseshoe" is defined by the separatrix of the averaged 1:1 Hamiltonian, which passes through L3 at ζ = 180° for a non-migrating planet. Sicardy & Dubois show that under migration (parameter α) L4 and L3 merge at α = 0.7265 (φ = 108.35°) (p6).
- **None give an operational counting criterion for N-body output.**

### Lucy 2021, Nesvorný & Vokrouhlický 2019 (binaries)
- No Trojan definition.

---

## 5. Summary of where each quoted N4/N5 lives

| Value | Sample (what is counted) | Source |
|---|---|---|
| 1.892 | MPCORB 22 Feb 2023: box (a 4.95–5.45, e < 0.6, i < 50°, 20° < \|σ\| < 130°) plus 1-Myr nominal libration; all H; families included | V24 p3 |
| 1.45 ± 0.05 | MPC raw, H ≤ 13.8, families removed, √N errors | V24 p2 |
| 1.43 ± 0.05 / 1.45 ± 0.05 | bias-corrected background model, H ≤ 13.8 (Fig. 25) or "H < 15" (abstract) or H = 15 (conclusions) | V24 pp1, 31, 34–35 |
| 1.483 | 3951/2664, debiased background at H ≤ 14.5 | V24 Table 3; V25 p22 |
| 1.894 | AstDyS synthetic-proper-element list ∩ JPL TJN box (4.6–5.5 au, e < 0.3), numbered + multi-opposition | Holt+2020b Table 2 |
| 1.89 / 1.56 | number / volume of the same sample | Holt p5 |
| 1.898 | JPL SBDB Aug 2022, 4.5–5.9 au, ≥95% of 1000 clones resonant for 0.5 Myr | Greenstreet p9 |
| 1.98 (≈1.97) | numbered JTs Mar 2013 (catalogue unnamed), all H | DS14 p4, p10 |
| 1.90 | catalogued (numbered + multi-opp) Mar 2012 | DS14 p2 |
| 1.892 / 1.859 | AstDyS numbered+multi-opp / MPC incl. single-opp, Aug 2019 | Hellmich pp1–2 |
| 1.82 | MPC list Jan 2020 | McNeill (file "Uehata+2022") p1 |
| 1.84 | MPC list 26 Jan 2022, all H | Li+2023a p3 |
| 1.69 | 4607/2721 "proper elements, Milani method" (source/date unstated) | Bottke+2023 Fig. 1 |
| 1.3 ± 0.1 | MPC, H < 12, Eurybates (9) removed: 361/279 | N13 p11 |
| 1.4 ± 0.1 | MPC list May 2018, H < 12, Eurybates (9) removed: 321/229 | Pirani+2019a p2 |
| 1.6 ± 0.1 (1.8 ± 0.2) | SDSS MOC3 kinematic candidates, sky-plane λ_Jup sign, Gaussian-shape fit | Szabó+2007 p7 |
| 1.34 | WISE, D > 50 km (55/41) | Grav+2012 p20–21 |
| 1.4 ± 0.2 | WISE, D > 10 km (Grav+2011, not local) | quoted by many |
| 1.40 ± 0.15 | Subaru HSC D > 2 km (Yoshida & Terai 2017; Uehata+2022, not local) | Yoshida+2026 p1 |
| 1.80 ± 0.26 | sim: (q, Q) within ±2.5 R_H of a_J at t = 5 Myr, horseshoe/QS included | Pirani+2019a Table 3 |
| 1.60 / 1.48 | sim at t = 5 Myr (J0 / JS0) | Pirani+2019b Table 1 |
| 1.50 → 1.68 | sim: "surviving" test Trojans, 1 Myr → 1 Gyr | Li+2023a |
| 1.3 ± 0.5, 1/1.4, 1/1.8 | sim: survived 4 Gyr Phase 3 | N13 p11 |

---

## 6. Definitions cited but not in the local library (to chase)

**Proper elements and the libration amplitude convention**
- Milani 1993, CeMDA 57, 59 (synthetic proper elements for Trojans; "full capacity" stability)
- Milani & Knežević 1992
- Beaugé & Roig 2001, Icarus 153, 391 (semi-analytic proper elements; used by DS14)
- Beaugé 2001 (cited by Holt)
- Brož & Rozehnal 2011, MNRAS 414, 565 (proper elements and families; Eurybates membership used by N13 and Pirani)
- Rozehnal+2016
- Holt+2020a, MNRAS 499, 3630 (Appendix A proper-element pipeline used by V24)
- Knežević & Milani 2017 (AstDyS proper-element database; the "stable Trojan" list used by Holt and Hellmich)
- Šidlichovský & Nesvorný 1996 (frequency-modified Fourier filtering)
- Erdi 1988 (d ↔ D conversion)
- Bien & Schubart 1987

**Stability-zone definitions**
- Levison, Shoemaker & Shoemaker 1997 (Nature)
- Tsiganis, Varvoglis & Dvorak 2005 (D as semi-amplitude; effective stability)
- Nesvorný & Dones 2002
- Robutel & Gabern 2006 (Paper I)
- Marzari et al. 2002 (Asteroids III review)
- Marzari+2003
- Hou+2014
- Zhou+2011 (NT stable windows used by Chen)

**Co-orbital classification algorithms**
- Greenstreet+2020
- Alexandersen+2013 (Science, Uranian Trojan)
- Alexandersen+2021 (Saturn horseshoe)
- DiRuzza+2023
- Wajer & Królikowska 2012
- Lykawka & Mukai 2007b (RESTICK)
- Parker+2013 (NT libration amplitudes)
- Horner+2004a,b (Centaur runs underlying Horner & Evans)
- Gladman, Marsden & VanLaerhoven 2008. **No local paper cites it for Trojans**; it is only cited for TNO classes (Nesvorný 2018 ARAA; N&V 2019). The Greenstreet criterion is the Trojan analogue.

**Observational ratios / sky-survey definitions**
- Grav+2011 (ApJ 742, 40; "1.4 ± 0.2"; the sample definition behind Grav+2012)
- Nakamura & Yoshida 2008
- Yoshida & Terai 2017
- Uehata+2022 (AJ 163, 213; the real one)
- Jewitt, Trujillo & Luu 2000 (velocity selection used by Szabó)
- Freistetter 2006 (historical ratio about 1.67)
- Vinogradova & Chernetenko 2015
- Wong+2014
- Wong & Brown 2015
- Shoemaker+1989
- Hendler & Malhotra 2020
- Vinogradova 2015, 2019

**Theory of tadpole/horseshoe under migration**
- Yoder+1983
- Morais 1999, 2001
- Fleming & Hamilton 2000
- Marzari & Scholl 1998
- Morbidelli+2005 (chaotic capture)
- Hou+2016
- Ogilvie & Lubow 2006
- Gomes 1998

**Catalogue-side definitions (no paper states them; check directly)**
- The MPC "Jupiter Trojans" list criterion (minorplanetcenter.net/iau/lists/JupiterTrojans.html), used by Li+2023a, Pirani+2019a, N13 and McNeill.
- The JPL SBDB "TJN" class (believed to be 4.6 < a < 5.5 au, e < 0.3; matches Holt's query; verify).

---

## 7. Cross-cutting observations (apples vs oranges)

1. **Four families of "Trojan" definitions coexist:**
   - **(a) Catalogue membership with no dynamics:** the MPC/JPL list or box (McNeill, Li+2023a observed, N13 and Pirani observed counts, Emery, Bottke).
   - **(b) Sky-plane or kinematic:** Szabó, Yoshida, Nakamura & Yoshida, Uehata. These contain **no orbit information at all**.
   - **(c) Dynamically confirmed librators:** V24 (1 Myr nominal), Holt and Hellmich (AstDyS ≥1 Myr proper elements), Greenstreet (0.5 Myr, ≥95% of 1000 clones).
   - **(d) Simulation-side capture cuts:**
     - Pirani 2019a: a geometric (q, Q) ±2.5 R_H cut at one epoch, **including horseshoe and QS**.
     - Chen: Δλ sector plus |Δa| < 2 au at one epoch.
     - Horner & Evans: 1 R_H in a for ≥800 P plus visual inspection, **horseshoes counted**.
     - Lykawka: RESTICK at end of run, **horseshoes counted separately**.
     - Parker: libration < 35°, e < 0.12, 100 kyr.
     - N13: survived 4 Gyr, criterion unstated.
     - Li+2023a,b: "survived" 1 Myr, criterion unstated.
     - R&N15: random-longitude box, no L4/L5.
2. **For the bright end, the dynamical-confirmation step barely changes counts.** V24 removed 26 + 12 of 12,118; Greenstreet found 14 non-resonant in the JPL/MPC box. The all-H observed ratio is about 1.89 in the MPC, AstDyS, V24 and Greenstreet samples alike. **The large spread in quoted ratios (1.3–2.0) comes from H/D cuts, family removal and debiasing, not from the Trojan definition.**
3. **H-limited counts are not stable across catalogue epochs.** For H < 12: N13 counted 361 L4 (Eurybates removed) and 279 L5; Pirani+2019a counted 330 L4 (before removal) and 229 L5 in 2018. MPC H revisions make "H < 12 complete" epoch-dependent.
4. **Libration amplitude conventions differ:**
   - D as full amplitude, the distance between extremes of λ − λ_J (N13, Nesvorný 2018).
   - D as semi-amplitude (DS14, after Tsiganis+2005).
   - λ − λ_J = ±π/3 + D cos θ, also a semi-amplitude (Hellmich).
   - RMS-based half-amplitude L11 (Parker; smaller than peak-to-peak).
   - Time-averaged max/min displacement (Lykawka).
   - Li's "resonant amplitude A", with no definition given.
   - V24's C (an energy-like integral, with the tadpole range 1.5–2.5).
   - **Amplitude cuts (e.g. 35°, 60°, 70°) are not directly transferable between papers.**
5. **Escape definitions differ, and DS14 vs DS19 shows they matter on the same sample:**
   - Holt: a outside [4.6, 5.5] au.
   - DS14: an encounter within 1 R_H.
   - DS19: an encounter within 2.1 R_H.
   - Hellmich: r < 3 au, r > 8 au, or within 2 R_H of Jupiter.
   - Greenstreet: within about 2 R_H, or failing the a-window test.
   - V24: leaving the libration region.
   - Robutel & Bodossian: an unquantified "co-orbital region".
   - DS14 and DS19 use the same 1975/997 sample: L5 escape 28.3% vs 25.1%.
6. **The L4/L5 assignment rule is almost never stated explicitly.**
   - Explicit: Greenstreet (hemisphere of φ11 over a 1 kyr window), Hellmich (sign of λ − λ_J), Szabó (sign of the sky-plane λ_Jup), Chen (Δλ sector), DS14 (representative-plane sign), and V24 (implicit sign of σ in the |σ| window).
   - Unstated, or taken from the MPC list: Holt, Pirani 2019a,b, Li 2023a,b, N13 simulations, McNeill, Bottke and Grav.
