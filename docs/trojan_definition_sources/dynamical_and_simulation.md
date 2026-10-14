# Dynamical, theoretical and simulation definitions of a "Jupiter Trojan" — literature survey

Compiled 2026-10-14 for the L4/L5 capture/loss asymmetry thesis. The question is whether the simulation's "Trojan" matches the observational "Trojan".

**Verification convention.** VERIFIED / [V] means the quote was copied from full text or an abstract fetched in this session (arXiv/ar5iv HTML, arXiv PDF, author preprint, or the publisher's page). SECONDARY means the claim comes from another paper's description of this one. RECALLED / [R] means it comes from memory and was not checked. Publisher sites (ADS, Springer, ScienceDirect, HAL, AstDyS) blocked automated access for several classic papers, so those entries are marked accordingly. No numbers were invented. Where sources disagree, both versions are given.

Layout: synthesis → compact summary table → Part A (classical theory and stability maps) → Part B (capture simulations) → Part C (long-term evolution, the definitional problem itself, catalogue definitions, and Greek/Trojan naming).

---

## 0. Synthesis: what the literature actually uses

1. **No single dynamical definition exists.** The literature uses five operational families of definition:
   - **(a) Tadpole libration of σ = λ − λ_J about the e- and I-dependent L4/L5 centre without reaching 180°.** Classical theory (Murray & Dermott; the C ∈ (1.5, 2.5) integral) and Levison+1997 (four-quadrant test). Also Nesvorný & Dones 2002, Lykawka & Horner 2010, Kortenkamp+2004, Morais & Namouni.
   - **(b) Libration plus a survival time.** Thresholds vary widely:
     - ≥800 P_J ≈ 9.5 kyr (Horner & Evans 2006)
     - 1 kyr windows lasting the whole 0.5 Myr, with ≥95% of clones (Greenstreet+2024)
     - 0.5–1 Myr (Lykawka & Horner 2010)
     - 1 Myr (Vokrouhlický+2024 observational filter; Li+2023)
     - 4 Gyr "stable Trojans" (Nesvorný+2013)
   - **(c) Catalogued Trojan with stable proper elements** (d, D, e_p, sin I_p; Milani 1993, Beaugé & Roig 2001, Brož & Rozehnal 2011, AstDyS). Holt+2020 uses this as its entry criterion.
   - **(d) Geometric snapshot boxes:**
     - Pirani+2019a: q and Q within a_J ± 2.5 R_H. This explicitly includes horseshoes and quasi-satellites.
     - Montesinos+2020: ±22.5° azimuth × ±1 R_H about ±60°.
     - JPL TJN: 4.6 < a < 5.5 au, e < 0.3.
     - MPC type 12: 4.8 < a < 5.4 au, e < 0.3.
     - Vokrouhlický+2024: 4.95 < a < 5.45 au, e < 0.6, I < 50°, 20° < |σ| < 130° at one epoch, then a 1 Myr libration check.
   - **(e) "Not yet escaped" exit rules that never test libration:**
     - Holt+2020: osculating a outside 4.6–5.5 au.
     - Di Sisto+2014: encounter within 1 R_H of a planet.
     - Di Sisto+2019: encounter within 2.1 R_H.
     - Hellmich+2019: within 2 R_H of Jupiter, or r < 3 au or r > 8 au.
     - Nesvorný & Dones 2002 and Robutel+: leaving the "co-orbital region".

   **In many papers the exit test is looser than the entry test.** Particles that become horseshoes or quasi-satellites, or that stop librating while staying near a_J, still count as Trojans.
2. **Horseshoes and quasi-satellites are handled inconsistently.**
   - Excluded: Levison+1997, Lykawka & Horner 2010, Kortenkamp+2004 (counted as lost), Vokrouhlický+2024 (|σ| window).
   - Kept as separate classes: Horner & Evans 2006, Lykawka+2009, Greenstreet+2020/2024, Namouni (who adds compound T–QS orbits).
   - Counted as Trojans: Pirani+2019a, whose L4 excess is generated on the horseshoe side and then frozen into tadpoles.
3. **L4/L5 assignment.** Four approaches are in use:
   - **Snapshot sign of σ or side of Jupiter:** Pirani+2019a, Montesinos+2020, and the Vokrouhlický+2024 selection.
   - **Hemisphere held over a running window** several libration periods long: Greenstreet+2020/2024.
   - **Libration centre:** Lykawka+2009 (60±5°, 300±5°, 180±5° for horseshoe) and Brož & Rozehnal 2011 (χ = ±60° in the proper-element fit).
   - **Starting swarm label:** Di Sisto, Hellmich, Hou, Li+2023.

   No paper found defines membership as the mean of σ over a libration cycle in those words. For tadpoles, the libration centre and the hemisphere are equivalent. Namouni, Christou & Murray 1999 show that at large e and I the L4 and L5 centres shift and are individually asymmetric, so "σ near ±60°" is only approximate.
4. **"Greek" vs "Trojan"** is purely the IAU/MPC naming convention: L4 objects are named for Greeks and L5 objects for Trojans. (624) Hektor (L4) and (617) Patroclus (L5) are pre-convention exceptions. No dynamical definition beyond L4 = leading, L5 = trailing was found.
5. **What models compare against.** Targets range over:
   - 1.3 ± 0.1: MPC H < 12 complete sample with Eurybates removed (Nesvorný+2013).
   - 1.4 ± 0.2: WISE D > 10 km (Grav+2011; Pirani+2019a,b; Bottke+2023).
   - 1.4 ± 0.1: MPC H < 12 recount (Pirani+2019a).
   - 1.6 ± 0.1: SDSS H < 13.8 (Szabó+2007; Li+2023).
   - ≈1.8: D > 2 km (Nakamura & Yoshida; Montesinos+2020) or raw catalogue counts.
   - The f_45 = N4/N5 debiased fit of Vokrouhlický+2024.

   The observed samples are built from osculating catalogue boxes or snapshot σ, sometimes with a short (1 Myr) libration filter. None applies a 4 Gyr survival requirement. Nesvorný+2013 is the only capture paper found that both filters its model population by 4 Gyr stability and compares against a stated, complete, family-corrected observed sample.
6. **Definitions of temporary and metastable objects.**
   - Horner & Evans 2006: temporary captures last 10–100 kyr.
   - Karlsson 2004: two catalogue "Trojans" are "falsely identified".
   - Greenstreet+2024: "'metastable' objects are here defined by undergoing many resonant argument librations before exiting the co-orbital state". Of the catalogue "Trojans", 27 co-orbitals are metastable (12 L4 and 4 L5 Trojans) and 14 MPC/JPL "Trojans" are non-resonant.
   - Bottke+2023: "The remaining unstable Trojans are likely to be temporary captures from the Jupiter family comet population".
7. **Hou, Scheeres & Liu 2016** (CeMDA 125, 451) is a **planar** study of the L4/L5 dynamical symmetry under Saturn during migration. It explicitly does not study vertical motion, so it does **not** pre-empt an inclination-dependent (cold-L4) result. Its conclusion is that the "Current observed asymmetry is only a random result if these mechanisms are the main reasons to cause it." Wang & Hou 2017 could not be located.

**Practical implication for the thesis.** A defensible comparison applies the same operational definition to the model as to the observed sample. One option is a Vokrouhlický+2024-style selection: an osculating box, 20° < |σ| < 130°, and libration over 1 Myr, with the swarm set by the hemisphere. A second option is to state explicitly that capture and loss are measured with a looser or stricter criterion, and to quantify the difference: horseshoe and QS occupancy, and objects inside the box that are not librating. The 1.67 interloper artefact recorded in the pilot shows that this matters.

---

## Compact summary table

| Paper | Context | Criterion for "Trojan" | HS/QS counted? | Time span | Observed sample compared | Key quote (source, status) |
|---|---|---|---|---|---|---|
| Murray & Dermott 1999 ch.3 | Averaged CR3BP | Tadpole: σ librates about L4/L5 inside separatrix through L3; C∈(1.5,2.5) | No (HS separate) | — | — | Book RECALLED; C-range VERIFIED in Vokrouhlický+2024: "for tadpole orbits C_min=1.5 and C_max=2.5" |
| Érdi 1978/1988 | Analytic 3D elliptic theory | Libration about L4; amplitude d is the expansion parameter | n/a | — | — | Mostly RECALLED/SECONDARY |
| Namouni 1999; Namouni, Christou & Murray 1999 | Co-orbital theory at large e, I | Modes T, H, RS/QS, plus compound and transition orbits | Separate classes; transitions occur | 200 kyr (NCM survey) | NEAs | NCM VERIFIED: "effective Lagrangian points L4 and L5 can be displaced appreciably" |
| Christou 2000 | Terrestrial transient co-orbitals | Namouni modes over time | Yes, as modes | ? | NEAs | SECONDARY/RECALLED |
| Morais & Namouni 2016 | Stability maps (MEGNO) | φ librating about ±60° (T), 180° (H), 0 (RS) | Separate | 5×10⁴ periods | — | VERIFIED: "we will not attempt to identify coorbital mode transitions or compound orbits" |
| Milani 1993 | Synthetic proper elements | Catalogued; d, D, e_p, sin I_p; stability of elements over ~1 Myr | n/a | ~1 Myr | Then-known Trojans (174 vs 147, sources conflict) | SECONDARY |
| Beaugé & Roig 2001 | Semi-analytic proper elements | Catalogued; a*, e*, I* | n/a | — | Numbered + multi-opposition | SECONDARY (via Di Sisto+2014) |
| Levison, Shoemaker & Shoemaker 1997 | Gyr stability, D–e_p grid | Remains in L4/L5 libration; exits if it visits all 4 quadrants of the rotating frame or is ejected | **HS excluded** | 1 Gyr (test particles), 4 Gyr (36 real) | 178 known; ~12% unstable | Preprint VERIFIED: "If a particle visited all four quadrants ... it was either in a nonresonant or a horseshoe orbit and was no longer a member of the Trojan swarms" |
| Marzari+2002 (Ast. III) | Review | — | — | — | — | Not fetched (R) |
| Robutel+2005; Robutel & Gabern 2006 | Frequency-map analysis, L4 | Diffusion index of libration frequency; escape = leaving co-orbital region | Not addressed | 10 Myr | Observed mapped onto web | VERIFIED (2009 method paper): "escaping the co-orbital region before the end of the integration (10 My)" |
| Tsiganis, Varvoglis & Dvorak 2005 | Lyapunov/escape maps (D,e) | "Effective stability region" | ? | 4 Myr / 4.5 Gyr | 667 numbered; ~17% unstable | SECONDARY |
| Nesvorný & Dones 2002 | Stability, J/S/U/N | Tadpole about σ_c(e); removed when released from co-orbital space | No | 7.7 Myr; 4 Gyr (Neptune) | Observed amplitudes ≲40° | VERIFIED: "we will only utilize the 25 bodies surviving on tadpole orbits" |
| Brož & Rozehnal 2011 | Resonant proper elements, families | d amplitude; L4/L5 via χ=±60° in fit | n/a | 1 kyr / 1 Myr filters | AstOrb 2010: 2647 L4, 1496 L5 | VERIFIED: "χ = ±60° for L4 or L5 respectively" |
| Holt+2020 (MNRAS 495, 4085) | 4.5 Gyr N-body of catalogue + clones | Entry: 4.6<a<5.5, e<0.3 & in AstDyS proper list; Exit: osculating a<4.6 or >5.5 au | Not separated | 4.5 Gyr | JPL SBDB 2018 | VERIFIED: "We define these escapes as occurring once the test particle obtains an osculating semi-major axis of less than 4.6 au or greater than 5.5 au" |
| Marzari & Scholl 1998 | Capture by Jupiter mass growth | Amplitude = max−min of critical argument over 5×10⁴ yr; side at capture | HS = pre-capture state | Until full mass | 223 Bowell Trojans (amplitudes) | VERIFIED (ADS scan): "≃40% of planetesimals, initially on horseshoe-type orbits ... captured as L4 or L5 Trojans" |
| Fleming & Hamilton 2000 | Analytic growth/migration | A = total angular extent of φ | HS→tadpole | — | Ratio ~2 discussed | VERIFIED: "A, is defined to be the total angular extent of these oscillations" |
| Kortenkamp, Malhotra & Michtchenko 2004 | Neptune Trojans, migration | Removed once amplitude >180° | HS = lost | ≤5τ | — | VERIFIED: "removed particles ... if their libration amplitude exceeded 180°" |
| Morbidelli+2005 | Chaotic capture (Nice) | Not verified (full text blocked) | ? | ? | Trojan mass ~10⁻⁵ M_E | RECALLED/SECONDARY |
| Lykawka+2009 | Neptune Trojans | Libration centre 60±5° (L4), 300±5° (L5), 180±5° (HS); amplitude max−min over clones, 10 Myr | HS separate | 10 Myr | Known Neptune Trojans | VERIFIED |
| Lykawka & Horner 2010 | Capture by all giants | Libration about L4/L5 ≥0.5 Myr (fast) / 1 Myr (slow), RESTICK | **HS excluded**; QS n/m | 5–10 Myr, then 4 Gyr | MPC ≥2-opposition (elements only) | VERIFIED: "these bodies [horseshoe] were not taken into account" |
| Nesvorný, Vokrouhlický & Morbidelli 2013 | Jumping Jupiter | Liberal selection at 10 Myr → 100 Myr → **4 Gyr survival** ("stable Trojans"); loses ~50%, mostly D>60° | Not stated | 10 Myr+100 Myr+4 Gyr | **MPC H<12 complete: 361/279 → 1.3±0.1** (Eurybates removed); D, e, i distributions; H<9 calibration | VERIFIED: "shaved off about 50% of Trojans ... typically had D > 60°, large e and/or large i" |
| Roig & Nesvorný 2015 | Belt under jumping Jupiter | Region box only | n/a | 500 Myr | — | VERIFIED |
| Gomes & Nesvorný 2016 | Neptune Trojans | Blocked | ? | to present | 11 known Neptune Trojans (snippet) | Mostly unverified |
| Pirani+2019a (A&A 623, A169) | Inward migration in gas | **q and Q within a_J ± 2.5 R_H** (no libration test); side-of-Jupiter counting | **Yes, HS and occasionally QS** | 5–200 Myr | Grav 1.4±0.2; Szabó 1.6±0.1; N&Y 1.8±0.57; MPC H<12 → 1.4±0.1 | VERIFIED: "We consider as Trojans all those particles that have aphelion and perihelion within ±2.5 Jupiter Hill radii ... tadpole ... horseshoe ... and, occasionally, quasi-satellite orbits" |
| Pirani, Johansen & Mustill 2019b (A&A 631, A89) | Stirred disc + embryos, 4.5 Gyr | Not restated (presumably as 2019a) | Presumably | 4.5 Gyr | Grav 1.4±0.2 | VERIFIED quote of target ratio |
| Li+2023a (A&A, jumping Jupiter) | Pre-built symmetric swarms | 2000 yr pre-run picks tadpole librators; label = start swarm; "survived" after 1 Myr (rule not stated) | HS = loss route | 1 Myr (→1 Gyr) | Szabó 1.6 as "unbiased value" | VERIFIED: "the 2000 yr evolution is long enough to identify the tadpole motion" |
| Li+2023b (free-floating planet) | As 2023a | As 2023a | — | 1 Myr | ~1.6 | VERIFIED |
| Montesinos+2020 | Gas+dust hydro | **Snapshot box** ±60°±22.5° azimuth × r_p±1 R_H | HS excluded geometrically | 10⁴ yr | ≈1.8 (D>2 km) | VERIFIED |
| Chanut, Winter & Tsuchida 2008; Kortenkamp & Hamilton 2001 | — | Not located/abstract only | — | — | — | Not verified |
| Nesvorný 2018 ARA&A | Review | Trojans δλ~±60°, D up to ≈70°; P_capture (stable Trojan)=(5±3)×10⁻⁷ | — | Gyr | D, e, i of known Trojans | VERIFIED |
| Di Sisto, Ramos & Beaugé 2014 (Icarus 243, 287) | 4.5 Gyr of 2972 numbered | Catalogue membership; escape = encounter < 1 R_H | No (tadpole only) | 4.5 Gyr | All numbered (Mar 2013), 1975 L4/997 L5 | [V]: "Each integration was stopped if the particle suffered a close encounter with a planet (i.e. mutual distance smaller than the corresponding Hill radius)"; L4 23%, L5 28.3% escape |
| Di Sisto, Ramos & Gallardo 2019 (Icarus 319, 828) | Same sample, Venus–Neptune | Escape = first encounter < 2.1 R_H | No | 4.5 Gyr | Same | [V]: L4 23.6%, L5 25.1% (≠2014's 28.3%: definition-sensitive) |
| Hellmich+2019 (A&A 630, A148) | Yarkovsky, 98304 clones | Escape = r<3 or >8 au or <2 R_H of Jupiter; no libration test | No | 1 Gyr | Numbered+multi-opp. 3642 L4 / 1925 L5 (AstDyS) | [V]: escaped L5/L4 = "1.106 ± 0.026" |
| Hou, Scheeres & Liu 2016 (CeMDA 125, 451) | Planar symmetry under migration | Stable = orbit "does not intersect with the x-z plane"; label from initial grid | No (only noted in intro) | 5 Myr; 10⁶ yr runs | — | [V, Springer page]: planar; "Current observed asymmetry is only a random result if these mechanisms are the main reasons to cause it." |
| Wang & Hou 2017 | — | Not located | — | — | — | — |
| Horner & Evans 2006 (MNRAS 367, L20) | Centaur temporary capture | a within 1 R_H of planet's a + Trojan-like in rotating frame for ≥800 P (≈9.5 kyr for J) | HS counted separately (J: 2 L4/5 L5/3 HS) | 3 Myr | — | [V]: "a minimum number of 800 orbital periods was required" |
| Karlsson 2004 (A&A 413, 1153) | Transitional/temporary objects | ±1 kyr integration of 1233 objects, 4.6–5.8 au | Yes (looks for HS transitions) | ±1 kyr | MPC catalogue Trojans | [V abstract]: "Two objects, 2000 HR24 and 2002 GP31, are falsely identified as Trojans" |
| Greenstreet+2020 (AJ 160, 144) | Transient co-orbitals from NEOs/Centaurs | Running window (5–9 kyr): ⟨a⟩ within 0.4 au of a_J, all a within 3.5 R_H; L4/L5 = σ stays in hemisphere all window; HS = crosses 180°; else QS | Yes, separate | up to Gyr | — | [V]: "If φ₁:₁ remains in the leading or trailing hemisphere for the duration of a running window, we assign the particle to the L4 or L5 state" |
| Greenstreet, Gladman & Jurić 2024 (ApJL 963, L40) | 11,581 real objects × 1000 clones | 1 kyr windows: ⟨a⟩ within 0.2 au, all within 0.65 au; "Trojan" = ≥95% clones resonant all 0.5 Myr; "transient" = ≥95% resonant ≥1 kyr then leave; metastable = many librations then exit | Yes: 2 HS, 8 QS among 27 metastable | 0.5 Myr (to 2 Gyr for transients) | JPL SBDB 4.5–5.9 au: 7482 L4 / 3941 L5 stable; 14 MPC/JPL "Trojans" non-resonant | [V]: "'metastable' objects are here defined by undergoing many resonant argument librations before exiting the co-orbital state" |
| Vokrouhlický+2024 (AJ 167, 138), observational reference | Debiased CSS model | 4.95<a<5.45, e<0.6, I<50°, 20°<\|σ\|<130° (snapshot, MJD 60000) + σ librating for 1 Myr; C∈(1.5,2.5) tadpole | Excluded by σ window | 1 Myr filter | MPC Feb 2023: 7903 L4 / 4177 L5; f₄₅=N4/N5 | [V]: "absolute value of the resonant angle σ larger than 20° and smaller than 130°" |
| Bottke+2023 (SSRv 219, 83) review | Origin review | L4/L5 librators in 1:1; most stable, some unstable = temporary JFC captures | — | — | D>100 km: 9 vs 9; D>10 km: 1.4±0.2 | [V]: "The remaining unstable Trojans are likely to be temporary captures from the Jupiter family comet population" |
| Di Ruzza, Pousse & Alessi 2023 (Icarus 390, 115330) | Averaged planar CR3BP map, ~600 yr ephemerides | TP / HS / QS / compound domains from integrable model | Yes, separate | ~600 yr (Jupiter) | JPL Horizons low-I objects | [V abstract] |
| JPL SBDB TJN / MPC type 12 | Catalogue classes | TJN: 4.6<a<5.5 au, e<0.3; MPC: 4.8<a<5.4, e<0.3 | Not distinguished | snapshot | — | [V]: "Asteroids trapped in Jupiter's L4/L5 Lagrange points (4.6 au < a < 5.5 au; e < 0.3)" |
| IAU naming (Greek/Trojan camps) | Convention | L4 = Greek names, L5 = Trojan names; Hektor (L4), Patroclus (L5) exceptions | — | — | — | Secondary sources; MPC L-column verified for Hektor/Patroclus |

---

# Part A: Classical theory and stability-map definitions of a "Jupiter Trojan"

Tags: **[VERIFIED]** means the quote was read in fetched full text or an abstract during this session (source given). **[SECONDARY]** means the claim comes from another paper's description, which was fetched. **[RECALLED]** means it is from memory and was not checked against the source. ADS, Springer, HAL, ScienceDirect and AstDyS all blocked automated access (bot walls), so several primary abstracts could not be read directly.

Working texts were extracted to a temporary folder (holt2020.txt, nd2002.txt, lss96c.txt, rb2009.txt, br2011.txt, ncm1999.txt, mn2016.txt, milani_ssc.txt [Pilat-Lohinger+1999], emery2015.txt, coorb.txt [Leleu thesis], ds2014.txt).

---

## 1. Classical / analytical theory

### Murray & Dermott 1999, *Solar System Dynamics*, ch. 3 (§3.8–3.9)
- **[RECALLED]** They use the averaged circular restricted problem with an energy-like integral in (σ = λ − λ', δa). Tadpole orbits librate about L4/L5. Horseshoe orbits enclose L3, L4 and L5. The boundary is the separatrix through L3. The limiting tadpole has a minimum angular separation from the planet of about 24°; horseshoe orbits come closer. I could not fetch the book, so equation numbers are not given.
- **[VERIFIED, equivalent statement elsewhere]** Leleu thesis (arXiv:1701.05585, after Robutel & Pousse 2013): "The tadpole domain lie within the black separatrix emanating from L3, hence for ζ0 ∈ ]ζs, 60°[ with ζs ≈ 23.9°. This domain has a maximum width of the order of ε^1/2 ... The horseshoe domain lies outside the separatrix, with a maximum width of the order of ε^1/3." Here ζ0 is "the minimal value of ζ on its trajectory".
- **[VERIFIED, same integral in observational paper]** Vokrouhlický+2024 (repo PDF, Eqs. 3–4): (da/a_J)² = (8µ/3)[C − f(σ)], with f(σ) = (1+4|sin σ/2|³)/(2|sin σ/2|). "for tadpole orbits C_min = 1.5 and C_max = 2.5", and "da_P ≃ a_J √(8µΔC/3)". Note that f(60°) = 1.5 is the L4 centre and f(180°) = 2.5 is the L3 separatrix. So C = 2.5 is the same as the 23.9° separatrix: **C < 2.5 is the classical tadpole (= Trojan) criterion.**
- **Implication for a Trojan definition:** in classical theory a Trojan is any tadpole orbit, i.e. σ librates about ±60° without reaching 180°. Horseshoes are a separate class. There is no time-span requirement because the circular restricted problem is integrable once averaged.

### Érdi 1978 (CeMech 18, 141), Érdi 1988 (CeMech 43, 303)
- **[VERIFIED abstract, IAU Symp. 172 text by Érdi]** "The motion of the Trojans is considered in the model of the three-dimensional elliptic restricted three-body problem ... An asymptotic solution is derived, by applying the multiple-timescale method". This is a perturbation theory around L4. It is not a membership criterion: Trojans are assumed to be librators about L4/L5.
- **[SECONDARY, Emery+2015 review]** Érdi (1988) gives g ≃ (27/8 + (129/26)d²)µn_p "at the second order in the libration amplitude d". So d (the angular libration amplitude, in radians) is the theory's small parameter.
- **[RECALLED]** Érdi 1978 treats 3D motion with small libration amplitude, e and i. I did not find a dynamical "is-a-Trojan" threshold in it.

### Namouni 1999 (Icarus 137, 293) and Namouni, Christou & Murray 1999 (PRL 83, 2506; astro-ph/9904016)
- **[SECONDARY, Namouni 1999 abstract via search summary]** Three orbit families matter in the co-orbital problem with close encounters: horseshoe, passing, and retrograde-satellite (quasi-satellite). In 3D, "orbit transitions between different families occur owing to the secular variations of the eccentricity and argument of pericenter."
- **[VERIFIED, NCM 1999 full text]**
  - "all the known Trojan objects are librating about L4 or L5 points and their orbits are often referred to as tadpoles (or T orbits)".
  - "When the angular amplitudes of the leading and trailing tadpoles are sufficiently large, the orbits merge near the unstable L3 point ... referred to as horseshoes (or H orbits)."
  - On large e and I: "the effective Lagrangian points L4 and L5 can be displaced appreciably from the equilateral configuration ... they no longer correspond necessarily to the same Jacobi constant, which fact implies an asymmetry in the distribution of T orbits."
  - "compound and transition orbits ... The first correspond to the merger of H or T orbits with RS orbits ... The second ... are not long-lived ... Transition orbits therefore appear in the orbital evolution only to permit the passage between different orbit families."
  - Fig. 2 caption: "Note the asymmetry in the location and the maxima of L4 and L5. Because S(λr, e, I, −ω) = S(−λr, e, I, ω), the coorbital motion also includes orbits that are symmetric".
  - Survey span used: "The integration time-span was set to be 200,000 years centered on the present".
- **Relevance:** at high e and I the libration centre is not at ±60°. Snapshot "σ near ±60°" is therefore not exact, and a particle can be a tadpole, a horseshoe, a quasi-satellite or a compound (T-RS, H-RS) orbit at different times. The individual L4/L5 asymmetry depends on ω, but the mirror orbit makes it vanish on average.

### Christou 2000 (Icarus 144, 1; transient co-orbitals of the terrestrial planets)
- **[SECONDARY only]** Cited for showing that Earth/Venus co-orbitals are dynamically transient and that NEAs pass between co-orbital modes. I could not fetch the paper's own classification criterion.
- **[RECALLED]** He classifies by the behaviour of λ − λ_planet over the integration (T, H, QS/RS, compound) using the Namouni framework. **Not verified.**

### Morais & Namouni (2013 CeMDA 117, 405; 2016 CeMDA 125, 91 = arXiv:1602.04755)
- **[VERIFIED, 2016 full text]**
  - "three possible 2D prograde coorbital modes: φ librating around 0 (retrograde-satellite orbits: RS), ±60° (tadpole orbits: T) or 180° (horseshoe orbits: H)", where φ = λ − λ_J.
  - 3D list: "T: asymmetric libration of φ around ±60° at e ≲ 0.4 and I ≲ 60°"; "H: libration of φ around 180° at e > 0.8 and I ≲ 70°" (stability maps for µ = 0.001, run 5×10⁴ periods, MEGNO).
  - "retrograde satellites combine with tadpoles or horseshoes to form the compound orbits, H-RS, RS-H, T-RS, RS-T and T-RS-T."
  - In their capture tests: "we will not attempt to identify coorbital mode transitions or compound orbits."
  - Escape and collision rule: "a collision occurs when the minimum distances to the primary or planet are less than 0.005 or 0.0005 ... Escape occurs when the barycentric distance exceeds the value 3."
- **Note:** they classify modes by which centre φ librates about. "Tadpole" is the mode with ±60° centre and is the only one called Trojan.

---

## 2. Proper elements and stability maps

### Milani 1993 (CeMDA 57, 59): "The Trojan asteroid belt: proper elements, stability, chaos and families"
- **[SECONDARY, search-result abstract fragments]**
  - About 1 Myr integrations with the 4 giant planets, then digital filtering into a "synthetic theory".
  - The libration amplitude "cannot be determined by a simple Fourier transform ... I therefore give another definition, which results in very good stability with time."
  - "For 87% of the computed orbits, the stability of the proper elements—at least over 1 Myr—is within" bounds (about 0.001 AU in d, 0.0025 in e_p and sin I_p).
  - The sample is reported as 174 Trojans ("numbered and multi-opposition"). Di Sisto+2014 says "147 bodies". **This number conflicts between sources; check the PDF.**
- **[VERIFIED, Pilat-Lohinger, Dvorak & Burger 1999, CeMDA 73, 117, using "a numerical method similar to the one by Milani (1993, p. 61ff)"]**
  - "proper elements d (the libration amplitude of the semi-major axis, i.e. a − aJ = d sin θ + ..., where θ is the argument of libration), D (the libration amplitude of the critical argument, i.e. λ − λJ = χ + D cos θ + ..., where χ denotes the position of the libration point), ep ... and sin(Ip)".
  - Pilat-Lohinger+ themselves integrated 100 Myr. Thersites "escaped after 26 million years".
- **[VERIFIED, Brož & Rozehnal 2011 footnote]** "Equivalently, we may compute the amplitude D of mean longitudes λ − λ′. Anyway, there is a linear relation between d and D."
- **Membership:** Milani's set is the bodies already catalogued as Trojans with reliable orbits. Proper-element "stability" here is a quality flag, not a membership test. AstDyS Trojan proper elements (Knežević & Milani) continue this. **[SECONDARY, Holt+2020]** "objects in this list require the completion of simulations spanning 1 × 10^6 years to generate the proper elements".

### Beaugé & Roig 2001 (Icarus 153, 391)
- **[SECONDARY]** Semi-analytical, asymmetric expansion of the Hamiltonian near the 1:1 resonance, with averaging and adiabatic invariants. Families found in L4 (Menelaus, Eurybates), and "No significant accumulations were found around L5" (Di Sisto+2014, verified).
- **[VERIFIED, Di Sisto+2014]** Their proper elements are written a*, e*, I*. Di Sisto+2014 compares them with Brož & Rozehnal and says "Since the definition of both sets are not exactly equal, we do not expect an exact linear trend". So **a* (B&R) and ā+d (Milani-style) are different conventions.**
- **[RECALLED]** a* is a proper semimajor-axis-like quantity tied to libration amplitude (roughly a_J + d_p). Input was the then-catalogued Trojans (numbered plus multi-opposition). The exact definition is not verified.

### Levison, Shoemaker & Shoemaker 1997 (Nature 385, 42) — the key operational definition
- **[VERIFIED, author preprint PostScript, boulder.swri.edu/~hal/PS/lss96.ps]**
  - Definition: "Trojans are asteroids that orbit the Sun about the L4 and L5 Lagrange points of Jupiter and are locked in the 1:1 mean motion resonance with Jupiter."
  - **Exit criterion (verbatim):** "An orbit was deemed unstable if the particle either suffered a close approach with Jupiter and was ejected from the solar system or visited all four quadrants of a coordinate frame that rotates with the Sun-Jupiter system. If a particle visited all four quadrants of such a frame than it was either in a nonresonant or a horseshoe orbit and was no longer a member of the Trojan swarms." **Horseshoes are explicitly excluded.**
  - D definition: "If φ is defined as the instantaneous difference between the mean longitudes of the asteroid and Jupiter, then the libration amplitude, D, is the amplitude of oscillations in φ. The proper eccentricity, ep, is defined as the amplitude of the oscillation in e cos(ϖ − ϖJ)". "(The actual location of these fixed points are a function of the proper eccentricity.)"
  - Grid and time span: 270 fictitious L4 particles, "27 values of ep ranging from 0 to 0.8 and 10 values of D between 0 and 140°", integrated "for 10^9 years or until they escaped". Also "36 real Trojan asteroids near the stability boundary" up to 4×10^9 yr: "Of the 36 objects studied, 21 were unstable in less than 4 billion years."
  - Observed comparison: "A total of 178 known Trojans are shown in Figure 1. Of these, 21 (12%) are unstable over a 4 billion year time interval". Orbits were supplied by E. Bowell and L. Wasserman (Lowell).
  - "the observed upper boundary in ep–D space occupied by Jupiter Trojans is time dependent ... it has been slowly shrinking with time."
  - Only L4 was sampled.
- **Discrepancy:** the SwRI web abstract says "over 200 escaped Trojans". The preprint abstract and body say "over 1200" (N_tot = rL = 1200). Check the printed Nature text before quoting a number.

### Marzari, Scholl, Murray & Lagerkvist 2002 (Asteroids III, 725–738)
- **[SECONDARY, Padua record]** Covers capture via Jupiter's mass growth, collisional diffusion and gas drag "in stable Trojan orbits".
- **[RECALLED, not verified]** The chapter reviews the D–e_p stability boundary of Levison+1997 and the analytic theory. I could not fetch the full text (LPI/UA Press), so **there is no verified quote.**

### Robutel, Gabern & Jorba 2005 (CeMDA 92, 53) and Robutel & Gabern 2006 (MNRAS 372, 1463)
- **[VERIFIED abstracts, zeus.maia.ub.edu preprint pages]**
  - RGJ: "a systematic study of the global dynamical structure of the Sun–Jupiter L4 tadpole region ... based on long-time simulations of the Trojans in the Sun, Jupiter, Saturn system and on the frequency analysis ... we are able to connect this global view of the phase space with the observed Trojans".
  - RG2006: "the slow transport mechanism that drives Trojans from the inner stable region to eventual ejections ... we relate this global view of the dynamics with the observed Trojans".
- **[VERIFIED, Robutel & Bodossian 2009, Paper II, arXiv:0809.3526, method "as it was established in Paper I"]**
  - "the escape rate: the number of Trojans escaping the co-orbital region before the end of the integration (10 My), divided by the initial number of Trojans".
  - "σν = (ν1 − ν2)/ν1, called diffusion index ... ν1 is the libration frequency computed on the first 5 My, while ν2 is calculated on the following 5 My ... blue ... stable regions (σν < 10^−6), to red for very chaotic regions (σν > 10^−2). In black, we display the particles that have been ejected from the Trojan swarms".
  - The precise geometric "escaped the co-orbital region" test is **not stated** in Paper II.
- Only L4 is sampled. Initial conditions are a grid in (a, e) at fixed angles, with the symmetry that lets them use a ≥ a_J, e ≥ e_J.
- **[SECONDARY, Emery+2015]** "Superposition of family I resonances is responsible for the large chaotic zone extending beyond 5.35 AU limiting the libration amplitude of Trojan orbits."

### Tsiganis, Varvoglis & Dvorak 2005 (CeMDA 92, 71): "Chaotic diffusion and effective stability of Jupiter Trojans"
- **[SECONDARY, search-engine rendering of the abstract; primary not fetched]**
  - About 3200 fictitious objects and 667 numbered Trojans integrated for 4 Myr with Lyapunov times. Chaotic orbits were followed to 1 Gyr.
  - An "effective stability region" was defined on Lyapunov-time and escape-time maps. 86% of numbered Trojans lie inside it and 14% outside.
  - A 4.5 Gyr run of the chaotic numbered Trojans found ~17% unstable. Holt+2020 also quotes "Tsiganis et al. (2005a) estimating 17%" **[VERIFIED in Holt]**.
- **[RECALLED]** Escape means a close encounter with Jupiter or leaving the libration region. The criterion was not verified.

### Nesvorný & Dones 2002 (Icarus 160, 271)
- **[VERIFIED, SwRI PDF]**
  - L4/L5 symmetry assumption: "the L4 and L5 points show no asymmetry concerning the dynamics and stability of orbits around them. For this reason, we chose to sample only the orbits around the L4 point, expecting the dynamics around the L5 point to be a mirror image".
  - Initial σ: "We initially set σ ≡ λ − λj = σc(e), where σc(e) is the center of the tadpole orbits. This center is near 60° for e ≈ 0 and has larger values for larger e (Namouni 1999)."
  - **Removal criterion:** "A test body was removed from the simulation if it was released from the coorbital space of a planet. Such a body is easily recognized because its orbit rapidly evolves to a ≠ aj by encounters with planets, which can be checked by its heliocentric distance." Survey spans: "7.7, 18.6, 39.4, and 87.1 Myr" for J, S, U, N. Bi-circular surveys: "The integration time span was 10^5 years. The test body was eliminated if it became a planet-crosser and was not phase protected from a collision."
  - Horseshoes: "the horseshoe orbits of Jupiter are unstable in this simple" model. Fig. 3 marks "limits of tadpole orbits ... and the limits of horseshoe orbits ... [and] distant retrograde satellites (e.g., Namouni 1999)". Only tadpoles are treated as Trojans.
  - Libration amplitude as used (quoting Weissman & Levison): "libration amplitudes (i.e., the full angular extent of a tadpole orbit)". **Note: this is the full extent, not the semi-amplitude, so it is a different convention from D.**
  - Observed comparison: "Trojan asteroids are mostly located in this region, usually having libration amplitudes ≲ 40°", and "To span the inclination distribution of the observed Trojans, whose mean inclination is ≈18°, we also ran an additional integration for Jupiter with i = 30°."
  - Long runs (Neptune): "From 27 bodies surviving over 4 Gyr, 25 bodies move on tadpole orbits at the end of the integration ... we will only utilize the 25 bodies surviving on tadpole orbits." **So survival alone is not membership; the final state must also be tadpole.**

### Brož & Rozehnal 2011 (MNRAS 414, 565; arXiv:1109.1109)
- **[VERIFIED, arXiv full text]**
  - Proper elements: "resonant proper elements: libration amplitude d of the a − a′ oscillations ... eccentricity e and inclination sin I ... We employ their definition from Milani (1993)."
  - **L4/L5 assignment (explicit):** "φ(t) = λ − λ′ − χ, where λ, λ′ are the mean longitudes of an asteroid and Jupiter and χ = ±60° for L4 or L5 respectively. The revolution of angle φ(t) must not be confined to the interval [0, 360°)". Frequency comes from a linear fit over a 1 kyr buffer, amplitude from a DFT, smoothed with a 1 Myr running average.
  - "Equivalently, we may compute the amplitude D of mean longitudes λ − λ′ ... there is a linear relation between d and D."
  - Sample: "The source of initial osculating elements is the AstOrb catalogue ... (Oct 31st 2010)". "We computed proper elements for 2647 L4 and 1496 L5 Trojan asteroids ... N(L4)/N(L5) ≃ 1.8" (for H ≲ 15).
  - "one-apparition orbits for the purposes of physical studies ... orbital studies require more precise multi-apparition data."
  - The catalogue's Trojan preselection rule is not stated in the paper.

### Holt et al. 2020 (MNRAS 495, 4085; arXiv:2005.03635)
- **[VERIFIED, arXiv PDF]**
  - **Definitional statement:** "many of those objects may be temporarily captured objects, rather than permanent members of the Trojan population. Whilst the 'true' Trojans move on stable orbits that keep them librating around the L4 and L5 Lagrange points on billion year timescales (e.g. Emery et al. 2015), temporarily captured objects would be expected to escape from the Trojan swarms on timescales of thousands or tens of thousands of years. To confirm that a given object is truly a member of the Trojan population requires confirmation that the object's proper orbital elements (Milani & Knežević 1992) are stable, and that the object is truly trapped in 1:1 resonance with Jupiter."
  - **Sample:** "selecting all objects with orbital semi-major axes between 4.6 au and 5.5 au and an orbital eccentricity less than 0.3. This process yielded an initial selection of 7202 objects, obtained on 17th April, 2018 ... We then filtered our sample to discard temporarily captured objects by limited selection to those objects present in the AstDys proper element database ... we were left with a total of 5553 nominally 'stable' Trojans ... 4780 numbered and 773 multioppositional objects."
  - **Escape criterion (purely a-based, no σ test):** "We define these escapes as occurring once the test particle obtains an osculating semi-major axis of less than 4.6 au or greater than 5.5 au." Span is 4.5 Gyr (Fig. 3 caption), with output every 1×10^5 yr ("escape times are only accurate" to that).
  - Results: L4 23.35% and L5 24.89% escape. Primordial populations ×1.31 and ×1.35. The "current 1.89 numerical asymmetry" (1.56 by volume).
  - Horseshoe/QS are not separated; anything inside 4.6–5.5 au counts as still in.
  - L4/L5 assignment is taken from the catalogue (presumably by σ sign). **[RECALLED/inferred, not stated in the passages read]**

---

## 3. Cross-cutting observations (Part A only)

1. **Three operational definitions coexist:**
   - (a) **Tadpole libration of σ about the (e-dependent) L4/L5 centre**, excluding horseshoes. This is classical theory, Levison+1997, Nesvorný & Dones 2002 and Morais & Namouni.
   - (b) **Having catalogued resonant proper elements** (Milani 1993, AstDyS, Brož & Rozehnal 2011). Here "stable proper elements over ~1 Myr" is the filter (Holt+2020).
   - (c) **Staying inside an a-window or the co-orbital region** (Holt+2020: 4.6 < a < 5.5 au; Nesvorný & Dones: "released from the coorbital space"; Robutel+: "escaping the co-orbital region"). These loss criteria are looser than the entry criterion.
2. **Horseshoe treatment:** only Levison+1997 explicitly reclassifies horseshoes as non-Trojans (four-quadrant test). The others either show horseshoes are unstable for Jupiter's mass ratio (Nesvorný & Dones) or lump them in via an a-window (Holt).
3. **Libration amplitude conventions differ:**
   - D is the semi-amplitude of λ − λ_J (Levison, Milani, Pilat-Lohinger).
   - d is the amplitude in a − a_J (Milani, Brož & Rozehnal; linear in D).
   - "Full angular extent" is used by Weissman & Levison as quoted in Nesvorný & Dones.
   - C ∈ (1.5, 2.5) is the energy-like constant (Vokrouhlický+2024; C = 2.5 is the separatrix, about 23.9° minimum separation).
   - a* is used by Beaugé & Roig.
   - Thresholds must be converted before comparison.
4. **L4/L5 assignment:** Brož & Rozehnal set χ = ±60° per swarm. Nesvorný & Dones put the centre at σc(e) > 60° for larger e. Namouni+1999 show L4 and L5 centres are displaced and asymmetric for given (e, I, ω), and symmetric under ω → −ω. None of the Part-A papers defines swarm membership by a time-averaged σ explicitly. The closest is the libration-centre fit (Brož & Rozehnal). Levison's quadrant test is the only explicit time-history test.
5. **Time spans:**
   - About 1 Myr for proper elements (Milani, AstDyS, Brož & Rozehnal's 1 Myr smoothing).
   - 10 Myr for frequency-map diffusion (Robutel+).
   - 4 Myr Lyapunov plus 1 Gyr and 4.5 Gyr escape runs (Tsiganis+).
   - 1–4 Gyr (Levison+).
   - 4–4.5 Gyr (Nesvorný & Dones, Holt).

---

# Part B: how capture simulations decide a particle is a Jupiter (or Neptune) Trojan

Method: I downloaded the arXiv PDFs and ran `pdftotext` on them. "VERIFIED" means the quote was copied from that extracted text. The extracted text was kept in a temporary folder only. "RECALLED" means it comes from memory and was not checked against fetched text. "NOT FOUND" means I searched the full text and the paper never states the item.

Access failures: A&A HTML and PDF (Cloudflare wall), ADS UI (HTTP 405), Nature (paywall) and HAL (bot wall) all blocked fetching. The exception is the ADS scanned-article server, which worked for Marzari & Scholl 1998 (A&A 339, 278).

---

## 1. Pirani, Johansen, Bitsch, Mustill & Turrini 2019a, A&A 623, A169 (arXiv:1902.04591)

**Context:** Jupiter's core grows by pebble accretion and migrates inward from about 18 au while gas is present, with test particles. N-body runs of 10 simulations each.

**Criterion (VERIFIED, Sect. 4.1):**
> "We consider as Trojans all those particles that have aphelion and perihelion within ±2.5 Jupiter Hill radii from the planet semimajor axis. This way we include almost all the main coorbital resonances: tadpole orbits (Trojans, that librate around ±60°), horseshoe orbits (objects that librate around 180° encompassing L4 and L5) and, occasionally, quasi-satellite orbits (asteroids that librate around 0°, i.e. around Jupiter's position), excluding interlopers with high eccentricity."

- **What the cut tests:** it is purely geometric (q and Q within a_J ± 2.5 R_H). It never tests the resonant angle or libration.
- **Horseshoe and QS orbits:** both are explicitly counted as "Trojans".
- **L4/L5 assignment:** the paper never states a rule (NOT FOUND). The mechanism is explained as the instantaneous side of the horseshoe orbit (VERIFIED, Fig. 11 caption): "The blue line is the number of particles in the L4 side of the horseshoe orbit (leading side) versus time and the green line is the number of particles in the L5 side of the horseshoe orbit (trailing side) versus time. Horseshoe orbit encompass both L4 and L4 [sic], hence an excess of leading particles means that particles spend more time in the leading part of the orbit." The asymmetry during migration is therefore a snapshot occupancy asymmetry. It survives only because mass growth "shrinks the horseshoe orbits into stable tadpole orbits" (VERIFIED).
- **Time span:** Table 2 reports N_tot and N_L4/N_L5 at 5, 10, 50, 100 and 200 Myr (e.g. 1.80 ± 0.26 at 5 Myr and 1.92 ± 0.29 at 200 Myr; VERIFIED). No multi-Gyr survival requirement.
- **Observed comparison (VERIFIED, Sect. 1):**
  - "Grav et al. (2011), with a sample detected by NEOWISE/WISE with sizes larger than 10 km, found a ratio of 1.4 ± 0.2"
  - Szabó et al. 2007 SDSS MOC, "H < 13.8m … 1.6±0.1"
  - Nakamura & Yoshida 2008, "1.8 ± 0.57 for … D > 2 km"
  - Vinogradova & Chernetenko 2015, "about 2.0 for … H < 19m"
  - Their own MPC count: "We also analysed the Trojans in the MPC with H<12 … there are 330 L4 objects and 229 L5 objects. As in Nesvorný et al. (2013), we excluded 9 Eurybates family members from L4 finding an asymmetry ratio of about 1.4 ± 0.1".
  - Their figures shade Szabó 1.6 ± 0.1, Grav 1.4 ± 0.2 and Nakamura & Yoshida 1.85 ± 0.57.
- **Number or mass constraint:** the abstract says the inclinations are near zero and the captured mass is too large. Pirani+2019b (VERIFIED) restates this as "a final mass of the Trojans that is 3-4 orders of magnitude higher than the current one".

## 2. Pirani, Johansen & Mustill 2019b, A&A 631, A89, "On the inclinations of the Jupiter Trojans" (arXiv:1910.01507)

**Context:** follow-up to 2019a covering a pre-stirred disc, embedded embryos and 4.5 Gyr evolution.

- **Criterion:** the paper does not restate a Trojan criterion (NOT FOUND for "Hill", "librat… amplitude", "criteri…"). It says the model is set up "as in Pirani et al. (2019)", so the ±2.5 R_H rule is presumably inherited. That is my inference.
- **Observed comparison (VERIFIED):** "an asymmetry ratio between the two swarms of 1.4 ± 0.2 for Trojans larger than 10 km in size (Grav et al. 2011)". Fig. 8 highlights the current ratio.
- **Time span:** the embryo runs go to 4.5 Gyr (VERIFIED: "the asymmetry ratio remains more or less constant during 4.5 Gyr of evolution").
- **Swarm assignment of embryos (VERIFIED):** the authors "completely randomised which and how many massive embryos end up in each swarm".

## 3. Li, Xia, Yoshida, Georgakarakos & Li 2023a, A&A 669?, "Asymmetry in the number of L4 and L5 Jupiter Trojans driven by jumping Jupiter" (arXiv:2211.13877)

The volume number is unverified. One web summary gave A&A 674, A138, but that may belong to Li+2023b.

**Context:** this is not a capture simulation. It assumes already-captured symmetric swarms and imposes a fast outward jump of Jupiter (swift_rmvs3).

**Setup (VERIFIED):**
- "In order to mimic the 'captured' Trojans, denoted by test Trojans hereafter, we conduct a series of pre-runs lasting 2000 yr … Since the Jupiter Trojans librate around the L4 or L5 points with periods of a few hundred years, the 2000 yr evolution is long enough to identify the tadpole motion."
- "For the L4 and L5 swarms, the values of σ0 are given by 60° + Δσ0 and −60° − Δσ0 respectively"
- "we restrict ourselves to consider only the rather stably librating particles with Δσ0 ≤ 80° in the pre-runs"
- Standard setting: "Δσ0 = 30°-80°" with 500 test Trojans per swarm.

**Rules:**
- σ = λ − λ_J (VERIFIED).
- L4/L5 assignment: by the starting swarm, since σ0 is set per swarm.
- Survival criterion at the end: NOT FOUND as an explicit rule. The authors report "the numbers (N4, N5) and the final resonant amplitudes (A4, A5) of survived Trojans" after 1 Myr. They describe an L5 Trojan that "crosses the L3 point and switches to a horseshoe orbit centered at σ = 180°" before ejection, so horseshoe transitions are treated as the route to loss.

**Time span (VERIFIED):** 1 Myr standard, extended to 1 Gyr: "R45 increases with time, from 1.50 at 1 Myr to 1.68 at 1 Gyr". They then apply a ×1.1 correction: "to achieve the asymmetry of R45 ∼ 1.6, we would require a value at the level of > 1.4 − 1.5 (i.e. ≳ 1.6/1.1) in the 1 Myr simulation."

**Observed comparison (VERIFIED):**
- "For the currently observed Jupiter Trojans, the leading-to-trailing number ratio N4/N5 is ∼ 1.8."
- Grav "N4/N5 = 1.34-1.4"
- Szabó "N4/N5 = 1.6 ± 0.1. Accordingly, in this paper we will consider N4/N5 = 1.6 as the unbiased value"
- For the observed sample they used MPC multi-opposition objects (10854 of 11552 on 2022 Jan 26) and computed resonant amplitudes by numerical integration.

**Caveat (VERIFIED, abstract):** "the resonant amplitudes of the simulated Trojans are excited to higher values compared to the current population."

## 4. Li, Xia, Georgakarakos & Yoshida 2023b, "The invasion of a free floating planet and the number asymmetry of Jupiter Trojans" (arXiv:2304.13598)

It is apparently the A&A 674 paper; the journal reference is unverified.

- **Method:** same as 2023a: Δσ0 = 30°–80°, 500 per swarm, 1 Myr (VERIFIED, Table 1 caption).
  > "the system is integrated for 1 Myr. We then derive the numbers (N4, N5) and final resonant amplitudes (A4, A5) of surviving Trojans"
- **Target (VERIFIED):** "R45 ∼ 1.6 (Szabó et al. 2007)".
- **Swarm-killing case (VERIFIED):** "at ΔaJ = 0.19 AU, R45 finally becomes infinite since some L4 Trojans can remain stable, but all the L5 ones have escaped".

## 5. Nesvorný, Vokrouhlický & Morbidelli 2013, ApJ 768, 45, "Capture of Trojans by Jumping Jupiter" (arXiv:1303.2900)

**Context:** jumping-Jupiter instability with 5×10⁷ disk particles. The run has three phases.

**Definitions (VERIFIED):**
- Phase 1, 10 Myr: "total integration timespan of 10 Myr; Phase 1"
- Phase 2, 100 Myr: "we identified Trojan 'candidates' by selecting particles with orbits near Jupiter's L4 and L5 at the end of Phase 1. This selection was very liberal in that most of the (hundreds of) selected bodies turned out not to be truly stable Trojans."
- Phase 3, 4 Gyr: "To test the long-term stability of Trojans surviving at the end of Phase 2, we performed an additional numerical integration over 4 Gyr. … We found that the long-term stability requirements shaved off about 50% of Trojans from the population that survived at the end of Phase 2. The removed particles typically had D > 60°, large e and/or large i."
- "captured Trojans that survived to the end of Phase 3 (hereafter the stable Trojans)"
- Background: "∆λ = λ − λJup ∼ ±60° … The angle ∆λ oscillates with a period of ∼150 yr and full libration amplitude up to D ≃ 70°"
- D definition (Fig. 7 caption): "The full libration amplitude D corresponds to the angular distance between extremes of λ − λJup during libration. The proper orbital elements shown here were computed by the method described in Brož & Rozehnal (2011). We found no significant difference between the orbital distributions of L4 and L5 Trojans, and clumped these distributions together."

**What is not stated:** the paper gives no explicit numeric "near L4/L5" window for Phase 1 (NOT FOUND) and no explicit L4/L5 assignment rule. "Stable Trojan" means surviving 4 Gyr past capture.

**Observed comparison (VERIFIED):**
- Count: "As there are 25 known Trojans with H < 9 (this sample is complete), this indicates that the planetesimal disk contained ∼ 25/(7 × 10−7) = 3.6 × 107 planetesimals with H < 9"
- Asymmetry: "To assess the significance of the asymmetry we find it best to use the population with H < 12 (complete sample according to Szabó et al. 2007). The number of the known L4 Trojans with H < 12 is 361, if 9 known Eurybates family members with H < 12 are removed (Brož & Rozehnal 2011). For comparison, there are 279 L4 [sic; L5 meant] Trojans with H < 12. This indicates that f45 = 1.3 ± 0.1"
- Model results: "f45 = 1.3 ±0.5 in Case 1, f54 ≡ 1/f45 = 1.4 ±0.4 in Case 2, and f54 = 1.8 ± 0.9 in Case 3"
- Distributions: D, e and i were K-S tested against the known Trojans with H < 12 (Fig. 8).
- Capture efficiency: "P = (7.0 ± 0.7) × 10−7".

## 6. Morbidelli, Levison, Tsiganis & Gomes 2005, Nature 435, 462 (M05)

Full text not obtainable (Nature paywall, HAL bot wall, no arXiv).

**Secondary, VERIFIED via NVM13 footnote 3:**
> "M05 instead reported, quoting, 'capture efficiencies per one particle cycled through the system as the planets migrate through unstable Trojan configurations'. … the capture efficiencies reported in M05, 1.8 × 10−5 to 2.4 × 10−6"

NVM13 also says "M05 estimated that the present mass of the Trojan population is MTro ∼ 10−5 MEarth."

**Secondary, VERIFIED via Nesvorný 2018:** "Chaotic capture takes place when Jupiter and Saturn pass, during their orbital migration, near the mutual 2:1 resonance".

**RECALLED, not verified:** M05 integrated captured particles past the end of migration to check long-term stability. They compared the captured objects' (D or libration amplitude, e, i) distribution with proper elements of the known Trojans, with a figure showing libration amplitude vs i, and the total mass with the ~10⁻⁵ M_E estimate. I cannot verify the exact libration or stability criterion.

## 7. Roig & Nesvorný 2015, AJ 150, 186, "The evolution of asteroids in the jumping-Jupiter migration model" (arXiv:1509.06105)

**Criterion:** region-based. There is no libration test (NOT FOUND for "resonant angle" or "critical angle").

**Verified quotes:**
- Initial Trojan set: "5.2 ≤ a ≤ 5.7 au, e < 0.2, I < 40°, for the Trojan swarms (hereafter TS set)"
- Survival: "less than 1% have survived the instability but all of them have been scattered out of the 3:2 and 1:1 mean motion resonances with Jupiter. We conclude that in an Early Instability hypothesis, the survival probability of a post Grand Tack population is … < 10−6 for the TS."
- Implantation: "For implanting TS particles, the probability is < 10−5-10−6."

**Simulation span:** 500 Myr (VERIFIED: "the simulation time span of 500 Myr"). No observed N4/N5 comparison.

## 8. Lykawka & Horner 2010, MNRAS 405, 1375, "The capture of Trojan asteroids by the giant planets during planetary migration" (arXiv:1003.2137)

**Criterion (VERIFIED):**
> "The detection routines were optimised to enable the detection of any Trojan librating around either the L4 or L5 Lagrange point that had a Trojan lifetime greater than ~0.5 Myr (fast migration) or ~1 Myr (slow migration). … Although our code was capable of detecting horseshoe Trojans for each of the giant planets, the resolution of the output data was such that it prevented us from accurately identifying all such objects, and avoiding false detections. Therefore, these bodies were not taken into account in the current investigation."

The detection tool is RESTICK (Lykawka & Mukai 2007).

- **Horseshoe orbits:** excluded. QS orbits are not mentioned.
- **Minimum libration lifetime:** 0.5–1 Myr. The authors say this makes the capture efficiencies "lower limits".
- **Counts (VERIFIED):** "The number of Trojans resident in the Jovian cloud was calculated using RESTICK after five and ten million years of integration time, yielding totals of 149 (5 Myr) and 131 (10 Myr) objects."
- **Long-term survival (VERIFIED):** followed for 4 Gyr: "25% (31 out of 131 bodies)" survived.
- **Observed comparison (VERIFIED, Fig. 1 caption):** "Currently known Trojans with orbits with at least two opposition observations … taken from the IAU Minor Planet Center … on 25th November, 2009 … 'large' Trojans (sizes > 50 km) with absolute magnitudes, H, less than 10.5 are represented by squares". The comparison covers orbital elements only. They make no N4/N5 comparison.

## 9. Lykawka, Horner, Jones & Mukai 2009, MNRAS 398, 1715, Neptune Trojans I (arXiv:0909.0404)

**Swarm assigned dynamically by libration centre (VERIFIED):**
> "The average libration amplitude, A, is calculated by sequentially measuring the individual maximum and minimum displacement values for each clone over the course of the integration, then averaging over these to get the best fit to this value. Effectively, then, A, gives the average of the libration amplitudes experienced by all clones of the object over the 10 Myr run."

> "Typical values for CL will typically lie within five degrees of the nominal location of the Lagrange point (i.e. 60±5°, 300±5° and 180±5° for L4, L5 and horseshoe orbits, respectively)."

**Horseshoe orbits are counted as a separate class (VERIFIED):**
> "we calculated the duration of the capture, the type of libration (L4, L5 or horseshoe) … the number of transitions between different libration styles, the number of periods spent as a horseshoe, L4 or L5 object".

- **Amplitude ranges (VERIFIED):** amplitudes run up to "~60-70° for L4/L5 orbits (in the limit of leaving the tadpole orbit) and 150-170° for horseshoe orbits."
- **Observed comparison:** the known Neptune Trojans of the time. The exact count is not extracted here.

## 10. Kortenkamp, Malhotra & Michtchenko 2004, Icarus 167, 347, Neptune Trojans during migration (astro-ph/0305572)

This entry is included as the nearest accessible Kortenkamp paper.

**Definition (VERIFIED):**
> "For a particle trapped in the planet's leading L4 or trailing L5 regions, φ1:1 librates about +60° or −60°, respectively. The full magnitude of the difference between the extremal values of φ1:1 is called the libration amplitude."

**Initial population (VERIFIED):**
> "All particles that maintained 0° < |φ1:1| < 180° for the first 25,000 years were considered members of the 'initial' resonant population. This initial 25,000 years is about four times the libration period"

**Loss criterion (VERIFIED):**
> "we removed particles from the simulations if their libration amplitude exceeded 180°. Thus, Neptune could not regain Trojans after they had been lost. … a small number of particles on large tadpole orbits … were found to be capable of transitioning to and from resonant horseshoe orbits as the planets migrated. However, recapture of lost Trojans by this process was a short-term effect."

**Horseshoe orbits:** effectively treated as lost, because they cross σ = 180° and so have amplitude > 180°.

**Initial conditions (VERIFIED):** Jupiter-Trojan elements were mapped onto Neptune. The source sample was "1171 asteroids … roughly evenly divided between Jupiter's leading L4 (57%) and trailing L5 (43%)" (MPC, January 2002).

## 11. Kortenkamp & Hamilton 2001, BAAS 33, 1086 (DPS abstract)

- **Existence:** VERIFIED only as a reference in Li+2023a ("Kortenkamp, S. J., & Hamilton, D. P. 2001, Bull. Am. Astron. Soc. 33, 1086"). It is cited for gas-drag capture into tadpole orbits.
- **Text:** the abstract could not be fetched because ADS was blocked. The criterion is unknown.

## 12. Marzari & Scholl 1998, A&A 339, 278, "The growth of Jupiter and Saturn and the capture of Trojans"

ADS scanned PDF, VERIFIED. Companion paper: Icarus 131, 41 (1998), not fetched.

**Mechanism (VERIFIED):**
> "the changing gravity field due to the mass growth of the proto-planet causes a large percentage (≃ 40%) of planetesimals, initially on horseshoe-type orbits, to be captured as L4 or L5 Trojans. Only a small percentage of planetesimals initially close to the resonance border (≃ 2%) are also captured."

**Amplitude definition (VERIFIED):**
> "For each orbit we compute the libration amplitude intended as the difference between the maximum and minimum values of the critical argument over a period of 5×10⁴ years."

**Swarm assignment (VERIFIED):** "The planetesimal is trapped in L4 instead of L5. This depends on the different angular distance from Jupiter at trapping."

- **Horseshoe orbits:** a pre-capture state, not counted as Trojans.
- **Observed comparison (VERIFIED):** "we integrated for 1 Myr the orbits of 223 Trojans derived from the asteroid orbit database of Bowell et al. (1993). We selected only those orbits with reliable orbital elements, i.e. with arcs longer than 60 days. The libration amplitude was then computed for each asteroid in the sample every 5 × 10⁴ years and these values were then averaged". The model amplitudes "are shifted towards high libration amplitudes" (VERIFIED). There is no N4/N5 comparison.

## 13. Fleming & Hamilton 2000, Icarus 148, 479 (astro-ph/0007296)

**Amplitude definition (VERIFIED):**
> "Note that A/2 is the amplitude of the φ oscillations. The libration amplitude, A, is defined to be the total angular extent of these oscillations"

Here φ is the difference of mean longitudes. Numerically, A is computed via "an analytic approximation (Yoder 1983, Shoemaker et al. 1989)" that assumes small A.

**Result (VERIFIED):** "Jupiter's predicted mass growth is sufficient to cause the capture of asteroids initially on horseshoe orbits into stable tadpole orbits."

**Observed context (VERIFIED):** "nearly twice as many asteroids have been observed librating about the L4 point as about the L5 point. This, however, may simply be the result of observational selection effects (Shoemaker et al. 1989)." Note that the observed swarms are described by libration, not by sky position.

## 14. Montesinos et al. 2020, A&A 642?, "Dust trapping around Lagrangian points in protoplanetary disks" (arXiv:2009.10768)

The volume is unverified.

**Context:** gas plus dust hydrodynamics (FARGO3D-type), with a 1–10 M_J planet. Dust only, not planetesimals.

**Criterion (VERIFIED):**
> "we consider that a particle will be in tadpole orbit around a Lagrangian point if it belongs to the cylindrical areas drawn in Figure 1 that surround L4 and L5. Each cylindrical area is delimited by the position of the Lagrangian points at ±π/3 adding (or subtracting) an angle of ±π/8 rad in the azimuthal direction, and between an effective capture radius given by the range rp ± RHill … We show a posteriori that the libration of particles around a Lagrangian point occurs in such a pre-defined area. At each time-step, we count particles that enter and leave the defined region."

- **What the box measures:** a snapshot azimuth ±22.5° and radius ±1 R_H box. It is an occupancy count, not a libration test.
- **Horseshoe orbits:** excluded by geometry. Horseshoe particles passing through the box would be counted while inside it.
- **Observed comparison (VERIFIED):** "observations of Jovian Trojans, which show a ratio of the number of asteroids N(L4)/N(L5) ≈ 1.8 for Trojans with diameters D > 2 km (Yoshida & Nakamura 2005)."
- **Result:** L5 holds more bulk mass. The run lasts 10,000 yr (≈460 orbits).

## 15. Nesvorný 2018, ARA&A 56, 137, "Dynamical Evolution of the Early Solar System" (arXiv:1807.06647)

**Definition (VERIFIED, Sect. 9):**
> "They hug two equilibrium points … L4 and L5, with a ≃ a5 = 5.2 au, e < 0.15, i < 40°, and δλ = λ − λ5 ∼ ±60° … The angle δλ librates with a period of ∼150 yr and full libration amplitude, D, up to D ≃ 70°."

**Calibration (VERIFIED):**
> "The capture probability (as a stable Trojan) was found to be P = (5 ± 3) × 10−7 for each particle in the original planetesimal disk".

Trojans are used to calibrate the disk SFD: "the size distribution of known Jupiter Trojans (the sample is incomplete for D < 10 km)". The K-S test against known Trojans gives "60%, 68% and 63%" for D, e and i.

## 16. Gomes & Nesvorný 2016, A&A 592, A146, "Neptune trojan formation during planetary instability and migration"

The full text was blocked (no arXiv version; A&A behind a bot wall).

- **VERIFIED, abstract via OpenAlex, first sentence only:** "We investigate the process of Neptune trojan capture and permanence in resonance up to the present time based on a planetary instability migration model."
- **Unverified (search-engine summary):** one run left about 5×10⁻⁵ M_E of Neptune trojans, and the paper counts 11 known Neptune trojans in November 2015, with 10 considered genuine.
- **Criterion:** unknown.

## 17. Chanut, Winter & Tsuchida 2008, A&A 481, 519

I could not locate the paper through web search, OpenAlex or ADS (blocked), so its content is NOT VERIFIED. I cannot confirm that it is a Trojan-capture paper. The caller should check the bibcode 2008A&A...481..519C.

## 18. Pirani & Johansen (other)

I found no Trojan paper by only these two authors. "Pirani 2019b" above is Pirani, Johansen & Mustill.

---

## Cross-cutting observations for the thesis

1. **Three families of operational "Trojan" definitions exist in capture work:**
   - **(a) Geometric snapshot boxes, with no libration test:**
     - Pirani+2019a: q and Q within a_J ± 2.5 R_H; horseshoe and QS included.
     - Montesinos+2020: azimuth ±π/8 around ±π/3 and r_p ± R_H.
     - Roig & Nesvorný 2015: an a/e/I region.
   - **(b) Libration-based with a minimum duration and no Gyr requirement:**
     - Lykawka & Horner 2010: libration about L4/L5 for ≥0.5–1 Myr; horseshoe excluded.
     - Marzari & Scholl 1998: max−min of the critical argument over 5×10⁴ yr.
     - Kortenkamp+2004: |φ| never reaches 180°.
     - Li+2023a,b: survivors after 1 Myr, extended to 1 Gyr.
   - **(c) Libration plus long-term survival ("stable Trojans"):**
     - NVM13: survive a 4 Gyr Phase 3. About 50% of the Phase-2 "Trojans" are removed, typically those with D > 60°.
     - Lykawka & Horner 2010 also report 4 Gyr survivors (25%).
2. **Horseshoe treatment differs sharply.** Pirani+2019a counts horseshoe orbits, and its asymmetry mechanism is literally a leading-side occupancy excess of horseshoe orbits. Lykawka & Horner 2010 excludes them. Lykawka+2009 keeps them as a third class (centre 180 ± 5°). Kortenkamp+2004 treats them as lost (amplitude > 180°).
3. **L4/L5 assignment is rarely stated explicitly:**
   - Libration centre (dynamical): Lykawka+2009 (CL = 60 ± 5°, 300 ± 5°, 180 ± 5°).
   - Instantaneous side or azimuth box (snapshot): Pirani+2019a, Montesinos+2020.
   - Starting swarm: Li+2023a,b.
   - Not stated: NVM13.
4. **Observed targets used:**
   - Szabó+2007, 1.6 ± 0.1: SDSS, H < 13.8, kinematically selected.
   - Grav+2011, 1.4 ± 0.2: WISE, D > 10 km.
   - Nakamura & Yoshida 2008 or Yoshida & Nakamura 2005, ≈1.8 (± 0.57): D > 2 km.
   - MPC H < 12 counts:
     - NVM13: 361 vs 279 with 9 Eurybates members removed, so 1.3 ± 0.1.
     - Pirani+2019a: 330 vs 229 minus 9 Eurybates, so 1.4 ± 0.1.
   - Li+2023a: MPC multi-opposition, ≈1.8 raw.
   - These observational catalogues select Trojans by orbit classification or kinematics, not by the simulation criteria above.
   - NVM13 is the only capture paper that compares a 4 Gyr-stable model population with a stated complete observational sample (H < 12) for both distributions and N4/N5. It also calibrates absolute numbers with the 25 known Trojans with H < 9.

---

## Part C — Long-term evolution, definitional-problem papers, catalogue definitions, L4/L5 naming

Verification legend: **[V]** = quoted from full text / abstract I fetched in this session (source URL given). **[R]** = from recall or secondary description only; not checked against the paper.
Note on quotes: full texts were fetched from ar5iv/arXiv HTML and converted to plain text. Inline math appears in `$...$` (LaTeX alt-text), so the quotes are verbatim except for that math rendering.

---

### C1. Di Sisto, Ramos & Beaugé 2014, Icarus 243, 287–295, "Giga-year evolution of Jupiter Trojans and the asymmetry problem" (arXiv:1604.05331) [V]
Source: https://ar5iv.labs.arxiv.org/html/1604.05331
- **Sample (observational):** "As of March 2013, there were 2972 numbered Jupiter Trojans, thus with fairly reliable orbits. Of these, 1975 (over 66%) display tadpole orbits around $L_4$, while 997 are associated to $L_5$. The population of Jupiter Trojans is believed to be complete up to absolute magnitude $H=12$ (Szabó et al. 2007); however for the purposes of the present study we will consider the complete (numbered) population regardless of the absolute magnitude." The paper does not state how a numbered object was classed as a Trojan or assigned to L4/L5. It takes the catalogue membership as given.
- **Representative plane for elements:** "each asteroid was integrated until it crossed the representative plane defined by the conditions $M-M_J=0$, $\varpi-\varpi_J=\pm 60^\circ$ and $\Omega-\Omega_J=0$."
- **Escape / loss criterion:** "Each integration was stopped if the particle suffered a close encounter with a planet (i.e. mutual distance smaller than the corresponding Hill radius)." The criterion is a Hill-radius encounter, not a check on libration.
- **D:** the paper uses $D$ = "the semi-amplitude of libration" (quoted while describing Tsiganis et al. 2005) and plots e vs D.
- **Results:** "approximately 23% were found to escape from the $L_4$ swarm, while this number increased to 28.3% for $L_5$" over 4.5 Gyr. The fictitious grid ($5.01\le a\le5.4$, $0\le e\le0.34$, $0\le i\le60^\circ$; $\varpi=\varpi_J\pm60^\circ$, $\Omega=\Omega_J$, $M=M_J$) shows "no significant difference between the two tadpole regions". Conclusion: the asymmetry "must be primarily primordial and characteristic of the capture mechanism"; "the original population ratio must have been much closer to unity".
- **Horseshoe / QS:** not counted. Only tadpoles are discussed.
- It also cites Hou et al. (2014): "a temporary asymmetry my [sic] be obtained with the same initial conditions in both tadpole regions. this asymmetry, however, is short-lived".

### C2. Di Sisto, Ramos & Gallardo 2019, Icarus 319, 828–839, "The dynamical evolution of escaped Jupiter Trojan asteroids, link to other minor body populations" (arXiv:1811.00352) [V]
Source: https://ar5iv.labs.arxiv.org/html/1811.00352
- **Same initial sample:** "all numbered Jupiter Trojan asteroids as of March 2013. Thereby, a numerical integration of 1975 $L_4$ Trojans and 997 $L_5$ Trojans", with planets Venus–Neptune, for 4.5 Gyr.
- **Escape definition (operational):** "The encounters at less than 2.1 Hill's Radius with the planets were registered ... If a Trojan has an encounter with a planet, usually Jupiter, the time of the first encounter is considered the 'escape time' and the Trojan would be an 'escaped Trojan'".
- **Results:** "We detect 466 (out of 1975) Trojans that escape from $L_4$, this is 23.6%, and 250 (out of 997) Trojans that escape from $L_5$, i.e. 25.1%". Abstract: "the rate of escape of Trojans from $L_5$, $\sim1.1$ times greater than from $L_4$".
- **Caveat:** the L5 escape fraction is 28.3% in the 2014 paper and 25.1% here, for the same 2972-object sample. The planet sets differ (4 giants vs Venus–Neptune) and so do the encounter thresholds (1 vs 2.1 R_H). The outcome is therefore sensitive to the operational escape definition.

### C3. Hellmich, Mottola, Hahn, Kührt 2019, A&A 630, A148, "Influence of the Yarkovsky force on Jupiter Trojan asteroids" (arXiv:1909.08584) [V]
Source: https://ar5iv.labs.arxiv.org/html/1909.08584
- **Sample:** clones of "numbered and multi-opposition Trojan orbits, consisting of 3642 L4 and 1925 L5 objects" (covariance from AstDyS), 98304 particles in total. L4 and L5 are equalised by cloning.
- **Escape definition:** "we flagged all particles when their heliocentric distance became smaller than 3 au, exceeded 8 au, or when the particles approached Jupiter closer than twice the planet's Hill radius. Such particles were excluded from further processing and considered escaped from the Trojan region."
- **Result:** "the ratio between escaped L5 and L4 particles after 1 Gyr was 1.106 ± 0.026, which shows that L5 objects are indeed depleted at a faster pace than L4 Trojans, although not fast enough to explain the currently observed asymmetry."
- **Horseshoe / QS:** not treated. Libration is not monitored, so a particle that leaves libration without approaching within 2 R_H of Jupiter and stays inside 3–8 au still counts as a survivor.

### C4. Hou, Scheeres & Liu 2016, CeMDA 125, 451–484, "Dynamics of the Jupiter Trojans with Saturn's perturbation when the two planets are in migration" (doi:10.1007/s10569-016-9692-1). Companion paper: Hou, Scheeres & Liu 2014, CeMDA 119, 119–142, "... in the present configuration of the two planets" [V for metadata (Crossref) and partial text via the Springer page; WebFetch only gives paraphrase plus short quotes]
Source: https://link.springer.com/article/10.1007/s10569-016-9692-1
- **What the paper is:** a study of dynamical symmetry between the two triangular libration points (TLPs) under Saturn's perturbation during migration, in a **planar** model. Short quoted fragments: it revisits "the problem of dynamical symmetry between two Jupiter triangular libration points (TLPs) with Saturn's perturbation". It names three asymmetry mechanisms: (1) short-time spatial asymmetry while Jupiter and Saturn are in resonance, (2) the changing planetary orbits during migration, and (3) the chaotic nature of Trojan orbits. It finds Yarkovsky differences between the TLPs too small to observe. Sect. 6.4: "Current observed asymmetry is only a random result if these mechanisms are the main reasons to cause it."
- **Trojan definition:** none beyond the textbook one (tadpole orbits around the TLPs; a few "jump" between the points on horseshoe-like orbits, a point made only in the Introduction). L4/L5 labels come from the initial grid, e.g. the Fig. 4 caption: "The unfilled circle indicates the stable orbits around the point L4 and '+' indicates the stable orbits around the point L5." The stability criterion in Sect. 3.4 is: "the orbit does not intersect with the x-z plane within the integration time" (5×10^6 yr). In other words, the orbit never crosses the Sun–Jupiter line, which is a no-horseshoe, no-switching requirement. Sect. 5 uses the libration amplitude only as a diagnostic.
- **Inclination:** NOT studied. The work is planar ("Due to the decoupling between the planar motion and the out-of-plane motion, we only discuss the planar motion"), and Sect. 7 acknowledges that "this restriction does not allow us to study some important properties of the vertical motions". **So Hou+2016 does not address an L4/L5 inclination asymmetry or a cold-L4 component.** It is a planar symmetry/asymmetry-mechanism paper. The "check Hou+2016 before claiming novelty" concern is therefore about the asymmetry mechanisms, not about inclination.

### C5. Wang & Hou 2017 — NOT LOCATED
Searches of Crossref, arXiv, Semantic Scholar and the web found no 2017 Wang & Hou paper on Jupiter Trojans, and I am not inventing a citation. Hou's verified Trojan papers are: Hou, Scheeres & Liu 2014 MNRAS 437, 1420 ("Saturn Trojans: a dynamical point of view"), 2014 CeMDA 119, 119, and 2016 CeMDA 125, 451.

### C6. Horner & Evans 2006, MNRAS 367, L20, "The capture of Centaurs as Trojans" (astro-ph/0511791) [V]
Source: https://ar5iv.labs.arxiv.org/html/astro-ph/0511791
- **Criterion:** "the data were searched for objects whose semimajor axis stayed within one Hill radius of that of Jupiter, Saturn, Uranus and Neptune ... These provisional candidates were then examined in the co-rotating frame of the planet's orbit to see whether they had Trojan-like or satellite-like behaviour. To ensure that an object was indeed captured, a minimum number of 800 orbital periods was required. For example, in the case of Jupiter, the object had to show Trojan-like or satellite-like behaviour for at least 9.5 kyr ($\sim800$ orbital periods of Jupiter)."
- **Horseshoe:** counted separately (Table 2 has L4 / L5 / Horseshoe columns; Jupiter: 2 / 5 / 3). "Objects that displayed two periods of Trojan behaviour in different regions gave a score of 0.5 in each region occupied."
- **On the definitional problem:** "simulations of the Jovian Trojans show that at least two objects currently classified as Trojans are experiencing only a brief visit to the region, rather than a prolonged stay (Karlsson 2004 ...)". Also: "the captured population has a different profile to the permanent one, with nearly equal likelihood of capture in the L4 or L5 regions."

### C7. Karlsson 2004, A&A 413, 1153–1161, "Transitional and temporary objects in the Jupiter Trojan area" (doi:10.1051/0004-6361:20031543) [V abstract only, via OpenAlex]
- "I have integrated 1233 objects with semimajor axes between 4.6 and 5.8 AU 1000 years back and forward in time. A search for Jupiter horseshoe Trojans in transition from tadpole orbits to non-Trojan orbits and objects in temporary horseshoe orbits was made. A total of 13 objects (5 comets, 3 asteroids and 5 Trojans) were found. ... Two objects, 2000 HR24 and 2002 GP31, are falsely identified as Trojans. ... Trojans in the second set have longer arcs and 5 of them are good candidates for being in transition from the Trojan area."
- This is an early statement that **catalogue "Trojans" (MPC list) contain non-librators and transitional objects**. The time span is only ±1 kyr.

### C8. Greenstreet, Gladman & Ngo 2020, AJ 160, 144, "Transient Jupiter co-orbitals from Solar System sources" (arXiv:2007.14973) [V]
Source: https://ar5iv.labs.arxiv.org/html/2007.14973
- **Co-orbital criterion:** a running window ("NEOs: 9 kyr, Centaurs: 5 kyr", "several times longer than the typical libration period at Jupiter"). "A particle was classified as a co-orbital if, within the running window, both its average semimajor axis $a$ was within 0.4 AU of Jupiter's average $a$ and no individual semimajor axis value differed by more than 3.5 times Jupiter's Hill-sphere radius $R_H=1.2$ AU from Jupiter's $a$."
- **Island assignment (dynamical, not a snapshot):** "If $\phi_{1:1}$ remains in the leading or trailing hemisphere for the duration of a running window, we assign the particle to the L4 or L5 state, respectively. If $\phi_{1:1}$ crosses $180^o$ at any time during the window interval, the co-orbital is labelled a horseshoe. All remaining orbits are classified as quasi-satellites".
- Error rates: the parameters were tuned until libration was "correctly identified $>90\%$ of the time". Misclassifications during transitions "affect $<10\%$".

### C9. Greenstreet, Gladman & Jurić 2024, ApJL 963, L40, "Jupiter's metastable companions" (arXiv:2309.06609) [V]
Source: https://ar5iv.labs.arxiv.org/html/2309.06609
- **Sample:** JPL SBDB objects with 4.5 < a < 5.9 au ("semimajor axis within twice Jupiter's Hill sphere radius") and arcs of at least 30 d: 11,581 objects × 1000 clones, integrated 0.5 Myr with SWIFT-RMVS4. Particles are removed within 1000 R_J ("0.48 au, or about 2 Hill spheres") of Jupiter.
- **Resonance test:** "we used a 1 kyr running window ... a particle's average semimajor axis must remain within 0.2 au of $a_J$ and no individual semimajor axis value may differ by no more than 0.65 au from $a_J$ within the 1 kyr window, if the object is to be considered resonant during that 1 kyr time period."
- **Classes (definitional):**
  - "Objects for which $\geq95$% of the 1000 clones are resonant for the duration of the 0.5 Myr integration, are classified as long-term stable 'Trojans'. We identify 7482 L4 Trojans and 3941 L5 Trojans".
  - "Objects that have $\geq95$% of their 1000 clones resonate for at least 1 kyr but then leave the resonance are classified as 'transient co-orbitals'."
  - "$<5$%" resonant → "non-resonant". This class "include[s] the 14 MPC/JPL-classified 'Trojans' ... which are clearly not long-term stable resonant L4/L5 primordial Trojans".
  - "insecure" covers the remaining objects.
- **Island assignment:** the same as in 2020: L4/L5 if "$\phi_{11}$ remains in the leading or trailing hemisphere ... during a 1 kyr running period". Horseshoe if it "crosses the direction $180^o$ away from Jupiter". Otherwise quasi-satellite.
- **Definition of "metastable":** "'metastable' objects are here defined by undergoing many resonant argument librations before exiting the co-orbital state". In the abstract, metastable Trojans are "recent captures from heliocentric orbits into moderately long-lived (10 kyr–100 Myr) metastable states". There are 27 such co-orbitals, of which "12 L4 Trojans, 4 L5 Trojans, 2 horseshoes, 8 quasi-satellites" plus 2015 BZ509. They "are thus required to be placed in a category of jovian co-orbitals separate from the primordial Trojans."
- **Relevance:** this is the clearest modern three-tier definition: stable Trojan, metastable/transient co-orbital, and catalogue-labelled non-resonant interloper. Note that the "stable Trojan" cut is only 0.5 Myr.

### C10. Vokrouhlický, Nesvorný, Brož et al. 2024, AJ 167, 138, "Orbital and absolute magnitude distribution of Jupiter Trojans" (arXiv:2401.15537) — the observational sample many models are now compared against [V]
Source: https://arxiv.org/html/2401.15537v1
- **Selection (verbatim):** "The orbital selection was: (i) semimajor axis $a$ in between 4.95 au and 5.45 au, (ii) eccentricity smaller than 0.6, (iii) inclination smaller than $50^\circ$, and (iv) absolute value of the resonant angle $\sigma$ larger than $20^\circ$ and smaller than $130^\circ$. This search resulted in 7,929 orbits around the L4 libration center and 4,189 orbits around the L5 libration center. A small number of these orbits, however, correspond to unstable Jupiter family comets masquerading as JTs. This led us to numerically propagate all selected orbits ... for a 1 Myr timespan. We found that 26 and 12 candidate Trojans near L4 and L5, respectively, escaped from the libration region ... The remaining 7,903 bodies near L4 and 4,177 bodies near L5 were found to reside in the resonant zone of their respective center, with their $\sigma$ librating for the entire timespan of the integration."
- **So L4/L5 = sign of the osculating σ at MJD 60000 (a snapshot), followed by a 1 Myr libration filter.** Horseshoes and QS are excluded by the |σ| window. The 1 Myr filter removes few objects: 38 of 12,118.
- **Libration variable:** $C$ from the integral $(da/a_J)^2 = (8\mu/3)[C - f(\sigma)]$ with $f(\sigma)=(1+4|\sin\sigma/2|^3)/(2|\sin\sigma/2|)$: "for tadpole orbits $C_{\min}=1.5$ and $C_{\max}=2.5$". The proper semimajor axis is $da_P \simeq a_J\sqrt{8\mu\,\Delta C/3}$. $A$ ≈ proper sin I and $B$ ≈ proper e, both quasi-proper "basically osculating" variables.
- **Asymmetry quantity:** "Denoting the magnitude-limited populations of the L4 and L5 Trojan swarm by $N_4$ and $N_5$, we define their ratio $f_{45}=N_4/N_5$". They caution that a magnitude-limited asymmetry is not necessarily a size-limited one.
- **Proper-element catalogue:** a separate step following Holt et al. (2020) App. A: "we used the MPC catalog of minor planet orbits as of April 15, 2023 ... and selected Jupiter Trojans using the same steps as described in Sec. II. We obtained 8,144 orbits in the L4 swarm, and 4,252 orbits in the L5 swarm."
- The abstract notes an inclination-distribution asymmetry: "the L4 population being tighter and the L5 population being broader".

### C11. Bottke, Marschall, Nesvorný & Vokrouhlický 2023, Space Sci. Rev. 219, 83, "Origin and evolution of Jupiter's Trojan asteroids" (arXiv:2312.02864) [V]
Source: https://arxiv.org/pdf/2312.02864
- **Definition:** "Jupiter Trojans (hereafter Trojans) are populations that orbit around the L4 and L5 Lagrange points of Jupiter, which are located 60° in front of and behind Jupiter, respectively. They are contained within Jupiter's 1:1 mean motion resonance".
- **Stable vs unstable:** "Most Trojans reside on stable orbits that librate around L4 and L5, but a small fraction is unstable over ages shorter than the lifetime of the Solar System (Levison et al. 1997; Tsiganis et al. 2005; Di Sisto et al. 2014; 2019; Holt et al. 2020)." Also: "The remaining unstable Trojans are likely to be temporary captures from the Jupiter family comet population. The regions around L4 and L5 are dynamically 'sticky'".
- **Their view of the asymmetry:** "Dynamically, there is no particular difference between the two regions that would lead to a pronounced asymmetric population by Trojan escape (e.g., Holt et al. 2020)." "The population of D > 100 km bodies ... contains 9 objects in both L4 and L5. ... If we examine D > 10 km bodies, the ratio of L4 to L5 objects moves to 1.4 ± 0.2 (Grav et al. 2011)". "These results are consistent with both populations being statistically comparable, but it is also possible that some modest asymmetry may exist".
- **Model-acceptance test:** a capture model "must place comparable numbers of bodies near Jupiter's L4 and L5 zones, but it must also explain how the populations obtained high inclinations". The capture probability is "~5 × 10^-7 (Nesvorný et al. 2013)". Proper elements in their Fig. 1 come from "the methods of Milani (1993)".

### C12. Di Ruzza, Pousse & Alessi 2023, "On the co-orbital asteroids in the solar system: medium-term timescale analysis of the quasi-coplanar objects" Icarus 390, 115330 (arXiv:2209.05219) [V abstract]
- An averaged planar circular RTBP in which "The domain of each co-orbital regime, that is, the quasi-satellite motion, the horseshoe motion and the tadpole motion, can be neatly defined by means of an integrable model and a simple bi-dimensional map". It uses JPL Horizons ephemerides over ~600 yr for Jupiter. Objects are labelled TP / HS / QS / CP (compound), with transitions such as "HS+TR". Greenstreet+2023 notes that this classification disagrees with theirs for 2015 YJ22 (Di Ruzza: horseshoe; Greenstreet: L5 Trojan).

### C13. Catalogue / naming definitions (what "observed Trojan" operationally means)
- **JPL SBDB orbit class TJN [V]** (https://ssd-api.jpl.nasa.gov/doc/sbdb_filter.html): "TJN Jupiter Trojan asteroid: Asteroids trapped in Jupiter's L4/L5 Lagrange points (4.6 au < a < 5.5 au; e < 0.3)". This is a pure osculating box with no libration test.
- **MPC orbit type 12 [V]** (https://docs.minorplanetcenter.net/mpc-ops-docs/orbits/orbit-types/): "Jupiter Trojan 12 4.8 < a < 5.4 and e < 0.3". The MPC "List Of Jupiter Trojans" page has an "Ln" column (L4/L5) but does not state on that page how Ln or list membership is determined [V that it is not stated there].
- **Greek / Trojan naming:** a mythological naming convention only. L4 objects are named for Greek heroes (the "Greek camp") and L5 objects for Trojans (the "Trojan camp"). (617) Patroclus (L5) and (624) Hektor (L4) predate the convention and are exceptions. [V that MPC lists Patroclus as L5 and Hektor as L4. Convention text from NASA/Lucy/Wikipedia-type secondary sources, R for the IAU primary text.] In 2018 the IAU allowed objects with H > 12 to receive Olympic-athlete names [secondary source; R]. **No dynamical definition of "Greek" vs "Trojan" beyond L4 vs L5 was found.**

### C14. Snapshot vs dynamical swarm assignment (what I found)
- **Dynamical (hemisphere over a time window):** Greenstreet+2020 and Greenstreet+2023 (σ stays in the leading/trailing hemisphere for a running window several libration periods long, 1–9 kyr). Horner & Evans 2006 used visual inspection of the co-rotating frame over ≥800 P_J and gave a half count to objects that visited both swarms.
- **Snapshot plus stability filter:** Vokrouhlický+2024 (sign of osculating σ in 20°<|σ|<130° at one epoch, then 1 Myr libration check).
- **Initial-condition label:** Di Sisto+2014/2019, Hellmich+2019 and Hou+2016 all label by the starting swarm and track only "escape", defined as a planet encounter or a heliocentric-distance cut, not loss of libration.
- **Pure catalogue box:** JPL TJN and MPC type 12, which test (a, e) only.
- I found no paper defining swarm membership by the mean of σ over a libration cycle as such. The proper-element approach (Milani 1993 / Beaugé & Roig / Brož & Rozehnal / Holt+2020; see Part A) effectively uses the libration centre and amplitude from filtered σ. For a tadpole librator, the hemisphere and the mean σ give the same answer.
