# Operational definitions of "Jupiter Trojan" across catalogues, databases, naming authorities and surveys

All pages accessed **2026-10-14**. Provenance tags:

- **[FETCHED]**: I downloaded the page or PDF (curl, or WebFetch for small pages) and the quote was extracted from the raw text. Quotes are verbatim apart from whitespace and PDF ligature cleanup.
- **[COMPUTED]**: I derived the number myself from a downloaded file. The script logic is described inline.
- **[RECALLED]**: From memory. Not verified this session; check before citing.

---

## 0. Summary table

| # | Source | Operational definition | Libration checked? | L4/L5 rule | N4 / N5 (ratio) | Tag |
|---|---|---|---|---|---|---|
| 1a | **MPC "List of Jupiter Trojans"** (JupiterTrojans.html, updated 2026 May 30) | "selected automatically using cuts in osculating element space". The cut values are **not published**. | **No** (MPC says so explicitly; libration is checked only for Earth/Mars/Uranus Trojans) | `Ln` column: "near the L4 or the L5 point… L4 is 60° preceding the planet and L5 is 60° following". The exact rule is not published. | **9823 / 5942 (1.653)**, total 15765 | FETCHED + COMPUTED |
| 1a′ | Same list, my reverse-engineering | Observed envelope: 4.952 ≤ a ≤ 5.450 au; e ≤ 0.297; i ≤ 55.4°; 20.4° ≤ \|λ−λJ\| ≤ 144.6°. Ln always equals the sign of λ−λJ. | n/a | consistent with Ln = sign(λ−λJ), with \|σ\| ≳ 20° | — | COMPUTED (inference) |
| 1b | **MPCORB.DAT orbit-type flag** (cols 162–165, low 6 bits; value 9 = Jupiter Trojan) | "based on cuts in osculating element space and is not 100% reliable". The cuts are **not published**. | No | No L4/L5 in the flag | — | FETCHED |
| 1c | **MPC new `orbit_type_int`** (mpc_orbits DB, Orbit-Type API, MPC Explorer) | **Code 12: `4.8 < a < 5.4 and e < 0.3`**. No i cut and no angle cut. | No | none | — | FETCHED + API-tested |
| 2 | **JPL SBDB class TJN** | "Asteroids trapped in Jupiter's L4/L5 Lagrange points (**4.6 au < a < 5.5 au; e < 0.3**)" | No (pure a/e) | none | — | FETCHED |
| 3a | **AstDyS `tro.syn`** (Knežević, Jun 2024) | Objects that have Milani-type synthetic Trojan proper elements. The **membership rule for entering the file is not documented** on the site. | Implicitly yes: computing the proper libration amplitude D requires a librating σ in a 5- or 50-Myr integration [RECALLED that non-librators are dropped] | `L` column = 4 or 5 | **7404 / 4463 (1.659)**, total 11867 | FETCHED + COMPUTED |
| 3b | Milani 1993 (CeMDA 57, 59) | Synthetic proper elements (da or D, e_p, sin I_p) from numerical integration | Yes (proper libration amplitude) | from the libration centre | 174 Trojans [RECALLED] | RECALLED |
| 3c | Beaugé & Roig 2001 (Icarus 153, 391) | Semi-analytic proper elements. Selection from the then-known numbered/multi-opposition Trojans | Yes (model of resonant motion) | from the libration centre | ≈533 objects [RECALLED] | RECALLED |
| 4a | **Lowell astorb** | **No Trojan class defined.** The crosser code excludes "Jupiter and Mars Trojans" without defining them. Jupiter's zone is 4.950–5.455 au. | — | — | — | FETCHED |
| 4b | **Rubin / LSST DP2 `mpc_orbits`** | Re-uses MPC `orbit_type_int` = 12 ("TJN"). No independent definition. | No | none | — | FETCHED |
| 4c | Brož & Rozehnal 2011 (MNRAS 414, 565) | Resonant proper elements from AstOrb (JD 2455500.5). The **initial pre-selection is not stated**. | Yes: φ = λ − λ′ − χ, χ = ±60°; the "revolution of angle φ(t) must not be confined…" (unfolding) | χ = +60° (L4) / −60° (L5) | **2647 / 1496** (paper quotes N(L4)/N(L5) ≃ 1.8 for H ≲ 15) | FETCHED |
| 4d | Holt et al. 2020 (MNRAS 499, 3630; arXiv 2009.06749) | Milani-type synthetic proper elements. Population "identified" from AstOrb (2020 Apr 28). **The criterion is not stated.** | Yes (proper libration amplitude) | by swarm | **4607 / 2721** (1.69) | FETCHED |
| 4e | **Vokrouhlický et al. 2024** (AJ 167, 138) | MPCORB 2023-02-22, osculating at MJD 60000: **4.95 < a < 5.45 au, e < 0.6, I < 50°, 20° < \|σ\| < 130°**, σ = λ − λJ. Then a **1-Myr** N-body integration. Bodies that escaped the libration region, hit Jupiter or were ejected are removed. | **Yes**: "σ librating for the entire timespan" (1 Myr) | sign of σ | box: 7929 / 4189. **Final: 7903 / 4177 (1.892)**. Appendix catalogue (Apr 2023): 8144 / 4252 | FETCHED |
| 4f | Vokrouhlický online catalogue (`sirrah.troja.mff.cuni.cz/~mira/tmp/trojans/`) | Same selection as 4e. Files `greeks-hcm_20230405_GS/` (L4) and `trojans-hcm_20230405_GS/` (L5). **No README in the directory.** | as 4e | directory name ("greeks" = L4, "trojans" = L5) | `follow.proper.pro`: 8155 lines (L4) / 4252 lines (L5) | FETCHED + COMPUTED |
| 5 | Gladman, Marsden & VanLaerhoven 2008 (SSBN, 43–57) | Resonance = resonant argument librates over a **10-Myr** integration for the nominal orbit plus 2 extremal-a clones (3/3 = secure, 2/3 = insecure, 1/3 = non-resonant). Only ≥3-opposition orbits. **Scope is TNOs/Neptune resonances; it does not classify Jupiter Trojans.** Objects with T_J < 3.05 and q < 7.35 au are "Jupiter-coupled". | Yes (for the resonances it treats) | n/a for Jupiter | — | FETCHED |
| 5′ | Khain+2020, Smullen & Volk 2020 (ML) | TNO/Neptune-resonance classifiers. **No Jupiter 1:1 treatment found.** | — | — | — | search only |
| 6 | **IAU WGSBN Rules & Guidelines v1.1 (2025-02-22)** | Naming theme only: "Jupiter Trojans: Names from the Trojan War (if Hv brighter than 12.0) or names of Olympians and Paralympians (if Hv fainter than or equal to 12.0)." **There is no orbital definition of "Jupiter Trojan"** and **no L4 = Greek / L5 = Trojan rule in the current text**. | — | **none in current rules** (the camp convention is historical, Nicholson 1961; Hektor (L4) and Patroclus (L5) are the exceptions) | — | FETCHED |
| 7a | Szabó et al. 2007 (MNRAS 377, 1393) SDSS MOC3 | Known Trojans for calibration: **5.0 < a < 5.4 au, e < 0.2** (ASTORB). Candidates selected by apparent **angular velocity vs opposition angle φ** (\|φ\| < 25°). | No | sky position relative to Jupiter (fits 2-D Gaussians at λ = ±60°) | 1187 detections of ≈860 candidates. **N(L4)/N(L5) = 1.6 ± 0.1** | FETCHED |
| 7b | Grav et al. 2011 (ApJ 742, 40) NEOWISE | "orbits with semi-major axis between 5.0 and 5.4 AU" and arc ≥ 18 d (MPC orbits) | No | "leading"/"trailing" (rule not stated) | LAT sample 981 / 770. Debiased N(lead)/N(trail) ∼ **1.4 ± 0.2** | FETCHED |
| 7c | Grav et al. 2012 (ApJ 759, 49) NEOWISE taxonomy | Same sample lineage as 7b | No | — | D > 50 km: L4/L5 = **1.34** | FETCHED |
| 7d | Jewitt, Trujillo & Luu 2000 (AJ 120, 1140) | **Sky-plane angular speed** near opposition (no orbits). L4 fields only. | No | L4 only | 93 L4 Trojans | FETCHED |
| 7e | Yoshida & Nakamura 2005 (AJ 130, 2900); Yoshida & Nakamura 2008 / Nakamura & Yoshida 2008 (PASJ 60) Subaru Suprime-Cam | Apparent-motion selection | No | field placement (L4 vs L5 pointing) | 51 L4 / 62 L5 detections (2006 JpGU abstract via search snippet) | search snippet + RECALLED |
| 7f | Yoshida & Terai 2017 (AJ 154, 71) HSC L4 | Motion-vector box from synthetic orbits with e 0–0.3, i 0–40° | No | L4 field | 631 L4 detected; 481 unbiased | FETCHED |
| 7g | Wong & Brown 2015 (AJ 150, 174) Subaru L4 | Objects in (λ, β, \|v\|) bins with Trojan fraction γ = 1; \|v\| contours 14″/hr–22″/hr. The bright end comes from the MPC catalogue. | No | L4 field | 557 L4 | FETCHED |
| 7h | Uehata et al. 2022 (AJ 163, 213) HSC L5 | Apparent motion (as Yoshida & Terai 2017) | No | L5 field | 189 L5. **N_L4/N_L5 = 1.40 ± 0.15** (D ≥ 2 km) | FETCHED |
| 7i | Slyusarev, Glezina & Belskaya, EPSC2018-807 | Source catalogue not stated in the abstract (presumably MPC) | not stated | not stated | **4599 / 2433 (1.9)** | FETCHED |

**Bottom line.** No naming authority or database publishes a dynamical (libration-tested) definition of a Jupiter Trojan:

- **MPC:** element cuts it describes as "not 100 percent reliable", with the numbers unpublished for the list and the legacy MPCORB flag. The new `orbit_type_int` uses 4.8 < a < 5.4, e < 0.3.
- **JPL:** 4.6 < a < 5.5, e < 0.3.
- **Lowell astorb:** no class.
- **WGSBN:** naming theme only.

Libration-checked samples exist only in research catalogues:

- AstDyS proper elements (membership rule undocumented)
- Brož & Rozehnal 2011
- Holt+2020
- Vokrouhlický+2024 (the only one with a fully stated recipe: box + σ window + 1-Myr libration test)

---

## 1. Minor Planet Center

### 1a. "List of Jupiter Trojans" — https://www.minorplanetcenter.net/iau/lists/JupiterTrojans.html [FETCHED]

The list page carries no criteria. Header text:

> "Here are explanations of the symbols used in the table." … "This list was last updated on 2026 May 30."

Columns: `Designation (and name)  Prov. Des.  Ln  q  Q  H  Epoch  M  Peri.  Node  Incl.  e  a  Opps.  Ref.`

Parent page, https://www.minorplanetcenter.net/iau/lists/Trojans.html [FETCHED, verbatim]:

> "At the present time, these lists of Trojan minor planets contain entries for a total of 15819 objects. More specifically, 1 Earth Trojan, 18 Mars Trojans, 15765 Jupiter Trojans, 0 Uranus Trojan and 35 Neptune Trojans."
>
> "For Earth, Mars and Uranus, the listed objects are only those that have been confirmed as Trojans by long-term integrations. For the other planets, the objects are selected automatically using cuts in osculating element space. This latter approach is not 100 percent reliable."
>
> "If we are missing any Trojans for Earth or Mars, or if we are listing non-Trojans for the other planets, please use this feedback form to alert us. Please ensure that you give a reference to a relevant paper in the astronomical literature (an ADS bibcode is sufficient) that supports your claim."
>
> "These lists were last updated on 2026 May 30"

Symbol explanations, https://www.minorplanetcenter.net/iau/lists/StatsExplanation.html [FETCHED, verbatim]:

> "Ln For Trojans, this column indicates whether the minor planet is near the L4 or the L5 point of the planet. L4 is 60° preceding the planet and L5 is 60° following."

**Libration is explicitly not checked for Jupiter Trojans.** The cut values are **not published anywhere I could find.**

**My reverse-engineering of the list (2026-05-30 version) [COMPUTED].** I parsed all 15765 rows.

- **Counts:** L4 = 9823, L5 = 5942, ratio 1.653.
- **a range:** L4 4.952–5.450 au; L5 4.955–5.433 au.
- **e:** max 0.297 (L4), 0.286 (L5).
- **i:** max 55.4° (L4), 43.6° (L5). This argues against an i < 50° cut.
- **σ test:** For the 15607 rows at the common epoch 2025-11-21 (JD 2461000.5), I computed λ = M + ω + Ω. I took Jupiter's mean longitude from the J2000 mean elements (L = 34.39644 + 3034.74613 T deg). This approximation is good to about 1°.
  - L4 has σ from +20.4° to +144.6° (median +62.1°).
  - L5 has σ from −144.1° to −32.0° (median −61.3°).
  - **Zero rows have the wrong sign** and **none has \|σ\| < 20°**.

This suggests the following, all **inference, not documented**:

- a window of about 4.95 < a < 5.45 au (equal to Jupiter's q..Q, which Lowell gives as 4.950–5.455 au)
- e < 0.3
- a σ exclusion of about \|σ\| > 20°
- Ln = sign(λ − λJ)

The list includes 25 objects with a ≥ 5.40 au. Those would *not* be type 12 under the new MPC `orbit_type_int` (see 1c). **So the published list and the new orbit_type are different definitions.**

Ln is assigned by orbital location, not by name. (624) Hektor is listed as L4 and (617) Patroclus as L5.

### 1b. MPCORB.DAT orbit-type flag — https://www.minorplanetcenter.net/iau/info/MPOrbitFormat.html [FETCHED via WebFetch summary; short quotes]

Columns 162–165 hold 4 hex-digit flags:

> "The bottom 6 bits (bits 0 to 5) are used to encode a value representing the orbit type (other values are undefined)"

The listed values are 1 Atira, 2 Aten, 3 Apollo, 4 Amor, 5 q < 1.665 AU, 6 Hungaria, 7 unused/internal, 8 Hilda, **9 Jupiter Trojan**, 10 Distant object. The page also says:

> "Note that the orbit classification is based on cuts in osculating element space and is not 100% reliable."
>
> "Note also that certain of the flags are for internal MPC use and are not documented."

**No numerical cuts are given.** There is no L4/L5 bit.

### 1c. New MPC orbit_type (mpc_orbits DB / Orbit-Type API / MPC Explorer) — https://docs.minorplanetcenter.net/mpc-ops-docs/orbits/orbit-types/ [FETCHED, verbatim table row]

> "Middle Solar System — Objects that occupy the orbital region of Mars-crossers up to that of the Jovian Trojans. … **Jupiter Trojan 12 4.8 < a < 5.4 and e < 0.3**"

Adjacent definitions on the same page:

- "Jupiter Coupled 20 a >= 1 and 2 < T < 3"
- "Neptune Trojan 21 29.8 < a < 30.4"
- "Centaur 22 a_Jupiter <= a < a_Neptune"
- constants "a_Jupiter = 5.204 AU"

The DB column is `orbit_type_int` ("Orbit classification"), per https://docs.minorplanetcenter.net/mpc-ops-docs/data-and-services/schema/mpc-orbits/.

**API test [FETCHED, 2026-10-14]**, POST to `https://data.minorplanetcenter.net/api/orbit-type`:

- `{"a":5.2,"e":0.1,"i":60}` returns `{"orbit_type": 12, "orbit_type_long": "Middle Jovian Trojan"}`. There is no inclination cut.
- `{"a":5.43,"e":0.1,"i":10}` returns `{"orbit_type": 20, "orbit_type_long": "Outer Jupiter Coupled"}`. An object that is in the MPC Trojan *list* is not type 12 here.

**No libration check, no σ, no L4/L5.**

### 1d. MPML / known misclassifications

Hsieh et al. 2020 (RNAAS; arXiv 2007.13945) [FETCHED] on P/2019 LD2, which was initially reported as a Jupiter Trojan:

> "similar analyses disputing P/2019 LD2's classification as a Jupiter Trojan were reported by Sam Deen and Tony Dunn in posts to the Minor Planet Mailing List (MPML…)"

A 2022 DPS abstract (BAAS, https://baas.aas.org/pub/2022n8i520p02; **403 on fetch, seen only as a search snippet**) reportedly says the MPC's "non-rigorous algorithm has resulted in the mis-classification of two objects in/near the Jupiter Trojan population", naming (514107) Ka'epaoka'awela and 2019 LD2. **Unverified.**

I found **no MPC statement, MPEC or MPML post giving the actual Trojan cut values.** There is also no indication that MPC checks libration for Jupiter Trojans.

---

## 2. JPL SBDB — https://ssd-api.jpl.nasa.gov/doc/sbdb_filter.html [FETCHED, verbatim]

> "OMB Outer Main-belt Asteroid asteroid Asteroids with orbital elements constrained by (3.2 au < a < 4.6 au)"
>
> "**TJN Jupiter Trojan asteroid Asteroids trapped in Jupiter's L4/L5 Lagrange points (4.6 au < a < 5.5 au; e < 0.3)**"
>
> "AST Asteroid … Asteroid orbit not matching any defined orbit class"
>
> "CEN Centaur Objects with orbits between Jupiter and Neptune (5.5 au < a < 30.1 au)"

The class is **purely a/e**. There is no i cut, no angle cut, no libration check and no L4/L5 split. The prose ("trapped in … L4/L5") is descriptive and not operational. Under this class any 4.6 < a < 5.5, e < 0.3 orbit counts as TJN, including non-librating ones. Note also that **there is no separate Hilda class**: Hildas at a ≈ 4.0 au fall in OMB.

---

## 3. AstDyS / Milani / Beaugé & Roig

### AstDyS proper-elements page — https://newton.spacedys.com/astdys/index.php?pc=5 [FETCHED]

Link text:

> "Trojan asteroids; the catalog contains Trojans." → `/~astdys2/propsynth/tro.syn`

Also listed: `tro.sig` (RMS), `all_tro.members`, `all_tro.famtab`, `all_tro.famrec`. References given on the page:

> "Milani, A. The Trojan asteroid belt: proper elements, stability, chaos and families, CMDA 57, 59-94, 1993 … presents the theory for synthetic Trojan proper elements. The paper Beaugé, C. and Roig, F. A Semianalytical Model for the Motion of the Trojan Asteroids: Proper Elements and Families. Icarus, Volume 153, Issue 2, pp. 391-415, 2001 … presents the results on semianalytic proper elements for Trojan asteroids."

### `tro.syn` header [FETCHED]

```
%Synthetic proper elements Knezevic Jun 2024: 11867 numbered & multiopp. Trojans
%Name        H   da(AU)  D(deg) f(deg/y) e     g("/y)  sin I   s("/y)   L  My
```

[COMPUTED] L = 4: 7404; L = 5: 4463 (ratio 1.659). Integration length "My" = 5 (8724 objects) or 50 (3143 objects). Max D = 40.2°.

**The selection rule for membership in tro.syn is not documented on AstDyS.** Only numbered and multi-opposition objects are included (per the header). Holt+2020 noted AstDyS was last updated in June 2017 with 5553 objects; it has since been updated to Jun 2024.

### Milani 1993 and Beaugé & Roig 2001 [RECALLED]

ADS returned HTTP 405, so these were not fetched.

- **Milani 1993:** computed synthetic proper elements (libration amplitude D or da, e_p, sin I_p) for 174 Trojans by numerical integration and digital filtering. The objects were taken from the then-known list.
- **Beaugé & Roig 2001:** semi-analytic proper elements for about 533 Trojans, which they used to identify families. Neither paper publishes a catalogue-membership rule beyond "known Trojans with good orbits".

---

## 4. Lowell astorb, Rubin, Brož & Rozehnal, Holt, Vokrouhlický

### Lowell astorb — https://asteroid.lowell.edu/astorb/ (same text at https://ftp.lowell.edu/pub/elgb/astorb.html) [FETCHED]

> "16 Outer-planet crossers (excluding Jupiter and Mars Trojans). Asteroids that cross or pass into the heliocentric distance zones between the perihelion and aphelion distances of Jupiter (4.950 to 5.455 AU), …"

**There is no Trojan flag or class definition.** "Jupiter Trojans" are excluded from the crosser code but are never defined. The 4.950–5.455 au zone matches the a-envelope of the MPC list exactly, which may be a coincidence or a shared heritage (inference).

### Rubin / LSST — https://dp2.lsst.io/tutorials/notebook/308/notebook-308-01.html [FETCHED]

> "The orbit_type_int field in the mpc_orbits catalog contains a set of integers used by the IAU Minor Planet Center (MPC) to define Solar System object orbit types … Jupiter Trojan | 12 … TJN = Jupiter Trojan"

Rubin re-uses MPC code 12 (4.8 < a < 5.4, e < 0.3). **There is no independent Rubin definition.**

### Brož & Rozehnal 2011 — arXiv 1109.1109 (ar5iv) [FETCHED]

> "The source of initial osculating elements is the AstOrb catalogue, version JD = 2455500.5 (Oct 31st 2010)."
>
> "We then estimate the libration frequency f by a linear fit of φ(t) = λ − λ′ − χ, where λ, λ′ are the mean longitudes of an asteroid and Jupiter and χ = ±60° for L4 or L5 respectively. The revolution of angle φ(t) must not be confined to the interval [0,360°) …"
>
> "We computed proper elements for 2647 L4 and 1496 L5 Trojan asteroids. … The ratio of populations valid for H ≲ 15 mag asteroids is thus N(L4)/N(L5) ≃ 1.8."

**The pre-selection from AstOrb is not stated in the paper.** Data: http://sirrah.troja.mff.cuni.cz/~mira/mp/

### Holt et al. 2020 — arXiv 2009.06749 [FETCHED]

> "Trojans from the AstOrb catalogue as of April 28, 2020, from which their population was also identified."
>
> "proper elements computed for 7328 Jovian Trojans (numbered and multiopposition objects as of April 2020) … the L4 swarm ('Greeks' leading Jupiter …) … contains 4607 objects. The smaller L5 swarm contains only 2721 known objects"

On AstDyS:

> "their last update occurred in June 2017 … 5553 numbered and multi-opposition Jovian Trojans"

**The identification criterion is not stated.**

### Vokrouhlický et al. 2024 — arXiv 2401.15537 (= AJ 167, 138) [FETCHED, verbatim]

> "First we downloaded the orbital catalog MPCORB.DAT from the Minor Planet Center website (the February 22, 2023 version). Next we selected all inputs using a simple criteria in osculating orbital elements which were available for a common epoch of MJD 60,000. The initial database represented about 1,245,000 orbits, but about 2% were rejected because they reflected poorly-determined single-opposition orbits given at different, mostly earlier epochs. The orbital selection was: (i) semimajor axis a in between 4.95 au and 5.45 au, (ii) eccentricity smaller than 0.6, (iii) inclination smaller than 50°, and (iv) absolute value of the resonant angle σ larger than 20° and smaller than 130°. This search resulted in 7,929 orbits around the L4 libration center and 4,189 orbits around the L5 libration center. A small number of these orbits, however, correspond to unstable Jupiter family comets masquerading as JTs. This led us to numerically propagate all selected orbits, together with all planets and the Sun, for a 1 Myr timespan. We found that 26 and 12 candidate Trojans near L4 and L5, respectively, escaped from the libration region about their Lagrange center, were discarded from the simulation by striking Jupiter, or were ejected from the Solar system. The remaining 7,903 bodies near L4 and 4,177 bodies near L5 were found to reside in the resonant zone of their respective center, with their σ librating for the entire timespan of the integration."

Definition of σ (Sec. III): "with σ = λ − λJ, da = a − aJ (aJ ≃ 5.206 au)".

Appendix A:

> "we used the MPC catalog of minor planet orbits as of April 15, 2023 containing about 1,275,000 entries, and selected Jupiter Trojans using the same steps as described in Sec. II. We obtained 8,144 orbits in the L4 swarm, and 4,252 orbits in the L5 swarm. … now we did not drop the orbits with short observation arcs (except the really poorly determined ones with arcs less than a few days)."

Catalogue: "available at https://sirrah.troja.mff.cuni.cz/~mira/tmp/trojans/".

**The user's paraphrase is confirmed.** Note "in between 4.95 au and 5.45 au" and "smaller than 0.6 / 50°", without ≤/≥ precision.

### Vokrouhlický online catalogue — https://sirrah.troja.mff.cuni.cz/~mira/tmp/trojans/ [FETCHED]

Directory listing:

- `ei2.eps`, `fAppA3.eps`, `figs.tar.gz`
- `greeks-hcm_20230405_GS/`, `trojans-hcm_20230405_GS/`
- `paper1.pdf`, `paper1.tex`, `rsync.sh`
- `trojans_L4_families_2023Apr/`, `trojans_L5_families_2023Apr/`

**There is no README.** Each `*-hcm_*` directory contains `astorb_follow.dat`, `astorb_follow.tmp`, `follow.proper.out_dfD` and `follow.proper.pro`. There is no column header.

[COMPUTED] Line counts: `follow.proper.pro` has 8155 lines (greeks = L4) and 4252 lines (trojans = L5). The L5 count matches the paper; the L4 count is 11 more than the paper's 8144.

Columns of `.pro` (inferred, not documented): number, H, mean a, e_P, sin I_P, then zeros, then a flag "+10". "greeks" = L4 and "trojans" = L5 follows the camp naming convention.

---

## 5. Gladman, Marsden & VanLaerhoven 2008

Source: "Nomenclature in the Outer Solar System", SSBN pp. 43–57. PDF: https://web.gps.caltech.edu/~mbrown/out/kbbook/Chapters/Gladman_Nomenclature.pdf [FETCHED, verbatim]

> "For each three-opposition object, the best-fit orbit is integrated, along with two other nominal orbits that are the extremes of the orbital uncertainty in semimajor axis … If in a 10-m.y. integration a particular resonance argument librates for all three initial conditions, the object has a 'secure' resonant classification. If two of these initial conditions show the resonant behavior the object will be classified as 'probably' in the resonance; if only one of the arguments is resonant, we classify the object according to the behavior of the other two trajectories"
>
> "If only one of three test particles librates the object is classed as nonresonant"
>
> "Note that the 10-m.y. future window used here does not require the object to be resonant over the age of the solar system, but only for a small number of resonant librations. This is in keeping with the philosophy that it is the current dynamics of the TNOs that we are classifying."
>
> "Currently only the eccentricity-type mean-motion resonances between TNOs and Neptune are securely known to be populated, ranging from the 1:1 Trojan resonance with Neptune …"

On Jupiter: JFCs are T_J < 3.05 with q < 7.35 au.

> "the combined TJ and q condition … tells us what is beyond Jupiter's reach and in need of classification"

**The scheme does not classify Jupiter Trojans.** It covers only Neptune resonances, including the Neptune 1:1. It is the template for a libration-based definition (nominal + 2 clones, 10 Myr, ≥3 oppositions) that no Jupiter-Trojan catalogue formally adopts.

ML follow-ups were checked only via search snippets. Khain et al. 2020 (AJ 159, 133; DES TNOs) and Smullen & Volk 2020 (arXiv 2007.03720) cover Neptune resonances; **no Jupiter 1:1 treatment was found.**

---

## 6. IAU WGSBN naming

Source: *Rules and Guidelines for Naming Non-Cometary Small Solar-System Bodies*, Version 1.1 (2025 February 22). https://www.wgsbn-iau.org/documentation/NamesAndCitations.pdf [FETCHED, verbatim]

Section 4, "Special names required for special objects":

> "In the table below, the defining orbital elements are indicated by their standard symbols: q, perihelion distance; and a, semi-major axis."
>
> "**Jupiter Trojans — Names from the Trojan War (if Hv brighter than 12.0) or names of Olympians and Paralympians (if Hv fainter than or equal to 12.0).**"
>
> "Neptune Trojans — Names of mythological Amazons."
>
> "For athlete names used for small Jupiter Trojans, the shortest unique version of the name (using just the surname or forename, if possible) is preferred over a concatenated full name."

Section 5:

> "for small Jupiter Trojans (see section 4), this means considering both male and female athletes who participated at summer or winter Olympics or Paralympics"

**Findings:**

- The table promises "defining orbital elements", but the Jupiter Trojan row gives **no orbital definition**.
- The **current rules contain no L4 = Greek / L5 = Trojan requirement.** I grepped for "L4", "L5", "Greek", "camp", "Hektor" and "Patroclus": no hits.
- The H threshold is **12.0** ("fainter than or equal to 12.0" → athletes). It is not 14.

History, from Wikipedia "Jupiter trojan" raw wikitext [FETCHED; secondary source]:

> "In 2018, at its 30th General Assembly in Vienna, the International Astronomical Union amended the naming convention for Jupiter trojans, allowing for asteroids with H larger than 12 … to be named after Olympic or Paralympic athletes"
>
> "Asteroids in the leading (L4) orbit are named after Greek heroes (the 'Greek node or camp' or 'Achilles group'), and those at the trailing (L5) orbit are named after the heroes of Troy (the 'Trojan node or camp').[Nicholson 1961] The asteroids 617 Patroclus and 624 Hektor were named before the Greece/Troy rule was devised"

WGSBN 2015–2018 triennial report (https://www.klet.org/csbn/docs/WGSBN2015-2018_rev.pdf) [FETCHED]:

> "A new guideline was adopted for the naming Neptunian Trojans: the names of Amazons will be used, with no differentiation between objects at L4 and L5."

The Jupiter Greek/Trojan camp split is therefore a **historical convention** (Nicholson 1961; Wyse 1938). It is **not in the current WGSBN text**. The athlete names on the Wikipedia list go to both camps.

---

## 7. Survey sample definitions

| Paper | Verbatim selection | N | Tag |
|---|---|---|---|
| **Szabó et al. 2007** (arXiv astro-ph/0703026) | "These 482 observations are identified in orbital space using constraints 5.0 AU < a < 5.4 AU and e < 0.2, and hereafter referred to as the Known Trojans (KT)." The candidates are then selected kinematically: "the selection algorithm must also include the longitudinal angle from the opposition, φ … for observations with −25 < φ < 25". | "1187 observations of about 860 unique candidate Jovian Trojan asteroids"; "N(L4)/N(L5) = 1.6 ± 0.1, independently of limiting object's size" | FETCHED |
| **Grav et al. 2011** (arXiv 1110.0280) | "As of 2011 April 5, there are currently 4846 known objects that the MPC has identified as Jovian Trojans. … NEOWISE observed 1751 objects that have orbits with semi-major axis between 5.0 and 5.4 AU with arcs long enough to securely identify them as Trojans (here set somewhat arbitrarily to 18 days), with 981 of these in the leading and 770 in the trailing cloud." | "N(leading)/N(trailing) ∼ 1.4 ± 0.2, lower than the 1.6 ± 0.1 value derived by Szabó et al. (2007)" | FETCHED |
| **Grav et al. 2012** (arXiv 1209.1549) | same sample lineage | "the leading cloud has a larger number of these large [D > 50 km] objects, L4/L5 = 1.34" | FETCHED |
| **Jewitt, Trujillo & Luu 2000** (arXiv astro-ph/0004117) | "we used the sky-plane angular speed to distinguish Trojans from main-belt asteroids. The main-belt asteroids move westward at rates generally higher than the more distant Trojans" | "Ninety-three Jovian Trojans were detected" (L4 only); N(L4, r ≥ 1 km) ~ 1.6×10⁵ | FETCHED |
| **Yoshida & Nakamura 2005** (AJ 130, 2900; L4) and **2008** (PASJ 60, 297; L5); **Nakamura & Yoshida 2008** (PASJ 60, 293; surface-density model) | Apparent-motion selection on Subaru Suprime-Cam fields | 2006 JpGU abstract (search snippet): "Fifty-one L4 Trojans and sixty-two L5 Trojans were detected" | snippet / RECALLED |
| **Yoshida & Terai 2017** (arXiv 1706.10017) | "we limit the motion range to be searched covering their motion distribution derived from orbits with eccentricity of 0.0–0.3 and inclination of 0°–40° … We regard the detected moving objects with motion vectors located on this area as JT/Hilda candidates." | "We detected 631 L4 JTs"; 481 unbiased | FETCHED |
| **Wong & Brown 2015** (arXiv 1510.03144) | "only those objects detected in regions of the (λ, β, \|v\|) space with Trojan fraction γ = 1 (i.e., zero expected contamination)". The bright end uses "all L4 Trojans contained in the Minor Planet Center (MPC) catalog brighter than H = 12.3". | 557 L4 | FETCHED |
| **Uehata et al. 2022** (arXiv 2204.08617) | "we identify L5 Jupiter Trojans based on their apparent motion (see Yoshida & Terai 2017 …) … the middle curve was used to distinguish between Hildas and Trojans" | "189 L5 Jupiter Trojans"; "NL4/NL5 = 1.40 ± 0.15" (D ≥ 2 km) | FETCHED |
| **Slyusarev, Glezina & Belskaya 2018** (EPSC2018-807) | catalogue source not stated | "4599 objects known in L4 and 2433 in L5 population, i.e. L4 Trojans are more numerous than L5 in 1.9 times"; "observed in the entire range of absolute magnitude values, including the area that not distorted by the effect of observational selection (H<14)" | FETCHED |

None of the surveys checks libration.

- **Deep pencil-beam surveys** (Jewitt+2000; Yoshida & Nakamura; Yoshida & Terai; Wong & Brown; Uehata) classify by **apparent sky motion**, with L4/L5 set by field pointing.
- **SDSS** (Szabó) and **NEOWISE** (Grav) use an **a-window of 5.0–5.4 au** (with e < 0.2 for SDSS calibration) on MPC/ASTORB orbits, plus kinematics for SDSS.

---

## 8. Observations relevant to the thesis

1. **The measured N4/N5 depends on which definition and which epoch you use:**
   - V24 (libration-cleaned, Feb 2023): 1.89
   - MPC list (May 2026): 1.65
   - AstDyS (Jun 2024): 1.66
   - Brož & Rozehnal (2010): ≈1.77 overall (1.8 for H ≲ 15)
   - Holt (2020): 1.69
   - Slyusarev (2018): 1.9
   - Szabó (SDSS): 1.6 ± 0.1
   - Grav (NEOWISE): 1.4 ± 0.2
   - Uehata (HSC, D ≥ 2 km): 1.40 ± 0.15

   Part of the drift between the 2023 and 2026 MPC-based counts is plausibly discovery geometry, since L5 has been better placed recently. That is inference.
2. The **official definitions are all osculating a/e boxes**: MPC new type 12 (4.8–5.4, e < 0.3) and JPL TJN (4.6–5.5, e < 0.3). They do not test libration and do not split L4/L5. Only the MPC *list* assigns L4/L5, and its rule is undocumented; it is empirically sign(λ−λJ) with \|σ\| ≳ 20°.
3. The only fully specified, libration-tested public recipe is **V24**: box + 20° < \|σ\| < 130° + 1-Myr libration. The **Gladman+2008** protocol (10 Myr, nominal + 2 a-extremal clones, ≥3 oppositions) is the outer-Solar-System analogue that no Jupiter-Trojan catalogue applies.
4. The WGSBN naming text has **no camp rule** and **no orbital definition**. The "Greek camp / Trojan camp" language is convention (Nicholson 1961). Hektor (L4) and Patroclus (L5) are the classical exceptions, and the MPC list confirms their Ln.
