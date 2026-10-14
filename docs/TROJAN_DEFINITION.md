# What is a Jupiter Trojan? Definitions in the literature, and the one we use

Written 2026-10-14 after Simon and I questioned whether our "Trojans" are the
observers' "Trojans". It collects every definition we could find, states the
catalogues' exact rules, measures how much the choice matters for our sample, and
proposes the definition used from now on.

The verbatim quotes, page numbers and URLs are in `docs/trojan_definition_sources/`:
- `local_library.md`: every PDF in `literature/`;
- `catalogues_and_surveys.md`: MPC, JPL, AstDyS, Lowell, Rubin, IAU, and the surveys;
- `dynamical_and_simulation.md`: theory, stability and capture papers.

Each claim there is tagged FETCHED/VERIFIED (quoted from text we downloaded) or
RECALLED (not checked). All web pages were accessed 2026-10-14.

**Short answer.**
- **No catalogue or naming authority tests libration.** MPC, JPL and Rubin use
  osculating-element boxes, and the MPC calls its own method "not 100 percent
  reliable". The IAU naming rules contain no orbital definition and no L4 = Greek rule.
- **Only research catalogues test libration.** These are Vokrouhlický+2024 (V24),
  AstDyS proper elements and Greenstreet+2024 (1000 clones). For bright, well-observed
  objects they all agree with the boxes.
- **On our sample the box labels are, in practice, a dynamical definition.** With good
  orbits (U ≤ 4), 99.6–99.9% of objects under *every* label (V24, MPC, JPL) are
  tadpoles of their labelled swarm for all 10 Myr of the pilot (§4). The present-day
  loss-rate result therefore compares like with like.
- **The mismatch is on the simulation and capture side.**
  - Pirani+2019a counts every particle with q and Q within a_J ± 2.5 Hill radii,
    horseshoe and quasi-satellite orbits included, 5 Myr after capture.
  - Our parked `migration/` code uses a position snapshot.
  - Neither result is directly comparable with the observed N4/N5.
- **Proposal (§5).** A Trojan is a tadpole librator: σ = λ − λ_J crosses neither 0° nor
  180° in any 10 kyr window over at least 1 Myr. A Greek (L4) has mean σ > 0; a
  Trojan-camp object (L5) has mean σ < 0. The same classifier is applied to real and
  simulated bodies. Horseshoe, quasi-satellite and transient co-orbitals are separate
  classes and never enter N4/N5.

---

## 1. Catalogues and naming authorities: their exact rules

| Source | Rule (verbatim where quoted) | Libration? | L4/L5 rule | N4 / N5 |
|---|---|---|---|---|
| **MPC, List of Jupiter Trojans** (updated 2026-05-30) | "For Earth, Mars and Uranus, the listed objects are only those that have been confirmed as Trojans by long-term integrations. For the other planets, the objects are selected automatically using cuts in osculating element space. This latter approach is not 100 percent reliable." The cut values are not published. In the list: 4.952 ≤ a ≤ 5.450 au, e ≤ 0.297, i ≤ 55.4°, 20.4° ≤ \|σ\| ≤ 144.6° (our inference from the data) | **No**, stated explicitly | `Ln`: "L4 is 60° preceding the planet and L5 is 60° following". Always matches sign σ | 9,823 / 5,942 = 1.65 |
| **MPCORB.DAT type flag** (value 9) | "cuts in osculating element space … not 100% reliable"; no numbers | No | — | — |
| **MPC new `orbit_type_int` = 12** (mpc_orbits, MPC Explorer) | "4.8 < a < 5.4 and e < 0.3". There is no i cut: the API returns 12 for i = 60°. a = 5.43 returns type 20 ("Jupiter Coupled"), yet 25 list members have a ≥ 5.40, **so the MPC list and the MPC orbit type are two different definitions** | No | — | — |
| **JPL SBDB class TJN** | "Asteroids trapped in Jupiter's L4/L5 Lagrange points (4.6 au < a < 5.5 au; e < 0.3)" | No | none | — |
| **Rubin/LSST** | Re-uses MPC type 12 and calls it "TJN" | No | — | — |
| **Lowell astorb** | No Trojan class. Jupiter's zone is 4.950–5.455 au | — | — | — |
| **AstDyS `tro.syn`** (Knežević, June 2024) | Milani-type synthetic proper elements for numbered and multi-opposition Trojans. **The rule for who enters the file is not documented** | Implicitly (a libration amplitude is computed) | `L` = 4 / 5 | 7,404 / 4,463 = 1.66 |
| **IAU WGSBN Rules v1.1** (2025-02-22) | "Jupiter Trojans — Names from the Trojan War (if Hv brighter than 12.0) or names of Olympians and Paralympians (if Hv fainter than or equal to 12.0)." **No orbital definition and no L4/L5 camp rule** | — | none | — |

**Greek vs Trojan camp.** "L4 = Greek camp, L5 = Trojan camp" is a naming tradition
(Nicholson 1961, via secondary sources). (624) Hektor (L4) and (617) Patroclus (L5)
predate it and are the classical exceptions. It is not in the current WGSBN rules,
and no paper defines the camps dynamically. Dynamically there is only "leading" and
"trailing".

**Correction to `CATALOGUE_DEFINITIONS.md` §1.** That note said the MPC publishes no
criterion. It does state the *kind* of criterion (osculating cuts, no integration),
though not the numbers.

## 2. Research catalogues and observational samples

| Source | Definition | Libration test | N4 / N5 |
|---|---|---|---|
| **Vokrouhlický+2024** (AJ 167, 138) | MPCORB of 2023-02-22, osculating elements at MJD 60000: "(i) a in between 4.95 au and 5.45 au, (ii) e < 0.6, (iii) i < 50°, (iv) 20° < \|σ\| < 130°", σ = λ − λ_J. Then 1 Myr integration, kept "with their σ librating for the entire timespan". Debiasing domain uses quasi-proper (A, B, C), tadpole C ∈ (1.5, 2.5) | Yes: 1 Myr, nominal orbit, no clones. Removed 26 L4 + 12 L5 of 12,118 | 7,903 / 4,177 = 1.89 (all H); **1.45 ± 0.05 debiased, families removed** |
| **Greenstreet+2024** (ApJL 963, L40; our file `Greenstreet+2023_…`) | JPL SBDB Aug 2022, 4.5 ≤ a ≤ 5.9 au, arc > 30 d. 1000 clones each, 0.5 Myr. Per 1 kyr window: ⟨a⟩ within 0.2 au of a_J and no excursion > 0.65 au. "Trojan" = ≥ 95% of clones resonant for the full 0.5 Myr. Separate classes: horseshoe, quasi-satellite, transient/metastable | Yes, the most careful in the literature | 7,482 / 3,941 = 1.90; 14 MPC/JPL "Trojans" are non-resonant |
| Holt+2020b (MNRAS 495, 4085) | JPL 4.6 < a < 5.5, e < 0.3 (April 2018), restricted to objects with AstDyS proper elements | Delegated to AstDyS | 1.89 |
| Brož & Rozehnal 2011 | AstOrb Oct 2010; pre-selection not stated; φ = λ − λ′ − χ with χ = ±60° | Yes (proper elements) | 2,647 / 1,496 |
| Milani 1993; Beaugé & Roig 2001 | Proper elements of the then-known Trojans | Yes | (RECALLED, not checked) |
| Szabó+2007 (SDSS) | Calibration set 5.0 < a < 5.4, e < 0.2; candidates from sky motion | No orbits | 1.6 ± 0.1 |
| Grav+2011/2012 (NEOWISE) | "semi-major axis between 5.0 and 5.4 AU", arc ≥ 18 d | No | ~1.4 ± 0.2 debiased |
| Subaru/HSC (Yoshida & Terai 2017; Uehata+2022; Yoshida+2026) | Apparent-motion windows, single swarm per field | No orbits | Uehata: 1.40 ± 0.15 (D ≥ 2 km) |
| Gladman+2008 (outer Solar System nomenclature) | Nominal orbit + 2 clones at the a-uncertainty extremes, 10 Myr. Libration in 3/3 = "secure", 2/3 = "insecure". Not applied to Jupiter's 1:1 (T_J < 3.05 objects are "Jupiter-coupled") | Yes | — |

**The definition hardly moves the all-H ratio; the H cut and the date do.**
- Every libration-tested all-H sample gives 1.89–1.90: V24, Greenstreet, Holt, and
  Hellmich's AstDyS sample.
- The complete or debiased samples give 1.3–1.5. The 1.3 is Nesvorný+2013's MPC
  H < 12 with Eurybates removed; that sample is not stable over time, since MPC H
  revisions dropped the counts by 2018.
- The MPC list of May 2026 gives 1.65. Since V24's Feb 2023 snapshot, L5 grew by about
  42% and L4 by about 24%. We guess recent survey geometry; this is unchecked.
- So a ratio must always be quoted with its H cut, family treatment and catalogue date.

## 3. Dynamical and simulation definitions

Five families are in use. Most papers are explicit about their entry test, but
looser about their exit test.

| Family | Criterion | Papers |
|---|---|---|
| **Tadpole libration** | σ librates about the (e, I-dependent) L4/L5 centre without reaching 180°. Levison+1997 drops an object once it "visits all four quadrants" | Murray & Dermott 1999; Levison+1997; Nesvorný & Dones 2002; Lykawka & Horner 2010 |
| **Libration + survival time** | Tadpole held for a set time: ~9.5 kyr (Horner & Evans 2006), 0.5 Myr (Greenstreet+2024), 1 Myr (V24; Li+2023a,b, whose "survived" is undefined), 4 Gyr (Nesvorný+2013 "stable Trojans", criterion unstated) | as listed |
| **Proper elements** | Catalogued object with stable proper d/D, e_p, sin I_p | Milani 1993; Beaugé & Roig 2001; Brož & Rozehnal 2011; Holt+2020 entry |
| **Geometric snapshot** | Box in position or elements at one time. **Pirani+2019a: "We consider as Trojans all those particles that have aphelion and perihelion within ±2.5 Jupiter Hill radii"**, so horseshoe and quasi-satellite orbits are included, with no stated L4/L5 rule, at t = 5 Myr. Montesinos+2020: ±22.5° × ±1 R_H around ±60° | Pirani+2019a,b; Montesinos+2020; Chen+2016 (Neptune) |
| **"Not yet escaped"** (exit only) | Di Sisto+2014: encounter < 1 R_H. Di Sisto+2019: < 2.1 R_H. Holt+2020: osculating a leaves 4.6–5.5 au. Hellmich+2019: r < 3 or > 8 au, or < 2 R_H. None of these tests libration | as listed |

**Where definitions change the answer.**
- **Escape criterion.** On the same 1,975 / 997 numbered Trojans, L5 escape over
  4.5 Gyr is 28.3% with Di Sisto 2014's criterion and 25.1% with Di Sisto 2019's (L4:
  23% and 23.6%).
- **Clone criterion.** Holt+2020: 23–25% escape counting all clones, about 5% if all 9
  clones of an object must escape.
- **Horseshoes.** Pirani+2019a's L4 excess arises on the horseshoe side and is then
  frozen into tadpoles. Its "Trojans" still include whatever horseshoe and
  quasi-satellite orbits remain at 5 Myr. How much of its N4/N5 is tadpole-only is
  never stated.
- **Libration amplitude.** Papers use the full peak-to-peak amplitude (Nesvorný+2013),
  the semi-amplitude (Di Sisto, Tsiganis), the RMS half-amplitude (Parker) or the C
  integral (V24). Amplitude cuts don't transfer between papers.
- **Swarm label.** Most papers use the snapshot sign of σ or the starting swarm.
  Greenstreet+2024 requires σ to stay in one hemisphere in every 1 kyr window.
  Lykawka+2009 uses the libration centre (60 ± 5°, 300 ± 5°).

## 4. Does it matter for our sample? Measured

`campaign/scripts/s3_membership.py` → `campaign/reports/pilot_10myr/membership_by_label.csv`.
The classifier is the pilot's own (`trojans/integrate.py`). In each 10 kyr window it
records whether σ crossed 0° or 180°. A window counts as a tadpole of the swarm given
by the sign of the circular mean of σ if σ crossed neither. Removal (< 1 R_H) counts
as failure. "Tadpole, 10 Myr" means a tadpole of the t0 swarm in *every* window.

| Sample | Swarm | n | Tadpole of own swarm, 1 Myr | Tadpole of own swarm, 10 Myr | Not tadpole (10 Myr) |
|---|---|---|---|---|---|
| V24 box, all U | L4 / L5 | 10,362 / 5,974 | 98.3% / 99.3% | 97.8% / 99.2% | 223 / 50 |
| MPC list, all U | L4 / L5 | 9,821 / 5,941 | 98.4% / 99.6% | 98.0% / 99.4% | 192 / 33 |
| JPL TJN, all U | L4 / L5 | 10,415 / 5,976 | 97.8% / 99.3% | 97.4% / 99.1% | 274 / 52 |
| **V24 box, U ≤ 4** | L4 / L5 | 8,815 / 5,528 | 99.8% / 99.7% | **99.8% / 99.6%** | 20 / 21 |
| **MPC list, U ≤ 4** | L4 / L5 | 8,765 / 5,512 | 99.9% / 99.9% | **99.9% / 99.9%** | 10 / 6 |
| JPL TJN, U ≤ 4 | L4 / L5 | 8,813 / 5,523 | 99.9% / 99.8% | 99.8% / 99.7% | 16 / 16 |
| **Complete, H ≤ 13.6** | L4 / L5 | 1,836 / 1,213 | 99.9% / 99.8% | **99.9% / 99.8%** | 2 / 3 |

**What this shows.**
- **With good orbits, the catalogue label and the dynamical definition agree to 0.1–0.4%.**
  The non-librators at U ≤ 4 are 41 objects out of 14,343. They are the pilot's escapers,
  which visit horseshoe, quasi-satellite and circulating states on the way out.
- **N4/N5 doesn't change.** Under the tadpole definition, U ≤ 4 gives 8,795 / 5,507 =
  1.597, against 1.595 for the label. The complete sample gives 1,834 / 1,210 = 1.516,
  against 1.514. Against the 562 extra cold L4 objects, 5 non-librators are
  negligible.
- **The disagreement lives at U ≥ 5.** That is where our earlier interloper problem
  came from, and it is already cut. JPL's label is the loosest: 80 JPL-only objects,
  all gone within 1 Myr (`CATALOGUE_DEFINITIONS.md` §1).
- **Leaving the tadpole and removal are the same event.** In every row of
  `cleaning.csv`, the objects that fail the 1 Myr libration test are exactly the
  objects removed at 1 R_H within 1 Myr. So our escape criterion (1 R_H) and a
  libration-based exit give the same loss rate. Unlike Di Sisto 2014 vs 2019, our
  result doesn't depend on that choice.

**Conclusion for the loss rate.** The present-day erosion result (L5/L4 escape ratio
1.00 ± 0.57 at U ≤ 4) compares observed Trojans with themselves under a definition
that agrees with every catalogue to better than 0.5%. It is not undermined. The fear
applies to the **capture** side:

| Our code / model | Definition | Compatible with observed sample? |
|---|---|---|
| Present-day campaign (`campaign/trojans/integrate.py`) | 10 kyr window tadpole classes: L4, L5, horseshoe, quasi-satellite, circulating | **Yes** |
| Parked migration code (`migration/simulation/config.py` `TrojanCriteria`) | Snapshot: \|r/a_J − 1\| < 0.07 and \|Δλ ∓ 60°\| < 25° (heliocentric distance, not a) | **No.** It misses tadpoles with \|σ − 60°\| > 25° at that instant, which V24 admits up to 130°, and Trojans with e ≳ 0.07. It counts horseshoes passing through the sector |
| Pirani+2019a | q, Q within a_J ± 2.5 R_H (implies e ≲ 0.17; ours) at 5 Myr; horseshoe and quasi-satellite orbits included | **No**, unless re-classified |

## 5. Proposed definitions

All angles use σ = λ − λ_J (heliocentric osculating mean longitudes, as in
`trojans/diagnostics.py`).

1. **Window class.** Each window of W = 10 kyr (about 70 libration periods) is one of:
   - tadpole L4 or tadpole L5: σ crosses neither 0° nor 180°; the swarm is the sign
     of the circular mean of σ;
   - horseshoe: σ crosses 180° but not 0°;
   - quasi-satellite: σ crosses 0° but not 180°;
   - circulating: σ crosses both.
2. **Jupiter Trojan.** A body whose every window over an interval of T ≥ 1 Myr is
   tadpole of the *same* swarm.
   - T = 1 Myr matches V24, the observational reference.
   - Going to T = 10 Myr changes membership by < 0.1% (§4). "Long-lived Trojan" is
     reserved for the full run.
3. **Greek (L4).** A Jupiter Trojan whose σ stays in (0°, 180°) (leading).
   **Trojan-camp object (L5).** A Jupiter Trojan whose σ stays in (−180°, 0°) (trailing).
   - Because a tadpole never crosses 0° or 180°, the snapshot sign, the window mean and
     the MPC `Ln` always agree. We found zero disagreements in 16,336 objects
     (`CATALOGUE_DEFINITIONS.md` §3).
   - In text, write "L4 (Greek camp)" and "L5 (Trojan camp)". Use "Trojan" alone only
     for the union, never for L5.
   - Hektor and Patroclus are naming exceptions with no dynamical meaning.
4. **Other co-orbitals, never counted in N4/N5.**
   - Horseshoe and quasi-satellite objects: their own window class for ≥ T.
   - Transient co-orbitals: co-orbital windows for less than T.
   - Box interlopers: inside the V24 box at t0 but not librating.
   - A swarm switch (tadpole L4 → horseshoe → tadpole L5) is booked as a loss from one
     swarm and a capture into the other.
5. **Capture and loss rates.** These are transitions of the *same* classifier.
   - Capture is the start of a run of tadpole windows lasting ≥ T.
   - Loss is the first non-tadpole window after membership. In the present-day runs
     this coincides with removal at 1 R_H (§4).
6. **Observed sample.** The candidate set is the V24 box at the common epoch (a
   superset; the cuts are only an entry gate). Then:
   - membership as in item 2, from our own integration;
   - U ≤ 4 for dynamics (orbit error < 0.3% of the tadpole width; our substitute for
     Gladman's clone test, since we run no clones);
   - H ≤ 13.6 (complete) for distributions and ratios.
7. **Simulated sample.** The same classifier, at the epoch matched to the observation
   (giants on their final orbits), with survival to that epoch.
   - Report N4/N5 for tadpoles only.
   - List horseshoe and quasi-satellite counts separately.
   - Bin in I (and e, libration amplitude), as for the observed cold-L4 excess.

## 6. What follows

- **Capture comparisons are meaningful only after re-classification.** Before any
  migration or capture run is compared with the observed N4/N5, replace `TrojanCriteria`
  with the window classifier of §5. When we quote Pirani's ratios, say they include
  horseshoe and quasi-satellite orbits and are read 5 Myr after capture.
- **Re-quote observed ratios with their full definition.** Ours: V24 box + 10 Myr tadpole,
  U ≤ 4: 1.60. Complete H ≤ 13.6: 1.52. V24 debiased: 1.45 ± 0.05. Literature all-H
  libration-tested: 1.89–1.90.
- **For the meeting.** Simon's worry is justified for the literature (Pirani, Li,
  Nesvorný+2013 all use different and sometimes unstated definitions) and for our
  parked migration code. It is not borne out for our present-day sample.
- **Library housekeeping** (found during this survey):
  - `literature/01_observations/Uehata+2022_…ATLAS.pdf` is actually McNeill+2021
    (arXiv:2101.04602). The real Uehata+2022 (AJ 163, 213) isn't local.
  - `RobutelGabern2009_…II.pdf` is Robutel & Bodossian 2009.
  - The local Holt paper is Holt+2020b. V24's proper-element method is in Holt+2020a
    (MNRAS 499, 3630), which isn't local.
- **Not verified:** the Milani 1993 and Beaugé & Roig 2001 sample definitions (paywalled);
  the reason for the MPC ratio drift from 1.89 (2023) to 1.65 (2026).
