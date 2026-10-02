# Capture Mechanisms, Inclination, and the Time-Symmetry Framing

Written 2026-10-02, as Step 4 of the plan in `docs/TIME_SYMMETRY_HYPOTHESIS.md`. It covers
what the literature says about (i) mechanisms that could make L5 hotter as well as smaller,
(ii) whether the "broken time symmetry" reading is new, and (iii) the control populations.
It does not repeat `docs/ASYMMETRY_REVIEW.md` (AR) or `docs/VOKROUHLICKY_DISISTO_REVIEW.md`
(VDR); it cites their sections instead.

**Reading rules used.** Every claim comes from the full text, with section, page, table or
equation. Page numbers are those of the arXiv/preprint PDF unless a journal page is given.
"(ours)" marks a number or argument produced here, not by the cited authors.
"**Unverified**" means I saw only a secondary citation or a search-engine abstract.

**Short answer.** *No published mechanism predicts that L5 is hotter.* None of them even
reports per-swarm inclinations; the one paper that discusses an L4/L5 inclination difference
(Li et al. 2023b) treats it as a defect to avoid. But the gap is narrower than it looks,
because **"L5 hotter" and "the asymmetry weakens as i rises" are the same fact**. In our
complete sample, families removed, N4/N5 = 2.01 ± 0.10 for I < 15° and **0.98 ± 0.06 for
I ≥ 15°** (ours, §1.0). The *entire* number excess is an excess of cold L4 objects; the hot
halves of the two swarms are equal. Li et al. (2023a, Table 5) get the right sign of this
dependence but about 5× too weak (my estimate: KS D ≈ 0.02, not 0.21). On time symmetry, the
ingredients are partly known. Sicardy & Dubois (2003) show that the asymmetry needs an
L4/L5-odd term: a differential torque, which flips with migration direction. Growth alone,
although it is time-dependent, gives none. Pirani et al. (2019a) flipped the sign by reversing
the migration. **Nobody I found** invokes the Roy–Ovenden mirror map (x, −y, z, −t), pairs
capture into L4 with escape from L5, or argues that steady state forces symmetric exchange
today. The framing is therefore **partly known, not stated**. It also needs one correction:
"an arrow of time" is necessary but not sufficient.

---

## 1. Mechanisms

### 1.0 First, what the inclination observable actually is (ours)

Proper I from V24, our catalogue `2026-09-15`, the same samples as `INITIAL_CONDITIONS_KS.md`
(scratch computation through `s3_distributions.load("pilot_10myr")`; not yet a repo script):

| Proper I bin | complete H ≤ 13.6: N4 / N5 | ratio | complete, no families: N4 / N5 | ratio |
|---|---|---|---|---|
| 0–5° | 282 / 134 | 2.10 ± 0.22 | 282 / 134 | 2.10 ± 0.22 |
| 5–10° | 560 / 209 | 2.68 ± 0.22 | 457 / 209 | 2.19 ± 0.18 |
| 10–15° | 389 / 216 | 1.80 ± 0.15 | 382 / 216 | 1.77 ± 0.15 |
| 15–20° | 248 / 233 | 1.06 ± 0.10 | 231 / 229 | 1.01 ± 0.09 |
| 20–25° | 182 / 180 | 1.01 ± 0.11 | 169 / 170 | 0.99 ± 0.11 |
| 25–30° | 101 / 168 | 0.60 ± 0.08 | 101 / 119 | 0.85 ± 0.11 |
| ≥ 30° | 74 / 71 | 1.04 ± 0.17 | 58 / 52 | 1.12 ± 0.21 |
| **< 15° / ≥ 15°** | 1231/559, 605/652 | **2.20 / 0.93** | 1121/559, 559/570 | **2.01 / 0.98** |

- **The excess N4 − N5 = 551 (no families) sits entirely below 15°** (562 there, −11 above).
  "L5 hotter" means **"L4 has an extra cold component"** (or "L5 lost its cold component
  without being heated"). It does not mean "L5 was heated": a one-sided heater would make the
  hot L5 count exceed the hot L4 count, and it doesn't.
- The 25–30° L5 spike with families is the Ennomos/Deiphobus clan (VDR §2.5).
- Consequence for every mechanism below: **ΔN and ΔI are one observable**, the I-dependence
  of N4/N5. A mechanism has to be strongly asymmetric for cold orbits and symmetric for hot
  ones. It also has to leave D symmetric (KS p = 0.11–0.33) and e almost symmetric.
- **Earlier statements of the inclination difference.** Slyusarev & Belskaya (2014) (raw
  data; via Pirani et al. 2019a §2.1, p. 2, and V24 §1, p. 3). Jewitt et al. (2000) found L4
  bimodal at ≃ 9° and 19° (via V24 §1, p. 3). V24 itself, Conclusion 7 (AR §1.3). V24 §1, p. 3
  says this difference was "discussed and analyzed in Pirani et al. (2019a)". That overstates
  it: Pirani lists it as an observed property (§2.1, p. 2) and her captured Trojans are flat
  (§4.1.3, p. 10). She never analyses it.

### 1.1 In-situ capture by a growing Jupiter (Marzari & Scholl 1998; Fleming & Hamilton 2000)

- **Setup.** Jupiter's mass grows adiabatically. Horseshoes shrink into tadpoles and
  libration amplitudes shrink by (M_J/M_J,c)^(−1/4).
- **Prediction: ΔN = ΔI = ΔD = Δe = 0.** Fleming & Hamilton (2000) find "orbits librating
  around … L4 behaved similarly to orbits around L5, as suggested by our analytic analysis"
  (§3, p. 484). For migration: "no apparent dependence on the choice of Lagrangian equilibrium
  point" (§4, p. 489). Growth leaves e and i invariant; 1 au of inward migration raises both
  by 4.5% (abstract; Fig. 12 and §4.2, p. 490–491), the same for both swarms. Pirani et al.
  (2019a) find in-situ growth gives 0.94 ± 0.35, and half the runs reversed (§4.1.1 and Fig. 8, p. 9).
- **Physical reason.** Mass change does not enter the guiding-centre Hamiltonian as an
  L4/L5-odd term (Sicardy & Dubois 2003, §9, p. 346; see §2.1 below).
- **Status:** quantitative (adiabatic invariants), symmetric. Marzari & Scholl (1998a,b) not
  read (paywalled). Source: author reprint, `astro.umd.edu/~hamilton/research/DPHreprints/FlemHam00.pdf`.

### 1.2 Gas drag on the particle (Peale 1993; Murray 1994; Kortenkamp & Hamilton 2001; Montesinos+2020)

- **Setup.** Nebular drag acts on the planetesimal, not on Jupiter.
- **Theory (verified).** In Sicardy & Dubois (2003, §8, p. 343), a torque on the particle
  enters as α = (ε_s − ε_p)/μ. "If the satellite and the particle receive the same specific
  torque, then α = 0 and the phase space is symmetrical." So drag on the particle produces an
  L4/L5 asymmetry of the *opposite* sign to the same torque applied to the planet. Leleu
  et al. (2019, §3.2.1, p. 5–6; §7.1, p. 18) generalise this to two massive bodies: "The
  tadpole (trojan) area is larger if the torque per mass unit applied on the leading planet is
  greater than the torque per mass unit applied on the trailing one."
- **Sign for drag capture is contested in secondary sources (Unverified).** Montesinos et al.
  (2020, §4, p. 12) cite Peale (1993) and Murray (1994) as showing that "in the presence of
  drag, the orbits of Trojans around L5 are more stable than those around L4". A search-engine
  rendering of Murray's abstract says the opposite ("the L4 point, if it exists, is less
  unstable than the L5 point"). Lhotka & Celletti (2015, §1, p. 2) say only that Murray "finds
  asymmetric stability indices". **I could not read Peale 1993, Murray 1994 or Kortenkamp &
  Hamilton 2001** (Icarus/DPS, paywalled; ADS blocked scripted access). Do not quote a sign
  until someone has read them.
- **Lhotka & Celletti (2015)** (PR drag, spatial elliptic problem). L4/L5 stability indices
  differ, but only by 10⁻⁹ to 10⁻³ (§5, p. 25–26). The equilibrium inclination stays i* = i₁
  for all β (§4.2, p. 21), so there is **no ΔI** at the equilibrium.
- **Montesinos et al. (2020), hydro plus dust.** L4 traps more μm–submm grains and L5 more
  mm–cm mass. The cause is a gas over-density on the trailing side; the effect shrinks as
  viscosity drops and does not depend on migration (Abstract; §4 and §5.1, p. 12).
  Planetesimal sizes were not modelled, and there are no inclinations (a 2-D disc).
- **ΔI, ΔD, Δe:** no prediction anywhere. Drag damps e and i; any capture-era cold
  component would be gas-era (cf. Bottke et al. 2023, §3, p. 7).
- **Status:** qualitative; sign unverified. Testable in our setup only with an added drag
  force (REBOUNDx `gas_dynamical_friction` / custom), not as a Jupiter-only migration.

### 1.3 Capture during inward gas-disc migration plus growth (Pirani et al. 2019a,b)

Already summarised in AR §2.2. New, verified points:

- **The mechanism is residence time on the horseshoe.** "Inward migration modifies the
  horseshoe orbits, increasing the width of the path of the particles in the L4 side … and
  decreasing [it] in the L5 side (Sicardy & Dubois 2003)", and mass growth freezes the excess
  into tadpoles (2019a §4.2, p. 10–11).
- **Direction test.** With outward migration (5 → 18 au) "we found an asymmetry ratio of
  NL4/NL5 ∼ 0.6 that is opposite to the one found in the inward migration" (§4.2, p. 11,
  Fig. 12). Drag on the particles is irrelevant: "without applying any gas drag … we obtained
  the same asymmetry" (p. 11).
- **ΔI: none.** The captured Trojans are flat (10⁻¹–10⁻² deg; §4.1.3, p. 10). Growth and
  migration preserve i (2019b §2.1, p. 3, citing Fleming & Hamilton). **ΔD, Δe:** not
  reported per swarm.
- **Inclination dependence is across runs, not within one.** Flat disc J0: N4/N5 = 1.60
  (2019b Table 1, p. 3). Pre-stirred discs: 1.15 → 1.11 (ab) and 1.14 → 1.11 (ckbo) over
  4.5 Gyr (Table 6, p. 8). "The low ratio is due to less efficiency in the mechanism
  generating the asymmetry in the pre-stirred case" (§3.3.1, p. 8). That is the same tendency
  as §1.0 (weaker at high i), but no per-I-bin ratio is given. Pre-stirred discs raise e and i
  together, so the i-effect is not isolated.
- **Embryos.** In run J250, "the leading swarm hosted the 4 most massive embryos and they
  depleted the leading swarm … reversing the asymmetry" (2019b §3.1, p. 7). That is a
  one-sided perturber (cf. §1.6). Per-swarm inclinations are not reported.
- **Status:** quantitative for N; silent on I, D and e per swarm. **Testable:** yes. Inward
  ȧ_J with growing M_J on a cold horseshoe population, binned in i.

### 1.4 Fast outward migration acting on existing Trojans (Li et al. 2023a)

- **Setup.** Sun + Jupiter, 500 test Trojans per swarm, Δσ₀ = 30–80°, ȧ_J = 1.5 × 10⁻⁴ au/yr,
  well below ȧ_crit = 3.8 × 10⁻³ au/yr (§2, p. 4; §3.2, p. 5).
- **Mechanism.** "The L4 sample has its resonant amplitude decreased and becomes more stable,
  while the L5 one has its resonant amplitude increased and finally gets ejected" (Fig. 3
  caption, p. 5; Fig. 4, p. 7). This is a *survival* asymmetry.
- **ΔN:** 1.50 at 1 Myr, 1.68 at 1 Gyr (Table 2, p. 7); grows with ȧ_J and e_J (Tables 3–4,
  p. 8–9).
- **ΔD, a specific prediction that the data contradict.** "For objects having resonant
  amplitudes smaller than 30°, the fractions of survived Trojans of the L4 and L5 swarms are
  ∼57 per cent and ∼37 per cent" (§3.2, p. 7; Fig. 5). The observed D distributions do not
  differ (V24 KS p = 0.17; ours p = 0.11–0.33). Li concedes all survivors have A ≳ 15° and
  proposes collisional damping (p. 8; AR §2.3).
- **ΔI.** Table 5 (p. 9) shows R45 = 1.50, 1.54, 1.38, 1.30 at i = 0, 10, 20, 30°; "we
  speculate that the cause may be the not very high i". Here is what that implies for ΔI
  (ours). Weight a common parent distribution, p(i) ∝ sin i·exp(−i²/2σ²) with σ = 8–16°, by
  each swarm's Table 5 survival fraction. The survivors then differ by KS D = 0.013–0.027 and
  by 0.1–0.7° in median. **The sign is right; the size is ~10× too small** (observed D = 0.21,
  about 5°). Equivalently, Li's R45 falls by a factor 1.15 from 0 to 30°, against the observed
  factor ≈ 2.
- **Δe:** not reported per swarm.
- **Status:** quantitative for N and D, with D in conflict with the data; ΔI implicit and too
  weak. **Testable:** yes. This is exactly our Jupiter-only MigrationDriver setup, with i as
  an axis.

### 1.5 Super-critical jump driven by a free-floating planet, indirect (Li et al. 2023b §2)

- ȧ_J up to 0.03 au/yr, above ȧ_crit = 3.8 × 10⁻³. L4 and L3 merge temporarily. Both swarms
  are excited, "the L4 one suffers less excitation" (§4, p. 12). R45 ≈ 1.6 at Δa_J = 0.12 au.
- Test particles at i = 0.01°, so **no ΔI**. ΔD: both excited, with L4 less. Status:
  quantitative for N only. Testable: yes, as a step in a_J.

### 1.6 One-sided perturbers: FFP through L5 (Li+2023b §3), ice giant through L5 (Nesvorný+2013 §3.4), embryos (Pirani+2019b)

- **Li et al. 2023b, direct case.** All 500 L4 survive and 269/500 L5, so R45 = 1.86 (§3.1,
  p. 10). L5 survivors keep their amplitudes "mainly below 30°" (Fig. 7, p. 10). Then comes
  the key sentence (§3.2.2, p. 11–12). The authors checked for inclination excitation, found
  none (all ≲ 2°), and wrote: "We think this outcome is reasonable, otherwise **the L4 and L5
  swarms would have quite different inclination distributions** because only the latter is
  strongly perturbed by the FFP in our model." **This is the only explicit link between a
  mechanism and an L4/L5 inclination difference that I found, and the authors thought such a
  difference would be a problem.** They were evidently unaware of V24 (posted 2024).
- **Nesvorný et al. 2013** (§3.4, p. 10–11). An ice giant on a Jupiter-crossing orbit
  "can traverse one of the Lagrange swarms and scatter captured bodies"; f45 > 1 "would
  indicate that the ice giant traversed the L5 cloud". The runs give f45 = 1.3 ± 0.5,
  f54 = 1.4 ± 0.4 and f54 = 1.8 ± 0.9: "not good enough to rule out f45 = 1 at more than 1σ".
  ΔI is not discussed. Captured inclinations "remained nearly constant" after capture (§3.2,
  p. 8), so I is set by pre-capture scattering, symmetrically.
- **Generic prediction of this class (ours).** A perturber that stays close to Jupiter's
  plane meets low-i Trojans far more often than high-i ones, which spend little time near
  the plane. Removal would then be concentrated in cold L5 orbits, giving **N4/N5 > 1 at
  low i and ≈ 1 at high i, the pattern of §1.0**. The catch is that partially scattered
  survivors would carry raised D and e in L5, which are not observed. This works only if
  removal is all-or-nothing. Nobody has tested it.
- **Status:** qualitative (stochastic, small-N). **Testable:** partly. It needs a massive
  perturber on a prescribed orbit, not just a migrating Jupiter.

### 1.7 Chaotic / jump capture itself (Morbidelli+2005; Nesvorný+2013)

"All capture mechanisms proposed so far, including chaotic capture of M05, are expected to
produce symmetric distributions" (Nesvorný+2013 §3.4, p. 10). Inclinations come from the
source population: "the inclination distribution of Jupiter Trojans is closely related to that
of planetesimals near 5 AU during the scattering phase" (§3.2, p. 8). That predicts ΔI = 0 at capture, and the hot,
symmetric half of §1.0 is what such a channel would leave behind. Bottke et al. (2023, §3,
p. 7): "the inclination distribution of the Trojans makes it difficult to accept any of the
proposed capture models" (gas-era ones).

### 1.8 Present-day erosion, Yarkovsky, collisions/families

- **Gravitational erosion.** Holt et al. (2020) find 23.35% (L4) and 24.89% (L5) escape over
  4.5 Gyr, and "Proper inclination … appears to have little effect on the general
  instability" (Abstract; §3, p. 5). Hellmich et al. (2019) find an L5/L4 escape ratio of
  1.106 ± 0.026 over 1 Gyr without Yarkovsky (§3.2, p. 6). Di Sisto: AR §3, VDR §3. Ours:
  1.00 ± 0.57. Erosion cannot make ΔI, because it is nearly I-blind.
- **Yarkovsky.** Wang & Hou (2017, MNRAS 471, 243, full text via OUP; §3.1) linearise about
  L4 and state "the same procedure applies for L5"; no L4/L5 difference. Hellmich et al.
  (2019): with Yarkovsky the escape ratio becomes 0.985 ± 0.016, so the effect *homogenises*
  the swarms for R ≲ 1 km (§3.2, p. 6–7; §4, p. 9). It is irrelevant at H ≤ 13.6 (D ≳ 10 km).
  Hellmich (§3.2, p. 6) cites Hou et al. (2016) for a "slightly different" Yarkovsky effect at
  the two points (Unverified).
- **Families.** Removing V24's families lowers the KS D from 0.211 to 0.174
  (`INITIAL_CONDITIONS_KS.md` §2). The low-i excess survives (§1.0: 2.01 vs 0.98). So
  families are not the cause.

### 1.9 Things that were asked for but are not mechanisms of ΔI

- **Hou, Scheeres & Liu (2014, CeMDA 119, 119; 2016, CeMDA 125, 451).** Both are
  **Unverified** (closed access). Di Sisto et al. (2014, §1, p. 3) report that Hou+2014
  "showed that a temporary asymmetry m[a]y be obtained with the same initial conditions in
  both tadpole regions". The 2016 abstract, seen only through a search-engine summary, says it
  "studies the symmetry problem between the Jupiter L4 and L5 Lagrangian points when the two
  planets are in migration". **This is the one paper that could pre-empt §2. Get it from
  Leiden first.**
- **Robutel & Bodossian (2009)** (the file named RobutelGabern2009). They map L4 only and use
  symmetries *within* a swarm (p. 9–10). There is no L4/L5 comparison, so the paper says
  nothing on ΔI.
- **Gomes (1998)**, claiming migration changes f45. Known only via Nesvorný+2013 footnote 4
  (p. 10), which attributes the effect to his initial orbits (Unverified).

---

## 2. The time-symmetry framing: who has said what

### 2.1 The relevant symmetry is already in Sicardy & Dubois (2003). It is spatial in the averaged problem.

- **What they wrote.** S&D Eq. (15)–(17), p. 330–331:
  H₄ = −(GM/J_s)²[(3/2)(J_φ/J_s)² + μ g(φ)], with g(φ) = f(φ) − αφ and α = ε_s/μ. Here
  f(φ) = −cos φ + 1/|2 sin(φ/2)| (Eq. 2, p. 325). They then say: "The term −αφ in g(φ) destroys
  the symmetry of the Hamiltonian H₄ with respect to φ = π … For that reason, α will be called
  the asymmetry parameter" (p. 330). And in §9, p. 346: "this asymmetry is caused by an
  explicit variation of J_s (i.e. by a torque applied to the satellite), and not by a
  variation of a_s per se. For instance, if the satellite migrates because of a variation of
  M₀ or m_s only, no asymmetry with respect to φ = π will be observed."
- **What follows (ours).** f is even and H₄ is quadratic in J_φ. So with α = 0 the
  guiding-centre problem is invariant under (φ, J_φ) → (−φ, −J_φ) **without reversing time**,
  even while M(t) and μ(t) vary. The map sends α → −α. Within the adiabatic theory, therefore:
  1. Any N4/N5 asymmetry is an **odd function of α = (ε_s − ε_p)/μ**: of the *differential*
     torque, so of the migration direction relative to the particles (§8, p. 343).
  2. Mass growth breaks time-reversal symmetry but produces **no** asymmetry. Fleming &
     Hamilton (§1.1) and Pirani's in-situ 0.94 ± 0.35 confirm this numerically.
- **Correction to `TIME_SYMMETRY_HYPOTHESIS.md` §3.** "Supplying an arrow of time" is
  *necessary* (by the mirror theorem) but *not sufficient*. The epoch must also break the
  L4/L5 reflection of the averaged problem, through a term odd in φ: a net torque (migration,
  drag) or a one-sided perturber. The two are linked, since t → −t flips the sign of J̇_s and
  so of α. A time-dependent but even perturbation (growth, symmetric mass loss) leaves the
  swarms equal.
- **The link to Trojans.** S&D also connect this directly to the observed asymmetry (§6,
  p. 335): "an inward migration of Jupiter during the formation of the Trojans increases the
  size of the L4 region with respect to the L5 region, and could in principle explain this
  asymmetry". They add a caution: "transferred bodies tend to be captured back to their
  original region when the migration stops."

### 2.2 Other prior statements, from strongest to weakest

| Source | Exact statement (location) | Overlap with our framing |
|---|---|---|
| Sicardy & Dubois 2003 | Quoted above (§6, p. 335; §9, p. 346) | Sign set by migration direction; needs a torque, not just time-dependence |
| Pirani et al. 2019a | Outward migration should give "an opposite asymmetry, i. e. NL4/NL5 < 1 … we found … ∼0.6" (§4.2, p. 11) | Numerical sign flip with migration direction |
| Leleu et al. 2019 | Tadpole area larger on the side with the larger torque per unit mass (§3.2.1, p. 5–6; §7.1, p. 18) | Same as S&D, two massive bodies |
| Lhotka & Celletti 2015 | "in the conservative setting the equilibria L4 and L5 are mirror symmetric … in the dissipative case, mapping small deviations from L4 symmetrically into the vicinity of L5 does not alter the signs of the dissipative terms … we can expect an asymmetry" (§4, p. 16) | **Closest in form.** A mirror-symmetry argument for why dissipation separates L4 from L5, but local (equilibria only) and without time reversal |
| Horner & Evans 2006 | "Newton's equations of motion are time-reversible, so some form of dissipation is required for permanent capture" (§3, p. 2). Captured Centaurs show "nearly equal likelihood of capture in the L4 or L5 regions" (§3, p. 3; Table 2: Jupiter 2 L4 / 5 L5, Saturn 25 / 26) | Time-reversibility ⇒ capture needs dissipation; present-day captures symmetric (our P2) |
| Nesvorný et al. 2013 | "Planetary migration is also not expected to change f45" (§3.4, p. 10) | The opposite view; predates Pirani and Li |
| Murray 1994 / Peale 1993 | Asymmetric L4/L5 stability under drag (via Leleu §3.2.1, Lhotka §1, Montesinos §4) | Unverified, and the secondary sources disagree on the sign |

### 2.3 What I did not find anywhere

1. Any use of the **Roy–Ovenden mirror theorem** or the reversibility (x, y, z, t) →
   (x, −y, z, −t) in the L4/L5 context. Roy & Ovenden (1955, MNRAS 115, 296) concerns
   commensurabilities. Lhotka & Celletti's "mirror symmetric" is a spatial statement about
   the equilibria.
2. The **pairing "capture into L4 ↔ escape from L5"**, and the steady-state argument that
   all four exchange rates must be equal today (hypothesis §2).
3. That **Pirani 2019a and Li 2023a are mirror images** (ours). Map Li's run (outward
   migration, more L5 *escapes*) through (y, t) → (−y, −t): it becomes inward migration with
   more L4 *captures*, which is Pirani's inward result. Pirani's outward run (more L5
   captures) maps to "inward migration should preferentially *drain* a pre-existing L4". For
   a given migration direction the sign therefore depends on **whether capture or loss
   dominates**. That explains why the two groups get opposite signs for outward migration
   (0.6 against > 1), which neither paper remarks on.
4. Any reading of the **L4/L5 inclination difference** as a record of the symmetry-breaking
   epoch.

**Verdict: partly known.** That the asymmetry's sign follows the direction of a
non-conservative or time-dependent drift is established (S&D 2003; Pirani 2019a; Leleu 2019).
So is the fact that conservative, time-reversible dynamics gives symmetric temporary captures
(Horner & Evans 2006). The *explicit* time-reversal formulation, the capture/escape pairing,
the steady-state corollary (H0/P2) and the Pirani–Li mirror reconciliation are, as far as I
can find, **unstated**. Two caveats: Hou, Scheeres & Liu (2016) is unread and is the main
risk, and ADS full-text search was unavailable. **Strongest prior statement:** Sicardy &
Dubois (2003, §6, p. 335): "an inward migration of Jupiter during the formation of the
Trojans increases the size of the L4 region with respect to the L5 region, and could in
principle explain this asymmetry."

---

## 3. Control populations (short)

- **Hildas (3:2).** V25 (Vokrouhlický et al. 2025, App. A): the Hildas are "a sister
  population of the Jupiter Trojans", captured in the instability from the primordial Kuiper
  belt. Vokrouhlický et al. (2016) "found approximately the same capture probability for
  Jupiter Trojans and Hildas". The broad e and I distribution "is a natural component of this
  capture scenario". Roig & Nesvorný (2015, §3.5, p. 23–25) find that pre-instability Hildas
  (survival < 10⁻⁵) and Trojans (< 10⁻⁶) do not survive the jump; < 5% of Hildas could be
  outer-belt implants. V25 gives a **bias-corrected Hilda I distribution**. Whether it has a
  cold excess like L4 is a cheap test of whether §1.0's cold component is capture-era. Not
  done by anyone.
- **Thule (4:3).** Found only via V25 App. A, citing Franklin et al. (2004): Hilda/Thule
  e and libration distributions plus the Hilda/Thule number ratio "require Jupiter's slow
  migration by at least 0.45 au", "but these authors did not investigate … the dispersed
  inclination distribution". No inclination constraint exists.
- **Neptune Trojans.** Observed then mostly in L4 because L5 lies toward the Galactic centre
  (Lykawka et al. 2009, §2, p. 5). Models: Pirani 2019a (§4.2, p. 12, Fig. 14) finds the
  leading excess on Neptune's horseshoes, but "the excess simply disappears" because
  Neptune's growth cannot freeze it: "Our simulations clearly predict no asymmetry for the
  Neptune Trojans". Parker (2015, §7.1, Fig. 13): "no captured Trojan population was found to
  be significantly asymmetric". Chen et al. (2016, Table 3, Fig. 9, Conclusions): similar
  numbers *and e, i distributions* at L4/L5, with only a "transient asymmetry" under damping.
  Order of magnitude (ours): α_N = ȧ_N/(2a_N n_N μ_N) ≈ ȧ_N/(1.2 × 10⁻⁴ au/yr) ≲ 10⁻²
  for smooth migration. S&D then predicts no asymmetry, and no paper links an NT asymmetry to
  migration direction.

---

## 4. Summary table

Signs use L4 minus L5 ("+": L4 more numerous, or larger quantity).

| Mechanism (§) | ΔN | ΔI | ΔD | Δe | Status | Testable with Jupiter-migration N-body? |
|---|---|---|---|---|---|---|
| Growth, in situ (1.1) | 0 | 0 | 0 | 0 | quantitative | yes: M_J(t) ramp, control run |
| Gas drag on particles (1.2) | ≠ 0, sign unverified | none predicted | — | — | qualitative | only with an added drag force |
| Inward migration + growth, capture (1.3) | + (1.6–1.8); − for outward (0.6) | none (flat); asymmetry falls in stirred discs | not reported | not reported | quantitative (N) | yes: inward ȧ_J, growing M_J, horseshoe seed, bin in i |
| Sub-critical outward jump, survival (1.4) | + (1.5–1.7) | + implied, ~10× too small (ours) | **+ predicted (L4 smaller A); contradicts data** | not reported | quantitative | **yes, directly** (MigrationDriver, i axis) |
| Super-critical jump / FFP indirect (1.5) | + (≈ 1.6) | none (planar) | both excited, L5 more | — | quantitative (N) | yes: step in a_J |
| One-sided perturber: FFP, ice giant, embryos (1.6) | + if L5 hit; − if L4 hit | authors expect ≠ 0; cold-L5 removal (ours) would match §1.0 | L5 larger if survivors excited | L5 larger | qualitative, stochastic | partly: needs a massive perturber |
| Chaotic/jump capture (1.7) | 0 ± stochastic | 0 | 0 | 0 | qualitative | no (needs the full instability) |
| Present-day erosion (1.8) | ×1.00–1.13 | ≈ 0 (I-blind) | small | small | quantitative | yes (pilot done) |
| Yarkovsky (1.8) | → 1 (homogenises) | — | — | — | quantitative, R ≲ 1 km | no (irrelevant for H ≤ 13.6) |
| **Observed** | **+, all at I < 15°** | **L5 hotter (D = 0.21)** | **0** | **≈ 0** | | |

---

## 5. Open threads

1. **Nobody explains why the asymmetry is confined to cold orbits.** That is the real content
   of the "inclination tension". It is not that L5 is hotter *and* smaller in some paradoxical
   way: in §1.0 the two are the same fact. Li's Table 5 and Pirani's Table 6 go the right way
   but far too weakly (§1.4). An i-dependent α (a 3-D Sicardy–Dubois asymmetry parameter) has
   never been derived. The co-orbital potential f(φ) loses its φ → 0 singularity at high i,
   so the islands' sensitivity to a torque should depend on i. That is a tractable analytic
   piece, and it is ours to do.
2. **The two-component reading (ours, speculative).** A cold, asymmetric component plus a
   hot, symmetric one fits §1.0 with no fine-tuning: the cold one from a torque- or
   perturber-driven epoch, the hot one from chaotic/jump capture. But Roig & Nesvorný (2015)
   and Nesvorný+2013 say pre-instability Trojans do not survive the jump. Then the cold
   asymmetric component must be made *during or after* capture, which favours Li-type
   survival or one-sided-perturber removal over Pirani-type gas-era capture. Jewitt et al.'s
   (2000) L4 bimodality (≃ 9° and 19°) is the observational hint of two components.
3. **ΔD = 0 is an underused discriminant.** Li 2023a predicts a large ΔD (57% against 37% at
   A < 30°), and excited-survivor perturber models predict L5 larger D and e. The data show
   neither. Only all-or-nothing removal, or capture-side asymmetry (Pirani-type, where the D
   distribution is set by growth damping, the same for both swarms), passes.
4. **Gas-drag sign.** Read Peale 1993, Murray 1994 and Kortenkamp & Hamilton 2001 before
   saying anything. The secondary sources contradict each other (§1.2).

**Constraints on our predictions (`TIME_SYMMETRY_HYPOTHESIS.md` §4):**
- **P1 (L5 escapers hotter).** Not tested anywhere. Holt et al. (2020) find erosion nearly
  I-blind. So if H1 holds, escapers should simply mirror the core: L5 escapers hotter, with
  the excess N4 escapes at low I. Hellmich (§3.3, p. 7) gives escaper D by swarm
  (21.75 ± 0.12° against 20.85 ± 0.10°), but not I.
- **P2 (present captures symmetric).** Supported in simulation: Horner & Evans 2006 Table 2
  (Saturn 25 / 26; Jupiter 2 / 5). Observationally mixed and small-N:
  - Greenstreet et al. (2023, §3, Table 1, p. 5–7) find 12 L4 against 4 L5 metastable
    Jupiter Trojans that "must be recent metastable captures". Two-sided binomial p ≈ 0.08
    against 1:1 (ours).
  - Our V24-box interlopers run 12 L4 against 16 L5.
  - Next step: fetch I for Greenstreet's 16 named objects (Table 1) and add them to the P2
    test.
- **P3 (fates match at fixed orbit).** No literature. Di Sisto et al. (2019) give fates
  without an L4/L5 split at fixed (e, i); see VDR §3.2.
- **H1 versus "erosion made it".** Further strengthened: every erosion study is I-blind and
  ×1.0–1.1, and none could produce a ratio of 2 at I < 15° alongside 1 above.

---

## 6. References

Local paths are relative to `literature/`. "New" means downloaded today (arXiv).

- Bottke, Marschall, Nesvorný & Vokrouhlický 2023, SSRv 219, 83; arXiv:2312.02864. `00_reviews/Bottke+2023_…REVIEW.pdf`
- Chen, Ma, Zheng et al. 2016, MNRAS (doi:10.1093/mnras/stw361); arXiv:1602.04303. New: `02_mechanisms/Chen+2016_Orbital_damping_and_Neptune_Trojan_distributions.pdf`
- Di Sisto, Ramos & Beaugé 2014, Icarus 243, 287; arXiv:1604.05331. `02_mechanisms/DiSisto+2014_…pdf`
- Fleming & Hamilton 2000, Icarus 148, 479; doi:10.1006/icar.2000.6523. Read from the author reprint (URL in §1.1); not saved.
- Greenstreet, Gladman & Jurić 2023, arXiv:2309.06609 (submitted to ApJL). New: `01_observations/Greenstreet+2023_Jupiters_metastable_companions.pdf`
- Hellmich, Mottola, Hahn, Kührt & de Niem 2019, A&A 630, A148; arXiv:1909.08584. New: `02_mechanisms/Hellmich+2019_Yarkovsky_force_on_Jupiter_Trojans.pdf`
- Holt, Nesvorný, Horner et al. 2020, MNRAS (doi:10.1093/mnras/staa1348); arXiv:2005.03635. New: `02_mechanisms/Holt+2020_Stability_of_Jovian_Trojans_and_families.pdf`
- Horner & Evans 2006, MNRAS 367, L20; arXiv:astro-ph/0511791. New: `02_mechanisms/HornerEvans2006_Capture_of_Centaurs_as_Trojans.pdf`
- Hou, Scheeres & Liu 2014, CeMDA 119, 119 (doi:10.1007/s10569-014-9544-9); 2016, CeMDA 125, 451 (doi:10.1007/s10569-016-9692-1). **Unread, closed access.**
- Leleu, Coleman & Ataiee 2019, A&A 631, A6; arXiv:1901.07640. `03_coorbital_theory/Leleu+2019_…pdf`
- Lhotka & Celletti 2015, Icarus (doi:10.1016/j.icarus.2014.11.039); arXiv:1412.1630. New: `03_coorbital_theory/LhotkaCelletti2015_PR_drag_on_triangular_Lagrange_points.pdf`
- Li, Xia, Georgakarakos & Yoshida 2023a, A&A 669, A68; arXiv:2211.13877. `02_mechanisms/Li+2023a_…pdf`
- Li, Xia, Georgakarakos & Yoshida 2023b; arXiv:2304.13598. `02_mechanisms/Li+2023b_…pdf`
- Lykawka, Horner, Jones & Mukai 2009, MNRAS 398, 1715; arXiv:0909.0404. New: `02_mechanisms/Lykawka+2009_Neptune_Trojans_I_formation_and_migration.pdf`
- Montesinos, Garrido-Deutelmoser, Olofsson et al. 2020, A&A 642, A224; arXiv:2009.10768. New: `02_mechanisms/Montesinos+2020_Dust_trapping_at_Lagrangian_points_in_discs.pdf`
- Murray 1994, Icarus 112, 465 (doi:10.1006/icar.1994.1198). **Unread.** Peale 1993, Icarus 106, 308 (doi:10.1006/icar.1993.1173). **Unread.** Kortenkamp & Hamilton 2001, BAAS 33, 1086. **Unread.** Marzari & Scholl 1998a, Icarus 131, 41 (doi:10.1006/icar.1997.5841). **Unread.** Gomes 1998, AJ 116, 2590. **Unread.**
- Nesvorný, Vokrouhlický & Morbidelli 2013, ApJ 768, 45; arXiv:1303.2900. `02_mechanisms/Nesvorny+2013_…pdf`
- Parker 2015, Icarus (doi:10.1016/j.icarus.2014.09.043); arXiv:1409.6735. New: `01_observations/Parker2015_Intrinsic_Neptune_Trojan_orbit_distribution.pdf`
- Pirani, Johansen, Bitsch, Mustill & Turrini 2019a, A&A 623, A169; arXiv:1902.04591. `02_mechanisms/Pirani+2019a_…pdf`
- Pirani, Johansen & Mustill 2019b, A&A 631, A89; arXiv:1910.01507. `02_mechanisms/Pirani+2019b_…pdf`
- Robutel & Bodossian 2009, MNRAS 399, 69; arXiv:0809.3526. `03_coorbital_theory/RobutelGabern2009_…pdf` (misnamed; AR §6)
- Roig & Nesvorný 2015, AJ 150, 186 (doi:10.1088/0004-6256/150/6/186); arXiv:1509.06105. New: `02_mechanisms/RoigNesvorny2015_Asteroids_in_jumping_Jupiter_model.pdf`
- Roy & Ovenden 1955, MNRAS 115, 296 (mirror theorem). Not read here; cited for the symmetry only.
- Sicardy & Dubois 2003, CeMDA 86, 321. `03_coorbital_theory/SicardyDubois2003_…pdf`
- Vokrouhlický, Nesvorný, Brož et al. 2024, AJ 167, 138; arXiv:2401.15537. `01_observations/Vokrouhlicky+2024_…pdf`
- Vokrouhlický, Nesvorný, Brož et al. 2025 (Hildas), arXiv:2503.04403 (submitted to AJ). New: `01_observations/Vokrouhlicky+2025_Orbital_and_magnitude_distribution_of_Hildas.pdf`
- Wang & Hou 2017, MNRAS 471, 243 (read as OUP HTML; no PDF saved).
