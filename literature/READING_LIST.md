# Reading list — L4/L5 Jupiter Trojan asymmetry under migration

Assembled 2026-08-24 for `MSc_PROPOSAL.md` / `MSc_PROJECT_PLAN.md`.
31 PDFs downloaded into the subfolders below. Paywalled items are listed at the
end with links — Leiden has access to all of them.

**How to use this:** Tier 0 is the week-one spine (5 papers, ~1.5 days). Tiers 1-5
are read-as-needed against the project phases. Each entry says *why you are
reading it* and *what to extract* — do not read any of these cover to cover
except the ones marked **FULL**.

---

## Tier 0 — the spine. Read these first, in this order.

| #   | Paper                                                                                                                                                           | File                   | What to extract                                                                                                                                                                                                                                                                                                                    |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | **Bottke, Marschall, Nesvorný & Vokrouhlický (2023)**, *Origin and Evolution of Jupiter's Trojan Asteroids*, SSRv 219, 83 — arXiv:2312.02864                    | `00_reviews/`          | **FULL.** The current review; your introduction's backbone. Note where it says the L4/L5 ratio is *unresolved* — that sentence is your gap statement.                                                                                                                                                                              |
| 2   | **Vokrouhlický et al. (2024)**, *Orbital and absolute magnitude distribution of Jupiter Trojans*, AJ 167, 138 — arXiv:2401.15537                                | `01_observations/`     | The number you must hit. Read §on debiasing and the L4/L5 comparison. Record the ratio **with its H-cut** and its error bar verbatim — this is the anchor of every comparison you will make. Also: the L4-vs-L5 *inclination* distributions and the family counts.                                                                 |
| 3   | **Li, Xia, Georgakarakos & Yoshida (2023a)**, *Asymmetry in the number of L4 and L5 Jupiter Trojans driven by jumping Jupiter*, A&A 669, A68 — arXiv:2211.13877 | `02_mechanisms/`       | **FULL** (you have read it — re-read Tables 3 and 5 and §2 setup). This is your Phase-1 validation target. Extract: the migration law, `Δa_J`/`τ` grid, seeding recipe, `α` definition, and the admitted amplitude problem.                                                                                                        |
| 4   | **Pirani, Johansen, Bitsch, Mustill & Turrini (2019a)**, A&A 623, A169 — arXiv:1902.04591                                                                       | `02_mechanisms/`       | **FULL.** The opposing (inward/capture) claim. Extract exactly *which* population is captured (horseshoe transients) and at what `Δa_J`. Companion paper 2019b (inclinations) is Tier 1.                                                                                                                                           |
| 5   | **Sicardy & Dubois (2003)**, *Co-orbital motion with slowly varying parameters*, CeMDA 86, 321                                                                  | `03_coorbital_theory/` | **FULL, with a pen.** The analytic backbone: adiabatic invariance of the libration action, the asymmetry coefficient, and the `α = 0.7265` island-merger limit. **Settle the `α ↔ ȧ_J` conversion here** — plan §11.4 flags that your table is internally inconsistent (`α = ȧ·190` vs `ȧ·262`). Do this before building any grid. |

---

## Tier 1 — the competing mechanisms (Phases 1-3)

`02_mechanisms/`

- **Pirani, Johansen & Mustill (2019b)**, *On the inclinations of the Jupiter Trojans*, A&A 631, A89 — arXiv:1910.01507.
  The inward-migration model's failure mode: near-zero inclinations. Relevant because Vokrouhlický+24 gives you an L4/L5 **inclination asymmetry** as a third observable, and Li+23's Table 5 is non-monotonic in `i` with no explanation. This is your unexploited hook.
- **Nesvorný, Vokrouhlický & Morbidelli (2013)**, *Capture of Trojans by Jumping Jupiter*, ApJ 768, 45 — arXiv:1303.2900.
  The standard chaotic/jump-capture model and the direct ancestor of Li+23. Note that it produces an asymmetry only through encounter geometry, not through a rate — contrast that with your framing.
- **Li, Xia, Georgakarakos & Yoshida (2023b)**, *The invasion of a free-floating planet and the number asymmetry of Jupiter Trojans* — arXiv:2304.13598.
  A **third** mechanism from the same group. One sentence in your introduction so the two-channel framing isn't presented as exhaustive. Do not simulate it.
- **Di Sisto, Ramos & Beaugé (2014)**, *Giga-year evolution of Jupiter Trojans and the asymmetry problem*, Icarus 243, 287 — arXiv:1604.05331.
  **The paper that licenses your whole project.** Fictitious L4 and L5 particles show *identical* escape rates; real Trojans give 23% vs 28.3% escape over 4.5 Gyr. Conclusion: long-term erosion cannot make the asymmetry — it must be primordial. Quote this number in the proposal.
- **Di Sisto, Ramos & Gallardo (2019)**, *The dynamical evolution of escaped Jupiter Trojan asteroids*, Icarus 319, 828 — arXiv:1811.00352.
  Where escapers go. Useful for sanity-checking your "escaped" classifier state and for the loss-rate discussion.

## Tier 2 — co-orbital theory: the machinery you have to implement (Phase 0)

`03_coorbital_theory/`

- **Robutel & Gabern (2009)**, *The resonant structure of Jupiter's Trojan asteroids II*, MNRAS 399, 69 — arXiv:0809.3526.
  Frequency-map picture of the tadpole/horseshoe/secular-resonance structure. Read this before you write the classifier: it tells you *which* boundaries in `(a, e, i, σ, A)` are real dynamical boundaries rather than arbitrary cuts.
- **Leleu, Coleman & Ataiee (2019)**, *Stability of the co-orbital resonance under dissipation*, A&A 631, A6 — arXiv:1901.07640.
  A general stability criterion for L4/L5 under **generic dissipation and slow mass change** — i.e. exactly your problem, written analytically. Predicts when libration amplitude grows vs damps, and the leading/trailing asymmetry. **This is the closest thing in the literature to an analytic prediction of your result**, and it is the natural theory to test your `C(t)`, `L(t)` against. Underused by the Trojan community; likely your best differentiator.
- Murray & Dermott, *Solar System Dynamics*, Ch. 3.11-3.13 (tadpole/horseshoe, libration period, `μ^{1/3}` scalings). Not a PDF here — use the book. You need the `T_lib ≈ 148 yr` derivation and the horseshoe turning-point distance for the binary argument (§10.2 of the plan).

## Tier 3 — migration physics: where the "do we need the Grand Tack?" question lives

`04_migration_physics/` — see the **Grand Tack** section below for the verdict.

- **Masset & Snellgrove (2001)**, *Reversing type II migration: resonance trapping of a lighter giant protoplanet*, MNRAS 320, L55 — arXiv:astro-ph/0101332.
  **FULL — it is 5 pages.** The actual physical mechanism for outward migration of a Jupiter-Saturn pair: overlapping gaps, mass-weighted torque imbalance. **Cite this, not "the Grand Tack", for your outward gas-disc case.**
- **Morbidelli & Crida (2007)**, *The dynamics of Jupiter and Saturn in the gaseous protoplanetary disk*, Icarus 191, 158 — arXiv:0704.1210.
  Where the outward migration **rate** comes from, and what disc parameters set it. This is what turns your `ȧ_J` from a free knob into a physical parameter — the claim your proposal makes.
- **Pierens, Raymond, Nesvorný & Morbidelli (2014)** — arXiv:1410.0543. Outward migration in 3:2 vs 2:1 in radiative discs. Shows the outcome depends on resonance and disc thermodynamics — i.e. your outward `ȧ_J` has a defensible *range*, not one value.
- **Griveaud, Crida, Petit, Lega & Morbidelli (2023)**, *Migration of pairs of giant planets in low-viscosity discs*, A&A — arXiv:2303.04652. The modern re-examination: outward migration of the pair often **fails** at low viscosity. Read for the honest range of `ȧ_J` you should sweep, and cite when justifying why you treat rate as a parameter.
- **Kanagawa et al. (2016)** — arXiv:1603.03853, and **Scardoni, Rosotti, Lodato & Clarke (2020)**, *Type II migration strikes back* — arXiv:1912.07313. The current state of the Type-II-rate debate (is migration locked to the viscous rate or not?). You need ~2 paragraphs from these to defend the inward `ȧ_J` range in your proposal. Do not go deeper.
- **Raymond & Morbidelli (2014)**, *The Grand Tack model: a critical review* — arXiv:1409.6340. You already have it. **Read only §on the outward-migration mechanism and the stated "Achilles heel".** Skip the terrestrial-planet material entirely.

## Tier 4 — binaries (supervisor's ask; plan §10.2)

`05_binaries/`

- **Nesvorný & Vokrouhlický (2019)**, *Binary survival in the outer solar system*, Icarus 331, 49 — arXiv:1905.02282.
  **The paper your binary chapter is built on.** Extract the survival criterion as a function of `a_B/R_Hill,B` and the encounter-velocity dependence — this is what you convolve with your recorded close-approach log. No new simulations needed.
- **Nesvorný, Vokrouhlický, Bottke & Levison (2018)**, *Evidence for very early migration of the Solar System planets from the Patroclus-Menoetius binary Jupiter Trojan*, Nat. Astron. 2, 878 — arXiv:1809.04007.
  Why a single surviving wide binary is a strong dynamical constraint. This is the argument template you are extending to an **L4-vs-L5 binary-fraction asymmetry**.

`06_methods_missions/`
- **Lucy mission** — arXiv:2104.04575 (instrumentation/ops). One paragraph for motivation: Lucy reaches L5 (Patroclus-Menoetius) in **2033**, so a predicted L4/L5 binary asymmetry is testable. Cite Levison et al. 2021 PSJ for the mission overview (link below).

## Tier 5 — observations you must match (Phase 4)

`01_observations/`

- **Szabó, Ivezić, Jurić & Lupton (2007)**, MNRAS 377, 1393 — arXiv:astro-ph/0703026. The historical N4/N5 = 1.6 ± 0.1. Read only to know what the field quoted **before** Vokrouhlický+24, so you can say clearly which number you target and why.
- **Grav et al. (2012)**, WISE/NEOWISE Jovian Trojan taxonomy — arXiv:1209.1549. The 1.34-1.40 ratio and the L4/L5 size distributions.
- **Uehata, Terai, Ohtsuki & Yoshida (2022)**, *Comparison of the physical properties of the L4 and L5 Trojan asteroids from ATLAS data* — arXiv:2101.04602. The direct L4-vs-L5 physical comparison.
- **Yoshida, Terai & Ohtsuki (2026)**, *Color and size distributions of small Jupiter Trojans* — arXiv:2603.16485. L4 only, no L5 comparison — confirms the physical-properties picture is still thin and that the asymmetry argument rests on **orbital** statistics, which is what you produce.

**Data, not papers:** AstDyS (`https://newton.spacedys.com/astdys/`) publishes a Trojan **proper-element catalogue that already includes libration amplitude `D`** — this may save you the short-integration step in Phase 4. Check it before writing that code.

## Tier 6 — general background, dip in as needed

`00_reviews/`

- **Emery, Marzari, Morbidelli, French & Grav (2015)**, *The complex history of Trojan asteroids*, Asteroids IV — arXiv:1506.01658. The pre-2015 picture; good for the historical framing of capture models (Marzari & Scholl, Fleming & Hamilton, Morbidelli+05).
- **Nesvorný (2018)**, *Dynamical evolution of the early Solar System*, ARA&A 56, 137 — arXiv:1807.06647. The instability / jumping-Jupiter context in one authoritative place.
- **Morbidelli et al. (2021)**, *Origin and dynamical evolution of the asteroid belt* — arXiv:2012.07932. Only if you need the main-belt constraint on `Δa_J` (the Pirani problem).

---

## Paywalled / not downloadable — get these through the Leiden library

Ordered by how much you need them.

1. **Fleming & Hamilton (2000)**, *On the origin of the Trojan asteroids: effects of Jupiter's mass accretion and radial migration*, Icarus 148, 479. doi:10.1006/icar.2000.6523.
   **Important.** Quantifies exactly your two effects separately: mass growth shrinks libration amplitudes by ~2.5×; 1 au of inward migration changes them by only ~4%. Their adiabatic-invariant argument (dimensionless tadpole area conserved) is the null hypothesis your Phase-2 sweep has to break.
2. **Marzari, Tricarico & Scholl (2003)**, *Clues to the origin of Jupiter's Trojans: the libration amplitude distribution*, Icarus 162, 453.
   The precedent for using the **amplitude distribution** as the discriminant — i.e. framing (C) in your plan §7. Read before committing to that chapter.
3. **Morbidelli, Levison, Tsiganis & Gomes (2005)**, *Chaotic capture of Jupiter's Trojan asteroids in the early Solar System*, Nature 435, 462.
   The canonical chaotic-capture paper. Cite once; superseded operationally by Nesvorný+2013.
4. **Marzari & Scholl (1998a)**, Icarus 131, 41 and **(1998b)**, A&A 339, 278. Capture by a growing proto-Jupiter, with and without gas drag. Historical, but they are the origin of the "horseshoes get pinched into tadpoles as Jupiter grows" argument that Pirani+19 reuses.
5. **Marschall, Nesvorný, Bottke et al. (2025)**, *Collisional evolution of Jupiter trojans after capture*, A&A — https://www.aanda.org/articles/aa/full_html/2025/09/aa54339-25/aa54339-25.html (open access HTML; the PDF endpoint blocked automated download). The newest constraint on how much collisional erosion is allowed post-capture — bounds how much of the ratio you are permitted to attribute to non-primordial processes.
6. **Levison, Olkin, Noll, Marchi et al. (2021)**, *Lucy mission to the Trojan asteroids: science goals*, PSJ 2, 171. Open access at IOP; blocked from scripted download.
7. **Milani (1993)**, CeMDA 57, 59 and **Beaugé & Roig (2001)**, Icarus 153, 391 — Trojan proper elements, including the definition of libration amplitude you should adopt so your `A` is comparable to catalogues.
8. **Peale (1993)**, Icarus 106, 308 and **Murray (1994)**, Icarus 112, 465 — nebular gas drag on co-orbitals. Only if you pursue the gas-damping test in the proposal's last sentence.
9. **Walsh, Morbidelli, Raymond, O'Brien & Mandell (2011)**, Nature 475, 206 — the original Grand Tack. See below: cite once, in one sentence, or not at all.

---

## Explicitly out of scope

`99_course_background/`
- Bošković et al. (2025), TOI-178 — the exoplanet resonant-chain extension. Plan §10.3 already ruled it out (it needs new simulations). Outlook paragraph only.
- AMUSE book pp. 320-338 — course background, not science input.
- Debes et al. (2025), metal-polluted white dwarfs with JWST/MIRI — white-dwarf planets, unrelated.

---

## On the Grand Tack: no, you do not need it

**Short answer: drop the name, keep the physics.** Nothing in this project depends
on the Grand Tack, and saying "Grand Tack" in the proposal buys you a contested
assumption you never use.

**Why it is separable.** Your independent variable is `(ȧ_J, direction, Δa_J)`.
The Grand Tack is not a physical mechanism — it is a *scenario* that happens to
contain an outward leg. The mechanism underneath that leg is Masset & Snellgrove
(2001): two gap-opening planets in resonance, overlapping gaps, net positive
torque. That mechanism stands entirely on its own, is quantified by
Morbidelli & Crida (2007), and is what actually sets your outward `ȧ_J`.

**The contested part is the part you don't use.** What people argue about is
Jupiter's deep excursion to ~1.5 au and back — an inner-Solar-System claim,
motivated by the small-Mars problem and asteroid-belt structure. Trojans do not
care. Raymond & Morbidelli's own review names the outward-migration mechanism as
the model's "Achilles heel", and Griveaud et al. (2023) show the pair often fails
to migrate outward at low viscosity. So: cite the *mechanism* with its honest
uncertainty range, and let `ȧ_J` be the swept parameter. That is a **stronger**
framing than "we simulate the Grand Tack", because your result then applies to
any outward history, whatever produced it.

**And it isn't even where the action is.** Plan §11.4 predicts the asymmetry peak
at `ȧ ~ 2×10⁻⁴` to `3×10⁻³ au/yr`. A Grand-Tack-style leg is `~4×10⁻⁵ au/yr` —
one point on the slow, rising side of your curve. It is a tick mark on the x-axis,
not a scenario needing its own chapter.

**Concrete edits.**

- `MSc_PROPOSAL.md` line ~57: *"outward drift driven by a Jupiter-Saturn resonant
  pair, as in the Grand Tack"* → **"outward drift driven by a Jupiter-Saturn
  resonant pair (Masset & Snellgrove 2001; Morbidelli & Crida 2007)"**. Deleting
  four words costs nothing and removes the whole argument.
- `MSc_PROJECT_PLAN.md` §1.4: retitle the Raymond & Morbidelli bullet
  *"gas-disc outward migration — rates"*, and keep only the `ȧ ~ 4×10⁻⁵ au/yr`
  number.
- `MSc_PROJECT_PLAN.md` §6 Phase 4: the optional direction-reversal run stays, but
  call it a **direction reversal** (a sign flip in `ȧ_J`), not a Grand Tack. Same
  simulation, no baggage.
- Introduction: at most one sentence — *"one specific realisation of such an
  outward phase is the Grand Tack, which additionally posits a deep inward
  excursion; that further assumption is not required here."* Or omit entirely.

**What you would lose by dropping it: nothing.** The only unique feature of the
Grand Tack for your purposes is inward-then-outward in sequence, and you can run
that as a parameterised reversal without invoking the model.
