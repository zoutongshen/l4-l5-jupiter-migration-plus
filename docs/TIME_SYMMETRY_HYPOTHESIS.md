# The L4/L5 asymmetry as a fossil of broken time symmetry

Written 2026-10-02, after Steps 1–3 (`docs/CATALOGUE_DEFINITIONS.md`,
`docs/INITIAL_CONDITIONS_KS.md`). This is a working hypothesis to test, not a result.
Whether anyone has stated it before is the first question for the literature review
(`docs/CAPTURE_ASYMMETRY_LITERATURE.md`).

---

## 1. The symmetry

The setting is the circular restricted three-body problem (Sun, Jupiter on a circle, a
massless Trojan). In Jupiter's rotating frame, with x along the Sun–Jupiter line, the
equations of motion

  ẍ − 2ẏ = ∂Ω/∂x,  ÿ + 2ẋ = ∂Ω/∂y,  z̈ = ∂Ω/∂z,  where Ω is even in y,

are unchanged under

  **(x, y, z, t) → (x, −y, z, −t)**

(the mirror theorem; Roy & Ovenden 1955). They are **not** unchanged under y → −y
alone, because the Coriolis terms change sign. A mirror image is a valid orbit only if
it is also run backwards.

What the map does:

- **It exchanges L4 and L5**, because y → −y.
- **It exchanges arriving and leaving**, because t → −t.
- **It keeps a, e and i**, and keeps prograde orbits prograde. It preserves the Jacobi
  constant, L_z and |L|.

So every route by which an orbit is **captured into L4** has a twin route by which an
orbit **escapes from L5**, at the same (a, e, i). Likewise capture into L5 pairs with
escape from L4. No object moves between the swarms: the pairing is between possible
orbits.

With Jupiter's eccentricity the symmetry survives if t is measured from Jupiter's
perihelion passage. Saturn, Uranus and Neptune break it, so in the real system it is
approximate.

## 2. Symmetry plus steady state

Suppose the planets never change (a "calm" system). Phase-space density is then
conserved along orbits (Liouville), and two things follow:

1. **The long-lived core of each swarm is closed.** Nothing from outside can reach it;
   only the chaotic fringe exchanges material with the outside.
2. **In steady state, the fringe of each swarm balances inflow and outflow at every
   (a, e, i):** capture L4 = escape L4, and capture L5 = escape L5.

The outside population (Jupiter-family comets, Centaurs) favours neither the leading
nor the trailing side. Combined with the symmetry, **all four rates (capture and escape
in L4 and in L5) are equal, cell by cell in (a, e, i).** Escapers from the two swarms
then have the same (e, i) distributions, both equal to the inflow's.

## 3. The hypothesis

- **H0, today is effectively calm.** Present-day exchange is close to L4/L5-symmetric.
  This is consistent with the pilot: objects that every catalogue calls Trojans escape
  at a ratio of 1.00 ± 0.57.
- **H1, the asymmetry is a fossil.**
  - The number ratio (N4/N5 ≈ 1.4–1.6) and the hotter L5 (proper-I KS D = 0.21) were
    set during an epoch that broke the time symmetry.
  - Candidate causes are a time-dependent potential (Jupiter migrating in one direction,
    a planetary encounter) or dissipation (gas drag).
  - Today the swarms drain a population that is not in balance with the present
    inflow.

This explains why published mechanisms have the shape they do. The sign of Pirani et
al.'s (2019) asymmetry depends on the *direction* of Jupiter's migration, and Li et
al. (2023) need an encounter. Both supply an arrow of time.

## 4. Predictions (Step 5)

**P1. Escapers inherit the fossil.** If H1 holds, escapes sample what each swarm
contains. L5 escapers should then be hotter than L4 escapers, as the cores are. In
steady state (where H1 fails) they would match.
- Test: KS on the escapers' inclinations (at escape and at t0), L4 against L5.
- Needs a few hundred escapers, so the 4.5 Gyr complete-sample run. The pilot has 13.

**P2. Captures do not inherit it.** Present-day captures are set by the inflow, so they
are symmetric in number and in inclination.
- Observational test: the observed transients (the 12 L4 and 16 L5 "interlopers" in
  V24's box) in L4 against L5.
- Simulated test: by the symmetry, L5 escape routes are L4 capture routes, so the fates
  of escapers give the capture statistics without integrating backwards.

**P3. Fates match at fixed orbit.** Escapers that leave with the same (e, i) end the
same way (ejection, Sun, planet, long-lived Centaur, recapture) whichever swarm they
left. A difference at fixed (e, i) would measure how much Saturn and the other planets
break the symmetry.

**What would falsify H1.**
- The transients (P2) turn out hot in L5 like the core. That would mean today's
  exchange can maintain the asymmetry.
- Or the 4.5 Gyr run shows erosion that turns an L4/L5-symmetric start into
  N4/N5 ≈ 1.5. Di Sisto 2014/2019 and the pilot make that very unlikely.

## 5. Caveats

- **The symmetry is exact only in the restricted problem.** How badly the real system
  breaks it is itself measurable (P3).
- **Steady state needs the fringe to have been refilled over times longer than its
  emptying time.** If it hasn't, the fringe is still a fossil even under H0.
- **The novelty of the framing is not yet established** (Step 4). "Trojan capture
  needs a time-dependent past" is old (Morbidelli et al. 2005; Nesvorný et al. 2013).
  What might be new is reading the *L4/L5 difference* as the measure of how strongly
  the symmetry was broken.

## 6. Revisions after the literature review (2026-10-02)

Source: `docs/CAPTURE_ASYMMETRY_LITERATURE.md`. The quotes from Sicardy & Dubois
(2003, §6, p. 335), Lhotka & Celletti (2015, §4) and Horner & Evans (2006, §3) were
checked against the PDFs.

**1. Breaking time symmetry is necessary but not sufficient.**
- Sicardy & Dubois's co-orbital Hamiltonian (Eqs. 15–17) stays symmetric under
  (φ, J) → (−φ, −J) without reversing time, *even while the masses change*.
- So a time-dependent epoch alone (for example, Jupiter growing) leaves L4 and L5
  equal. Their §9 (p. 346) puts it as: the asymmetry "is caused by an explicit
  variation of J_s (i.e. by a torque applied to the satellite), and not by a variation
  of a_s per se".
- The epoch must also supply a forcing that is odd under the mirror map: a migration
  or drag torque, or a perturber on one side.
- Lhotka & Celletti (2015, §4) make the same point for Poynting–Robertson drag: the
  conservative L4 and L5 are "mirror symmetric", and the dissipative terms don't change
  sign under the mirror map.

**2. The pieces are known; the synthesis appears to be new.**
- Already published:
  - inward migration enlarges the L4 region (Sicardy & Dubois §6);
  - reversing the migration reverses the sign (Pirani et al. 2019a §4.2, outward gives
    N4/N5 ≈ 0.6);
  - "Newton's equations are time-reversible, so some form of dissipation is required
    for permanent capture" (Horner & Evans 2006 §3).
- Not found anywhere:
  - the Roy–Ovenden map used for Trojans;
  - pairing capture into L4 with escape from L5;
  - the steady-state argument;
  - the observation that Pirani's capture-dominated run and Li's loss-dominated run
    are mirror images.
- Open risk: Hou, Scheeres & Liu (2016, CeMDA 125, 451) "studies the symmetry problem".
  It is closed access and not yet read. **Get it from Leiden before claiming novelty.**

**3. The asymmetry is an extra cold L4 component (new measurement).**
In the complete sample, binned by proper inclination:

| Proper I | N4 | N5 | N4/N5 (families removed) |
|---|---|---|---|
| < 15° | 1,121 | 559 | 2.01 ± 0.10 |
| ≥ 15° | 559 | 570 | 0.98 ± 0.06 |
| 0–10° | 739 | 343 | 2.15 ± 0.14 |
| 10–20° | 613 | 445 | 1.38 ± 0.09 |
| 20–30° | 270 | 289 | 0.93 ± 0.08 |
| > 30° | 58 | 52 | 1.12 ± 0.21 |

Source: `campaign/reports/pilot_10myr/ratio_by_inclination.csv`; the same holds with
families included and with osculating I.

- **The hot halves of the two swarms are equal.** The whole number excess, about 550
  objects, is cold L4 objects.
- **So "L5 is hotter" and "N4 > N5" are one fact, not two.** This also dissolves the
  tension noted from Pirani et al. 2019b and Li et al. 2023a, that the asymmetry
  weakens as inclination rises: that is exactly what an extra cold L4 component looks
  like.
- **Survey bias doesn't explain it.** H ≤ 13.6 is the complete range, and V24's
  debiased model has the same KS distance (0.221).

**4. The target for a capture mechanism, sharpened.** Explain an extra cold
(I ≲ 15–20°) population in L4, about the size of L5's cold population, on top of two
swarms that are otherwise equal. No published mechanism reports inclinations swarm by
swarm:
- Li et al. 2023a's Table 5 has the right sign but is far too weak (estimated KS D of
  0.01–0.03 against the observed 0.21).
- Li 2023a also predicts that L4 survivors have smaller libration amplitudes. The data
  show no difference (V24 p = 0.17; ours p = 0.11–0.33), which is a discriminant
  against it.

**5. A lead for P2.** Greenstreet et al. 2023 (Table 1) list 12 L4 and 4 L5 recently
captured, short-lived Jupiter co-orbitals. Their inclinations are a direct observational
test of whether present-day captures are symmetric.
