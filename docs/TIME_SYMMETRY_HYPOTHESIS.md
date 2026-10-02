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
