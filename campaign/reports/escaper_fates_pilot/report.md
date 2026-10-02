# escaper_fates_pilot

Written 2026-10-02. Step 5a of the capture-asymmetry plan (`docs/TIME_SYMMETRY_HYPOTHESIS.md`).
Script: `campaign/scripts/s5_escaper_fates.py`. Per-object data: `fates.csv` here;
tracks on the SSD under `runs/escaper_fates_pilot/`.

**The sample.** All 41 escapers of `pilot_10myr` in the well-determined sample
(V24-selected, U ≤ 4):
- 13 Trojans (8 from L4, 5 from L5);
- 28 interlopers (12 from L4, 16 from L5), the comet-like objects that only V24's box admits.

**The method.**
1. **Replay.** Each escaper is replayed with the pilot's own setup (WHFast) up to its
   first entry within 2.1 R_H. All 41 entry times match the pilot exactly.
2. **Follow.** Each is then followed with TRACE (an integrator for close encounters)
   until it reaches r > 1000 au or collides, or for up to 100 Myr. Elements beyond
   30 au are barycentric.

**Short answer.**
- Every escaper leaves the planetary region within a few Myr: median 0.3 Myr after
  escape, longest 14.5 Myr.
- 26 leave on hyperbolic orbits. 15 pass 1000 au still bound, which makes them
  Oort-cloud candidates.
- None hits the Sun or a giant planet.
- On the way out they spend their time as Jupiter-family comets, Centaurs and
  scattered objects.
- 8 of the 41 are briefly recaptured as Jupiter co-orbitals, into the same swarm or
  the opposite one. These phases last 1–120 kyr, and none is permanent.
- With 13 real Trojan escapers, nothing distinguishes L4 from L5. This sample can't
  test predictions P1–P3; it shows the pipeline works and what the exchange looks like.

## Fates

| Group | From | n | Hyperbolic | Bound past 1000 au | Median life after escape |
|---|---|---|---|---|---|
| Trojan | L4 | 8 | 4 | 4 | 0.37 Myr |
| Trojan | L5 | 5 | 4 | 1 | 0.15 Myr |
| Interloper | L4 | 12 | 7 | 5 | 0.33 Myr |
| Interloper | L5 | 16 | 11 | 5 | 0.28 Myr |

**Encounters.** The median number of entries within 1 R_H is 7–18 for Jupiter and
5–8 for Saturn.

**Time in each orbital class** (mean fraction of each object's life after escape):

| Group | From | JFC (2 < T_J < 3) | Centaur | Outer (a > a_N) | Co-orbital |
|---|---|---|---|---|---|
| Trojan | L4 | 0.34 | 0.23 | 0.40 | 0.03 |
| Trojan | L5 | 0.32 | 0.18 | 0.46 | 0.01 |
| Interloper | L4 | 0.35 | 0.24 | 0.32 | 0.04 |
| Interloper | L5 | 0.65 | 0.15 | 0.15 | 0.04 |

**After escape, Trojans and interlopers behave alike.** This supports reading the
interlopers as members of the same exchange population: comets passing through the
co-orbital region.

## Temporary recaptures

A co-orbital phase means |a − a_J| < 5% of a_J and 20° < |σ| < 130°, held for at least
1 kyr. Episodes less than 10 kyr apart are merged into one phase.

| Object | Group | Escaped from | Recaptured into | Starts after escape | Lasts |
|---|---|---|---|---|---|
| 54097899 | Trojan | L4 | L5 | 18.6 kyr | 4.4 kyr |
| 54053729 | interloper | L4 | L5 | 14.6 kyr | 40.1 kyr |
| 54515418 | interloper | L4 | L4 and L5, horseshoe-like, 3 phases | 0–109 kyr | 199 kyr in total |
| 54385769 | interloper | L5 | L4 and L5 | 2.3 kyr | 17.3 kyr |
| 54386675 | interloper | L5 | L5, twice | 0.4 and 30.5 kyr | 4.6 and 1.3 kyr |
| 20487496 | interloper | L5 | L4 | 1.2 kyr | 1.2 kyr |
| 20613709 | interloper | L5 | L5 | 0.1 kyr | 1.8 kyr |
| 54292826 | interloper | L5 | L5, then L4 twice | 0.4–42 kyr | 16.3 kyr in total |

- **Escapers can be recaptured by the opposite swarm.** That is the exchange the
  time-symmetry argument describes, seen directly. Four objects leave one swarm and
  are temporarily held by the other.
- **Every recapture is temporary.** This fits the Liouville argument: from outside,
  the chaotic fringe can be reached, the long-lived core cannot.
- **Object 54515418 is a long-lived high-inclination co-orbital.** It held I ≈ 44°
  and a = 5.20 au on average for about 230 kyr, swinging between the L4 and L5 sides.

## Caveats

- **The terrestrial planets are folded into the Sun.** 22 of the 41 escapers pass
  inside 1.5 au, where real encounters and impacts with the terrestrial planets are
  missing.
- **Comets aren't modelled physically.** There is no sublimation, fading or splitting,
  so the time spent as a JFC is an upper limit on the physical lifetime.
- **Ejection is a distance criterion.** "Bound past 1000 au" is an end state here, not
  a test of whether the object reaches the Oort cloud.
- **The sample is small.** 13 Trojans can't test P1–P3. The 4.5 Gyr complete-sample
  run would give hundreds of escapers.
