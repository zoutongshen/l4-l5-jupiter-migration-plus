# The Pirani model, stage by stage

Written 2026-10-09 for the Tuesday meeting. Sources: Pirani et al. 2019a (A&A 623, A169;
"P19a") and 2019b (A&A 631, A89; "P19b"), both in `literature/02_mechanisms/`. Page and
table numbers are from the arXiv versions. "(ours)" marks our own reading or arithmetic.

**Short version.** Pirani runs a forward model: a symmetric disc, then migration plus
growth, then an asymmetric population, then long-term evolution. She *varies* only the
first stage (the migration rate). The later stages are *checked* to leave the ratio
roughly unchanged, but never fitted. Simulated N4/N5 at t = 5 Myr is compared directly
with the observed total ratio.

---

## Stage 0: initial disc (symmetric)

- Massless particles in 0.5 au annuli: 4–26 au (P19a), or 15.5–20.5 au around Jupiter's
  seed (P19b). Surface density ∝ r⁻¹, e < 0.01, i < 0.01°.
- No asymmetry is put in. Before migration starts, the horseshoe population is split
  about 50/50 between the L4 and L5 sides (P19a Fig. 11).
- P19b also runs "pre-stirred" discs (e ≈ 0.15, i ≈ 8.5°; asteroid-belt- or KBO-like).

## Stage 1: capture during gas-era migration and growth (the stage that is varied)

- Jupiter's seed (10⁻² M⊕) starts at about 18 au and follows pebble-accretion growth tracks
  (Bitsch+2015-like; Ida+2016 disc, α = 10⁻³). It grows and migrates inward to about
  5 au between 2.3 and 3 Myr, when the gas disc photoevaporates.
- **Mechanism (P19a §4.2).** Inward drift widens the L4 side of horseshoe orbits and
  narrows the L5 side (Sicardy & Dubois 2003), so particles spend longer on the L4 side.
  Jupiter's growth shrinks horseshoes into tadpoles and freezes that excess.
  - Gas drag on the particles is irrelevant: switching it off gives the same asymmetry
    (p. 11).
  - Growth is essential. Neptune makes the same horseshoe excess but does not grow
    enough to freeze it, so it disappears (Fig. 14).
- **Parameter scan (P19a Table 3, Fig. 8; read at t = 5 Myr).**

  | Rate × nominal | in situ | 0.15 | 0.25 | 0.5 | 0.75 | 1.0 | 1.25 | 1.5 | 2.0 |
  |---|---|---|---|---|---|---|---|---|---|
  | N4/N5 | 0.94 ± 0.35 | 1.23 ± 0.40 | 1.60 ± 0.44 | 1.63 ± 0.29 | 1.73 ± 0.29 | 1.80 ± 0.26 | 1.44 ± 0.15 | 1.63 ± 0.21 | 1.77 ± 0.27 |

  Above a threshold (v_lim ≈ 0.17 qΩr, about 1 au/Myr for a 1 M⊕ core) the ratio
  **saturates** at about 1.4–1.8. Outward migration reverses the sign (N4/N5 ≈ 0.6;
  Fig. 12).
- **Output:** flat Trojans (i ≈ 0.01°–1°) with 10³–10⁴ times today's mass. In P19b,
  pre-stirred discs give weaker asymmetries: 1.15 at the nominal start, and 1.3–1.4 when
  Jupiter starts at 23–29.5 au (Table 6, Fig. 8). The e and i effects are not separated.

## Stage 2: post-disc rearrangement or instability (checked, not fitted)

- P19a §5: Jupiter and Saturn are moved from a compact configuration to their current
  orbits, either smoothly (τ = 0.5 Myr or 5 kyr) or by a single instantaneous jump of
  Jupiter (Δa = 0.16 or 0.23 au).
- The result is that "the asymmetry ratio does not change significantly". 40% (0.16 au) or
  20% (0.23 au) of Trojans survive the jump. From this they conclude the ratio is
  "primordial".
- **Not tested:** an ice giant crossing one swarm (Nesvorný+2013 §3.4), survival effects
  of a fast jump (Li+2023a), a free-floating planet (Li+2023b), or new objects captured
  during the instability (jump capture). P19b simulates no instability.

## Stage 3: long-term erosion (checked, not fitted)

| Run | N4/N5 evolution | Note |
|---|---|---|
| P19a Table 2 | 1.80 → 1.92 (5–200 Myr) | Jupiter–Saturn in 1:2 resonance "preferentially destabilises L5" |
| P19b J0 | 1.60 → 1.60 (4.5 Gyr) | Jupiter only, 0.3% lost |
| P19b JS0 | 1.48 → 1.61 (4.5 Gyr) | Jupiter + Saturn, 39% lost; S ≈ 1.09 (ours) |
| P19b pre-stirred, JS | 1.28 → 1.14; 1.38 → 1.06 | about 170 survivors, "not particularly meaningful" |
| P19b embryos | random, sometimes reversed | set by which swarm hosts the largest embryos |

No run starts from symmetric swarms, and none splits the ratio by inclination.

## What is compared with observations

Simulated N4/N5 at **t = 5 Myr**, for every particle within ±2.5 Hill radii of Jupiter's
semimajor axis (tadpole, horseshoe and quasi-satellite orbits), is compared with the
literature totals: Grav+2011 1.4 ± 0.2, Szabó+2007 1.6 ± 0.1, Nakamura & Yoshida 2008
1.85 ± 0.57. Stages 2 and 3 enter only as consistency checks.

---

## Where this leaves us (ours)

For each inclination bin, today's ratio combines the three stages. P4 and P5 are the
primordial counts from Stage 1, s4 and s5 the fractions surviving Stage 2, and C the
number captured into each swarm during Stage 2, assumed symmetric:

    N4/N5(I) = [ (P4·s4 + C) / (P5·s5 + C) ](I)  ×  S_erosion(I)

1. **Only the left-hand side is observed**, so fitting Stage 1 and the later loss rates
   together is degenerate. Pirani avoids the problem by assuming s4 = s5, C = 0 and
   S_erosion = 1.
2. **S_erosion is measurable without any formation model.** Pilot: 1.0015 ± 0.0010
   over 10 Myr. Di Sisto: 1.02–1.07 over 4.5 Gyr. P19b JS0: ≈1.09. None of these is
   split by I yet. A long run from mirror-symmetric initial conditions, binned in I,
   closes that gap (the "symmetric-start control" in `ASYMMETRY_LOGIC_CHAIN.md` §6).
3. **The Stage 2 terms are what remain.** They are constrained by the shape in I (a
   cold, asymmetric primordial component diluted by a hot, symmetric captured one), and by
   libration-amplitude, D and e distributions that show no difference between the swarms.
4. **Stage 1 constrains a rate only weakly.** Because the ratio saturates, the total ratio
   gives only a lower bound on migration speed relative to growth. Whether the
   mechanism still works for objects at about 5–10° inclination with small e is
   untested. P19b's pre-stirred discs change e and i together.
