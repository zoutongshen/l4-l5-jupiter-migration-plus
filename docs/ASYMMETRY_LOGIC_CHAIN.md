# The L4/L5 asymmetry is an initial condition, not erosion

Written 2026-10-07 for Simon. Numbers come from the 10 Myr pilot
(`campaign/reports/pilot_10myr/`); each one's source is given next to it.

**Claim.** L4's excess is an extra population of cold (I_p < 15°) L4 Trojans. Today's
dynamics erode both swarms equally and can't build it, so it was set at capture.

---

## 1. The sample: why U ≤ 4

**U is a property of the observations, not of the asteroid.**
- U turns dT and dP, the *fit* errors in perihelion time and period, into the 3σ drift
  along the orbit after 10 years. Each step is a factor of 4.42.
- It is set by the arc length, the number of oppositions and the astrometric quality
  (and through those, by brightness and observing geometry). Dynamics don't enter.
- So a simulated particle has no U.

**U ≤ 4 means multi-opposition** (median arc 1,245 d at U = 4, 69 d at U = 5), with
σ_a ≤ 4 × 10⁻⁴ au, about 0.3% of the tadpole width (`CATALOGUE_DEFINITIONS.md` §2).

**The U ladder tests the cut** (`uladder.csv`, figure below).
- Escape fraction in 10 Myr, L4 and L5 pooled:

  | U | 0 | 1–4 | 6–7 | 8 | 9 |
  |---|---|---|---|---|---|
  | Escaped in 10 Myr | 0.06% | 0.2–0.8% | ≈ 1% | 3% | 19% |

- The L5/L4 escape ratio is consistent with 1 for every cut up to U ≤ 7 (1.00 ± 0.57
  at U ≤ 4). Adding U = 8 fakes an L4 excess (0.59 ± 0.24), because L4 holds 3× more
  of these short-arc discoveries.

**Interlopers removed.** 28 comet-like objects (12 L4, 16 L5) in V24's box but on
neither the JPL nor the MPC list all left within a few kyr. They alone produced the
earlier "L5/L4 = 1.67 ± 0.52" (`INITIAL_CONDITIONS_KS.md` §3).

**Working samples.**

| Sample | N(L4) | N(L5) | N4/N5 |
|---|---|---|---|
| U ≤ 4, Trojans by V24 *and* JPL/MPC (dynamics) | 8,803 | 5,512 | 1.60 |
| Complete, H ≤ 13.6, V24-selected (distributions) | 1,836 | 1,213 | 1.51 |

In the complete sample U doesn't matter: only 1 of its 3,049 objects has U ≥ 5.

![U ladder](../campaign/reports/pilot_10myr/fig_uladder.png)

## 2. The expectation: a calm system can't make L4 differ from L5

(`TIME_SYMMETRY_HYPOTHESIS.md` §1–2, §6)

- **Symmetry.** The restricted problem is invariant under (x, y, z, t) → (x, −y, z, −t)
  (Roy & Ovenden 1955). Every route into L4 is the time-reverse of a route out of L5,
  at the same (a, e, i).
- **Steady state.** With fixed planets (Liouville) and a source population that favours
  neither side, capture and escape balance cell by cell, and all four rates are equal.
- **Breaking it** needs a forcing that is *odd* under the mirror map, such as a migration
  or drag torque. A time-dependent potential alone (Jupiter growing) is not enough
  (Sicardy & Dubois 2003 §9). Reversing the migration reverses the sign (Pirani et al.
  2019a: outward gives N4/N5 ≈ 0.6).

## 3. The observation: an extra cold L4 component

**N4/N5 by proper inclination** (complete sample, families removed;
`ratio_by_inclination.csv`):

| I_p | N4 | N5 | N4/N5 |
|---|---|---|---|
| < 15° | 1,121 | 559 | **2.01 ± 0.10** |
| ≥ 15° | 559 | 570 | **0.98 ± 0.06** |
| 0–10° | 739 | 343 | 2.15 ± 0.14 |
| 10–20° | 613 | 445 | 1.38 ± 0.09 |
| 20–30° | 270 | 289 | 0.93 ± 0.08 |
| > 30° | 58 | 52 | 1.12 ± 0.21 |

- **The hot halves of the two swarms are equal.** The whole excess (1,121 − 559 = 562
  objects) is cold L4.
- **"N4 > N5" and "L5 is hotter" are one fact.** KS on I_p: D = 0.211, p = 6 × 10⁻²⁹
  (`ks_tests.csv`; V24's debiased model: 0.221). Libration amplitude D doesn't differ
  (p = 0.11), and e_p only marginally (p = 0.05).
- It holds with families included and with osculating I, and the sample is complete.

![Cold L4 excess](../campaign/reports/pilot_10myr/fig_cold_l4.png)

## 4. Erosion today is symmetric

**10 Myr, U ≤ 4** (`escape_split.csv`, `ratios.csv`): 8 of 8,803 escape from L4 and
5 of 5,512 from L5, so Ṅ/N = 0.91 per 10⁴ Trojans per Myr in both. The escape ratio is
1.00 ± 0.57; the amplifier S4/S5 (the factor applied to N4/N5) is 1.0015 ± 0.0010.

**Ṅ per U class** (`s3_ndot_by_u.py` → `ndot_by_u.csv`; escapes per 10⁴ Trojans per Myr, 1σ Wilson interval):

| U | n L4 / L5 | k L4 / L5 | Ṅ/N L4 | Ṅ/N L5 | L5/L4 |
|---|---|---|---|---|---|
| 0 | 7,749 / 4,755 | 4 / 3 | 0.52 (0.32–0.85) | 0.63 (0.36–1.1) | 1.22 ± 0.93 |
| 1–4 | 1,054 / 757 | 4 / 2 | 3.8 (2.3–6.2) | 2.6 (1.3–5.3) | 0.70 ± 0.60 |
| **≤ 4** | 8,803 / 5,512 | 8 / 5 | **0.91 (0.64–1.3)** | **0.91 (0.58–1.4)** | **1.00 ± 0.57** |
| 5–7 | 312 / 199 | 4 / 1 | 13 (7.8–21) | 5.0 (1.9–13) | 0.39 ± 0.44 |
| 8 | 290 / 91 | 10 / 2 | 35 (25–47) | 22 (11–43) | 0.64 ± 0.49 |
| 9 | 939 / 151 | 183 / 21 | 195 (182–208) | 139 (113–170) | 0.71 ± 0.15 |

- **L4 and L5 agree within errors in every class up to U = 8.** U = 9 (0.71 ± 0.15) is
  orbit error: those orbits are essentially unknown.
- From U ≥ 5, Ṅ rises because the fit worsens. Why U = 1–4 escapes faster than U = 0
  is untested; they are later, hotter discoveries (median I 17° against 11°).

**Over the age of the Solar System.** Di Sisto's 4.5 Gyr amplifiers are
1.071 ± 0.025 (2014) and 1.020 ± 0.023 (2019). Erosion would need 1.5–1.6 overall and
2.0 in the cold bin.

**Fates** (`escaper_fates_pilot/report.md`): all 41 escapers leave the planetary region
(median 0.3 Myr; 26 hyperbolic, 15 bound beyond 1000 au). 8 are recaptured as
co-orbitals for 1–120 kyr, some into the opposite swarm, and none stays. Today's exchange
reaches only the chaotic fringe, not a long-lived cold core.

![Escaper fates](../campaign/reports/escaper_fates_pilot/fig_fates.png)

## 5. Conclusion

1. The asymmetry is measured on samples that are well determined (U ≤ 4) and complete
   (H ≤ 13.6).
2. A calm system keeps L4 = L5 by symmetry.
3. The data show L4 = L5 *except* for an extra 562 cold L4 objects.
4. Present-day loss is equal per object in both swarms. Over 4.5 Gyr erosion changes
   N4/N5 by ≤ 7%, against the factor 2 needed at I < 15°.

**So the asymmetry is an initial condition, laid down by a mirror-odd forcing.** The
next question is which forcing makes extra *cold* L4 objects. Jupiter's migration is the
natural candidate (its sign follows the direction of migration; Pirani 2019a). Li et al.
2023a is disfavoured: it predicts smaller L4 libration amplitudes, and the data show
none (V24 p = 0.17; ours p = 0.11–0.33).

## 6. Caveats

- 10 Myr and 13 Trojan escapers: the per-swarm Ṅ carries about 50% errors.
- Erosion hasn't been measured by inclination (Di Sisto's isn't split by I). A 4.5 Gyr
  run split by inclination, or a symmetric-start control, would close this gap.
- The symmetry is exact only in the restricted problem; the outer giants break it
  slightly.
- Novelty: Hou, Scheeres & Liu 2016 (CeMDA 125, 451) is still unread.
