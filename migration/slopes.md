<!-- RETIRED 2026-08-31 — do not use as a verification target. -->

> **These numbers are not trustworthy.** They come from AMUSE-era runs using the
> `compute_trojan_counts` detector, which keys on instantaneous heliocentric
> distance and longitude rather than the resonant angle (see
> `MSc_PROJECT_PLAN.md` Section 3.2), and at ~150 particles per swarm they are
> noise-dominated. Worse, the **labels below cannot be trusted**: the notebook
> that generated them built its label as `f"{DURATION_KYR}kyr_{DIRECTION[:2]}"`,
> which maps both `inward` and `insitu` to `in`, so the `_in` / `_insitu` rows
> were separated by hand after the fact (Section 11.2). Four of the nine values
> are duplicated pairs. Kept for provenance only.

50kyr_out: L4/L5 ratio slope: 2.481e-07 yr⁻¹
100kyr_in: L4/L5 ratio slope: 1.238e-06 yr⁻¹
50kyr_in: L4/L5 ratio slope: -1.830e-07 yr⁻¹
500kyr_in: L4/L5 ratio slope: 1.477e-07 yr⁻¹
100kyr_out: L4/L5 ratio slope: 2.481e-07 yr⁻¹
500kyr_out: L4/L5 ratio slope: 1.238e-06 yr⁻¹
50kyr_insitu: L4/L5 ratio slope: 2.279e-07 yr⁻¹
100kyr_insitu: L4/L5 ratio slope: 5.468e-07 yr⁻¹
500kyr_insitu: L4/L5 ratio slope: 7.152e-08 yr⁻¹