# Stage 8 figures, group meeting 2026-11-10

Regenerate with `python campaign/scripts/s8_summary_figures.py --out updates/11102026`
(`--only 03 09` for single figures; the observed sample is cached in `observed_proper_I.csv`, so no SSD needed).
All pages are also in `stage8_figures.pdf`. Observed = complete sample (H ≤ 13.6),
V24 families removed, proper I. Simulated = Trojans at 1.2 Myr (10.7 Myr in fig. 8), final I.

| # | file | shows |
|---|---|---|
| 1 | 01_tracks_nu_mgap.png | the knobs: ν(m) = ν_I / (1 + (m/M_gap)³), Jupiter's a(t), m(t) |
| 2 | 02_ratio_vs_nu.png | cold-start N4/N5 against ν_I and against ν at 10 / 20 / 30 M⊕ (common subset e0 < 0.05, i0 < 15°, all 11 runs); table in 02_ratio_vs_nu.csv |
| 3 | 03_buildup_vs_mass.png | co-orbital L4-side / L5-side against Jupiter's mass and time: builds from 3 M⊕, peaks at ≈ 0.65 M_gap (vertical lines), relaxes, frozen after the gas |
| 4 | 04_counts_by_I.png | actual L4 and L5 counts by inclination: observed vs D6, D8, D7 |
| 5 | 05_cdf_by_I.png | inclination CDFs per swarm, KS D and medians |
| 6 | 06_ratio_by_I.png | N4/N5 by I for every inclination run vs observed; (b) reweighted to 7 disc models |
| 7 | 07_e0_i0_maps.png | N4/N5 (and counts) in starting (e0, i0) for D8 and D7 |
| 8 | 08_frozen_10myr.png | the same Trojans at 1.2 and 10.7 Myr (C2→D4, D7→D10, D8→D11) |
| 9 | 09_isolation_mass.png | E1 / C2 / E1b: pebble isolation mass 10 / 20 / 30 M⊕ with the same ν(m). The peak stays at 16–17 M⊕ while its time moves 158 → 230 kyr: the gap brake sets it, not growth |
