# Self-gravitating Trojan swarm (swarm)

200 Trojans -- the largest per swarm, which carry the swarm's mass -- for 1.00 Myr, run twice: once with real H-derived masses and full mutual gravity, once massless. Identical objects, identical initial conditions, WHFast, dt = 1% P_J.

Total mass carried: **1.332e-11 Msun = 4.44e-06 M_Earth** (albedo 0.07, density 1000 kg/m^3).

## Verdict

**Self-gravity is real but dynamically negligible over 1.00 Myr.** It perturbs 132/200 objects, against 10/200 for the largest meaningless nudge, so this is a force doing work and not a chaotic handful separating. The work is tiny: the biggest semi-major-axis change is 1.01e-03 au, 1.9e-04 of a, no object changed libration class, and all 200 survived in both arms.

## Against the chaos floor

Each calibration arm is the identical massless system started a known, physically meaningless distance away. Trojan orbits are chaotic well inside 1 Myr, so two runs separate no matter what starts them apart; these arms measure how much separation a given seed buys, and self-gravity is then quoted in the same units.

| arm | seed at t = 0 | objects moved | largest \|Δa\| (au) | largest Δa/a | largest Δamp (deg) |
|---|---|---|---|---|---|
| **self-gravity** | 4.44e-06 M_Earth of real mass | **132/200** | 1.011e-03 | 1.9e-04 | 0.810 |
| nudge | 1e-18 au | 0/200 | 0.000e+00 | 0.0e+00 | 0.000 |
| nudge | 1e-15 au | 9/200 | 3.715e-04 | 7.1e-05 | 0.569 |
| nudge | 1e-12 au | 10/200 | 3.433e-04 | 6.6e-05 | 0.358 |

"Moved" means the stored window-mean a changed by at least one float32 step, 4.8e-07 au near 5.2 au (`integrate.py` writes these diagnostics as float32). It is a floor on what can be seen, not a claim about the dynamics, and 38 of self-gravity's 132 sit within two steps of it. Every arm is measured through the same floor, so the counts compare.

The 1e-18 au arm moves nothing: that displacement is below what a float64 coordinate near 5 au can even represent, so it is the control run repeated. It reaching 0/200 is the check that these integrations are deterministic and that the other rows are caused rather than noise.

- Libration class changed for **0/200** objects (and 0 across all calibration arms).
- Survivors: 200 self-gravitating vs 200 massless vs 200 nudged.

## Caveats

- This is the top 100 per swarm, not all 16,445. They carry most of the mass but not all of it, and the many small bodies are not represented.
- Masses come from one albedo and one density; good to a factor of a few per object.
- 1.00 Myr, not 4.5 Gyr. A null here bounds the short-term effect, and mutual scattering accumulates, so it does not licence the assumption over the full campaign.

