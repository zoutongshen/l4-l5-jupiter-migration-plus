# Mass ladder (mass_ladder)

100 catalogue Trojans, 14 masses, 1e+06 yr each, IAS15, one massive Trojan per simulation. Same objects at every rung (ic `2026-09-15_giants`, seed 20260915).

## Where the threshold falls

- Empirical: survival drops below 50% between **1 and 3 M_Jup**.
- Routh (1875) linear limit for this model: **40.9 M_Jup** (mu = 0.038484).
- Liberato & Winter (2020): mu = 0.023313, i.e. **24.0 M_Jup**.

The control rung above Routh (50 M_Jup) leaves 0% librating -- as it must, or nothing below it is trustworthy.

Rungs finished: 14/14.

## Where the real Trojans sit

(624) Hektor, the largest known Trojan, is ~1e-9 M_Jup. At that rung 97% still librate, against 97% for massless test particles -- the same objects, the same three losses.

Real Trojan masses are ten orders of magnitude below the threshold, so treating them as massless costs nothing for single-body dynamics. Whether their *mutual* gravity matters is a separate question -- see the self-gravitating swarm run.

## Per rung

| m_jup | mu | n | n_librating | f_librating | f_L4 | f_L5 | z_L4_L5 | n_horseshoe | n_circulating | n_removed | median_t_leave_yr | max_dE_rel |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.0009539 | 100 | 97 | 0.97 | 0.9524 | 1 | -1.348 | 0 | 3 | 3 | 1.36e+04 | 6.101e-14 |
| 1e-09 | 0.0009539 | 100 | 97 | 0.97 | 0.9524 | 1 | -1.348 | 1 | 2 | 3 | 8805 | 9.819e-14 |
| 1e-06 | 0.0009539 | 100 | 97 | 0.97 | 0.9524 | 1 | -1.348 | 0 | 3 | 3 | 6090 | 8.425e-14 |
| 0.001 | 0.0009548 | 100 | 97 | 0.97 | 0.9524 | 1 | -1.348 | 0 | 3 | 3 | 3.736e+04 | 8.438e-14 |
| 0.01 | 0.0009634 | 100 | 97 | 0.97 | 0.9524 | 1 | -1.348 | 0 | 3 | 3 | 4.161e+04 | 8.012e-14 |
| 0.1 | 0.001049 | 100 | 97 | 0.97 | 0.9524 | 1 | -1.348 | 0 | 3 | 3 | 6060 | 8.841e-12 |
| 0.3 | 0.00124 | 100 | 95 | 0.95 | 0.9524 | 0.9459 | 0.1426 | 1 | 4 | 3 | 7.1e+05 | 8.074e-14 |
| 1 | 0.001906 | 100 | 74 | 0.74 | 0.7302 | 0.7568 | -0.2928 | 0 | 24 | 23 | 4.947e+05 | 2.389e-11 |
| 3 | 0.003805 | 100 | 48 | 0.48 | 0.5397 | 0.3784 | 1.559 | 1 | 47 | 35 | 1.231e+05 | 1.591e-10 |
| 10 | 0.01039 | 100 | 30 | 0.3 | 0.381 | 0.1622 | 2.305 | 0 | 70 | 7 | 2.173e+04 | 2.831e-11 |
| 24 | 0.02331 | 100 | 0 | 0 | 0 | 0 | nan | 2 | 88 | 5 | 112.5 | 4.759e-12 |
| 30 | 0.02875 | 100 | 0 | 0 | 0 | 0 | nan | 0 | 88 | 4 | 97.5 | 7.706e-12 |
| 40.9 | 0.03847 | 100 | 1 | 0.01 | 0 | 0.02703 | -1.311 | 1 | 80 | 2 | 75 | 6.135e-12 |
| 50 | 0.04643 | 100 | 0 | 0 | 0 | 0 | nan | 1 | 80 | 5 | 75 | 7.178e-12 |

`f_L4` and `f_L5` are the surviving fractions in each swarm, `z_L4_L5` the two-proportion z for the difference between them.

## Does the asymmetry depend on mass?

Largest |z| is **+2.31** at 10 M_Jup (L4 38.1% vs L5 16.2%). The rungs below the knee repeat one outcome rather than testing separately, so this is about 6 independent looks, and |z| > 2 among them arises by chance roughly 24% of the time. A hint worth checking with a bigger sample, not a detection -- and it sits at a mass no real Trojan has.

Below the knee the two swarms are flat and identical rung to rung -- the same objects leave at every mass -- so within this sample there is no mass axis on the asymmetry where real Trojans live.

## Cost

1,400 integrations, 22.32 CPU-h, 57.4 CPU-s each. Worst |dE/E| over the whole ladder: 1.6e-10.

