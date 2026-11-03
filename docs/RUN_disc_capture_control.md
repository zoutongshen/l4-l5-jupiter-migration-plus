# Run plan: capture from a symmetric disc, without migration (arms 0 and 1)

Drafted 2026-10-23 for review **before** the production run. The code is in
`campaign/trojans/disc.py`, `campaign/scripts/s6_run_disc.py` and
`campaign/scripts/s6_disc_report.py`. Only smoke tests have run so far.

**In one line.** Start from a disc that is exactly L4/L5 mirror-symmetric, leave
Jupiter where it is (fixed, or growing in place), and see whether the Trojans it
ends up with are still symmetric in every (D, I, e) bin. Migration is out on
purpose: it hasn't been discussed with Simon.

---

## 1. Why

- **Loss:** symmetric. The pilot measures it as 1.00 ± 0.57 per object at U ≤ 4,
  and Di Sisto's 4.5 Gyr erosion changes the ratio by ≤ 7%.
- **The data:** N4/N5 = 2.0 below I_p = 15° and 1.0 above. The excess is cold L4.
- **If gain without migration is also symmetric:** a symmetric start can't make
  the cold L4 excess. Something mirror-odd has to have happened: migration during
  capture, or an asymmetric event later (the instability). This run does not
  choose between those two.
- **Why the run isn't trivial:** the mirror map (x, y, z, t) → (x, −y, z, −t)
  maps a *growing* Jupiter to a *shrinking* one. So symmetric capture under
  growth is not guaranteed by the theorem. The only published number is Pirani
  2019a in situ, 0.94 ± 0.35: noisy, and not split by inclination.

## 2. Setup

| | Arm 0 `fixed` | Arm 1 `grow` |
|---|---|---|
| Bodies | folded Sun + Jupiter only | same |
| Jupiter's orbit | circular, planar, a = 5.2026 au | same; velocity reset to circular after every mass change, so a cannot drift |
| Jupiter's mass | 1 M_J from t = 0 | 10 M⊕ → 1 M_J, exponential over t_grow = 0.1 Myr |
| Integrator | WHFast, dt = 1% P_J (as the campaign) | same |
| Removal | within 1 Hill radius of Jupiter (current mass) or beyond 100 au | same |

**The disc** is identical in both arms (seed 20261023).
- a uniform in 4.9–5.5 au; e uniform in 0–0.3; I uniform in 0–40°; Ω, ω, M
  uniform.
- e and I are uniform rather than physical, so every bin is filled.
- **Each particle has a mirror twin** (x, −y, z, −vx, vy, −vz). With Jupiter on
  the +x axis, this makes the starting disc *exactly* symmetric. Checked: 1000
  particles with σ > 0 and 1000 with σ < 0; twin σ sum ≤ 1e-13; identical a, e, I.

**Trojan.** The `docs/TROJAN_DEFINITION.md` classifier: alive at the end, and a
tadpole of one swarm in every 10 kyr window of the last 1 Myr. No catalogue is
used anywhere.

**Binned output** (`s6_disc_report.py`):
- N4 and N5 per bin, binned separately in D (half-amplitude, as in
  `s3_distributions.py`), I and e, using window means over the last 1 Myr.
- An I < 15° / ≥ 15° split, to compare with the observed 2.0 / 1.0.
- Each count also as a fraction of the disc. This shows whether a difference is
  extra L4 or missing L5.
- Twin fates: whether each mirror pair ends as L4+L5, L4+L4 and so on.

## 3. Smoke tests (done)

| Test | Result |
|---|---|
| Existing test suite after the `ChunkRunner` change (planets and forcing hook) | 54 pass. 2 layout failures were already there (`.claude/`, `docs/figures/`, `docs/trojan_definition_sources/`) |
| Mirror twins | exact, as above |
| Jupiter's a during growth | 5.202600 au throughout, e_J ~ 1e-16 |
| Both arms, 1000 particles, 50 kyr, disc 4.6–5.8 au | runs end to end; fixed 59/58, grow 57/52 at 25 kyr (meaningless; plumbing only) |
| Where Trojans come from (that smoke) | fixed arm: a₀ = 5.0–5.4 au. Grow arm: a₀ = 5.1–5.3 au. **None from a₀ < 5.0 or > 5.6 au.** Without migration nothing flows in; growth only freezes what is already co-orbital. Hence the 4.9–5.5 disc |
| Grow arm, full schedule (0.1 Myr growth, 1.2 Myr, 1000 particles, old 4.6–5.8 disc) | Works with the real 100-window classifier. N4/N5 = 41/48 = 0.85 ± 0.18; I < 15°: 19/25; I ≥ 15°: 22/23. Twins: 29 L4+L5, 8 L5+L5, 3 L4+L4. Consistent with symmetric, but too small to say more |

**Cost from the full smoke.**
- About 650 CPU-s per chunk of 125 particles. Most of that is fixed per-check
  Python overhead (1.26 × 10⁶ checks in 1.2 Myr), not the particles: only 12% were
  still alive at the end.
- Wall time was 4,070 s, six times the CPU time. That run was launched from a
  background shell, which macOS throttles. The pilot, run from your terminal, had
  CPU/wall ≈ 0.97 per core. **Launch from your own terminal.**

## 4. Production run

**Launched 2026-10-23** (approved by the user): `disc_fixed` and `disc_grow`, run
side by side, 4 chunks × 5,000 particles on 4 workers each (8 of the 10 cores).
Logs are in `data/runs/disc_*.log`. A 20 kyr throttle check gave CPU/wall ≈ 0.97 when
the env's python is called directly, not through `conda run`.

**Reweighted physical view** (`binned_physical.csv`): the same particles weighted to
Pirani+2019b's pre-stirred asteroid-belt disc (e ~ N(0.15, 0.07), I ~ N(8.5°, 7°),
cut at 0). Pirani's nominal disc (e < 0.01, I < 0.01°) is too cold to reach by
reweighting and would need its own run.

The commands as first planned (8 chunks, one arm at a time):

```bash
conda activate trojan-rb          # SSD mounted; data -> /Volumes/X10-Pro-4T/trojans
cd "<repo>"
python campaign/scripts/s6_run_disc.py --run-id disc_fixed --arm fixed \
    --n-pairs 10000 --t-end-yr 1.2e6 --n-chunks 8 --workers 8
python campaign/scripts/s6_run_disc.py --run-id disc_grow --arm grow \
    --n-pairs 10000 --t-end-yr 1.2e6 --t-grow-yr 1e5 --n-chunks 8 --workers 8
python campaign/scripts/s6_disc_report.py --run-id disc_fixed
python campaign/scripts/s6_disc_report.py --run-id disc_grow
```

- **Size:** 20,000 particles per arm, 8 chunks of 2,500.
- **Expected yield:** 9% of the wide smoke disc ended as Trojans. The narrower
  disc should do better, so roughly 2,000–4,000 Trojans per arm. That gives ±5%
  on N4/N5 in the I < 15° bin and about ±10% per finer bin. The observed 2.0 would
  stand out at ≫ 10σ.
- **Time (estimate, not measured at this N):** about 0.5–1 h wall per arm on 8
  workers, both arms ≈ 1–2 h. It is resumable (`--resume`). Run arm 0 first:
  if its wall time is far above 1 h, stop and we re-plan.
- **Disk:** about 50 MB per arm (the smoke test used 2.4 MB for 1,000 particles).

## 5. How to read the outcome

- **Arm 0 symmetric in every bin.** This is the expected baseline: a symmetric
  start stays symmetric in today's-style dynamics.
- **Arm 1 symmetric in every bin.** Growth alone makes no asymmetry, so the cold
  L4 excess needs a mirror-odd event. That is the result to take to Simon, as
  the motivation for the migration arms.
- **Arm 1 asymmetric.** This would be surprising and interesting. First check it
  against the twin fates and the bin populations, and then check whether it is
  cold-only.
- **Either way**, compare the I distribution of captures with the observed swarms.
  Growth capture from a uniform disc makes no prediction about *where* in I the
  Trojans sit: that is set by the disc we chose.

## 6. Caveats

- **Jupiter alone, circular and planar.** There is no Saturn, so the restricted
  problem's symmetry is as clean as it can be. Saturn would be the next control.
- **The disc is warm by construction** (e to 0.3, I to 40°), unlike Pirani's cold
  disc. That is what makes binning possible, but it is not a model of the gas-era
  disc.
- **One growth law** (exponential, 0.1 Myr, from 10 M⊕). Pirani's tracks are slower
  and start from a much smaller core. A second t_grow (for example 0.5 Myr)
  would test sensitivity.
- **No gas drag.** Pirani found it irrelevant to the asymmetry (2019a p. 11).
- **1 Myr classification.** A Trojan here means 1 Myr in tadpole, not 4.5 Gyr.

---

## 7. Results (2026-10-24; runs `disc_fixed`, `disc_grow`)

Reports: `campaign/reports/disc_fixed/`, `campaign/reports/disc_grow/`
(`summary.json`, `binned.csv`, `binned_physical.csv`, `formation.csv`).
Cost: 6.35 and 5.86 CPU-h; about 2.3 h wall for each arm, run side by side on 8 cores.

| | fixed | grow |
|---|---|---|
| Trojans (1 Myr tadpole) N4 / N5 | 2293 / 2287 | 1796 / 1780 |
| N4/N5 | 1.003 ± 0.030 | 1.009 ± 0.034 |
| I < 15° | 1.004 ± 0.047 | 1.022 ± 0.050 |
| I ≥ 15° | 1.001 ± 0.038 | 0.998 ± 0.045 |
| Capture fraction of the disc | 22.9% | 17.9% |
| Reweighted to the P19b pre-stirred disc | 1.003 | 1.021 (n_eff ≈ 1650) |
| Twin fates L4+L5 / L4+L4 / L5+L5 | 2251 / 0 / 0 | 1059 / 245 / 245 |

**What it shows:**
- **Swarms form without migration.** In the grow arm, tadpoles rise from 887/882 at
  10 kyr to 2004/1962 when growth ends (0.11 Myr). They then erode slowly, to
  1802/1782 at 1.1 Myr.
- **Every D, I and e bin, in both arms, is consistent with N4/N5 = 1.** The largest
  deviation is grow e ∈ [0.05, 0.1), at 1.107 ± 0.083 (1.3σ), and there are about 30
  bins. Capture probability falls with starting e (0.27 → 0.13) and with starting
  I ≥ 30° (0.11), identically in both swarms.
- **The cold bin.** The observed cold-bin N4/N5 = 2.01 ± 0.10 is excluded by the
  grow arm's 1.02 ± 0.05 at about 9σ. Growth of a non-migrating Jupiter cannot make
  the cold L4 excess.
- **The fixed arm reproduces the mirror theorem:** every surviving twin pair is
  L4+L5.
- **Growth breaks the symmetry of individual orbits, not of the ensemble.** In the
  grow arm, 490 twin pairs end in the same swarm, but L4+L4 = L5+L5 = 245. Growth is
  not mirror-odd, as Sicardy & Dubois 2003 §9 argue.
- **Errors are Poisson.** Twins are correlated, so these errors are conservative.

**So:** symmetric loss (pilot) plus symmetric gain (this run) means the asymmetry
needs a mirror-odd event: migration during capture, or something asymmetric
afterwards (the instability). Caveats §6 apply: Jupiter alone and circular, one
growth law, a warm disc, 1 Myr classification.
