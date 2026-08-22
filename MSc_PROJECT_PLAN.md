# MSc Research Project — Working Plan

**Status:** draft for discussion. Written 2026-08-22.
**Purpose:** shape the scope well enough to fill in the Leiden MSc research project
registration form (working title + short description), and to serve as the
running plan afterwards.
**Repo:** `L4-L5-Jupiter-Migration-Plus`, branch `rebound-port`.

---

## 0. Context in one paragraph

The project grew out of the SIMOA group report (`SIMOA_report/`, with Colby
Malcom et al.), which asked whether Jupiter's migration can produce the observed
excess of L4 over L5 Trojans. The simulation engine has since been ported from
AMUSE to REBOUND (`analysis/simulation/`), with a custom O(1) Jupiter-only
`MigrationDriver` replacing REBOUNDx. Four production migration runs exist
(`analysis/results/batch_summary.txt`). The task now is to turn this into a
defensible MSc research project rather than a longer version of the course
project.

---

## 1. What the literature leaves open

All four PDFs in `literature/` were read in full.

Every published treatment of the L4/L5 asymmetry reports **one number**: the
end-of-run count ratio N4/N5. Nobody publishes the time-resolved behaviour, and
nobody publishes what is actually happening in the co-orbital phase space.

### 1.1 Observational target

- MPC raw ratio N4/N5 ~ 1.8 (11 552 Trojans, Jan 2022).
- Debiased value adopted field-wide: **N4/N5 = 1.6 +/- 0.1** (Szabo et al. 2007).
- WISE-based estimate: 1.34-1.40 (Grav et al. 2011, 2012).
- Observed libration ("resonant") amplitudes peak well **below 15 deg**.
- Gyr-scale differential erosion contributes only ~10% (Di Sisto et al.).
  So the asymmetry must be primordial.

### 1.2 Two competing mechanisms, with opposite direction dependence

|  | **Pirani et al. 2019** | **Li et al. 2023** |
|---|---|---|
| Direction | fast **inward** (gas disk) | fast **outward** jump (instability) |
| Channel | **capture** — relative drift piles particles onto the L4 side of *horseshoe* orbits; Jupiter's subsequent mass growth pinches horseshoes into tadpoles | **survival** — outward drift *contracts* the L4 island (amplitudes shrink, stabilise) and *expands* the L5 island (amplitudes grow, cross L3, become horseshoe, escape) |
| Population acted on | **transients** | **permanent residents** |
| Result | N4 > N5 | N4 > N5 |
| Stated problem | needs Delta a_J > 3.5 AU inward, which overfills the main belt (Deienno et al. 2022) | all survivors end with amplitude A >~ 15 deg, **larger than the observed Trojans**; Li flags this as unresolved and waves at collisions |

Two mechanisms, same sign of the answer, **opposite sign of the driver**, acting
on **two different populations**.

### 1.3 Li et al. 2023 — numbers worth having to hand

- Control model: outward, `Delta a_J = 0.5 AU`, e-folding `tau = 1000 yr`,
  `Delta t = (10/3) tau`, hence **`adot_J = 1.5e-4 AU/yr`**. Gives R45 = 1.50 at
  1 Myr, 1.68 at 1 Gyr.
- Migration law: `a_J(t) = a_J + Delta a_J [1 - exp(-t/tau)]`.
- Dimensionless rate (Sicardy & Dubois 2003): `adot_J = (2 a_J n_J mu) * alpha`.
  L4 and L3 merge and the leading island vanishes at `alpha = 0.7265`, i.e.
  **`adot_crit = 3.8e-3 AU/yr`**.
- Test Trojans seeded directly at `sigma_0 = +/-(60 + Delta sigma_0)`, standard
  `Delta sigma_0 = 30-80 deg`, 500 per swarm, swift_rmvs3, `dt = 0.5 yr`.
- R45 rises steeply with Jupiter's eccentricity: 1.50 / 1.70 / 2.47 at
  `e_J = 0.05 / 0.10 / 0.15`.
- **R45 vs Trojan inclination is already non-monotonic** in their own Table 5
  (1.50, 1.54, 1.38, 1.30 at i = 0, 10, 20, 30 deg) and they cannot explain it.

### 1.4 Supporting context

- **Raymond & Morbidelli 2014 (Grand Tack review):** Jupiter goes inward to
  ~1.5 AU then back out to ~5.4 AU in the gas disk, ~1e5 yr per leg
  (`adot ~ 4e-5 AU/yr`). The natural physical setting for a direction
  **reversal** — i.e. both mechanisms firing in sequence.
- **Boskovic et al. 2025 (TOI-178):** co-orbital regions in a 2:4:6:9:12 Laplace
  chain are ~30% wider than Dermott & Murray theory (52% for the non-chain
  planet b). Hook for an exoplanet extension; pairs with the leftover AMUSE
  scripts `generate_resonant_chain.py` / `resonance_chain_planetary_system.py`.
- `Debes2025.pdf` at repo root is about white-dwarf planets and is **unrelated**
  to this project.
- `literature/AMUSE pages 320 - 338.pdf` is the AMUSE-book bridge chapter —
  course background, not science input.

---

### 1.5 Literature refresh (web check, 2026-08-22)

The four PDFs in `literature/` are not the whole current picture. Four items
matter, none of them in the folder:

- **Vokrouhlicky et al. 2024** (arXiv:2401.15537), Catalina Sky Survey,
  bias-corrected: **N4/N5 = 1.45 +/- 0.05 for H < 15**. This *supersedes*
  Szabo+07's 1.6 +/- 0.1 as the number to hit — lower, and 2x tighter. Same paper
  reports an **inclination-distribution asymmetry: L4 tighter, L5 broader**, plus
  new proper elements and families (9 at L4, 4 at L5).
  *Consequence:* there are now **three** observables to match jointly —
  N4/N5, the libration-amplitude distribution, and the inclination distributions.
  And Li+23's own Table 5 already shows R45 non-monotonic in inclination
  (1.50, 1.54, 1.38, 1.30 at i = 0, 10, 20, 30 deg) with no explanation. That is a
  direct, unexploited hook: the model that explains the i-dependence should also
  predict the observed i-asymmetry.
- **Li, Xia & Zhou 2023b** (arXiv:2304.13598), *The invasion of a free-floating
  planet and the number asymmetry of Jupiter Trojans* — a **third** mechanism from
  the same group. Worth one sentence in the introduction so the two-channel
  framing is not presented as exhaustive; not worth simulating.
- **Bottke, Marschall, Nesvorny & Vokrouhlicky 2023**, *Origin and Evolution of
  Jupiter's Trojan Asteroids*, Space Sci. Rev. 219, 83 — the current review, and
  the right citation for "primordial capture during the giant-planet instability".
  Use it for framing; it does not resolve the L4/L5 ratio.
- **Yoshida, Terai & Ohtsuki 2026** (arXiv:2603.16485), Subaru colours/sizes of
  small Trojans — **L4 only**, no L5 comparison. Confirms the comparative L4-vs-L5
  *physical*-properties picture is still thin; the asymmetry argument therefore
  rests on orbital statistics, which is exactly what this project produces.

Nothing found supersedes Li+23 as the dynamical state of the art on this specific
question. The gap identified in Section 1 stands: still nobody publishes the
time-resolved channel decomposition.

---

## 2. The central hypothesis

> **The L4/L5 asymmetry is the sum of two channels — differential *capture*
> (favoured by inward migration, operating on transient horseshoe librators) and
> differential *survival* (favoured by outward migration, operating on resident
> tadpole librators). Their relative weight is set by the migration rate and
> direction. Where the two channels compete, N4/N5 is non-monotonic in the
> migration rate.**

This directly matches the three instincts driving the project:

| Instinct | Formal quantity |
|---|---|
| "asymmetry in the **rate** AND **amount**, not merely number" | escape/capture **flux** per swarm (rate) vs the **libration-amplitude distribution** and phase-space area (amount). The "amount" is precisely Li's admitted failure. |
| "are they **permanent residents or jumping around**?" | per-particle state history: tadpole-L4 / tadpole-L5 / horseshoe / quasi-satellite / escaped. The capture channel lives in transients, the survival channel in residents — so this *is* the mechanism decomposition. |
| "change in asymmetry is **not monotonic** in direction and rate" | the handoff between the two channels as `alpha` is swept. |

Nobody has drawn the R45(alpha, direction) curve, because nobody measures the
two channels separately.

---

## 3. Honest assessment of the current setup

**The numbers in `analysis/slopes.md` and `analysis/results/batch_summary.txt`
are consistent with noise and should not be used as evidence for anything.**
Two independent reasons.

### 3.1 Wrong migration regime

The runs move Jupiter 0.4 AU over 50-500 kyr, i.e. `adot_J = 8e-6` down to
`8e-7 AU/yr`:

| | `adot_J` (AU/yr) | `alpha` | `tau_mig / T_lib` |
|---|---|---|---|
| This project, 50 kyr | 8e-6 | 2.1e-3 | 4400 |
| This project, 500 kyr | 8e-7 | 2.1e-4 | 44 000 |
| Li+23 control | 1.5e-4 | 4e-2 | 230 |
| Grand Tack leg | 4e-5 | 1e-2 | 880 |
| Critical (island merger) | 3.8e-3 | 0.7265 | 9 |

(Jupiter's tadpole libration period `T_lib ~ 148 yr`.)

The current runs are **20-200x slower than the slowest case known to produce an
asymmetry**, and ~1e3 below critical. This is the deeply adiabatic regime, where
the libration action is an adiabatic invariant and theory predicts **no**
asymmetry.

### 3.2 The Trojan detector cannot measure Trojans

`analysis/simulation/analysis.py: compute_trojan_counts` uses the
**instantaneous heliocentric distance** `r` (not the semi-major axis `a`) and the
**instantaneous longitude** within +/-25 deg of +/-60 deg.

- Real tadpole librators swing +/-40 deg or more in the resonant angle, so this
  misses genuine Trojans at the ends of their libration **and** counts
  non-resonant passers-by.
- The resonant angle `sigma = lambda - lambda_J` is **never computed anywhere in
  the codebase**.
- Only ~150-200 of 10 000 particles land in each swarm, so the 1-sigma Poisson
  noise on N4/N5 is ~0.12 — the same size as the effects in the batch table.
  Seeds 42 and 1042 flip the sign of the "slope", which is exactly what pure
  noise looks like.

### 3.3 So: is the remembered non-monotonicity real?

Unknown, and it must be treated as unknown. At `alpha ~ 2e-3` with N ~ 150 per
swarm, those particular runs are noise-dominated. **The reason to believe a real
non-monotonicity exists is the two-mechanism argument in Section 2, not those
runs** — plus the unexplained non-monotonicity already sitting in Li's Table 5.
Phase 0 exists to settle this.

---

## 4. Open item: dropping the terrestrial planets "doesn't work"

Reported 2026-08-22; cause not yet diagnosed. **Needs a definition of "doesn't
work" before the next discussion** — crash, wrong output, no speed-up, or
changed physics?

Leading hypotheses, in order, all *unverified*:

1. **Hardcoded `jupiter_index=4` in call sites that don't pass it.** With
   `planet_set='giants'` the massive list is Sun, Jupiter, Saturn, Uranus,
   Neptune, so Jupiter is index **1** and index 4 is **Neptune**. Confirmed
   offenders:
   - `analysis/run_batch.py`, in `do_migration`: `compute_jupiter_orbit(...)` is
     called with **no** `jupiter_index`, so it defaults to 4 and would report
     Neptune's semi-major axis as `jupiter_a_final`.
   - `analysis/01_burnin.py`: `plot_snapshot(...)` is called with no
     `jupiter_index`, so Trojan zones would be drawn around the wrong planet.
   - Defaults of 4 also sit in `simulation/analysis.py`,
     `simulation/visualization.py`, `simulation/migration.py`.
   `simulation/runner.py` itself passes `config.jupiter_index` correctly, so the
   *integration* is probably fine and only the *reporting* is wrong — which would
   look exactly like "it doesn't work".
2. **Checkpoint / `planet_set` mismatch.** `sim_from_state` rebuilds from saved
   massive arrays; continuing an 8-planet burn-in into a 4-planet migration run
   (or vice versa) mixes array shapes silently.
3. **Comparability, not correctness.** `create_solar_system` draws random angles
   per planet in order, so changing the planet set changes the RNG stream and
   every run at a given seed becomes a different realisation. Not a bug, but it
   makes A/B comparison at fixed seed meaningless.
4. Genuine physics (unlikely): the terrestrials are irrelevant to Jupiter's
   co-orbital region; Saturn is the one that matters and is retained in
   `'giants'`.

**Action:** reproduce with a short run, capture the actual error/output, and work
down this list. If it is (1), it is a ~10-line fix.

---

## 5. Compute: ALICE changes the picture

**New as of 2026-08-22: access to ALICE (Leiden HPC) secured.**

This removes compute as the binding constraint, and changes the plan in two
ways:

- The efficiency argument (restricted 3-body setup, direct tadpole seeding) is
  now about **statistics and correctness**, not about fitting on a laptop. It is
  still worth doing — seeding directly into tadpole/horseshoe orbits raises the
  useful particle yield from ~1.5% to ~100%, which is what buys the error bars —
  but it is no longer make-or-break.
- The **rate sweep can be far denser and the integrations far longer**. Li et al.
  ran 1 Myr routinely and 1 Gyr once, with 500 particles per swarm. On ALICE,
  1 Myr with 1e4 particles per swarm across a few hundred (alpha, direction,
  Delta a_J) grid points is realistic. That is a genuinely better experiment than
  anything published on this question.
- The runs are embarrassingly parallel (massless test particles, prescribed
  Jupiter), so this maps onto a job array with near-perfect scaling.

Still worth doing regardless of ALICE:

- Store `(a, e, i, sigma)` per particle per snapshot instead of full 6D
  Cartesian state — this is what permits a snapshot every ~5 yr instead of every
  250 yr, and **libration cannot be resolved without it**. `T_lib ~ 148 yr`, so
  the current 250 yr cadence is below Nyquist for the very quantity the project
  is about.
- Keep the terrestrials if that is what works; the cost is now affordable.

**To do:** confirm ALICE account/allocation details, queue limits, and whether
REBOUND is available or needs a local conda env built on the cluster.

---

## 6. Proposed phases

Durations assume a ~7-month project; compress or extend once the EC / duration
is known.

**Phase 0 — Instrument the co-orbital dynamics (~3 weeks).**
Compute `sigma = lambda - lambda_J`. Classify each particle over a running
~3-libration-period window into tadpole-L4 / tadpole-L5 / horseshoe /
quasi-satellite / non-co-orbital. Measure libration amplitude `A` and libration
centre. Emit a **transition event log** (particle, time, from-state, to-state).
Bootstrap error bars on every ratio.
*Deliverable: an honest error bar on the existing four runs — does the
non-monotonicity survive?*

**Phase 1 — Validate (~4 weeks).**
Reproduce Li et al. 2023 Table 3 (R45 vs `Delta a_J` and `tau`) with the new
pipeline. Non-negotiable: nothing downstream is believable without it. Also
reproduce the `alpha = 0.7265` island-merger limit as a code test.

**Phase 2 — The asymmetry map (~6 weeks).**
Sweep `alpha` over four decades (1e-4 to 0.7, i.e. `adot_J ~ 5e-7` to
`3.8e-3 AU/yr`), **both directions**, with `Delta a_J` as a second axis to test
whether R45 is a function of `alpha` alone.
*Deliverable: R45(alpha, direction) with error bars — the first such curve in
the literature, and the place where the non-monotonicity is or is not.*

**Phase 3 — Decompose rate vs amount (~6 weeks).**
From the event log: capture rate and escape rate per swarm vs time; L4<->L5
exchange flux through L3; residence-time distributions giving permanent vs
transient fractions. From the amplitudes: `A_4(t)` and `A_5(t)` distributions,
and the libration action `J ~ (1/2pi) * closed-integral(delta a d sigma)` — test
where adiabatic invariance breaks.
*Deliverable: the asymmetry attributed to capture vs survival. This is the
actual scientific claim.*

**Phase 4 — Confront observations (~4 weeks).**
Pull MPC Trojan elements, compute their real `A` distribution by short
integration, and apply the **joint** constraint: match N4/N5 ~ 1.6 *and* the
amplitude distribution simultaneously. This is Li's stated open caveat and the
highest-value single result available. Optionally add a Grand-Tack-style
direction reversal (inward then outward, ~4e-5 AU/yr per leg) — the one scenario
in which both channels fire in sequence.

**Phase 5 — Write-up (~4 weeks).**

**Stretch, only if time allows:** exoplanet co-orbitals in resonant chains —
TOI-178's 30-52% enhanced co-orbital widths (Boskovic et al. 2025), reusing
`generate_resonant_chain.py`. Presumably the "Plus" in the repo name.
Recommendation: keep this **out** of the registered scope, mention as outlook.

---

## 7. Three framings to choose between

These lead to genuinely different projects.

**(A) Mechanism decomposition** — *"Is the asymmetry made by capture or by
survival?"* Strongest science, cleanest narrative, most likely to become a
paper. Risk: if the two channels separate trivially, there is less to say.

**(B) Regime map** — *"Where is the adiabatic / non-adiabatic transition, and how
fast must Jupiter have moved?"* Safest: guaranteed a result even if it is a null
one (a constraint on `adot_J`). Least novel.

**(C) The amplitude problem** — *"Can any migration history match both
N4/N5 = 1.6 and the observed libration amplitudes?"* Highest risk, highest
reward; attacks the stated open problem in Li et al. 2023 head-on. Could
plausibly answer "no", which is itself a real result.

**Recommendation: (A) as the spine, with (C) as the closing chapter. (B) falls
out of Phase 2 for free.**

---

## 8. Open questions before the title and description can be written

1. **Framing** — A, B, C, or a blend?
2. **Supervision** — supervisor and daily supervisor. Is this Simon Portegies
   Zwart, given the AMUSE lineage? Does he want AMUSE kept in the loop, or is
   the REBOUND port fine? Who is the second reader/examiner?
3. **Duration and EC** — 6 months / 42 EC, or the longer variant? The phases
   above total ~7 months.
4. **The form itself** — does it ask for anything beyond title + description
   (research question, methods, deliverables, timetable, literature list)? What
   is the word limit? Deadline?
5. **Exoplanet extension** — actively wanted in scope by you or the supervisor,
   or a leftover to be dropped?
6. **"Dropping the terrestrials doesn't work"** — what exactly happened? (See
   Section 4.)
7. **ALICE** — allocation size, queue limits, whether REBOUND is already
   available there.

---

## 9. Changelog

- **2026-08-22** — initial draft. Literature read in full; regime mismatch and
  detector limitations diagnosed; two-mechanism hypothesis formulated. Noted:
  dropping terrestrial planets reportedly does not work (undiagnosed); ALICE
  access secured, which lifts the compute constraint.
- **2026-08-22 (later)** — literature refreshed against the web (Section 1.5):
  observational target updated to N4/N5 = 1.45 +/- 0.05 (Vokrouhlicky+24), which
  also adds an L4/L5 *inclination*-distribution asymmetry as a third observable.
  Proposal title + two-paragraph scope drafted in `MSc_PROPOSAL.md`.
