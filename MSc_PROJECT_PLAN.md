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

- **Gas-disc outward migration — rates:** the outward leg is the
  Masset & Snellgrove (2001) resonant-pair mechanism (two gap-opening planets in
  resonance, overlapping gaps, net positive torque), with rates from
  Morbidelli & Crida (2007) and the low-viscosity failure modes of
  Griveaud et al. (2023). Order `adot ~ 4e-5 AU/yr` over ~1e5 yr. Run in
  sequence after an inward leg, this is the natural setting for a direction
  **reversal** — both mechanisms firing one after the other. (The Grand Tack is
  one realisation of such a history; it additionally posits a deep excursion to
  ~1.5 AU, which this project does not use and does not need.)
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
| This project, 50 kyr | 8e-6 | 1.5e-3 | 4400 |
| This project, 500 kyr | 8e-7 | 1.5e-4 | 44 000 |
| Li+23 control | 1.5e-4 | 2.9e-2 | 230 |
| Gas-disc outward leg | 4e-5 | 7.6e-3 | 880 |
| Critical (island merger) | 3.8e-3 | 0.7265 | 9 |

(Jupiter's tadpole libration period `T_lib ~ 148 yr`; `tau_mig = a_J / adot_J`.)

The current runs are **20-200x slower than the slowest case known to produce an
asymmetry**, and ~1e3 below critical. This is the deeply adiabatic regime, where
the libration action is an adiabatic invariant and theory predicts **no**
asymmetry.

**`alpha` conversion, settled 2026-08-31** (was flagged inconsistent in Section
11.4). Sicardy & Dubois (2003) define `alpha = eps_s / mu` with
`eps_s = (1/n_s)(Jsdot/Js)` and `Js = sqrt(G M a_s)`, so `Jsdot/Js = adot/(2a)`
and therefore

    alpha = adot_J / (2 a_J n_J mu),   i.e.   adot_J = 2 a_J n_J mu * alpha

With `a_J = 5.2028 AU`, `n_J = 0.5297 rad/yr`, `mu = 9.5388e-4`:
`2 a_J n_J mu = 5.2575e-3 AU/yr`, hence **`alpha = adot_J x 190.2`** and
`alpha = 0.7265` gives `adot_crit = 3.82e-3 AU/yr` — reproducing Li+23's quoted
`3.8e-3` exactly. The four non-critical rows previously used `x 262`, which is
`190.2 / 0.7265`: they had been normalised by `adot_crit` (so that critical read
`alpha = 1`) rather than by `2 a_J n_J mu`. All were 1.376x too large; corrected
above. The Section 6 Phase 2 sweep (`alpha = 1e-4` to `0.7` <-> `5e-7` to
`3.8e-3 AU/yr`) already used the correct factor and is unchanged.

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
highest-value single result available. Optionally add a **direction reversal**
(inward then outward, ~4e-5 AU/yr per leg) — the one scenario in which both
channels fire in sequence.

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
- **2026-08-23** — supervisor input recorded (Section 10): binaries in L4/L5,
  static-vs-dynamic membership. Sweet-spot provenance traced (Section 11).
- **2026-08-24** — literature assembled: 31 PDFs in `literature/`, tiered
  reading order in `literature/READING_LIST.md`.
- **2026-08-31** — `alpha <-> adot_J` conversion settled against Sicardy &
  Dubois (2003); Section 3.1 `alpha` column corrected (was 1.376x too large in
  four rows). Grand Tack removed from the framing per supervisor preference:
  Sections 1.4, 3.1 and 6 now cite Masset & Snellgrove (2001) /
  Morbidelli & Crida (2007) for the outward leg and call the inward-then-outward
  run a direction reversal. Section 11.2 slope duplication diagnosed: the
  archived notebook's `DIRECTION[:2]` label collapsed `inward` and `insitu`, so
  `slopes.md`'s labels are unreliable and it is retired as the port's
  verification target.
- **2026-08-31 (later)** — novelty claim audited against the PDFs (Section 12).
  The blanket "nobody has measured capture and loss individually" is **false**:
  Di Sisto et al. (2014) fit per-swarm fractional loss rates, Li et al. (2023a)
  report per-swarm survival fractions, Pirani et al. (2019a) plot per-side
  population time series. The surviving claims are narrower and checked. Two
  consequences: Pirani's channel requires a **growing** Jupiter and so cannot
  appear in the current fixed-mass setup, and "Type II" should leave the title
  because both published thresholds sit above the Type II ceiling for a
  full-mass Jupiter. `MSc_PROPOSAL.md` brief description rewritten accordingly.

---

## 10. Supervisor input (2026-08-23) — binaries and static-vs-dynamic

Two suggestions from Simon Portegies Zwart (supervisor), recorded with an
assessment. Project length confirmed as **1 year**.

### 10.1 "Are the swarms static or dynamic?" — already core, make it explicit

This is the same instrument as the capture/loss measurement, and it adds a third
term to the rate equation:

    dN4/dt = C4 - L4*N4 + T(5->4) - T(4->5)

Transfer T requires a particle to leave its tadpole island, cross to a horseshoe
orbit, pass L3 and be re-trapped. The sigma-classifier plus event log measures it
for free. Residence-time distributions give the permanent-vs-transient split.
**Verdict: include, name it in the description. Zero extra compute.**

### 10.2 Binaries — worth it, but only in the post-processed form

`analysis/Binary_detector.py` cannot work as written: it computes two-body
binding energy between **massless** test particles, using an assumed mass
`m_planetesimal = 1e-10 Msun` that plays no role in the integration (and which
corresponds to D ~ 700 km, larger than any real Trojan). The particles never feel
each other. Finding no binaries was guaranteed by construction and is **not** a
result.

Two ways to do it properly:

- **Direct integration** (each Trojan = 2 mutually bound massive bodies).
  Infeasible for a population: a Patroclus-Menoetius-like binary has an orbital
  period of ~4 days against Jupiter's 12 yr, a ratio of ~1000, and needs a
  close-encounter integrator (MERCURIUS/TRACE, not WHFast). Not a thesis.
- **Encounter recording + published survival criterion.** Record each particle's
  close approaches to Jupiter (in Hill radii) along the trajectory; apply the
  survival criterion of Nesvorny & Vokrouhlicky (2019), which depends on
  a_B/R_B (binary separation over effective radius); convolve with an assumed
  primordial separation distribution. **Feasible, ~2 months, no new runs.**

Physical hook that makes this non-trivial and ties it to the core measurement:
**horseshoe orbits turn at ~mu^(1/3) from the planet, i.e. within a few Hill
radii of Jupiter, whereas tadpoles never approach.** So time spent on horseshoe
orbits is exactly the exposure that strips wide binaries — and the two swarms
spend different amounts of time there. Prediction: a binary-fraction asymmetry
between L4 and L5 that tracks the loss channel.

Risks, stated honestly: (i) the pre-capture scattering phase from 20-30 AU is
common to both swarms and may dominate, washing out any swarm-dependent signal
(a null result, still worth reporting before Lucy reaches L5 in 2033);
(ii) the primordial binary separation distribution is an assumed input, so the
result is conditional; (iii) observed L4/L5 binary fractions are currently too
poorly known to decide the test — LSST is the observational path, Lucy gives
ground truth on individual systems only.

### 10.3 Scope verdict for 12 months

Core (C and L across speed x direction, with validation) is ~6 months once the
classifier exists. 10.1 costs ~1 month of analysis, 10.2 ~2 months, write-up ~2.
Total ~11 months with slack. Both additions reuse the **same trajectories** and
require no new simulations — that is the test an add-on must pass. The exoplanet
resonant-chain extension fails that test and stays out of scope.

---

## 11. The "sweet spot" — the observation the project actually grew from

Recalled 2026-08-23: in the original grid over migration timescale, the L4/L5
behaviour peaked at some intermediate rate and then fell again, suggesting a
handover between mechanisms. This is the origin of the project and of Simon's
interest. Found the contemporaneous note — `SIMOA_report/sn-article.tex`, lines
158-159, in the Results section:

    %L4/L5 changing ratio graph and why: non-adiabatic vs adiabatic?
    %Essentially just the loss rate of L5 is hitting sweetspot at 100kyr?

So the remembered feature is: **a maximum at tau = 100 kyr, attributed to the L5
loss rate, framed as adiabatic vs non-adiabatic.** Note that this is already a
statement about **L**, not about the population ratio — the instinct was correct
from the start.

### 11.1 Why a maximum is expected on theoretical grounds

Independent of the old data, a peak is the generic prediction:

- **Too slow (adiabatic):** the libration action is an adiabatic invariant, the
  islands deform slowly, particles follow, nothing is lost. Asymmetry -> 0.
- **Too fast:** above the critical rate the L4 island merges with L3 and ceases to
  exist (`adot_crit = 3.8e-3 AU/yr`); an impulsive change also hits both swarms
  nearly equally. Asymmetry -> 0.
- **In between:** where the migration timescale approaches the libration
  timescale, adiabatic invariance breaks *asymmetrically* — leading and trailing
  islands respond differently. Maximum asymmetry.

The peak should therefore lie between the onset of measurable asymmetry (Li+23
already get R45 = 1.50 at `adot = 1.5e-4 AU/yr`) and island destruction at
`3.8e-3 AU/yr`. **Predicted window: `adot ~ 2e-4` to `3e-3 AU/yr`.**

The old sweet spot at tau = 100 kyr with `Delta a = 0.4 AU` is
`adot = 4e-6 AU/yr` — **50 to 1000 times slower than that window.** So the
remembered peak is very unlikely to be the physical one.

### 11.2 Two reasons to treat the old peak as motivation, not evidence

1. **The in-situ control peaks in the same place.** From `analysis/slopes.md`,
   inward runs give 1.83e-7 / 1.24e-6 / 1.48e-7 at 50 / 100 / 500 kyr — a peak at
   100 kyr. But the **no-migration** control gives 2.28e-7 / 5.47e-7 / 7.15e-8 —
   also a peak at 100 kyr. A control with no migration cannot show a migration
   sweet spot. Whatever produced the bump is a function of run length, not rate.
2. **Four of the nine slopes are two duplicated pairs.** `2.481e-07` appears for
   both 50 kyr and 100 kyr outward; `1.238e-06` appears for both 100 kyr inward
   and 500 kyr outward — identical to four significant figures across different
   runs. That is not a coincidence.

   **Diagnosed 2026-08-31.** The slope arithmetic is fine —
   `simulation/analysis.py::compute_l4_l5_ratio_slope` is a plain
   `np.polyfit(t, N4/N5, 1)`. The fault is in the **labels**. The archived
   notebook that produced these numbers
   (`analysis/_archive/03_migration.ipynb`, cell 11) printed its slopes.md line
   as

       slope_label = f"{DURATION_KYR}kyr_{DIRECTION[:2]}"

   and `DIRECTION[:2]` maps **`inward` and `insitu` to the same string `in`**
   (and `outward` to `ou`, which does not match slopes.md's `_out` either). So
   every console line for a given duration read `50kyr_in` whether the run was
   inward or in-situ, and the `_in` / `_out` / `_insitu` labels in `slopes.md`
   were retyped by hand from output that could not distinguish them. Duplicated
   values across rows are what that looks like.

   **Consequence:** the label-to-value mapping in `slopes.md` is unverifiable,
   and the runs behind it are AMUSE-era with the Section 3.2 detector, so it
   cannot be recovered. Point 1 above — "the in-situ control peaks in the same
   place as the inward runs" — may simply be the same value written into two
   rows. **`slopes.md` is retired as a verification target for the REBOUND
   port.** (`analysis/results/batch_summary.txt` already disagrees with it in
   sign on three of four scenarios; that is now expected, not alarming.) The
   port's correctness rests on the physics validation recorded in the migration
   driver work — Jupiter `a(t)` against the analytic exponential, and the
   REBOUNDx-operator cross-check — not on reproducing these slopes.

### 11.3 The key logical point: a peak does not prove a handover

**A single mechanism with a resonant response produces a maximum. Two competing
mechanisms handing over also produce a maximum. The R45(rate) curve alone cannot
distinguish them.** This is precisely why C and L must be measured separately: if
the peak sits in the capture curve, it is one mechanism; if C and L peak at
different rates and cross, it is a handover. The old observation is therefore the
motivation for the method, and the method is what makes the observation
interpretable.

### 11.4 Consequence for the grid

The predicted window (`2e-4` to `3e-3 AU/yr`) sits **at or above the fast end of
Type II** (`~1e-4 AU/yr`). If that holds, gas-disc migration can only ever place
Jupiter on the slow, rising side of the curve, and the peak is reachable only by
an instability-driven jump. That is a real result in itself, and the grid must
span both regimes for it to be decidable.

---

## 12. Literature audit of the novelty claim (2026-08-31) — the claim was too strong

Prompted by the question "Li 2023a/b and Pirani 2019 *do* track influx, retention
and efflux — didn't you say they didn't?". Checked directly against the PDFs in
`literature/`. **The questioner is right and the earlier blanket claim — "no study
has measured the capture rate and loss rate individually" — is false as written.**
What follows is the corrected map, and the narrower claim that survives it.

### 12.1 What has actually been measured

| Quantity | Measured by | In what regime | Form |
|---|---|---|---|
| **Fractional loss rate `L`, per swarm, time-resolved** | **Di Sisto et al. 2014** | present-day, **no migration**, 4.5 Gyr | `d(N_surv/N_0)/dt` fitted for `t > 2 Gyr` |
| **Survival fraction, per swarm** | Li et al. 2023a | fast outward jump, 1 Myr - 1 Gyr | end-to-end fraction, not `L(t)` |
| **Population time series per side** | Pirani et al. 2019a | inward gas-disc migration + growth | `N_lead(t)`, `N_trail(t)`, **net** |
| **Capture efficiency vs migration rate** | Pirani et al. 2019a | inward, 9 growth-track speeds | `N_tot` and `R45` at `t = 5 Myr` |
| **Gross in- and out-flux separated, same run** | *nobody* | — | — |
| **Inter-swarm / horseshoe transfer counts** | *nobody* | — | — |
| **`C` and `L` on one common `alpha` axis** | *nobody* | — | — |

**Di Sisto et al. 2014 is the paper that kills the old wording.** They give, for
`t > 2 Gyr`:

- real Trojans: `s_L4 = -3.9929e-11 +/- 2.45e-13 yr^-1`,
  `s_L5 = -5.1166e-11 +/- 4.14e-13 yr^-1` — L5 depletes **28% faster**;
- synthetic (uniformly seeded) Trojans: `s_fL4 = -1.10675e-11`,
  `s_fL5 = -1.07656e-11` — **statistically identical**.

That pair of results is the important one, and it is a *constraint on this
project*, not just a citation. It says the two tadpole regions have **identical
phase-space dynamics**; the differential loss seen in the real swarms comes
entirely from the **distribution** of the objects inside them (Di Sisto points at
the L4 inclination excess). They conclude Gyr evolution supplies only `~10%` of
the total asymmetry (`~25%` of the present unbiased value), implying a primordial
ratio `N0(L4)/N0(L5) = 1.3 +/- 0.2`.

Consequence: **`L` alone, measured on a population, is not a mechanism.** Two runs
with identical dynamics and different amplitude/inclination distributions give
different `L`. Any loss rate this project reports must be reported *conditioned on
the libration-amplitude distribution*, or it repeats Di Sisto's own warning.

### 12.2 Li et al. 2023a — what they do and do not have

They do report a per-swarm survival fraction, and they call it a "surviving rate":
255/312 L4 (81.7%) vs 151/209 L5 (72.2%) from 1 Myr to 1 Gyr, "the surviving rate
of the L4 swarm is about 1.13 times greater". So a loss measurement exists.

But:

- **`C = 0` by construction.** 500 test Trojans are seeded directly at
  `sigma_0 = +/-(60 + Delta sigma_0)`; nothing is ever added. The setup cannot
  produce a capture rate, so `C` and `L` are not comparable within it.
- It is an **integrated fraction over a fixed window**, not `L(t)`. The 1 Myr and
  1 Gyr endpoints are the only time resolution.
- Section 4 point (2) explicitly **assumes** capture is symmetric across jumps
  ("assuming that the same quantity of new Jupiter Trojans would be captured
  around each Lagrangian point"). That assumption is untested and is precisely
  what a joint `C`/`L` measurement would test.
- Transitions to horseshoe are shown for **example objects** (their Fig. 3;
  2023b Fig. 3-4) but never counted as a flux.

### 12.3 Pirani et al. 2019a — the mechanism is not "capture" in the simple sense

Their Fig. 11 (inward) and Fig. 12 (outward) plot the number of particles on the
**leading and trailing side of horseshoe orbits versus time** — genuinely
time-resolved and per-side. But the caption states the meaning plainly: *"an
excess of leading particles means that particles spend more time in the leading
part of the orbit."* This is a **residence-time asymmetry within horseshoe
orbits**, not a capture-rate asymmetry. It is frozen into membership only by a
**second, separate step** — Jupiter's mass growth pinching horseshoes into
tadpoles. Without that step it is reversible: *"when the migration stops the
excess simply disappears because the horseshoe orbits are back symmetric."*

Three consequences, all of them design-level:

1. **A fixed-mass migrating Jupiter cannot reproduce Pirani's mechanism at all.**
   The current REBOUND setup has fixed Jupiter mass, so the horseshoe excess it
   builds during migration must vanish when migration stops. Any asymmetry the
   present code reports at the end of a run is therefore *not* Pirani's channel.
   Either Jupiter's mass must grow, or the paper's channel is out of scope and the
   proposal must say so.
2. The observable that carries Pirani's signal is the **horseshoe residence-time
   split**, which is measurable continuously from `sigma(t)` and has never been
   reported as a rate.
3. Their control is decisive and reusable: repeating without gas drag gives the
   same asymmetry, so the driver is the **relative drift**, not the drag.

### 12.4 Pirani's Fig. 13 already brackets the "sweet spot" — and confirms `x190.2`

Section 11 predicted the asymmetry window from Sicardy & Dubois plus Li. Pirani
Fig. 13 ("Constraints on the migration rate for asymmetric Trojans") already
draws both bounds, in a different normalisation:

- upper: `v_crit ~ 1.45 q Omega r` (Ogilvie & Lubow 2006), L3/L4 merge;
- lower: `v_lim ~ 0.17 q Omega r`, empirical, below which no asymmetry appears.

For Jupiter at 5.2 AU, `q Omega r = 2.628e-3 AU/yr`, so `v_crit = 3.81e-3 AU/yr`
— **identical to Sicardy & Dubois' `alpha = 0.7265` -> `3.82e-3`**. The two
normalisations differ by exactly a factor 2 (`1.45 / 0.7265 = 1.996`), which is an
independent confirmation of the `alpha = adot_J x 190.2` conversion settled in
Section 3.1. Section 11's predicted window `2e-4 - 3e-3 AU/yr` is essentially
Pirani's `[v_lim, v_crit] = [4.5e-4, 3.8e-3] AU/yr`. **Credit it to Pirani; do not
present it as new.**

The scaling `v ~ q Omega r` also sharpens Section 11's punchline. The threshold is
a condition on `adot_J / (q Omega r)`, not on `adot_J`. Pirani clears it with a
*slow* absolute drift because her `q` is a growing **core** (`1 M_E` at 18 AU:
`v_lim ~ 1 AU/Myr`, which she states). Applied to a **full-mass Jupiter at
5.2 AU** the same threshold demands `adot_J >~ 4.5e-4 AU/yr` — 4.5x above the fast
end of Type II (`~1e-4`). Li's own control sits at `1.5e-4 AU/yr`, also above the
Type II ceiling, and works because his channel acts on the marginally stable
large-amplitude tadpoles rather than on horseshoes.

**So the two mechanisms have different thresholds in `alpha` and act on different
sub-populations.** That is a sharper, testable version of the handover hypothesis
than Section 2 states, and it is directly measurable with the amplitude-resolved
diagnostic already planned.

### 12.5 Direct consequence for the proposal wording and the title

- **Drop** "no study has measured the capture rate and loss rate individually".
  Di Sisto measured `L`; Li measured a survival fraction; Pirani measured net
  population per side. Anyone on the signing committee who knows the field will
  catch it.
- **Keep**, because it is true and checked: capture and loss have never been
  measured *in the same integration*, gross fluxes have never been separated from
  net, inter-swarm transfer has never been counted, and no `R45(alpha, direction)`
  curve exists.
- **"during Type II Migration" in the title is a liability.** Both published
  thresholds sit at or above the Type II ceiling for a full-mass Jupiter. A thesis
  promising Type II may be promising a regime where the answer is "nothing
  happens". Frame the axis as dimensionless `alpha` spanning Type II *through* the
  jump regime instead; "Type II is too slow, by this factor, for these reasons" is
  then a result rather than a failure.

### 12.6 What this does to the scope

It does not shrink it. It replaces one over-broad claim with four narrow ones,
adds a genuine constraint from Di Sisto (identical dynamics -> the asymmetry lives
in the distributions), and hands the project a concrete, previously unnoticed
prediction: **the two channels switch on at different `alpha`.** It also surfaces
a real defect in the current code — fixed Jupiter mass cannot host Pirani's
channel — which is better found now than in month 8.
