# Run note — mass ladder, and the self-gravity check

Prepared 2026-09-28. Branch `mass-threshold`, commits `9c0072b`…`270d828`.
Machine-generated results: `campaign/reports/mass_ladder/`, `campaign/reports/swarm/`.
Literature derivation: `docs/MASS_THRESHOLD.md`.

**In one line.** Simon's suggested mass threshold is real and analytically known
(40.9 M_Jup), but measured on real catalogue orbits the co-orbital region **fails 8–13×
earlier** — and real Trojan masses sit nine orders of magnitude below any of it.

---

## 1. Question, and an important separation

The suggestion (2026-09-22) was that below some Trojan mass an object could stay in L4/L5
forever, and above it everyone eventually leaves. **Both halves are true but they are
unrelated results, and only the first is about mass.**

**(a) A mass threshold exists.** Gascheau (1843), rederived by Routh (1875): the
equilateral configuration of three *finite* masses is linearly stable iff

```
27 (m₁m₂ + m₂m₃ + m₃m₁) < (m₁ + m₂ + m₃)²
```

Unlike the restricted problem — where the third body is massless by construction — this
constrains the Trojan's own mass. For our Sun and Jupiter it solves to
**m₃ < 0.03905 M☉ = 40.9 M_Jup**. Implemented and unit-tested as
`constants.routh_trojan_mass_limit()`.

**(b) Finite lifetimes are not a mass effect.** Trojans leave by chaotic diffusion driven
by the giants, which operates at *exactly zero mass* — Levison, Shoemaker & Shoemaker
(1997); Robutel & Gabern (2006); Di Sisto+ (2014), 23% (L4) / 28.3% (L5) over 4.5 Gyr.

So "no lifetime is infinite" needs no mass at all. Conflating the two would attribute a
diffusive process to a stability boundary ten orders of magnitude away.

**What was actually unmeasured:** where between those two the massless test-particle
approximation — which every long-term Trojan study uses, ours included — breaks for real
orbits under four giants.

---

## 2. Design

Same 100 catalogue Trojans, same initial conditions as the pilot (ic `2026-09-15_giants`,
seed 20260915), integrated **1 Myr at each of 14 masses** from 0 to 50 M_Jup.

**One massive Trojan per simulation.** This is the point of the design: putting 100 bodies
at 1 M_Jup in one box places 100 M_Jup in the co-orbital region, which is a different
experiment. Each run is therefore an exact **six-body problem** — Sun, four giants, one
Trojan.

Rungs (M_Jup): `0, 1e-9, 1e-6, 1e-3, 0.01, 0.1, 0.3, 1, 3, 10, 24, 30, 40.9, 50`

- `0` — massless control, bridges to the pilot
- `1e-9` — approximately (624) Hektor, the largest real Trojan
- `24` — Liberato & Winter's (2020) practical co-orbital limit, μ = 0.023313
- `40.9` — Routh's linear limit
- `50` — **above Routh; must come apart.** This is the falsifiable check that the setup
  reproduces known physics. Nothing below it is worth believing until it does.

---

## 3. N-body integration — specifics

### 3.1 Integrator: IAS15, not WHFast

| | |
|---|---|
| Code | REBOUND 4.6.0 |
| Integrator | **IAS15** — 15th-order Gauss–Radau, adaptive step, non-symplectic |
| Timestep | adaptive (IAS15 sets it; no fixed dt) |
| Sampling | every 15 yr, against a libration period of ≈ 148 yr |
| Span | 1 Myr per run |
| Achieved \|ΔE/E\| | **1.6 × 10⁻¹⁰ worst across all 1,400 integrations** |

This is the one place this project departs from WHFast, and it is not a preference.

### 3.2 Why WHFast cannot do this run

Measured directly, one massive Trojan, 20 kyr, relative energy error:

| integrator | 10 M_Jup | 24 M_Jup | 40.9 M_Jup | 50 M_Jup |
|---|---|---|---|---|
| WHFast (Jacobi) | 2.9e−6 | **9.2e−2** | 2.1e−4 | **1.8e−1** |
| WHFast (democratic heliocentric) | 6.0e−2 | 1.2e−1 | 3.9e−1 | 1.5e−1 |
| MERCURIUS | 1.5e−7 | 6.2e−5 | 1.3e−6 | 9.3e−5 |
| **IAS15** | **8.9e−15** | **5.1e−15** | **5.5e−15** | **4.8e−15** |

WHFast carries the Trojan as a semi-active particle in Jacobi coordinates — an
approximation that fails once its mass is planetary. Democratic-heliocentric coordinates
are *worse*, not better.

**Worse than the energy error:** at 10 M_Jup WHFast and IAS15 **disagree about whether the
object survives at all**. WHFast would have placed the knee at the wrong mass.

IAS15 costs the same as WHFast at six bodies (≈ 22 s per object per Myr, since cost is
dominated by the sampling callback rather than force evaluation), so there was no reason
to accept the approximation. It also **agrees with WHFast 8/8 on fate at zero mass**, so
the ladder's massless rung bridges consistently to the pilot.

**The 4.5 Gyr campaign keeps WHFast** — correctly. There the Trojans really are massless
and none of this applies.

### 3.3 How mass is carried

```
sim.N_active          = 5      # Sun + four giants
sim.testparticle_type = 1      # the Trojan acts on the planets
```

With **exactly one** test particle, `testparticle_type = 1` is not an approximation: "test
particles do not act on each other" is vacuous when there is only one, so this is the
exact six-body problem. A guard in `initial.build_simulation` raises if more than one
massive Trojan is requested without explicit self-gravity, so the design cannot be
violated silently.

Massive bodies, units, G, frame and epoch are **identical to the pilot** — see
`RUN_present_day_pilot.md` §3.2–3.3.

### 3.4 What is measured

Not the pilot's Hill-radius escape: above the threshold a massive Trojan *rearranges* the
co-orbital region rather than politely leaving it. The observable is **whether σ keeps
librating**, classified exactly as the campaign does (`integrate.classify_windows`), so a
Trojan that becomes a horseshoe is not counted as still being in L4/L5.

Removal still applies at 1 R_Hill, r > 100 au, or non-finite state.

---

## 4. Result: survival against carried mass

| m (M_Jup) | μ | librating | f_L4 | f_L5 | z | circulating | removed |
|---|---|---|---|---|---|---|---|
| 0 | 0.00095388 | 97 | 0.952 | 1.000 | −1.35 | 3 | 0 |
| 1e−9 | 0.00095388 | 97 | 0.952 | 1.000 | −1.35 | 2 | 0 |
| 1e−6 | 0.00095388 | 97 | 0.952 | 1.000 | −1.35 | 3 | 0 |
| 1e−3 | 0.00095483 | 97 | 0.952 | 1.000 | −1.35 | 3 | 0 |
| 0.01 | 0.00096341 | 97 | 0.952 | 1.000 | −1.35 | 3 | 0 |
| 0.1 | 0.00104916 | 97 | 0.952 | 1.000 | −1.35 | 3 | 0 |
| 0.3 | 0.00123968 | 95 | 0.952 | 0.946 | +0.14 | 4 | 0 |
| 1 | 0.00190593 | 74 | 0.730 | 0.757 | −0.29 | 24 | 3 |
| **3** | 0.00380461 | **48** | 0.540 | 0.378 | +1.56 | 47 | 17 |
| 10 | 0.01039350 | 30 | 0.381 | 0.162 | +2.31 | 70 | 63 |
| **24** | 0.02331320 | **0** | 0.000 | 0.000 | — | 88 | 95 |
| 30 | 0.02874750 | 0 | 0.000 | 0.000 | — | 88 | 96 |
| 40.9 | 0.03846670 | 1 | 0.000 | 0.027 | −1.31 | 80 | 97 |
| 50 | 0.04643310 | 0 | 0.000 | 0.000 | — | 80 | 95 |

100 objects per rung (L4 63 / L5 37). 1,400 integrations, **22.3 CPU-h**.

**Control passes.** 50 M_Jup, above Routh, leaves 0/100 librating.

### Three readings

**(i) Flat across seven decades.** 0 → 0.1 M_Jup gives 97% survival at every rung, with
*the same three objects* leaving each time. Real Trojans sit nine orders of magnitude
below the knee. The campaign's massless test particles are safe with enormous margin —
now measured rather than assumed.

**(ii) The knee is at 1–3 M_Jup, not at 40.9.** Survival passes 50% between 1 and 3
M_Jup: **8× below** Liberato & Winter's limit and **13× below** Routh's.

**(iii) The asymmetry does not depend on mass.** Largest |z| is +2.31 at 10 M_Jup, but the
sub-knee rungs repeat one outcome rather than testing independently — about 6 independent
looks, so |z| > 2 arises ~24% of the time by chance, at a mass no real Trojan has.

---

## 5. Comparison with the literature

| source | configuration | μ limit | m_T (M_Jup) | this work |
|---|---|---|---|---|
| Gascheau 1843 / Routh 1875 | linear, 3 finite masses | 0.038521 | 40.9 | already 0 by 24; 48% at 3 |
| **Liberato & Winter 2020** | numerical CR3BP | **0.023313** | **24.0** | **0/100 survive at exactly this μ** |
| Laughlin & Chambers 2002 | equal-mass pair, tadpole | 0.038120 | ≈40 | consistent with Routh |
| Laughlin & Chambers 2002 | equal-mass pair, horseshoe | 0.000400 | ≈0.4 | horseshoes never exceed 2/100 |
| Ćuk, Hamilton & Holman 2012 | horseshoe, ratio ≳ 1200 | — | ≈0.8 | tadpoles persist well past it |
| **This work** | catalogue orbits, 4 giants | 0.0019–0.0038 | **1–3** | 50% crossing |

**The agreement is exact, at the endpoint.** At 24 M_Jup — μ = 0.023313, Liberato &
Winter's published value to five figures — survival is precisely zero here. Their μ_crit
marks where the co-orbital region stops holding anything, confirmed independently on real
orbits.

**The onset is nowhere near, and should not be.** Those analyses ask whether an
*equilibrium point* is linearly stable. A real Trojan is not at that point: it carries
eccentricity, inclination and a finite libration amplitude, and it is forced by Saturn,
Uranus and Neptune. Each widens the region of phase space the object must survive, so the
co-orbital domain empties from its edges long before the central equilibrium destabilises.

The published thresholds are therefore **upper bounds on where a co-orbital region can
persist, not predictions of where a real population is lost.** For exoplanet work the
distinction is material: a 5 M_Jup Trojan companion sits comfortably inside every
published limit and is gone here within 1 Myr.

---

## 6. Second experiment: does mutual gravity matter?

The ladder answers a question about *one* body. The collective question is separate.

**Setup.** The 200 largest catalogue Trojans (100 per swarm, H = 7.32–11.12) given real
masses from H (albedo 0.07, ρ = 1.0 g cm⁻³), **fully self-gravitating** (`N_active = N`),
against the identical objects massless. Total mass carried: 4.44 × 10⁻⁶ M⊕. WHFast is
correct here — these masses are ~10⁻¹² M☉, so there is no mass-ratio problem.

**The methodological trap.** Trojan orbits are chaotic well inside 1 Myr, so two runs of
the same system separate regardless of what started them apart. A raw per-object Δa
measures the Lyapunov time, not the force. The comparison is therefore made against
**calibration arms**: the identical massless system displaced at t = 0 by a known,
physically meaningless amount.

| arm | seed at t = 0 | moved | max \|Δa\| (au) | max Δa/a | class changes |
|---|---|---|---|---|---|
| **self-gravity** | 4.44e−6 M⊕ real mass | **132/200** | 1.01e−3 | 1.9e−4 | 0 |
| calibration | 1e−18 au | 0/200 | 0 | 0 | 0 |
| calibration | 1e−15 au | 9/200 | 3.72e−4 | 7.1e−5 | 0 |
| calibration | 1e−12 au | 10/200 | 3.43e−4 | 6.6e−5 | 0 |

"Moved" means the stored window-mean `a` changed by ≥ one float32 step (4.8 × 10⁻⁷ au near
5.2 au) — a floor on what the diagnostic resolves, applied identically to every arm.

The 10⁻¹⁸ au arm is below what a float64 coordinate near 5 au can represent, so it is the
control run repeated; its 0/200 confirms the integrations are deterministic and the other
rows are caused rather than numerical noise.

**Finding: self-gravity is real but dynamically negligible over 1 Myr.** It perturbs 132 of
200 objects where a 0.15 mm displacement perturbs 10 — a force doing work, not a chaotic
handful separating. The work is small: largest Δa/a = 1.9 × 10⁻⁴, no libration class
changed, all 200 survived both arms.

---

## 7. Caveats

- **1 Myr, not 4.5 Gyr.** §6 bounds a short-term effect; mutual scattering accumulates, so
  it does not license the massless approximation over the full campaign span.
- **100 objects per rung.** Enough to locate the knee. The three sub-knee losses are
  small-number statistics and run opposite to Di Sisto+ (2014)'s L5-dominant escape, so
  the sub-knee asymmetry is a sampling artefact, not a measurement.
- **Masses from one albedo and one density** — reproduces (624) Hektor to a factor of 3.
  Adequate where the question turns on orders of magnitude.
- **§6 uses the top 200 by mass**, not all 16,445; the many small bodies are unrepresented.
- **Gravity only.** No collisions, Yarkovsky, gas or migration.

## Reproduce

```
python campaign/scripts/s4_run_ladder.py --workers 8            # ~22 CPU-h
python campaign/scripts/s4_ladder_report.py --run-id mass_ladder

python campaign/scripts/s4_run_swarm.py --n-per-swarm 100
python campaign/scripts/s4_run_swarm.py --n-per-swarm 100 --massless
for j in 1e-18 1e-15 1e-12; do
  python campaign/scripts/s4_run_swarm.py --n-per-swarm 100 --massless --jitter-au $j
done
python campaign/scripts/s4_swarm_report.py --run-id swarm
```

Both runners resume from their last checkpoint with `--resume`.
