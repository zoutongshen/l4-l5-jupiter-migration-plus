# MSc Research Project — Proposal

**Draft 2026-08-22.** Title and two-paragraph scope, for signature by supervisor
and examiner. Background reasoning: `MSc_PROJECT_PLAN.md`.

---

## Working title

**Capture Rates, Loss Rates, and the Origin of the L4/L5 Jupiter Trojan
Asymmetry during Type II Migration**

*Alternatives:*
- Capture or Loss? A Rate-Resolved Study of the L4/L5 Jupiter Trojan Asymmetry
- How Jupiter's Migration Through the Gas Disk Builds the Trojan Asymmetry

---

## Description (brief — this is the version being submitted)

Jupiter's L4 Trojan swarm contains about 45% more asteroids than its L5
counterpart, a primordial signature of Jupiter's migration. Two explanations
exist, and they require opposite directions of drift: L4 has a higher capture
rate as Jupiter migrates inward (Pirani et al. 2019), or L5 has a higher loss
rate as it migrates outward (Li et al. 2023). Both report only the final
population ratio; neither rate has been measured on its own. This project will
use time-resolved N-body integrations across migration directions and speeds to
determine which rate builds the asymmetry.

---

## Long version (kept for reference / for the thesis introduction)

### Scope, expanded

Jupiter's leading (L4) Trojan swarm contains roughly 45 per cent more asteroids
than its trailing (L5) swarm — a bias-corrected ratio of 1.45 +/- 0.05
(Vokrouhlicky et al. 2024) that four billion years of collisional and dynamical
erosion cannot produce, and which must therefore date from the epoch when Jupiter
was still migrating. Two explanations exist, and they attribute the imbalance to
different physical rates. Writing the evolution of each swarm as
dN/dt = C(t) - L(t) N, with C the capture rate and L the fractional loss rate,
the inward-migration model of Pirani et al. (2019) is a claim that C4 > C5, while
the fast outward-jump model of Li et al. (2023) is a claim that L4 < L5. Every
published study to date reports only the final population ratio, which is the
time-integrated outcome of both rates together; neither C nor L has ever been
measured separately, and the balance between them as a function of Jupiter's
drift speed and direction is unknown. This project will measure the two rates
directly and determine which of them builds the observed asymmetry.

The work uses REBOUND N-body integrations of massless test particles in the
Sun-Jupiter-Saturn system, with Jupiter migrating under a Type II prescription in
which the drift rate and the total displacement are both set by the disc
viscosity, so that the migration speed is a physical parameter rather than a free
knob. Three scenarios are compared: inward Type II drift; outward drift driven by
a Jupiter-Saturn resonant pair, as in the Grand Tack; and, as a post-gas
reference case, a fast outward jump of the kind invoked for the giant-planet
instability. The central methodological step is to instrument the co-orbital
dynamics directly. Each particle is classified continuously from its resonant
angle sigma = lambda - lambda_J as librating about L4, librating about L5, on a
horseshoe orbit, or escaped, and its libration amplitude is tracked at a cadence
resolving the ~150 yr libration period. The resulting log of state transitions
yields C(t) and L(t) for each swarm separately, with bootstrap uncertainties from
10^4 particles per swarm on the ALICE cluster. After validation against a
published control run, the deliverables are: capture and loss rates as functions
of migration speed and direction; an attribution of the observed ratio of 1.45 to
capture versus loss; and a test of whether libration-amplitude damping by disc
gas, which is available during Type II migration but not after the gas has
dispersed, can reconcile the surviving population with the narrow libration
amplitudes actually observed — an open problem in the current literature.

---

## Indicative timetable (12 months) — only if the form asks

| Months | Work |
|---|---|
| 1-2 | Build the co-orbital diagnostic: resonant angle, state classifier, libration amplitude, transition-event log. Validate against a published control run. |
| 3-5 | Inward Type II sweep across disc viscosity; extract C and L per swarm. |
| 6-8 | Outward cases: Grand-Tack-style resonant pair, and the fast post-gas jump for comparison. |
| 9-10 | Confront observations: number ratio, libration amplitudes, inclinations; the gas-damping test. |
| 11-12 | Write-up. |

## One-line version, for saying out loud

*"The L4/L5 Trojan imbalance is either because more asteroids were caught on the
leading side, or because more were lost from the trailing side. Nobody has
measured those two rates separately. I will."*
