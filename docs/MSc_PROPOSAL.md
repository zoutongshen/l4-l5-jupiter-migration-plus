# MSc Research Project — Proposal

**Draft 2026-08-22.** Title and two-paragraph scope, for signature by supervisor
and examiner. Background reasoning: `docs/MSc_PROJECT_PLAN.md`.

---

## Working title

**Capture, Loss and Exchange: A Rate-Resolved Origin for the L4/L5 Jupiter
Trojan Asymmetry**

*Alternatives:*
- Capture or Loss? Separating the Two Channels Behind the L4/L5 Trojan Asymmetry
- The Migration Rate That Builds the Trojan Asymmetry

*Dropped 2026-08-31:* "...during Type II Migration". Both published thresholds
for producing the asymmetry (Pirani's `v_lim ~ 0.17 q Omega r`, Li's control at
`1.5e-4 AU/yr`) sit at or above the Type II ceiling for a full-mass Jupiter, so a
title promising Type II promises a regime where the answer may be "nothing
happens". See `docs/MSc_PROJECT_PLAN.md` Section 12.4-12.5.

---

## Description (brief — this is the version being submitted)

Jupiter's L4 Trojan swarm contains about 45% more asteroids than its L5
counterpart, a signature of Jupiter's migration. Two mechanisms are proposed,
requiring opposite directions of drift: an excess captured on the leading side
during inward migration through the gas disc (Pirani et al. 2019), or an excess
lost from the trailing side during a fast outward jump (Li et al. 2023). Each has
been demonstrated only in its own regime, never against the other. This project
will use N-body integrations to measure capture, loss and exchange between the
swarms within single runs, and to locate the migration rate at which the
asymmetry is strongest.

---

## Long version (kept for reference / for the thesis introduction)

### Scope, expanded

Jupiter's leading (L4) Trojan swarm contains roughly 45 per cent more asteroids
than its trailing (L5) swarm — a bias-corrected ratio of 1.45 +/- 0.05
(Vokrouhlicky et al. 2024) that four billion years of collisional and dynamical
erosion cannot produce, and which must therefore date from the epoch when Jupiter
was still migrating. Two explanations exist, and they attribute the imbalance to
different physical rates. Writing the evolution of each swarm as
dN/dt = C(t) - L(t) N + T(t), with C the capture rate, L the fractional loss rate
and T the net exchange with the other swarm, the inward-migration model of Pirani
et al. (2019) is a claim about C and about residence time on horseshoe orbits,
while the fast outward-jump model of Li et al. (2023) is a claim that L4 < L5.
Pieces of this ledger exist. Di Sisto et al. (2014) fit per-swarm fractional loss
rates for the present-day, non-migrating system and conclude that four billion
years of erosion supply only about a tenth of the asymmetry, so the rest must be
primordial; Li et al. (2023) report per-swarm survival fractions for a fast jump,
in a setup where capture is zero by construction and symmetric capture is assumed
rather than measured; Pirani et al. (2019) plot per-side populations during
inward migration, but as net numbers and for horseshoe residence rather than
swarm membership. No study has measured capture and loss inside the same
integration, separated gross flux from net, counted exchange between the swarms,
or placed the two mechanisms on a common dimensionless migration-rate axis. This
project will do so, and determine which channel builds the observed asymmetry.

The work uses REBOUND N-body integrations of massless test particles in the
Sun-Jupiter-Saturn system. The control parameter is the dimensionless migration
rate alpha = adot_J / (2 a_J n_J mu) of Sicardy & Dubois (2003), which is the
quantity both published mechanisms are actually governed by, and it is swept from
the Type II regime set by disc viscosity (alpha ~ 1e-3) up to the island-merger
limit alpha = 0.7265 reached only by an instability-driven jump. Three scenarios
are compared: inward drift through the gas disc, with Jupiter's mass growing, as
that growth is what converts a transient horseshoe excess into permanent tadpole
membership; outward drift driven by a Jupiter-Saturn resonant pair (Masset &
Snellgrove 2001; Morbidelli & Crida 2007); and, as a post-gas reference case, a
fast outward jump of the kind invoked for the giant-planet instability. The central methodological step is to instrument the co-orbital
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
leading side, or because more were lost from the trailing side. The two have only
ever been studied in separate regimes, so nobody knows where one takes over from
the other. I will measure both in the same simulation and find the crossover."*
