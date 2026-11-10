# Stage 9: full reproduction of Pirani et al. 2019a

Simon's ask (2026-11-10): reproduce Pirani+2019a (A&A 623, A169) as she ran it, then extend.
Plan: `~/.claude/plans/vivid-drifting-hummingbird.md`. Compute: haagsebeek (STRW, ~24 cores).

## 1. Her set-up (from the paper, incl. the appendices)

| item | value |
|---|---|
| code | MERCURY hybrid, dt = 50 d |
| disc | 4–26 au, 44 rings × 1000 = 44,000 particles; e ≤ 0.01, i ≤ 0.01°; 10 re-randomised repeats per scenario |
| clock | 0 → 5 Myr; J, S, U, N start at 2.31 / 2.56 / ~2.70 / ~2.70 Myr; gas gone at 3 Myr |
| gas | Ida+2016 disc, α = 1e-3, Ṁ(t) from Hartmann+1998 |
| planets | a(t) imposed by the tracks; e, i free, damped below 30 M⊕ (Cresswell & Nelson 2008) |
| particles | Adachi+1976 gas drag, r = 50 km, ρ = 1 g/cm³ |
| Trojans | q and Q within ±2.5 R_H of a_J, at 5 Myr |

Code for this (committed c0444de): `trojans/gasdisc.py`, `trojans/pirani_forces.py`, MERCURIUS and impact removal in `trojans/integrate.py`, and the ring disc in `trojans/disc.py`. Tests are in `tests/test_pirani.py`.

## 2. Her growth tracks, reproduced and verified (2026-11-10)

### 2.1 Her curves, exactly
**Extraction.** Figs. 1, 3 and 13 are vector graphics in the arXiv PDF. `scripts/s9_digitize_pirani.py` reads each curve's coordinates straight from the PDF (PyMuPDF, installed in trojan-rb), into `configs/tracks/pirani19a/fig*.csv`. This is not pixel digitising. Each page-to-data map is anchored on the tick labels.
- **Fig. 1:** the a(m) paths of J, S, U and N, with solid (core) vs dashed (gas) phases.
- **Fig. 3:** Jupiter's path at 8 rates.
- **Fig. 13:** Jupiter's |ȧ|(m) at nominal, ×0.25, "×0.15" and ×2.

**The rate scan is one track, stretched.**
- **Paths:** every Fig. 3 curve satisfies a_k(m) − a_end,k = k·(a_nom(m) − a_end,nom). The fitted k is constant to 1–2% over the whole curve.
- **Speeds:** every Fig. 13 curve is exactly k × the nominal rate (ratios 0.126, 0.251 and 1.997, constant in m).
- **Her "×0.15" is k = 0.125.** Fig. 8's Δa = 1.7 au = 0.125 × 13.45 au confirms it.
- **Starting radii:** the starts read from the figures match Table 1.

### 2.2 The model (`trojans/pirani_tracks.py`): BLJ15's recipe on the Ida+2016 disc

| piece | prescription |
|---|---|
| pebbles | Ida+2016 eqs. 59–61; flux from eq. 54 × FLUX_FACTOR; Stokes number from eq. 47 |
| isolation | BLJ15 eq. 16, 20 (h/0.05)³ M⊕, × ISO_FACTOR; 10% envelope before it |
| contraction | Piso & Youdin 2014 (BLJ15 eq. 17) while M_env < M_core |
| runaway | Machida+2010 (BLJ15 eqs. 18–19), capped at 0.8 Ṁ* |
| migration | PBK11 torque (Bell & Lin ice opacity), Crida & Morbidelli f(P), BLJ15 type II |

**Three calibrations.** Pirani gives none of these numbers.

| | value | fitted to |
|---|---|---|
| FLUX_FACTOR | 8 × Ida eq. 54 | Jupiter's clock (her Figs. 1 + 13) and its 306 M⊕ at 3 Myr |
| ISO_FACTOR | 1.066 | where her dashed lines start, for J and S together |
| Ice-giant starts (her "~2.70 Myr") | U 2.721 Myr, N 2.739 Myr | their final masses |

The BLJ15 corrigendum (A&A 609, C2) says only that their pebble flux was wrong, so the flux has to be calibrated.

### 2.3 Verification (`scripts/s9_pirani_tracks.py` → `reports/pirani_tracks/`)

| check | model | Pirani |
|---|---|---|
| J mass at 3 Myr | 305 M⊕ | 306 |
| S mass at 3 Myr (prediction, not fitted) | 116 M⊕ | 113 |
| J / S gas phase starts | 20.1 / 22.2 M⊕ | 20.6 / 22.7 |
| J clock vs her Figs. 1 + 13, relative to t(5 M⊕), at 2 / 10 / 20 / 30 / 100 M⊕ | +3 / +1 / +9 / +11 / +32 kyr (out of 690) | — |
| type-I speed along her path, 0.2–8 M⊕ | 0.42–0.48 × Tanaka+2002, constant | — |
| Forward model (migration integrated, PBK11 × 0.6): final masses | J 304, S 113, U 17.7, N 14.1 M⊕ | 306, 113, 18.1, 14.4 |
| Forward model: final a | J 3.8, S 6.0, U 17.4, N 20.0 au | 5.1, 8.0, 16.0, 18.6 |

**Reading.**
- **Growth is verified.** One flux constant reproduces Jupiter's clock and all four final masses. Saturn's mass is a true prediction.
- **Migration physics is only approximate.** PBK11 on the Ida disc is about 1.2–1.7× her speed, and the partial gap opens later than hers. The forward model's paths miss by 1–2 au.
- **So the N-body tracks use her exact paths with the model's clock.** The physics supplies only the clock, which her figures don't show.

### 2.4 The N-body tracks (`configs/tracks/pirani19a/track_*.csv`, columns t_yr, a_au, m_earth)

**How they're built.** Each track starts at t = 0 with the seed parked at its starting a, then follows the path with the model's clock until 3 Myr.
- **Paths:** her Fig. 1 curves, smoothed (log-m bins, monotone spline; within 0.21 au of her curve, worst at the isolation corners).
- **Jupiter below 3 M⊕:** the path is built from her Fig. 13 rate along the clock, because her plotted path is too flat there to differentiate. That puts Jupiter's seed at 18.26 au instead of 18.40.

**The tracks.**
- `nominal_{jupiter,saturn,uranus,neptune}`.
- `rate_x{0.15…2.00}_{jupiter,saturn}`:
  - Jupiter uses the scaling rule, with the Fig. 3 end points;
  - Saturn uses the same rule on its nominal path, with the end chosen so it starts at its Table 1 value. The paper gives no Saturn rate tracks.
- `insitu_{jupiter,saturn}`: at 5.4 and 8.6 au, with the same clocks.
- `outward_jupiter`: the nominal path mirrored (5.08 → 18.26 au). Which planets her outward run had is not stated.

**Check against Fig. 13** (panel d of `verification.png`). The tracks' |ȧ|(m) follows her Fig. 13 at all four rates from 0.2 to 100 M⊕. A ±30% wobble remains at 20–40 M⊕, from the isolation corner and the gas-phase switches in the clock. Her own clock shows kinks at the same masses.

**Open (ask Pirani):** her actual track files; the disc extent for the fast rates (×1.25–2 start beyond 26 au); which planets her outward run had.
