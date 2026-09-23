# Is there a Trojan mass below which L4/L5 is forever?

Written 2026-09-23, after Simon's suggestion on 2026-09-22 that there might be a
threshold in Trojan mass: below it a Trojan could stay in L4/L5 indefinitely,
above it everyone eventually leaves.

**Short answer: both halves of that are true, but they are two different results,
and only one of them is about mass.** There is a real, exact mass threshold, and
it sits about ten orders of magnitude above any real Trojan. And no Trojan's
lifetime is infinite anyway — but that is chaotic diffusion, which is already
there at exactly zero mass.

---

## 1. The mass threshold is real, and it is Gascheau's

Gascheau (1843), later rederived by Routh (1875), gives the condition for the
equilateral (L4/L5) configuration of **three finite masses** to be linearly
stable:

$$27\,(m_1 m_2 + m_2 m_3 + m_3 m_1) < (m_1 + m_2 + m_3)^2 .$$

This is the general three-body statement, so — unlike the restricted problem,
where the Trojan is massless by construction — it does constrain the Trojan's own
mass $m_3$. Writing $S = m_1 + m_2$ it rearranges to

$$m_3^2 - 25\,S\,m_3 + \left(S^2 - 27 m_1 m_2\right) > 0 ,$$

and the configuration is stable below the smaller root. For the Sun + Jupiter
masses this project actually uses (`trojans.constants`: folded Sun
$1.0000060\,M_\odot$, Jupiter system $9.5479\times10^{-4}\,M_\odot$):

$$\boxed{m_{3,\mathrm{max}} = 0.03907\,M_\odot = 40.9\,M_{\rm Jup}}$$

Implemented and tested as `constants.routh_trojan_mass_limit()`. In the massless
limit the same criterion reduces to the familiar $m_1/m_2 > 24.96$, i.e.
$\mu < \mu_G = 0.0385209$ — Routh's value.

So: **yes, there is a threshold, and above ~41 Jupiter masses no Trojan can
stay.** It is just nowhere near the asteroids.

## 2. In practice the limit is lower than linear theory says

Linear stability is necessary, not sufficient. Three results narrow it:

- **Liberato & Winter (2020)**, MNRAS 496, 3700, integrated the co-orbital region
  across $9.5\times10^{-5} < \mu < 2.3\times10^{-2}$ and found stability in the
  co-orbital region actually disappears at **$\mu = 0.023313$** — 24.0 $M_{\rm Jup}$
  for Sun + Jupiter, well below Gascheau's 0.0385. They also give polynomial fits
  for the angular and radial width of the stable region as a function of $\mu$,
  which is a ready-made analytic prediction to test a simulation against.
- **Ćuk, Hamilton & Holman (2012)**, MNRAS 426, 3051: horseshoes are far more
  fragile than tadpoles, needing a primary-to-secondary mass ratio $\gtrsim 1200$,
  because a horseshoe must pass within ~5 Hill radii of the secondary while a
  tadpole stays outside ~4.
- **Laughlin & Chambers (2002)**, AJ 124, 592, for two equal-mass co-orbital
  planets: tadpoles stable for $\mu < 0.03812$, horseshoes only for
  $\mu < 4\times10^{-4}$.

**Sicardy (2010)**, CeMDA 107, 145, covers what happens just past Gascheau's
value: stable periodic orbits survive slightly beyond $\mu_G$, then a
period-doubling cascade takes the region into chaos at $\mu_\infty = 0.0463$.

## 3. Real Trojans are ten orders of magnitude below all of this

| | mass | as $M_{\rm Jup}$ |
|---|---|---|
| (624) Hektor, the largest Trojan | $7.9\times10^{18}$ kg | $\sim 4\times10^{-9}$ |
| Entire Trojan population | $\sim 2\times10^{-5}\,M_\oplus$ | $\sim 6\times10^{-8}$ |
| Liberato & Winter practical limit | $0.0229\,M_\odot$ | 24.0 |
| Gascheau/Routh linear limit | $0.0391\,M_\odot$ | 40.9 |

For **single-body** dynamics, a Trojan's own mass is irrelevant by a margin of
$10^{9}$. Treating them as massless test particles — which is what this project
and every long-term study does — costs nothing.

## 4. Nobody's lifetime is infinite, but that is not a mass effect

The second half of Simon's suggestion is correct and independently established,
and it holds at **exactly zero mass**. Massless Trojans leave through slow chaotic
diffusion driven by the giant planets:

- **Levison, Shoemaker & Shoemaker (1997)**, Nature 385, 42 — the first very
  long-term integrations; the swarms are not indefinitely stable.
- **Robutel & Gabern (2006)**, MNRAS 372, 1463 — the frequency-map picture of
  which resonances do the leaking. (We hold paper II, not I.)
- **Tsiganis, Varvoglis & Dvorak (2005)** — 17% of real L4 Trojans escape over the
  age of the Solar System; chaotic diffusion is the cause.
- **Di Sisto, Ramos & Beaugé (2014)**, Icarus 243, 287 — **23% (L4) vs 28.3% (L5)**
  escape over 4.5 Gyr, while *fictitious* particles escape at identical rates from
  both swarms. Already Tier 1 in `literature/READING_LIST.md`.

So "no lifetime is infinite" needs no mass at all. Conflating it with the
Gascheau threshold would be a mistake.

## 5. What is actually unmeasured

Every long-term Trojan study treats them as massless and neglects their mutual
gravity. Two things follow that nobody appears to have checked:

1. **Where on the mass axis does the massless assumption break?** The analytic
   limits above are for the *co-orbital* mass in an idealised three-body setting,
   not for real catalogue orbits under four giant planets.
2. **Does the L4/L5 asymmetry itself depend on mass?** That is this thesis's
   question with a new axis on it, and it is cheap to ask.

`campaign/scripts/s4_run_ladder.py` measures both: the same 100 catalogue
Trojans, integrated for 1 Myr at 14 masses from 0 to 50 $M_{\rm Jup}$, one massive
Trojan per simulation, reporting the fraction still librating and the L4/L5 split
at each rung. Results in `campaign/reports/mass_ladder/report.md`.

### What it found

**The knee is at 1–3 $M_{\rm Jup}$, an order of magnitude below the published
limits.** Survival is flat at 97% from zero mass to 0.1 $M_{\rm Jup}$ — seven
orders of magnitude, the same three objects leaving at every rung — then falls to
48% at 3 $M_{\rm Jup}$ and to zero by 24. Liberato & Winter's $\mu = 0.023313$
turns out to mark where survival reaches *zero*, not where it starts to fall, and
Routh's 40.9 $M_{\rm Jup}$ is 13× past the 50% point.

That gap is the interesting part. The published limits come from the planar
circular restricted problem with a particle at the exact equilateral point; real
catalogue Trojans carry eccentricity, inclination and finite libration amplitude
under four giant planets, and come apart far earlier. The idealised threshold is
an upper bound, not the operative one.

For the thesis the consequence is the flat part: real Trojans sit nine orders of
magnitude below the knee, so the campaign's massless test particles are safe with
enormous margin — now measured rather than assumed. The L4/L5 split is flat in
mass too (largest $|z| = 2.31$ at 10 $M_{\rm Jup}$, ~6 independent looks, ~24% by
chance, and at a mass no real Trojan has).

### And their mutual gravity?

`s4_run_swarm.py` gives the 200 largest catalogue Trojans their real H-derived
masses and full mutual gravity, against the identical objects massless.
**Self-gravity is real but dynamically negligible over 1 Myr**: it perturbs
132/200 objects where a physically meaningless 0.15 mm nudge perturbs 10, so it is
a force doing work rather than chaos — but the largest change in $a$ is
$1.9\times10^{-4}$ of $a$, nothing changed libration class, and every object
survived both arms. Details and caveats in `campaign/reports/swarm/report.md`.

A methodological note that cost some time: Trojan orbits are chaotic well inside
1 Myr, so two runs of the same system separate regardless of what started them
apart. A raw per-object $|\Delta a|$ between two arms measures the Lyapunov time,
not the force. The calibration arms — the same system displaced by a known,
meaningless amount at $t_0$ — are what make the comparison mean anything.

### A methodological finding worth keeping

**WHFast cannot do this experiment.** Measured over 20 kyr with one massive
Trojan: at 24 $M_{\rm Jup}$ WHFast loses $|\Delta E/E| \sim 9\times10^{-2}$, and at
10 $M_{\rm Jup}$ it disagrees with IAS15 about whether the Trojan survives at all —
it would have put the knee in the wrong place. Democratic-heliocentric
coordinates are worse, not better. IAS15 costs the same at six bodies (~22 s per
object per Myr) and holds $|\Delta E/E| \sim 10^{-14}$, and agrees with WHFast 8/8
on fate at zero mass, so it bridges cleanly to the campaign. The ladder runs on
IAS15; the 4.5 Gyr campaign, where Trojans really are massless, keeps WHFast.

---

## References

- Gascheau, M. (1843), *Examen d'une classe d'équations différentielles...*, C. R. Acad. Sci. Paris 16, 393.
- Routh, E. J. (1875), Proc. London Math. Soc. 6, 86.
- Laughlin, G. & Chambers, J. E. (2002), AJ 124, 592 — arXiv:astro-ph/0204091.
- Sicardy, B. (2010), CeMDA 107, 145 — *Stability of the triangular Lagrange points beyond Gascheau's value*.
- Ćuk, M., Hamilton, D. P. & Holman, M. J. (2012), MNRAS 426, 3051 — arXiv:1206.1888.
- Liberato, L. & Winter, O. C. (2020), MNRAS 496, 3700 — arXiv:2007.13814.
- Levison, H. F., Shoemaker, E. M. & Shoemaker, C. S. (1997), Nature 385, 42.
- Robutel, P. & Gabern, F. (2006), MNRAS 372, 1463.
- Tsiganis, K., Varvoglis, H. & Dvorak, R. (2005), CeMDA 92, 71.
- Di Sisto, R. P., Ramos, X. S. & Beaugé, C. (2014), Icarus 243, 287 — arXiv:1604.05331.
