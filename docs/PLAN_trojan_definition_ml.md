# Plan: a snapshot definition of a Jupiter Trojan, learned from integration labels

Drafted 2026-10-27 after the meeting with Simon, on branch `trojan-definition-ml`.
The code is set up and smoke-tested; nothing has run on real data yet, because the
SSD was not mounted.

**In one line.** Learn which single-epoch orbits are tadpole librators, from labels
that come from integrating them. Then score that rule, and the catalogue boxes, on
the real catalogue.

---

## 1. The question, and why it is not circular

- **Inputs:** only what a snapshot gives you: the osculating elements at t0, and
  quantities derived from them and from Jupiter at the same epoch.
- **Labels:** what the orbit does next. An object is an L4 (L5) Trojan if σ = λ − λ_J
  stays in (0°, 180°) (or (−180°, 0°)) in every one of its first k windows of 10 kyr,
  and it is not removed. This is the `docs/TROJAN_DEFINITION.md` classifier.
- **Not circular.** Catalogue labels are computed from the snapshot (a box in a and
  e), so a model trained on them would only learn the box back. Integration labels
  come from the future, so learning snapshot → label is a real question.
- **k is a choice, not a feature.** It sets the label's timescale ("librating now" at
  small k). It is stated with every result, and the result is checked at k = 1, 5
  and 10. Lifetime (1 Myr, 10 Myr …) is a separate property of an object, not part
  of the definition.
- **A perfect rule exists in principle.** The dynamics are deterministic, so the
  snapshot decides the label. The study measures how simple that rule is and which
  quantities it needs. It also measures how big the band near the edge is where
  chaos makes the snapshot insufficient.

## 2. Data

| Rows | What | Label from |
|---|---|---|
| **grid** | Test particles drawn uniformly in a 4.6–5.8 au, e < 0.6, I < 60°, σ, Ω, ϖ, at the pilot's t0 with the pilot's giants (`trojans/coorbital.py`) | `s7_run_map.py --source grid` |
| **real, in the pilot** | Catalogue objects under any Trojan label (`in_run`) | the pilot's windows (`pilot_10myr`) |
| **real, not in the pilot** | Catalogue objects inside the SBDB superset box (4.9–5.5 au, e < 0.6, I < 50°) but under no label | `s7_run_map.py --source catalogue` (new, short run) |

- **Why the grid:** near a_J almost every real object is a Trojan, because the region
  next to the tadpole zone has been emptied. A model needs examples on both sides
  of the edge, so the edge is mapped with test particles.
- **The grid is a map, not a population.** Where the label flips is a property of
  the dynamics. How many particles fall anywhere is our choice. So grid precision
  depends on the grid's prior; **only the real-catalogue scores are physical.**
- **Same coordinates as the catalogue:** heliocentric osculating elements with
  mu = G M_sun (the SBDB convention used in `catalogue.states_at_t0`), at the same
  epoch.

## 3. Features, models, scoring (`trojans/definition.py`, `s7_train.py`)

- **Feature sets:**
  - `raw`: the six elements as catalogues publish them.
  - `physical`: a − a_J, e, I, and σ, ϖ − ϖ_J and Ω − Ω_J as sin/cos.
  - `minimal`: a − a_J, e, I, sin/cos σ.

  Comparing them answers whether σ (which needs Jupiter's position) is essential, and
  whether anything beyond σ matters.
- **Models:** decision trees at depth 2–8, which are readable rules (the depth-4 tree
  is exported as text), plus gradient boosting as the accuracy ceiling. Permutation
  importance on held-out grid particles.
- **Scored on:** held-out grid (25%), the real catalogue (all), and the real catalogue
  with U ≤ 4.
- **Metrics:** accuracy; Trojan precision and recall (swarm ignored); counts of false
  Trojans, missed Trojans and wrong-swarm assignments.
- **Catalogue boxes on the same rows and labels:** JPL TJN, the MPC envelope as
  inferred from its members, and the V24 box.

## 4. Smoke test (done, fake planets)

- **Setup:** the test-suite giant orbits (not an ephemeris), 1,600 grid particles,
  20 kyr (2 windows), 8 workers. Run in a scratch data root, with no real data.
- **Cost:** 12 s wall, 2.6 CPU-s per particle-Myr (most particles are removed early).
- **Labels:** k = 1: 63 L4 + 69 L5 of 1,600; k = 2: 40 + 48. Many orbits near the
  edge librate for one window but not two, so k matters near the edge.
- **Training:** the pipeline runs end to end. With only 22 held-out Trojans the
  numbers mean nothing yet.
  - The one pattern that showed up: `raw` features recall far fewer Trojans than
    `physical` (boosted 0.23 vs 0.68). That is what you would expect if σ has to be
    built from Ω + ω + M and Jupiter.
  - On the uniform grid the boxes have high recall (0.95) and low precision (≤ 0.34).
    That precision reflects the grid's prior, not the sky.
- **Tests:** `campaign/tests/test_definition.py`, 4 tests, all pass. The 2 `test_layout`
  failures were already there before this branch.

## 5. Production (to run; SSD needed)

```bash
conda activate trojan-rb        # SSD mounted
python campaign/scripts/s7_run_map.py --source grid --n 40000 --t-end-yr 1e5 --run-id map_grid --workers 8
python campaign/scripts/s7_run_map.py --source catalogue --catalogue-id 2026-09-15 --t-end-yr 1e5 --run-id map_catalogue --workers 8
python campaign/scripts/s7_dataset.py --out-id definition_v1 --grid-runs map_grid \
    --catalogue-id 2026-09-15 --catalogue-run map_catalogue --k 1 5 10
python campaign/scripts/s7_train.py --out-id definition_v1 --k 5
```

- **Grid:** 40,000 particles × 0.1 Myr. That is about 3 CPU-h from the smoke rate, or
  roughly 30 min wall on 8 workers. It should yield about 2,000 Trojans.
- **Catalogue run:** the `~in_run` objects. Their number is unknown until the SSD is
  mounted; expect hundreds, so minutes.
- **Then:**
  - Look at `map_slice.png` (the edge in σ, a − a_J for cold orbits) and the depth-4
    trees.
  - If the edge is under-resolved, run a second grid concentrated where the boosted
    model is uncertain.

## 6. What prior work already covers (literature check 2026-10-27)

The closest papers, with what each does and does not do:

- **Ciacci et al. 2023** (MNRAS 526, 600; arXiv:2309.10603). The closest ML work.
  Classifies co-orbitals of Venus, Earth and Jupiter into quasi-satellite, horseshoe,
  L4 and L5. The input is the θ(t) time series, not a snapshot. About 98–99.9%
  accuracy. It does not address the definition question.
- **Smirnov & Markov 2017** (MNRAS) and **Smirnov 2024** (Icarus). ML on **proper**
  elements with integration labels, for three-body resonances.
- **Smullen & Volk 2020** and **Volk & Malhotra 2024.** Kuiper belt classification from
  features of short integrations.
- **Carruba et al. 2021 onward.** CNNs and ViTs on images of resonant arguments.
- **Analytic criteria:**
  - Dermott & Murray 1981, and Murray & Dermott 1999 §3.9: the averaged
    separatrix, valid at low e and I.
  - Namouni 1999: high e and I.
  - **Di Ruzza, Pousse & Alessi 2023** (Icarus 390, 115330): regime domains for
    quasi-coplanar co-orbitals from an integrable model. The strongest competitor.
- **Greenstreet, Gladman & Jurić 2024** (ApJL; arXiv:2309.06609). Box contamination,
  shown with 1000 clones × 0.5 Myr: 14 MPC/JPL "Trojans" are non-resonant. No ML.
- **Stability maps** in proper elements: Levison+1997; Marzari+2003; Robutel & Gabern
  2006.

**Verdict:** no published snapshot → tadpole rule for Jupiter Trojans scored against
the catalogue boxes. But a referee will say:
- the physics of the edge is the known separatrix;
- box contamination is already shown (Greenstreet+2024).

**What the study must add:**
- the edge at real e and I, where the analytic criteria stop being valid;
- an interpretable rule with measured error rates on the real catalogue;
- a benchmark against Murray & Dermott and Di Ruzza+2023, not only against the boxes.

**Not yet verified:** Smirnov & Markov 2017 volume and pages; the Greenstreet+2024 volume;
the full text of Di Ruzza+2023. All of the above are from abstracts; read the papers
before citing.

## 7. Caveats

- A snapshot rule in osculating elements depends on the epoch and the frame. It is
  valid for this epoch and heliocentric elements. Test whether the rule is simpler
  in proper elements.
- Near the separatrix the dynamics are chaotic. Report that fuzzy band as a result.
- The grid's prior sets grid precision. Quote only real-catalogue error rates as
  physical.
- At 10 kyr windows the label means "librating now", not long-term stability.

## 8. First results (2026-10-27; `definition_v1`, label k = 5, i.e. 50 kyr)

**Runs:**
- `map_grid`: 40,000 particles × 0.1 Myr, 0.89 CPU-h. Yield 893 L4 / 895 L5.
- `map_catalogue`: the 34 unlabelled real objects. 33 were removed within 0.1 Myr.
- Real labels for the pilot objects come from `pilot_10myr/merged`.

**Results** (`campaign/reports/definition_v1/`):

| Rule | Real, U ≤ 4 (14,376; 14,315 Trojans): false / missed | Real, all (16,479): false / missed |
|---|---|---|
| JPL TJN box | 22 / 1 | 222 / 1 |
| MPC envelope (inferred) | 5 / 2 | 126 / 2 |
| V24 box | 30 / 2 | 160 / 2 |
| Boosted, `minimal` (a − a_J, e, I, σ) | **1 / 7** | 14 / 50 |
| Depth-8 tree, `minimal` | 2 / 90 | 40 / 147 |
| Boosted, `raw` six elements | 0 / 8,238 | 3 / 9,376 |

**What it shows:**
- **The raw elements cannot define a Trojan.** Without σ, which needs Jupiter's longitude
  at the same epoch, recall on real objects is 0.42 and swarms are confused (1,576
  wrong-swarm assignments).
- **ϖ − ϖ_J and Ω − Ω_J add nothing:** `physical` ≈ `minimal`.
- **Importance:** a − a_J > e > σ > I.
- **The edge is fuzzy.** On held-out grid particles the best model has recall 0.80 and
  precision 0.90. Real objects mostly sit deep inside, so they score far better.
- **Shallow trees (depth ≤ 4) predict "none" almost everywhere.** An axis-aligned box can't
  follow the curved separatrix in (σ, a − a_J), and only 4.5% of the grid are Trojans.
  - Next step: add the analytic separatrix variable (Murray & Dermott's (δa)² against
    f(σ), `diagnostics.libration_integral_f`) as a feature. Check whether a depth-2 tree
    on it plus e and I matches the boosted model. That would give a readable definition.
- **Box error rates on real objects are small at U ≤ 4.** But boxes don't test libration,
  and the grid shows how much non-librating orbit space they admit. The real sky
  happens to be empty there.

**Data integrity (found 2026-10-27).** The SSD (renamed X10-Pro-4T) had silent
corruption:
- **Restored:** V24 raw (6 files, re-downloaded and matching the original sha256) and
  `catalogue/2026-09-15/master_table.parquet` (rebuilt from raw and verified against
  the pilot).
- **Still corrupt:** the pilot's chunk files (the merged files are intact), the benchmarks,
  and the Horizons manifest.
- **Horizons data is still trustworthy:** it reproduces the pilot's initial states
  bit for bit.

## 9. The rule applied to every known small body (2026-10-27)

**Pipeline:**
1. `s7_fetch_population.py`: all of SBDB. 1,577,707 asteroids and 4,082 comets.
2. `s7_screen.py`: Kepler-propagate each orbit to t0, apply the boosted `minimal` rule
   and the boxes. Domain: a 4.6–5.8 au, e < 0.6, I < 60°.
3. `s7_fetch_states.py`: Horizons states at t0 for everything uncertain.
4. `s7_run_map.py --source targets`: integrate them for 0.1 Myr (`map_screen_targets`).

**Screen:** 18,028 objects with 4 < a < 6.5 au; 16,730 inside the domain.

| Group | n | Integration verdict (k = 5) |
|---|---|---|
| Rule says Trojan, no catalogue label | **0** | — |
| Unlabelled, inside the domain (all of them, any probability) | 256 | **0 Trojans** |
| Near a_J, outside the domain (e ≥ 0.6, I ≥ 60°, retrograde) | 110 | **0 Trojans** |
| Labelled in Sept, rule says no | 316 | pilot: 254 not librating, 62 librating |
| Labelled in Sept, rule says yes | 16,129 | pilot: 16,116 librating, 13 not |
| Not integrable | 2 | 101P/Chernykh (split comet), C/2015 D1 (unbound) |

**What it shows:**
- **No hidden Trojans.** No known small body outside the catalogues librates as an L4/L5
  tadpole. The labels miss nothing.
- **The catalogues over-include instead.** About 254 labelled objects are not librating
  at k = 5, mostly U = 9 orbits.
- **Of the 62 Trojans the rule rejects:**
  - 50 have U = 9 and SBDB epochs a median of 11 yr from t0, so the Kepler screen moved
    σ by about 1°.
  - On the exact t0 snapshot the rule recovers 18 of them.
  - The other 44 are edge orbits (|a − a_J| up to 0.23 au).
- **Non-Trojan co-orbitals that no catalogue lists.** These held a co-orbital state for
  the whole 0.1 Myr, or until the end:
  - (2011 VS15): 10/10 windows quasi-satellite.
  - (2003 WG166): 10/10 quasi-satellite.
  - 487P/Siding Spring: 10/10 quasi-satellite.
  - (2018 PZ27): 9 quasi-satellite + 1 horseshoe.
  - (2019 GP197): 4 quasi-satellite.
  - 514107 Ka`epaoka`awela: retrograde co-orbital, 8 windows read as "horseshoe" by a
    classifier written for prograde orbits.
  - **Check the literature** (e.g. Jupiter quasi-satellite papers) before calling any of
    these new.
- **Classifier quirk:** a window in which an object is removed after a few samples can
  read as "tadpole". k = 1 labels are therefore unreliable for objects removed early;
  k ≥ 2 is not affected.
