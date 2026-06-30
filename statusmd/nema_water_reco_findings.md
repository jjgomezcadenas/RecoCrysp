# NEMA-water reconstruction: CONCLUDED findings

**Branch:** `osl-safeguard` (off `reco`). **Concluded:** 2026-06-30.
**Doc:** `recoExamples/nema/doc/nema_water.tex` (4 pp, built). **Supersedes** the old
`nema_water_methods_wip.md` (deleted).

## Headline (do NOT rediscover this)
On the full-physics water NEMA contrast phantom (`recoExamples/nema/nema_la_water_bgo/`):
1. **The reconstruction question is closed.** A reconstruction of the *true*
   coincidences (gold, attenuation-corrected) already reproduces the clinical CR curve
   to within ~1%. The algorithm/prior is **not** the bottleneck.
2. **Every converging method collapses to one linear contrast-vs-BV frontier** — MLEM+
   post-filter, De Pierro quadratic, OSL Huber, and convergent BSREM-RDP (Q.Clear) all
   sit on the same line. Edge-preserving priors buy **nothing** here.
3. **The whole gap to clinical is the corrections, specifically scatter.** Our
   scatter+randoms correction recovers only **1.6 of the 25** CR points scatter removes.
   Fix the **scatter (multiple-scatter) correction** — worth ~24 CR points. **Do NOT
   re-scan reconstruction priors.**

## Methods table (10 mm sphere, corrected)
| Method | Class | Stable here? | 10 mm CRC/BV |
|---|---|---|---|
| MLEM + post-filter (f0/4/6/8) | linear smoother | yes | 40/19.8 … 16/10.3 |
| De Pierro quadratic | convergent, quadratic | yes | 15/10.9 (β10) |
| OSL Huber | one-step-late, edge | β≤10 only (clamp) | 30/14.4 (β3) |
| OSL Logcosh / RDP | one-step-late, edge | OSL unstable under att. | — |
| **BSREM + RDP** (Q.Clear) | convergent, edge | yes (no clamp) | 31/15.3 … 27/12.9 |
| Gold (trues, AC), any method | — | yes | ≈ clinical (+1) |
| clinical 10 mm | TOF-OSEM ref | — | 47/16.4 |

## Decomposition table (mean ΔCRC over 6 spheres, BSREM-RDP β=0.5)
| Step | Mean ΔCRC | Meaning |
|---|---|---|
| clinical − gold | +1.0 | reconstruction is fine (gold = clinical) |
| gold − uncorr | +25.4 | raw contamination damage |
| corr − uncorr | +1.6 | recovered by our correction model |
| gold − corr | +23.8 | residual the model fails to recover |

## What got built (this investigation)
- **Core `bsrem`** (`src/penalized.jl`): convergent relaxed preconditioned MAP (Q.Clear /
  Ahn–Fessler), `x⁺ = x + relaxₙ·(x/sens)·[Aᵀ(mult·counts/pred) − sens − ∇R(x)]`,
  `relaxₙ = relax/(1+relax_gamma·(n−1))`. Prior in the numerator → stable under
  attenuation, no clamp. Exported. Works with any `Prior` (RDP/Huber/Logcosh/quadratic).
  Test in `test/test_penalized.jl` (NoPrior,relax=1,γ=0 ≡ mlem; RDP cuts bg std, stable).
- **`compare_methods.jl`**: `bsrem` method branch + `build_prior` helper; config has
  `[[variant]]` `bsrem_rdp_b{03,05,1,2}` (β sweep) and the mlem fwhm sweep
  `mlem_f{0,4,6}`. AC sensitivity cached once (`out/_cache_sens.npz`).
- **`gold_vs_clinical.py`** (NEW, formal): prints gold/uncorr/corr CRC vs clinical per
  sphere + the decomposition (clinical−gold, gold−uncorr, corr−uncorr, gold−corr).
- **Figures** (`recoExamples/nema/nema_la_water_bgo/figures/`):
  `nema_la_water_bgo_compare_mlem_vs_bsrem.png` (the frontier collapse),
  `nema_la_water_bgo_bsrem_rdp_b05.png` (gold=clinical, corr far below).
- Label fix: compare_plot.py gold label is now "best case / upper bound", not "our target".

## Next direction (the only one that moves the needle)
Fix the **scatter correction** so it actually subtracts the ~20.5% pedestal in/around the
spheres (currently a near no-op: recovers 1.6/25). Candidates: a single-scatter-simulation
(SSS)-style analytic estimate (STIR has one), or revisit the smoothed-sinogram scatter
model's normalization/parametrization. Validate by re-running `gold_vs_clinical.py` and
watching `corr − uncorr` climb toward `gold − uncorr` (+25). Randoms (3.2%) is not the
problem. Reconstruction (MLEM/BSREM/priors) is **done** — leave it.

## Housekeeping / open
- `papers/NEMA_GE_Discovery.txt` untracked (reference). Clinical: Q.Clear = RDP β=50
  (chosen to *match OSEM noise*, not max CR; β not transferable to our units), γ=2.
- `osl-safeguard` not merged to `reco` yet.
- Memory: [[mc-data-examples-plan]], [[software-status]] (statusmd/).
