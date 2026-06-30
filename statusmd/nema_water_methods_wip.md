# WIP: NEMA-water reconstruction-method comparison (resume after compact)

**Branch:** `osl-safeguard` (off `reco`), pushed. HEAD `f36ce94`.
**Goal:** find the reconstruction method/operating-point for the full-physics water
NEMA contrast phantom (`recoExamples/nema/nema_la_water_bgo/`) that best matches the
clinical NEMA standard, benchmarked with the proper metric (NEMA Background Variability).
**Last updated:** 2026-06-30.

## Where we are (one paragraph)
We built the cylinder-attenuation water-NEMA scenario; De Pierro quad β=1000 *washed*
the spheres → switched to MLEM+8mm post-filter (committed earlier on this scenario).
Then, adapting STIR (Apache-2.0), we added an OSL denominator **clamp** (safeguard) and
the **Logcosh** + **RDP** priors to the core (`src/penalized.jl`, committed `c64408e`).
We then built a **systematic comparison framework** and switched the noise metric to
**NEMA BV** (committed `f36ce94`). We read the GE Discovery MI NEMA paper for the
clinical benchmark. Current best is `huber_b3` but the deciding experiment (mlem-fwhm
frontier) is not yet run.

## The clinical benchmark (Vandendriessche 2019, papers/NEMA_GE_Discovery.txt)
Discovery MI 3-ring, TOF-OSEM, NEMA NU2-2012 IQ (4:1). Per sphere 10→37 mm:
- **CR**: 47.4 / 59.3 / 67.0 / 77.0 / 82.5 / 85.1 %
- **BV**: 16.4 / 12.1 / 9.1 / 6.6 / 5.1 / 3.8 %
- Activities: spheres 21 kBq/cc, background 5.3 kBq/cc, +120 MBq out-of-FOV scatter, ~3m20s.
- **Q.Clear (RDP via BSREM, β=50) improves BOTH CR and BV vs OSEM** — the standard.
Caveats: theirs is **TOF + tuned commercial**; ours is **non-TOF on CRYSP**. But CRYSP's
~102 cm AFOV gives a sensitivity SNR gain (~√7≈2.6×) that ≳ TOF's gain (~1.9× for a 20cm
phantom), and resolution is comparable — so the scanner is NOT the excuse; the CR gap is
reconstruction-side (over-smoothing). Count check: our 13.8M trues ≈ clinical order (tens
of M) — **not statistics-starved**.

## Metric (IMPORTANT)
NEMA **BV** = std of size-matched background-ROI *means* / their mean (per sphere size);
falls with diameter. This is NOT per-voxel CoV (we used to report CoV; it's ~larger and
not comparable to literature). `nema_background_variability(img,n,org,vs)` in
`recoExamples/src/nema_phantom.jl` (exported). CRC unchanged (matches NEMA CR).

## Results so far (corrected recon, 10 mm sphere CRC / BV)
| variant | CRC | BV | note |
|---|---|---|---|
| mlem (8mm filter) | 16% | 10.3% | baseline; over-smoothed (down-left of clinical) |
| quad_b10 | 15% | 10.9% | ≈ mlem (De Pierro≈MLEM+filter, documented) |
| quad_b30/100/300 | 5/-2/-2% | — | washes contrast (raise β kills CR) |
| huber_b10 (δ0.2) | 18% | 11.5% | ≈ mlem regime (clamp 0%) |
| **huber_b3 (δ0.2)** | **30%** | **14.4%** | **best so far**; sharp small spheres, grainier bg |
| huber_b30+ | unstable | CoV>1 | OSL breaks down at β≥30 even with clamp (clamp 0→67%) |
| **clinical 10mm** | **47%** | **16.4%** | the target |

Key reads: (1) mlem ≈ De Pierro quad — single linear-smoother frontier, no free lunch
(`figures/..._compare_mlem_delpiero.png`). (2) OSL Huber only stable at β≤10 (≈mlem);
unstable above → can't reach strong edge preservation under attenuation. (3) huber_b3 is
up-and-right toward clinical but we can't yet say it BEATS the linear smoother at matched
BV. (4) even our `gold` (trues) trails clinical CR (non-TOF + recon).

## THE DECIDING NEXT STEP (do this first after compact)
Run an **MLEM fwhm scan** to extend the linear-smoother frontier up to BV~16% and overlay
`huber_b3`. If a less-smoothed mlem reaches ~CRC 30% at BV ~14% on its own → Huber's edge
preservation buys nothing; if it undershoots → Huber wins.
- Add to `config.toml` `[[variant]]`: `mlem_f0` (fwhm 0), `mlem_f4` (4), `mlem_f6` (6). (mlem=8 exists.)
- `julia -t auto --project=recoExamples recoExamples/nema/nema_la_water_bgo/compare_methods.jl mlem_f0 mlem_f4 mlem_f6`
- `python3 .../compare_plot.py --name nema_bv mlem_f0 mlem_f4 mlem_f6 mlem quad_b10 quad_b30 huber_b3_d20 huber_b10_d20`
- Read `figures/..._compare_nema_bv.png`: is huber_b3 above the mlem-fwhm curve?

## THEN (agreed action b): OSSPS + RDP — the principled path
The clinical answer (Q.Clear) is **RDP + a convergent algorithm (BSREM/OSSPS)**, not OSL.
OSL-Huber is unstable here; OSL-RDP likely too. So implement **OSSPS** (improvement "C"
from the STIR assessment): additive relaxed SPS update `x⁺ = x + relaxₙ·subgrad/(D_data+
D_prior)`, `relaxₙ=relax/(1+γn)`, floored at 0; needs a precomputed data-term curvature
(approx Hessian Aᵀ·diag·A on ones) + the prior's parabolic-surrogate curvature. Pair with
**Logcosh** (already has the surrogate `tanh(sΔ)/(sΔ)` — see STIR PriorWithParabolicSurrogate)
or extend RDP. STIR refs: `~/Projects/STIR/src/iterative/OSSPS/`, `src/recon_buildblock/
{RelativeDifferencePrior,LogcoshPrior}.cxx`. Apache-2.0, OK to port.

## How the framework works (commands)
- Driver: `compare_methods.jl [tag ...]` (no args = all variants). Builds+caches the 500M
  AC sens to `out/_cache_sens.npz` (2.3MB) → later runs are ~25s each (recon only).
- Variants declared in `config.toml` as `[[variant]]` (tag + method + params). Methods:
  `mlem`(+fwhm_mm), `quadratic`(beta), `huber`(beta,delta), `rdp`(beta,gamma,epsilon),
  `logcosh`(beta,scalar). niter defaults to `[recon].niter`=30; override per variant.
- Outputs: `out/nema_la_water_bgo_<tag>.npz` (gitignored), figures via `plot.py <tag>`
  (per-variant slices+CRC+BV vs clinical) and `compare_plot.py [--name L] [tags]` (frontier).
- Core priors (`src/penalized.jl`): `osl_mlem(...; denom_clamp=10, clamp_frac=Ref)`;
  `HuberPrior`, `LogcoshPrior`, `RelativeDifferencePrior`, plus De Pierro `penalized_mlem`.

## Housekeeping / open
- `papers/NEMA_GE_Discovery.txt` is untracked (reference, not committed).
- `osl-safeguard` not merged to `reco` yet.
- Possible later: harmonize CRC to use the NEMA background ROIs (currently central ROI);
  the BV ROI placement is an approximation to NEMA's hand-placed 12/slice.
- Memory: [[mc-data-examples-plan]], [[software-status]] (statusmd/).
