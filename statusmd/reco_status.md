# reco_status — `recoExamples/src/` support layer software status

**Scope:** `RecoExamples`, the Monte-Carlo-study support layer (`recoExamples/src/`):
PTCRYSP listmode readers, contamination/attenuation estimators, phantom ROIs, plus the
shared `tools/`. Kept out of the lean `RecoCrysp` core (see `pp_status.md`).
**Reviewed:** 2026-06-30 (branch `statusmd`, off `reco`).

Loaded via `julia --project=recoExamples`; Julia package/module name `RecoExamples`
(directory `recoExamples`). Per-study drivers live under `recoExamples/<phantom>/<scenario>/`.

## Modules (`recoExamples/src/`)

| File | Purpose | Key exports | Last touched | State |
|---|---|---|---|---|
| `mc_listmode.jl` | PTCRYSP `lors_det.h5` reader + truth-flag classifiers | `read_coincidences`, `endpoints`, `is_true`(0)/`is_scatter`(1)/`is_random`(2)/`is_single_scatter`/`is_multiple_scatter` | 2026-06-26 | stable |
| `attenuation.jl` | analytic survival `a=exp(-μ·chord)` | `sphere_chord`, `attenuation_factors` | 2026-06-27 | **sphere only — gap below** |
| `background.jl` | class-agnostic smoothed sinogram model for **scatter (`truth==1`) or randoms (`truth==2`)**, calibrated to a flagged total; Gaussian post-filter | `lor_sinogram_coords`, `background_estimate`, `background_sinograms`, `gaussian_postfilter` | 2026-06-28 | stable |
| `nema_phantom.jl` | NEMA NU-2 IQ phantom ROIs + ground truth | `NEMA_SPHERES`, `NEMA_HOT_RATIO`(4), `NEMA_BODY_R_MM`(100), `NEMA_BODY_HALF_MM`(90), `nema_sphere_masks`, `nema_background_mask`, `nema_true_image` | 2026-06-28 | stable |
| `randoms.jl` | singles-based randoms `r∝S_i·S_j` | `singles_element_counts`, `randoms_estimate` | 2026-06-26 | **legacy — superseded** by the sinogram model in `background.jl` (singles.h5 no longer kept) |
| `norm_lors.jl` | exploratory normalization-measure samplers | `emission_sens_lors`, `surface_doi_lors`, `ideal_sphere_lors` | 2026-06-27 | exploratory/diagnostic |

## Tools (`recoExamples/tools/`, reusable across studies)

- `lors_summary.py <lors_det.h5>` — class breakdown (run first on every new dataset).
- `sinogram_dump.jl <config> <scatter|randoms>` + `sinogram_plot.py <npz>` — inspect a
  background model in sinogram coords before trusting it.
- `recon_noise.py`, `scan_optimum.py` — background CoV / CRC-vs-noise helpers.

## The locked recipe (matches the NEMA study; see reco_nema_status.md)

`2.5 mm` grid · `n_sens = 5×10⁸` sensitivity LORs · `niter = 30` · regularized update.
- **Vacuum** scenarios → `HuberPrior(β=1000, δ=0.05)` via `osl_mlem`.
- **Attenuation** scenarios → `QuadraticSmoothnessPrior(β=1000)` via `penalized_mlem`,
  because **OSL Huber diverges under attenuation** (`Aᵀ(a)` small at the centre).
- `n_sens` must be large: `Aᵀ(1)` is itself a Monte-Carlo estimate whose 1/√n_sens
  sampling noise is the dominant background mottle. Normalization is a sizeable MC job.

## Scenario layout convention

`recoExamples/<phantom>/<scenario>/`: `config.toml` (data path, scanner, grid, `[sens]`,
`[recon]`, contamination blocks, `[backend]` cpu|metal) · `run.jl` (read → classify →
sensitivity → recon → ROIs → `out/<tag>.npz`) · `plot.py` (`<tag>.png` → `figures/`).
PNGs committed in place; `.npz` gitignored. Diagnostics are saved `scan_*.jl`+`*_plot.py`
pairs, never throwaway snippets.

## Data schema (PTCRYSP `lors_det.h5`)

One row per coincidence: endpoints `x1..z2`, true origin `x0..z0`, energies, element
indices, `truth∈{0,1,2}` (true/scatter/random), per-gamma `nscat1,nscat2`. Int fields
scaled by `xyz_scale_mm`(0.1)/`e_scale_keV`(0.1). `singles.h5` is no longer produced
(the randoms model is singles-free).

## Gap for the next scenario (`nema/nema_la_water_bgo`)

`attenuation.jl` has only `sphere_chord` (used by the water *sphere*). The NEMA body is
a uniform **water cylinder** (`NEMA_BODY_R_MM=100`, `|z|≤NEMA_BODY_HALF_MM=90`), one
material throughout, so attenuation is exact and analytic via a new **`cylinder_chord`**
(LOR ∩ radial disk ∩ z-slab) + a cylinder `attenuation_factors`. This is the only new
library code needed; everything else (reader, sinogram scatter+randoms, NEMA ROIs,
De Pierro recon) is reusable. μ_water(511 keV) = 0.096 cm⁻¹.
