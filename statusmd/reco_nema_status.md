# reco_nema_status — `recoExamples/nema/` studies status

**Scope:** the NEMA NU-2 image-quality contrast-phantom arm of the Part-III MC studies —
six hot spheres (Ø10–37 mm) at 4:1 in a body cylinder (R=100 mm), where randoms/scatter
are no longer degenerate with the source (unlike the uniform sphere).
**Reviewed:** 2026-06-30 (branch `statusmd`, off `reco`). See `reco_status.md` for the
shared helpers/recipe and `reco_spheres_status.md` for the sphere arm it grew out of.

## Built scenario: `nema_la_air_bgo` (vacuum, low activity, 2.9% randoms)

The contrast phantom in vacuum (n=1, no attenuation/scatter; only randoms). Two roles:
1. **Contrast baseline** — `run.jl` (gold/uncorr/corr), CRC per sphere + background CoV.
   Randoms from the singles-free sinogram model. At 2.9% the three are near-identical
   (the gentle end; the randoms-control contrast wants the high-activity twin).
2. **Reconstruction-noise investigation** (the deep dive) — saved scans, each a
   `scan_*.jl` + `*_plot.py` pair, figures committed under `figures/`:

| Scan | Question | Finding |
|---|---|---|
| `scan_niter` | noise vs iteration | CoV grows ~linearly; mean converges (semi-convergence) |
| `scan_counts` | noise vs data statistics | **count-INDEPENDENT** — not Poisson |
| `scan_voxel` | noise vs voxel size | flat 1.5–4 mm — not a grid effect |
| `scan_fwhm` | post-filter trade | CRC/CoV is degenerate — don't free-optimize |
| `scan_penalized` | quadratic-prior β sweep | prior bounds CoV vs iteration |
| `scan_huber` | Huber β sweep (OSL) | edge-preserving variant |
| `scan_nsens` | noise vs sensitivity LORs | **THE source** — bg CoV 0.30(20M)→0.10(200M)→0.08(500M), ~1/√n_sens, ≈5× the sens CoV |
| `compare_methods`/`method_compare` | CRC-vs-noise frontier | **post-filter ≈ Huber > quadratic ≈ MLEM** (edge-preservation no clear win on low-contrast spheres) |
| `spheres_nsens` | clean recon at high n_sens | `spheres_nsens.jl <n_sens>`; at 500 M all six spheres clean, CRC 43%→106% |

**The headline finding:** the dominant background mottle is the **Monte-Carlo sampling
noise of the sensitivity image Aᵀ(1)**, not data statistics or voxel size — sampled at
only 20 M LORs it injects ~30% CoV (amplified ~5× by the MLEM denominator), falling as
1/√n_sens. Fixed by sampling n_sens large (≥5×10⁸). This drove the **locked recipe** now
used across all studies. Figures: `figures/nema_la_air_bgo_{nsens_scan,method_compare,spheres_500M}.png`.

## Available datasets (PTCRYSP `prod/`), not yet reconstructed

| Dataset | Body | Activity / randoms | Status |
|---|---|---|---|
| `nema_la_air_bgo` | vacuum | low / 2.9% | **built** (above) |
| `nema_air_bgo` | vacuum | high / 22.9% | not run — the **randoms-control** twin of `nema_la_air_bgo` |
| `nema_la_water_bgo` | water | low / 3.2% | not run — **full physics** (atten + 20.5% scatter + randoms); the payoff case |
| `nema_water_bgo` | water | high | not run |

## Open / next

- **`nema_la_water_bgo`** — the natural next build: full-physics contrast phantom where
  attenuation + scatter + randoms corrections should all be *visible*. Reuses the reader,
  sinogram scatter+randoms, NEMA ROIs, and De Pierro recon; the **only new code** is a
  `cylinder_chord` in `recoExamples/src/attenuation.jl` (NEMA body = water cylinder
  R=100, |z|≤90; μ_water=0.096 cm⁻¹). Recon must be De Pierro quadratic (OSL Huber
  diverges under attenuation).
- **Randoms-control**: run `nema_air_bgo` (22.9%) vs `nema_la_air_bgo` (2.9%) at one
  locked config — the original NEMA-arm goal, still open.
- Findings currently live in the NEMA section of `recoExamples/sphere/doc/sphere.tex`;
  a dedicated NEMA write-up may follow once the water/randoms scenarios are in.
