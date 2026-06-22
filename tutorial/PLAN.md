# RecoCrysp Tutorial — Design & Decisions

Persistent design record for the RecoCrysp tutorial series. Captures structure,
formatting decisions, the technical/physics content to be written, and the
example specifications, so work can resume after any context loss.

## 1. Goal & structure

A tutorial that teaches PET image reconstruction with the RecoCrysp library,
**theory-first / spec-before-code**.

```
tutorial/
  PLAN.md                      # this file
  docs/
    recocrysp_tutorial.sty     # shared LaTeX style/preamble for the whole series
    tutorial_basis.tex         # DRIVER: \input{}s tbsrc/*.tex (foundational theory)
    tutorial_example1.tex      # DRIVER for example 1 spec (later; own te1src/ or tbsrc)
    Makefile                   # pdflatex x2
    tbsrc/                     # section sources for tutorial_basis (one file per section)
    figures/                   # figures (TikZ inline now; result figures from code later)
  example1/                    # CODE implementing tutorial_example1.tex (AFTER the .tex)
  example2/  ...
```

- `tutorial_basis.tex` is a **driver** only: title/abstract/TOC + `\input{tbsrc/<section>}`.
  Develop & review section by section by editing the individual `tbsrc/*.tex`.
- Each `tutorial_exampleN.tex` describes example N; its code lives in `tutorial/exampleN/`.
- Write the `.tex` (basis, then example1) **before** writing example code.

## 2. Formatting decisions (confirmed)

- `article`, 11pt, a4; shared `recocrysp_tutorial.sty`.
- pdflatex; inline `thebibliography` (no bibtex step); TikZ schematics; `listings`
  for short Julia snippets.
- Conceptual figures = TikZ (PDF builds standalone, no Julia run). Result figures
  (reconstructions, convergence curves) belong in the *example* docs, generated
  by that example's code.
- Build incrementally in 4 stages, compiling/ reviewing after each:
  1. sty + driver + §1–3 (intro, geometry, grid)
  2. §4–6 (projection, resolution, datamodel) + TikZ figures
  3. §7–9 (statistics, mlem, acceleration)
  4. §10–11 + appendices (refs already done)

## 3. tutorial_basis.tex — section outline (tbsrc file → content)

1. `intro.tex` — PET physics (annihilation→511 keV back-to-back→coincidence→LOR);
   reconstruction as ill-posed inverse problem; the full data model previewed
   `ȳ_i = n_i a_i (A G x)_i + s_i + r_i`; scope (non-TOF, listmode, RecoCrysp,
   Float32, CPU/Metal); notation table. **[WRITTEN — first draft]**
2. `geometry.tex` — endpoint-driven model (resolution decoupled from crystal
   size); both detector models: pixelated (RegularPolygonPETScannerGeometry) and
   continuous/monolithic (ContinuousPET, sample_lors, CRYSP 774/1024 mm, 3.5 mm);
   world frames/mm (axial = z); listmode vs sinogram, the (x1,y1,z1,x2,y2,z2)
   event format. **[WRITTEN — first draft]**
3. `grid.tex` — voxel grid; voxsize two roles (world↔voxel mapping; path-length
   correction factor Δ/cosθ); **voxel size ≠ resolution** (⅓–½·FWHM rule);
   anisotropy; FOV; units/frame consistency gotcha; 1-based vs C 0-based index
   remark. **[WRITTEN — first draft]**
4. `projection.tex` — line integral ∫f dℓ; Joseph's method (principal axis,
   plane stepping, bilinear interp, cf=Δ/cosθ); system matrix A; back-projection
   & **matched adjoint** Aᵀ (atomic scatter; why matched matters); Float32;
   validation (adjointness ~1e-10, analytic line integrals). **[WRITTEN — first
   draft]** physics migrated from the retired `docs/tex/joseph3d_note.tex`
   (port/Metal/implementation depth now lives in the Documenter pages, not the
   tutorial).
5. `resolution.tex` — blur sources (positron range, non-collinearity,
   positioning, DOI); pixelated crystals: σ=l/√12 per endpoint, ≈ l/2 FWHM at
   FOV centre, **library derives none of it**; monolithic+SiPM+ML (DNN) → 3.5 mm
   (CRYSP), decoupled from 50 mm block; PSF operator G (image-space isotropic
   Gaussian, self-adjoint, σ=FWHM/2.355), A_psf = A·G. Resolution injected at
   SIMULATION time: ȳ = A·(G·x_true); reconstruction uses plain A and converges
   to G·x_true (resolution-limited). MC/endpoint-smear route is equivalent (no G
   at all); putting G in the recon = resolution recovery, OUT OF SCOPE.
   **[WRITTEN — first draft]**
6. `datamodel.tex` — full model ȳ = n·a·(A G x) + s + r. Attenuation a=exp(−∫μ dℓ),
   μ-map, multiplicative, **computed by projecting the μ-map** (a=exp(−A·μ));
   μ_water≈0.0096 mm⁻¹ @511keV; non-TOF ⇒ single factor per LOR. Normalization n
   (efficiencies), multiplicative. Scatter (single+multiple) & randoms: additive,
   estimated EXTERNALLY (SSS/Watson, MC); library accepts via contamination.
   Map to library fields: **mult = n·a, contamination = s + r**; recon predicts
   with plain A (G stays in data), sens = Aᵀ(mult). For monolithic, n≈1 (no
   crystals; geometric sensitivity already in Aᵀ𝟙). **[WRITTEN — first draft]**
7. `statistics.tex` — Poisson counting; Poisson log-likelihood; listmode vs
   sinogram likelihood & equivalence (the ⟨sens,x⟩ term); negative-log-likelihood
   objective f(x) = ⟨sens,x⟩ − Σ counts·log(pred); gradient sens − Aᵀ(n·counts/pred).
8. `mlem.tex` — ML estimation; EM derivation → multiplicative update; **MLEM as
   preconditioned gradient step** x − (x/sens)⊙∇f (the form RecoCrysp uses);
   sensitivity image sens=Aᵀ(n·a) (preconditioner+normalizer); properties:
   non-negativity (clamp), monotone likelihood, convergence; full update with
   pred = n·a·(A G x)+s.
9. `acceleration.tex` — OSEM (subsets, subset sensitivity Aᵀ_sub(n), the
   speed/accuracy caveat); noise & **semi-convergence** (MLEM amplifies noise,
   200 iters worse than 50; early stopping; relerr-vs-iteration curve); brief
   regularization note (preconditioned-gradient form generalizes).
10. `pipeline.tex` — two-script structure (acquisition/calibration → recon);
    simulation outputs (listmode events, sensitivity/normalization image, μ-map);
    recon loop (mlem/osem); quality metrics (relerr, likelihood, visual).
11. `summary.tex` — the **example ladder** table (what each example switches on).
- `appendices.tex` — A: EM derivation; B: notation glossary; C: API quick ref. [STUB]
- `references.tex` — bibliography. **[DONE]** (Joseph82, SheppVardi82, LangeCarson84,
  HudsonLarkin94, Watson00, Moses11, SchrammThielemans24).

## 4. Substantive technical decisions (already agreed, to be written into the text)

- **Endpoint-driven projector**: joseph3d_fwd/back = thin-line integral between
  supplied endpoints. Crystal size NEVER enters the core projector. No PSF baked
  in. → handles monolithic+DNN naturally; no crystal-size→resolution coupling.
- **Voxel size**: required input (3-tuple mm). Roles: world↔voxel mapping &
  path-length cf. Independent of resolution; pick ≈ ⅓–½·FWHM. Anisotropy allowed.
  img_origin = world coord of voxel (1,1,1) centre; same mm frame as endpoints.
- **Resolution**: ≈ l/2 FWHM (pixelated, at centre) or 2.5 mm (monolithic+DNN);
  always USER-supplied (endpoint sampling stats and/or PSF FWHM), never derived.
- **Data model**: ȳ = n·a·(A G x) + s + r. attenuation & normalization →
  multiplicative (mult = n·a); scatter+randoms → additive (contamination).
  Attenuation computed in-library by projecting a μ-map; scatter needs external
  model (SSS/MC).
- **Reconstruction**: listmode Poisson; sens = Aᵀ(n·a) over ALL geometric LORs
  (decoupled from event list); MLEM = preconditioned gradient; non-negativity
  clamp; OSEM; semi-convergence.

## 5. Example ladder & Example 1 spec

| Example | resolution G | attenuation a | scatter/randoms | normalization n |
|---|---|---|---|---|
| 1 | 2.5 mm PSF | 1 (vacuum) | 0 | 1 (sens = Aᵀ𝟙) |
| 2 | 2.5 mm | water μ-map (a=exp(−∫μ)) | 0 | 1 |
| 3 | 2.5 mm | water | scatter model / MC | 1 |
| 4 | 2.5 mm | water | scatter | per-crystal efficiencies |

**Example 1 (to spec in tutorial_example1.tex):**
- Scanner: standard bore — interpret 700 mm as **detector ring diameter**
  ⇒ R ≈ 350 mm (CONFIRM if it's bore vs ring). Axial length 300 mm.
- "Crystals" 50×50 mm **monolithic** modules read by SiPM matrix + DNN ⇒
  intrinsic resolution **2.5 mm isotropic**, decoupled from module size.
  Thickness irrelevant (parametric). Axial: 6 module rows (300/50). Around ring:
  ≈ π·700/50 ≈ 44 modules; adjust radius for exact pitch (R≈350 mm).
- **Do NOT** sample LOR endpoints at 44 discrete crystals (would cap resolution
  at ~25 mm). Model the detector as **continuous**: sample event endpoints
  continuously on the cylinder, apply the 2.5 mm resolution model. The 50 mm is
  just module size, irrelevant to resolution.
- Phantoms (three, separate reconstructions): **sphere, cylinder, Derenzo**
  (rods scaled to ~resolution, e.g. {40,32,25,20,16,12} mm sectors — demonstrates
  the resolution limit). Uniform activity.
- **"Filled with vacuum"** = emission only, no attenuation/scatter (a=1, s=r=0);
  later examples fill with water etc.
- Normalization n=1 for example 1; "normalization image" = geometric sensitivity
  sens = Aᵀ𝟙.

## 6. Open decisions (pending user confirmation)

- [ ] 700 mm = ring diameter (assumed) or patient bore?
- [ ] Simulation method: **Monte-Carlo emission** (sample emission→back-to-back→
      intersect cylinder→blur endpoints) vs **analytic** (project phantom over
      dense LOR set→Poisson→blur). Leaning analytic for a first example.
- [ ] **Add the PSF operator G to the library** (image-space isotropic Gaussian,
      self-adjoint, FWHM as data; default none) — needed before example 1 code.
- [ ] Example-1 voxel size (≈1.0–1.25 mm ideal for 2.5 mm res; cost trade-off vs
      coarser grid).
- [ ] Intermediate data format: **HDF5** (proposed; PET-standard, matches real LM)
      vs JLD2. Plotting: **CairoMakie** (proposed). Example code: plain
      well-commented scripts (the .tex carries the teaching).

## 7. Library status (context anchor)

- Library: `~/Projects/RecoCrysp` (renamed from JosephProjectors; remote
  origin git@github.com:jjgomezcadenas/RecoCrysp.git). Latest pushed work:
  projectors + geometry + reconstruction (MLEM/OSEM) + normalization machinery
  (mult/contamination). Tests 35/35. Depends only on KernelAbstractions; Metal
  test-only. PSF operator G **not yet implemented**.
- Python reference: `~/Projects/parallelproj`. Feasibility scratch: `~/Projects/PP`.

## 8. Progress tracker

- [x] Scaffold: sty, driver, Makefile, tbsrc stubs, references, PLAN.md
- [x] §1 intro.tex (first draft)
- [x] §2 geometry.tex (first draft; both detector models)
- [x] §3 grid.tex (first draft) — stage 1 complete
- [x] §4 projection.tex (first draft; physics from the retired joseph3d_note)
- [x] §5 resolution.tex (first draft; simulation-time Gaussian smear, no recon-side G)
- [x] §6 datamodel.tex (first draft) — stage 2 complete
- [ ] §7 statistics, §8 mlem, §9 acceleration (stage 3)
- [ ] §10 pipeline, §11 summary, appendices (stage 4)
- [ ] tutorial_example1.tex (after basis)
