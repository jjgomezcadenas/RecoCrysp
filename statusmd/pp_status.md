# pp_status — RecoCrysp core (`src/`) software status

**Scope:** the `RecoCrysp` core library (`src/`) — Joseph 3D projectors, scanner
geometry, phantoms, the resolution operator, and listmode Poisson reconstruction.
**Reviewed:** 2026-06-30 (branch `statusmd`, off `reco`).
**Heritage:** Julia port of the non-TOF kernels of libparallelproj (KUL-recon-lab,
Apache-2.0; Schramm & Thielemans 2023).

## Design invariants (stable)

- **Single-source kernels** — projectors written once with KernelAbstractions; run
  unchanged on multithreaded CPU and GPU (Apple Metal, CUDA), backend chosen from the
  array type at call time. Only hard dependency: KernelAbstractions.
- **Float32 throughout** (Apple GPUs have no Float64).
- **Endpoint-driven** — the projector sees only LORs (world-space endpoint pairs) and
  voxels, never crystal geometry; detector resolution enters separately. One code path
  serves pixelated and monolithic/ML-positioned scanners.
- **Matched adjoint** — `joseph3d_back` is the exact transpose of `joseph3d_fwd`
  (⟨Ax,y⟩=⟨x,Aᵀy⟩ to Float32 precision), as MLEM/OSEM require.

## Layers

| File | Purpose | Key exports | Last touched | State |
|---|---|---|---|---|
| `projectors.jl` | Joseph forward `A` + matched adjoint `Aᵀ` (KA `fwd_kernel!`/`back_kernel!`) | `joseph3d_fwd[!]`, `joseph3d_back[!]` | 2026-06-16 | stable |
| `pixelated_pet.jl` | regular-polygon ring scanner + sinogram LOR descriptor (discrete crystals) → `(3,nlor)` endpoints | `RegularPolygonPETScannerGeometry`, `…LORDescriptor`, `get_lor_endpoints`, `get_lor_coordinates` | 2026-06-16 | stable |
| `continuous_pet.jl` | continuous cylindrical scanner (monolithic); samples LOR endpoints on the detector surface | `ContinuousPET`, `sample_lors` | 2026-06-16 | stable |
| `phantoms.jl` | digital phantoms | `uniform_sphere`, `uniform_cylinder`, `derenzo` | 2026-06-22 | stable |
| `psf.jl` | image-space Gaussian resolution operator G | `gaussian_blur` | 2026-06-22 | stable |
| `reconstruction.jl` | listmode Poisson MLEM/OSEM | `sensitivity_image`, `ListmodePoissonModel`, `predicted`, `neg_log_likelihood`, `em_update`, `mlem`, `osem`, `subset_models` | 2026-06-28 | stable |
| `penalized.jl` | regularized (MAP-EM) reconstruction | `penalized_mlem`, `osl_mlem`, `Prior`, `NoPrior`, `QuadraticIntensityPrior`, `QuadraticSmoothnessPrior`, `HuberPrior` | 2026-06-28 | newest; see notes |

## Reconstruction model (the contract)

Forward model per LOR/event: `pred_e = n_e·(A_LM x)_e + s_e` with `n` the per-LOR
multiplicative factor (efficiency × geometric × attenuation) and `s` additive
background. Objective `f(x)=⟨sens,x⟩ − Σ counts·log(pred)`; MLEM is the
preconditioned gradient step `x⁺ = x − (x/sens)⊙∇f = (x/sens)⊙Aᵀ(n·counts/pred)`.

- **`sensitivity_image(...; weights, scale)`** — `sens = scale·Aᵀ(w)`. The `scale`
  kwarg is the one subtle knob: it puts the sensitivity on the per-LOR scale of the
  event list. `scale=1` when `sens` shares the event LOR set (sampling noise cancels
  in the ratio); `scale = n_events/n_sens` for true listmode where `sens` is a
  **separate Monte-Carlo sample** — its `1/√n_sens` noise then does NOT cancel and
  imprints on the image (the headline of the MC studies: sample `n_sens` large).
- **`ListmodePoissonModel`** — carries `counts` (default 1/event), `contamination`
  (additive `s`, default 0), `mult` (`n`, default 1), and the precomputed `sensitivity`.
  All arrays must share one backend.

## Regularized reconstruction (`penalized.jl`, added 2026-06-22, KA priors 2026-06-28)

Minimizes `f(x)+βR(x)` while keeping EM's non-negativity.
- **`penalized_mlem`** — De Pierro MAP-EM for **quadratic** priors; closed form
  `x⁺=2t/(√(B²+4Gt)+B)` (the prior sets `(B,G)`; β=0 ≡ MLEM). **Monotone.**
  Priors: `QuadraticIntensityPrior(β,z)`, `QuadraticSmoothnessPrior(β)`.
- **`osl_mlem`** — One-Step-Late for **non-quadratic** priors: `x⁺=t/(sens+∇R)`.
  Prior: `HuberPrior(β,δ)` (edge-preserving). **NOT provably monotone** — stable at
  moderate β, can diverge at large β, and **diverges under attenuation** when
  `sens=Aᵀ(a)` is small somewhere (use De Pierro quadratic there instead).
- Neighbour sums / Huber influence via KA stencil kernels (CPU/Metal), mirroring the
  projector idiom.
- β has no universal value — it scales with the magnitude of `sens` (≈10²–10³ in the
  studies → β ~ O(10²–10³)). Tuned per study, like the iteration count.

## Tests

- `test/runtests.jl` — projector analytic accuracy + adjointness (<1e-3), rays missing
  the image, PET scanner geometry, reconstruction; CPU + Metal.
- `test/test_penalized.jl` — `NoPrior`≡`mlem`, De Pierro cost monotone (both quadratic
  priors), smoothness/Huber reduce variance.
- `test/crossval/` — cross-validation against the reference.

## Notes / open

- `penalized.jl` is the only core file still settling; the OSL-Huber-under-attenuation
  divergence is the live gotcha (documented in its docstring and worked around in the
  water studies by using De Pierro quadratic).
- Methods write-up: Part-I tutorial `tutorial/docs/tbsrc/{mlem,acceleration,regularization}.tex`
  + Appendices A (EM) and B (De Pierro surrogate).
