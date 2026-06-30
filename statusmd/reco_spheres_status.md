# reco_spheres_status — `recoExamples/sphere/` studies status

**Scope:** the uniform-sphere arm of the Part-III MC studies — reconstructing PTCRYSP
listmode data of a uniform sphere to validate each correction (normalization, randoms,
attenuation, scatter) against the truth flags.
**Reviewed:** 2026-06-30 (branch `statusmd`, off `reco`).
**State:** all four scenarios re-run on the locked recipe 2026-06-28; findings written
up in `recoExamples/sphere/doc/sphere.tex` (9 pp, builds clean). See `reco_status.md`
for the shared helpers and the recipe.

Phantom throughout: uniform R=80 mm sphere of ¹⁸F at the origin, CRYSP BGO scanner
(geometry matched to the MC). Recipe: 2.5 mm grid, n_sens=5×10⁸, niter=30; **vacuum →
Huber, water → De Pierro quadratic** (OSL Huber diverges under attenuation).

## Scenarios

| Scenario | Physics isolated | Driver(s) | Recon | Headline result |
|---|---|---|---|---|
| `air_bgo_100kBq` | **normalization** (n=1) + measure tilt | `run.jl` (+ `diagnostics/`) | Huber | flat in noise, interior **CoV 0.10** at 500 M sens; a **~25% centre-low measure tilt** (real, grid-resolved, not the regularizer) |
| `air_bgo_1MBq` | **randoms** 0.94% | `run.jl` | Huber | gold/uncorr/corr degenerate on the uniform sphere; sinogram randoms calibrate exactly |
| `air_bgo_10MBq` | **randoms** 8.7% | `run.jl` | Huber | still degenerate interior; only signature is the outside-sphere halo the correction pulls partway back |
| `water_bgo_1MBq` | **attenuation** + **scatter** | `run_att_only.jl`, `run_att_scatter.jl` | De Pierro quad | AC flattens the cupping (centre/edge ~1.1, interior 0.87–1.03); scatter correct but far-field-only (interior <0.1%) |

Each scenario: `config.toml` + `run*.jl` + `*_plot.py`; outputs `out/<tag>*.npz`
(gitignored) and `figures/<tag>*.png` (committed). `air_bgo_100kBq/diagnostics/`
holds the normalization probes (`scan_niter`, `empirical_sens`, `confirm_measure`,
`compare_norm`).

## What the arm established (the through-line)

A uniform, centred sphere is **spatially degenerate** with smooth additive backgrounds
(randoms, scatter): the corrections calibrate exactly and act physically in the far
field, but the interior-mean normalization divides out the pedestal, so they are nearly
invisible. **Attenuation is the exception** — multiplicative and radially structured —
which the sphere displays fully (cupping → flat). The degeneracy is precisely why the
contrast (NEMA) phantom is needed for a visible randoms/scatter demonstration.

Two corrections to earlier claims, settled on the locked recipe (2026-06-28):
- The interior noise once called "≈0.29 Poisson" is the **sensitivity-image MC sampling**
  noise; at n_sens=5×10⁸ it drops to **0.10** (count/voxel-independent).
- The normalization tilt is **~25%**, not the once-quoted ±8% — the fine 2.5 mm grid
  resolves a real LOR-measure tilt the coarse 4 mm voxels averaged away; unregularized
  MLEM reproduces it (so it is not the regularizer). Which LOR measure is correct to
  normalize with is the remaining open question (geometry, no MC).

## Findings doc

`recoExamples/sphere/doc/sphere.tex` → `sphere.pdf` — standalone write-up (Setup,
Normalization, Randoms, Attenuation, Scatter, a NEMA contrast-phantom section, and a
Synthesis table). Pulls each scenario's PNG via `\graphicspath` (relative paths, builds
clean). `.aux/.log/.out/.pdf` are gitignored.

## Open / next

- The arm is complete. The natural continuation is the **contrast phantom** (`nema/`,
  see `reco_nema_status.md`), where randoms/scatter are no longer degenerate.
- Side studies parked: energy-window relaxation on the water dataset (would raise the
  ~0.13 interior scatter fraction); the LOR-measure question behind the ~25% tilt.
