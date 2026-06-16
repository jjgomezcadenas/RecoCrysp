"""
    RecoCrysp

Backend-agnostic Joseph 3D projectors and listmode Poisson reconstruction for
PET/SPECT, a Julia port of the non-TOF kernels of libparallelproj
(KUL-recon-lab, Apache-2.0; Schramm & Thielemans 2023).

# Design

- **Single-source kernels.** The projectors are written once with
  KernelAbstractions.jl and run unchanged on multithreaded CPUs and on GPUs
  (Apple Metal, CUDA); the backend is selected from the array type at call time.
  The only dependency is KernelAbstractions.
- **Float32 throughout**, as in the reference kernels — Apple GPUs have no Float64.
- **Endpoint-driven.** The core projector sees only LORs (pairs of world-space
  endpoints) and voxels. Detector resolution enters separately as a point-spread
  operator, so the same code serves pixelated and monolithic/ML-positioned
  scanners.
- **Matched adjoint.** `joseph3d_back` is the exact transpose of `joseph3d_fwd`
  (⟨Ax, y⟩ = ⟨x, Aᵀy⟩ to machine precision), as MLEM/OSEM require.

# Layers (`src/`)

- `projectors.jl` — the Joseph forward operator A and matched adjoint Aᵀ.
- `pet_geometry.jl` — regular-polygon ring scanner and sinogram LOR descriptor
  that produce the `(3, nlors)` endpoint matrices A consumes.
- `reconstruction.jl` — listmode Poisson MLEM/OSEM built on the projectors.

The technical note `docs/tex/joseph3d_note.tex` gives the formal derivation.
"""
module RecoCrysp

using KernelAbstractions
using KernelAbstractions: @atomic

const KA = KernelAbstractions

# Dependency order: geometry produces LORs; the projectors map image ↔ LOR;
# reconstruction iterates the projectors.
include("pet_geometry.jl")
include("projectors.jl")
include("reconstruction.jl")

export joseph3d_fwd!, joseph3d_fwd, joseph3d_back!, joseph3d_back
export RegularPolygonPETScannerGeometry, RegularPolygonPETLORDescriptor,
       get_lor_endpoints, get_lor_coordinates
export sensitivity_image, ListmodePoissonModel, predicted, neg_log_likelihood,
       em_update, mlem, osem, subset_models

end # module
