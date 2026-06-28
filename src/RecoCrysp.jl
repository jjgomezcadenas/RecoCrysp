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
  endpoints) and voxels, never crystal geometry. Detector resolution enters
  separately (a simulation-time smear or an external model), so the same code
  serves pixelated and monolithic/ML-positioned scanners.
- **Matched adjoint.** `joseph3d_back` is the exact transpose of `joseph3d_fwd`
  (⟨Ax, y⟩ = ⟨x, Aᵀy⟩ to machine precision), as MLEM/OSEM require.

# Layers (`src/`)

- `projectors.jl` — the Joseph forward operator A and matched adjoint Aᵀ.
- `pixelated_pet.jl` — regular-polygon ring scanner and sinogram LOR descriptor
  (discrete, pixelated-crystal detectors) that produce the `(3, nlors)` endpoint
  matrices A consumes.
- `continuous_pet.jl` — continuous cylindrical scanner (monolithic detectors)
  that samples LOR endpoints on the detector surface.
- `phantoms.jl` — digital phantoms (uniform sphere/cylinder, Derenzo).
- `psf.jl` — image-space Gaussian resolution operator `G` (`gaussian_blur`).
- `reconstruction.jl` — listmode Poisson MLEM/OSEM built on the projectors.

See the package documentation (`docs/`) for the projection algorithm and
backend details.
"""
module RecoCrysp

using KernelAbstractions
using KernelAbstractions: @atomic
using Random

const KA = KernelAbstractions

# Dependency order: geometry produces LORs; the projectors map image ↔ LOR;
# reconstruction iterates the projectors.
include("pixelated_pet.jl")
include("continuous_pet.jl")
include("phantoms.jl")
include("psf.jl")
include("projectors.jl")
include("reconstruction.jl")
include("penalized.jl")

export joseph3d_fwd!, joseph3d_fwd, joseph3d_back!, joseph3d_back
export RegularPolygonPETScannerGeometry, RegularPolygonPETLORDescriptor,
       get_lor_endpoints, get_lor_coordinates
export ContinuousPET, sample_lors
export uniform_sphere, uniform_cylinder, derenzo, gaussian_blur
export sensitivity_image, ListmodePoissonModel, predicted, neg_log_likelihood,
       em_update, mlem, osem, subset_models
export penalized_mlem, osl_mlem, Prior, NoPrior,
       QuadraticIntensityPrior, QuadraticSmoothnessPrior, HuberPrior

end # module
