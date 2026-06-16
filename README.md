# RecoCrysp.jl

Fast, backend-agnostic **Joseph 3D matched forward and back projectors** for
tomographic (PET/SPECT) image reconstruction in Julia. A single
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
kernel source runs multithreaded on the CPU and on GPUs — including **Apple
Metal**, which the original C/CUDA library does not support.

This is a Julia port of the non-TOF projectors of
[libparallelproj](https://github.com/KUL-recon-lab/libparallelproj)
(G. Schramm et al., Apache-2.0; see
[Schramm & Thielemans, Front. Nucl. Med. 2023](https://doi.org/10.3389/fnume.2023.1324562)).

## Status

- ✅ Non-TOF Joseph 3D forward projector (`joseph3d_fwd`)
- ✅ Non-TOF Joseph 3D back projector, exact matched adjoint (`joseph3d_back`)
- ✅ Backends: CPU (threads), Apple Metal; CUDA expected to work via the same
  KernelAbstractions code path (untested here)
- 🔜 Planned: TOF sinogram / listmode projectors, scanner geometry layer,
  `ChainRulesCore` adjoints for deep-learning reconstruction

## Installation

```julia
using Pkg
Pkg.develop(path = "path/to/RecoCrysp")
```

## Usage

```julia
using RecoCrysp

n = (128, 128, 128)                    # image dimensions
voxsize = (2.0f0, 2.0f0, 2.0f0)        # mm
org = @. -(n - 1) * voxsize / 2        # world coords of voxel (0,0,0) center
img = ones(Float32, n)

xstart = Float32[-300.0; 0.0; 0.0;;]   # (3, nlors) LOR start points
xend = Float32[300.0; 0.0; 0.0;;]      # (3, nlors) LOR end points

proj = joseph3d_fwd(xstart, xend, img, org, voxsize)          # forward
bimg = joseph3d_back(xstart, xend, proj, n, org, voxsize)     # adjoint
```

To run on the GPU, load a GPU package and pass device arrays — nothing else
changes:

```julia
using Metal   # (or CUDA on an NVIDIA machine)
proj = joseph3d_fwd(MtlArray(xstart), MtlArray(xend), MtlArray(img), org, voxsize)
```

All arithmetic is `Float32` (Apple GPUs have no `Float64`), matching the
original library.

## Validation and performance

`Pkg.test()` checks exact line integrals against analytic values and the
adjointness identity `⟨Ax, y⟩ = ⟨x, Aᵀy⟩` (machine precision, ~1e-10 relative)
on every available backend. Measured with `benchmark/throughput.jl`
(128³ image, best of 12 timed runs):

| Backend                  | forward      | back         |
| ------------------------ | ------------ | ------------ |
| CPU, M1 Max (8 threads)  | 19.9 Mlors/s | 3.3 Mlors/s  |
| CPU, M5 Pro (6 threads)  | 74 Mlors/s   | 6.0 Mlors/s  |
| CPU, M5 Pro (18 threads) | 153 Mlors/s  | 6.1 Mlors/s  |
| Metal GPU, M1 Max        | 57.3 Mlors/s | 36.0 Mlors/s |
| Metal GPU, M5 Pro        | 170 Mlors/s  | 171 Mlors/s  |

## Documentation

- HTML docs: `docs/` (build with `julia --project=docs docs/make.jl`)
- Technical note (algorithm derivation, port design, validation):
  `docs/tex/joseph3d_note.tex`

## License

Apache-2.0, as a derivative work of
[libparallelproj](https://github.com/KUL-recon-lab/libparallelproj).
