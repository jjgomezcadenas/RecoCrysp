# JosephProjectors.jl

Backend-agnostic **Joseph 3D matched forward and back projectors** for
tomographic (PET/SPECT) image reconstruction. A single
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)
kernel source runs multithreaded on the CPU and on GPUs, including Apple Metal.

The projectors are a Julia port of the non-TOF kernels of
[libparallelproj](https://github.com/KUL-recon-lab/libparallelproj)
(Schramm & Thielemans 2023, Apache-2.0).

## The model

The forward projector computes, for each line of response (LOR) ``i`` with
endpoints ``\mathbf{x}_s^{(i)}`` and ``\mathbf{x}_e^{(i)}``,

```math
p_i = \int_{\mathrm{LOR}_i} \tilde{f}(\mathbf{x})\, \mathrm{d}\ell ,
```

where ``\tilde{f}`` is the trilinear interpolation of the voxel image ``f``.
Joseph's method evaluates this integral by stepping through the voxel planes
perpendicular to the principal axis of the ray (the axis with the largest
direction cosine), bilinearly interpolating in each plane, and scaling the
plane sum by the correction factor

```math
c = \frac{\Delta_{\mathrm{dir}}}{\lvert\cos\theta_{\mathrm{dir}}\rvert} ,
```

with ``\Delta_{\mathrm{dir}}`` the voxel size along the principal axis. The
back projector applies the exact transpose of the same weights (atomic
scatter), so the pair satisfies the adjointness identity

```math
\langle A x, y \rangle = \langle x, A^{\mathsf T} y \rangle
```

to machine precision — the property MLEM/OSEM convergence relies on.

## Conventions

- The image is an `(n0, n1, n2)` `AbstractArray{Float32,3}`. Voxel `(0,0,0)`
  is **centered** at world coordinates `img_origin`; voxel sizes are `voxsize`.
- LOR endpoints are `(3, nlors)` matrices in world coordinates.
- All arithmetic is `Float32` (Apple GPUs have no `Float64`), matching the
  original library.

## Quick start

```julia
using JosephProjectors

n = (128, 128, 128)
voxsize = (2.0f0, 2.0f0, 2.0f0)
org = @. -(n - 1) * voxsize / 2     # image box centered at the world origin
img = ones(Float32, n)

xstart = Float32[-300.0; 0.0; 0.0;;]
xend = Float32[300.0; 0.0; 0.0;;]

proj = joseph3d_fwd(xstart, xend, img, org, voxsize)
bimg = joseph3d_back(xstart, xend, proj, n, org, voxsize)
```

See [Backends and GPU usage](backends.md) for running the same code on Metal
or CUDA.

## Validation

The test suite (`Pkg.test()`) verifies on every available backend:

- analytic line integrals through a uniform image (exact to `Float32` eps),
- the adjointness identity (relative deviation ``\sim 10^{-10}``),
- rays that miss the image volume produce exact zeros,
- CPU and GPU results agree (forward projections bitwise identical on Metal).

## Roadmap

- TOF sinogram and listmode projectors (ports of the corresponding
  libparallelproj kernels; the required `erf` device approximation is already
  validated on Metal).
- Scanner geometry / LOR descriptor layer.
- `ChainRulesCore` rules so the projectors can be used in
  Flux/Lux reconstruction pipelines.
