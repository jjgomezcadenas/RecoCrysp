# Backends and GPU usage

RecoCrysp depends **only on KernelAbstractions.jl** — not on Metal.jl,
CUDA.jl or any other GPU stack. This page explains how backend selection works
and why the package is designed this way.

## How backend selection works

The kernels are written once, in backend-neutral form, with
`KernelAbstractions.@kernel`. The backend decision happens at *call* time:

```julia
backend = KernelAbstractions.get_backend(img)   # inspects the array TYPE
fwd_kernel!(backend)(proj, xstart, xend, img, ...; ndrange = nlors)
```

- `img isa Array` → `CPU()` → the kernel runs as a multithreaded Julia loop
  (start Julia with `-t auto` to use all cores).
- `img isa MtlArray` → `MetalBackend()` → the kernel is JIT-compiled to Metal
  IR by Metal.jl the first time it is called, then cached.
- `img isa CuArray` → `CUDABackend()` → likewise via CUDA.jl.

The GPU-specific machinery lives in the GPU package, not here:
`get_backend(::MtlArray)` is a method *Metal.jl* defines, and the lowering of
`@atomic` to Metal atomic instructions is provided by the `AtomixMetalExt`
extension that Metal.jl loads. RecoCrysp never references any GPU type.

## Using a GPU

Load the GPU package in *your* environment and pass device arrays. Nothing
else changes:

```julia
using RecoCrysp, Metal

d_img = MtlArray(img)
d_xs = MtlArray(xstart)
d_xe = MtlArray(xend)

proj = joseph3d_fwd(d_xs, d_xe, d_img, org, voxsize)   # runs on the GPU
```

All arrays of one call must live on the same device. Results come back as
device arrays; use `Array(proj)` to copy to the host.

On an NVIDIA machine, replace `Metal` with `CUDA` and `MtlArray` with
`CuArray`.

## Why not depend on Metal directly?

- **Platform portability.** A hard Metal dependency would load (or fail to
  load) Metal.jl on every Linux/CUDA machine; a hard CUDA dependency would
  pull a multi-gigabyte artifact stack onto Macs. With the agnostic design,
  each machine's environment pulls only its own GPU stack.
- **Ownership of array types.** The concrete array type is the caller's
  choice; the package is generic library code over `AbstractArray`.
- **Compat isolation.** Metal.jl and CUDA.jl release frequently; the package
  does not need to track their versions.

Metal appears in exactly one place: the *test* environment
(`test/Project.toml`). `Pkg.test()` runs the GPU testsets only where
`Metal.functional()` is true, and skips them elsewhere.

## Precision

All kernels use `Float32`: Apple GPUs do not support `Float64`, and the
original libparallelproj kernels are `float` throughout, so nothing is lost
relative to the C/CUDA implementation. Host-side reductions in the test suite
accumulate in `Float64` for reference comparisons.

## Performance notes

- The back projector accumulates with atomic `Float32` adds; on Apple silicon
  these are native (MSL 3) and fast.
- On unified-memory Macs, host↔device transfers are cheap, but keeping data
  resident as `MtlArray`s across repeated projections is still preferable.
- Measured with `benchmark/throughput.jl` (128³ image, best of 12 timed runs):

  | Backend                  | forward      | back         |
  | ------------------------ | ------------ | ------------ |
  | CPU, M1 Max (8 threads)  | 19.9 Mlors/s | 3.3 Mlors/s  |
  | CPU, M5 Pro (6 threads)  | 74 Mlors/s   | 6.0 Mlors/s  |
  | CPU, M5 Pro (18 threads) | 153 Mlors/s  | 6.1 Mlors/s  |
  | Metal, M1 Max            | 57.3 Mlors/s | 36.0 Mlors/s |
  | Metal, M5 Pro            | 170 Mlors/s  | 171 Mlors/s  |

  The back projector's atomic scatter serializes across CPU threads (so it does
  not speed up with more cores) but parallelizes on the GPU — hence the large
  Metal advantage on back projection.
