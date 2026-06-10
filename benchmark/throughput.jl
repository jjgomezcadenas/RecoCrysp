# Throughput benchmark for the Joseph 3D projectors on CPU and (if available) Metal.
#
# Run from the package root (first time: instantiate the benchmark environment):
#   julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   julia -t auto --project=benchmark benchmark/throughput.jl

using Random
using Printf
using JosephProjectors
using KernelAbstractions

const KA = KernelAbstractions

function make_lors(rng, nlors, radius)
    xs = randn(rng, Float32, 3, nlors)
    xe = randn(rng, Float32, 3, nlors)
    xs .*= radius ./ sqrt.(sum(abs2, xs; dims = 1))
    xe .*= radius ./ sqrt.(sum(abs2, xe; dims = 1))
    return xs, xe
end

centered_origin(n, vs) = @. -(n - 1) * vs / 2

function bench(to_dev, label; n = (128, 128, 128), nlors = 500_000)
    rng = MersenneTwister(3)
    vs = (2.0f0, 2.0f0, 2.0f0)
    org = centered_origin(n, vs)

    img = to_dev(ones(Float32, n))
    xs, xe = make_lors(rng, nlors, 320.0f0)
    d_xs, d_xe = to_dev(xs), to_dev(xe)

    proj = joseph3d_fwd(d_xs, d_xe, img, org, vs)  # warm-up / compile
    t_fwd = @elapsed joseph3d_fwd!(proj, d_xs, d_xe, img, org, vs)

    bimg = similar(img)
    fill!(bimg, 0.0f0)
    joseph3d_back!(bimg, d_xs, d_xe, proj, org, vs)  # warm-up / compile
    fill!(bimg, 0.0f0)
    t_back = @elapsed joseph3d_back!(bimg, d_xs, d_xe, proj, org, vs)

    @printf("%-28s %4d^3 image, %7d LORs: fwd %7.2f Mlors/s, back %7.2f Mlors/s\n",
            label, n[1], nlors, nlors / t_fwd / 1.0e6, nlors / t_back / 1.0e6)
    return nothing
end

println("Julia ", VERSION, " (", Threads.nthreads(), " threads)")
bench(identity, "CPU (KernelAbstractions)")

import Metal
if Metal.functional()
    bench(Metal.MtlArray, "Metal (" * String(Metal.device().name) * ")")
else
    println("Metal not functional — GPU benchmark skipped")
end
