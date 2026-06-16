# Throughput benchmark for the Joseph 3D projectors on CPU and (if available) Metal.
#
# Timing is best-of-`nruns`: a single @elapsed around one projector call is
# dominated by kernel launch + synchronize overhead at small LOR counts (a
# 500k-LOR forward finishes in a few ms on the GPU), so we take the minimum over
# several timed runs of a larger workload to measure steady-state throughput.
#
# Run from the package root (first time: instantiate the benchmark environment):
#   julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#   julia -t auto --project=benchmark benchmark/throughput.jl   # -t auto = performance cores
#   julia -t 18   --project=benchmark benchmark/throughput.jl   # all cores (set to your count)

using Random
using Printf
using RecoCrysp
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

function bench(to_dev, label; n = (128, 128, 128), nlors = 2_000_000, nruns = 12)
    rng = MersenneTwister(3)
    vs = (2.0f0, 2.0f0, 2.0f0)
    org = centered_origin(n, vs)

    img = to_dev(ones(Float32, n))
    xs, xe = make_lors(rng, nlors, 320.0f0)
    d_xs, d_xe = to_dev(xs), to_dev(xe)

    proj = joseph3d_fwd(d_xs, d_xe, img, org, vs)  # warm-up / compile
    t_fwd = Inf
    for _ in 1:nruns
        t_fwd = min(t_fwd, @elapsed joseph3d_fwd!(proj, d_xs, d_xe, img, org, vs))
    end

    bimg = similar(img)
    fill!(bimg, 0.0f0)
    joseph3d_back!(bimg, d_xs, d_xe, proj, org, vs)  # warm-up / compile
    t_back = Inf
    for _ in 1:nruns
        fill!(bimg, 0.0f0)
        t_back = min(t_back, @elapsed joseph3d_back!(bimg, d_xs, d_xe, proj, org, vs))
    end

    @printf("%-28s %4d^3 image, %8d LORs: fwd %7.1f Mlors/s, back %7.1f Mlors/s  (best of %d)\n",
            label, n[1], nlors, nlors / t_fwd / 1.0e6, nlors / t_back / 1.0e6, nruns)
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
