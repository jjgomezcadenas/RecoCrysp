# Case (a) diagnostic: does the central radial gradient fill in with iterations
# (under-convergence, normalization fine) or stick (a real sensitivity bias)?
# One load + one sensitivity, then MLEM to 40 snapshotting the normalized radial
# profile at 10/20/40 iterations. Reuses config.toml (grid/geometry/backend/data).
#
#   julia -t auto --project=recoExamples recoExamples/sphere/scan_niter.jl
#
# Writes niter_scan_results.npz (read by scan_niter_plot.py).

using RecoExamples
using RecoCrysp
using Random
using NPZ
import TOML

cfg = TOML.parsefile(joinpath(@__DIR__, "..", "config.toml"))
const BACKEND = lowercase(get(get(cfg, "backend", Dict()), "device", "cpu"))
if BACKEND == "metal"
    @eval using Metal
    to_dev(x)  = MtlArray(x)
    sync_dev() = Metal.synchronize()
else
    to_dev(x)  = x
    sync_dev() = nothing
end
println("backend = ", BACKEND); flush(stdout)

c      = read_coincidences(cfg["data"]["lors"])
xs, xe = to_dev.(endpoints(c, is_true(c)))
nev    = size(xs, 2)

vs  = ntuple(_ -> Float32(cfg["grid"]["voxsize"]), 3)
n   = Tuple(Int.(cfg["grid"]["n"]))
org = ntuple(i -> -(n[i] - 1) / 2 * vs[i], 3)
R   = Float32(cfg["phantom"]["radius_mm"])

sc = ContinuousPET(diameter = 2 * Float32(cfg["scanner"]["sample_radius_mm"]),
                   afov = Float32(cfg["scanner"]["afov_mm"]))
nsens    = Int(cfg["sens"]["n_sample_lors"])
gxs, gxe = to_dev.(sample_lors(sc, nsens; rng = MersenneTwister(Int(cfg["sens"]["seed"]))))
sens     = sensitivity_image(gxs, gxe, n, org, vs; scale = nev / nsens)
x0       = Float32.(sens .> 0)

# radial-profile machinery (normalized to the interior mean), computed once
cx = Float32[org[1] + (i - 1) * vs[1] for i in 1:n[1]]
cy = Float32[org[2] + (j - 1) * vs[2] for j in 1:n[2]]
cz = Float32[org[3] + (k - 1) * vs[3] for k in 1:n[3]]
rr = Float32[sqrt(cx[i]^2 + cy[j]^2 + cz[k]^2) for i in 1:n[1], j in 1:n[2], k in 1:n[3]]
inner = rr .< (R - 2 * vs[1])
nb    = ceil(Int, (R + 5) / vs[1])
radii = Float32[(b - 0.5) * vs[1] for b in 1:nb]
function profile(rec)
    recn = rec ./ (sum(rec[inner]) / count(inner))
    prof = zeros(Float64, nb); cnt = zeros(Int, nb)
    for idx in eachindex(rr)
        b = floor(Int, rr[idx] / vs[1]) + 1
        b <= nb || continue
        prof[b] += recn[idx]; cnt[b] += 1
    end
    return Float32.(prof ./ max.(cnt, 1))
end

# Is the central radial tilt iteration-driven (under-convergence) or fixed? The
# locked-recipe recon (Huber, niter=30) shows ~26% centre depression vs the old
# coarse/under-converged ±8%. Scan UNREGULARIZED MLEM over niter and read off the
# centre/peak ratio at each checkpoint (a Gaussian post-filter is low-pass and
# leaves this radial profile essentially unchanged, so MLEM is the clean probe).
checkpoints = [1, 5, 10, 20, 30, 40]
ninr = floor(Int, (R - 2 * vs[1]) / vs[1])           # interior bins (for the peak)
profs = Dict{Int,Vector{Float32}}()
function snapshot(k, x)
    if k in checkpoints
        sync_dev()
        p = profile(Array(x)); profs[k] = p
        ctr = p[1]; pk = maximum(@view p[1:ninr])
        println("  niter $(lpad(k,2))  centre $(round(ctr; digits=3))  " *
                "peak $(round(pk; digits=3))  centre/peak $(round(ctr/pk; digits=3))"); flush(stdout)
    end
end

model = ListmodePoissonModel(xs, xe, sens; img_origin = org, voxsize = vs)
mlem(model, x0; niter = maximum(checkpoints), callback = snapshot)

out = Dict{String,Any}("radii" => radii, "radius_mm" => R)
for k in checkpoints
    out["prof_$k"] = profs[k]
end
npzwrite(joinpath(@__DIR__, "niter_scan_results.npz"), out)
println("wrote niter_scan_results.npz")
