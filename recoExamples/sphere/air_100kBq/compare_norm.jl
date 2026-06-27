# Case (a) normalization comparison: build the sensitivity two ways and see how
# much the analytic refinement flattens the radial tilt.
#   (A) surface  : uniform endpoint pairs on a single radius (sample_lors)
#   (B) emission : isotropic emission from the FOV + DOI depth (emission_sens_lors)
# Everything else identical (same trues, grid, iterations); only the sensitivity
# differs. Reconstruct case (a) with each and overlay the radial profiles.
#
#   julia -t auto --project=recoExamples recoExamples/sphere/compare_norm.jl
#
# Writes compare_norm_results.npz (read by compare_norm_plot.py).

using RecoExamples
using RecoCrysp
using Random
using NPZ
import TOML

cfg = TOML.parsefile(joinpath(@__DIR__, "config.toml"))
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
niter = 40   # run to convergence so the surface/emission comparison is fair
nsens = Int(cfg["sens"]["n_sample_lors"])
seed  = Int(cfg["sens"]["seed"])
afov  = Float32(cfg["scanner"]["afov_mm"])

# --- (A) surface sampling at a single radius ------------------------------------
scA = ContinuousPET(diameter = 2 * Float32(cfg["scanner"]["sample_radius_mm"]), afov = afov)
gxsA, gxeA = to_dev.(sample_lors(scA, nsens; rng = MersenneTwister(seed)))
sensA = sensitivity_image(gxsA, gxeA, n, org, vs; scale = nev / nsens)

# --- (B) surface endpoint-pair sampling WITH along-track DOI ---------------------
nm = cfg["norm"]
gxsB, gxeB = surface_doi_lors(nsens;
    r_inner = Float32(nm["r_inner_mm"]), wall = Float32(nm["wall_mm"]),
    halflength = afov / 2, att_length_mm = Float32(nm["att_length_mm"]),
    rng = MersenneTwister(seed))
sensB = sensitivity_image(to_dev(gxsB), to_dev(gxeB), n, org, vs; scale = nev / nsens)

# --- reconstruct with each sensitivity ------------------------------------------
function reco(sens)
    x0 = Float32.(sens .> 0)
    m  = ListmodePoissonModel(xs, xe, sens; img_origin = org, voxsize = vs)
    r  = Array(mlem(m, x0; niter = niter)); sync_dev(); r
end
recA = reco(sensA); println("surface done"); flush(stdout)
recB = reco(sensB); println("emission+DOI done"); flush(stdout)

# --- radial profiles (normalized to interior mean) ------------------------------
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
cov(rec) = (m = sum(rec[inner]) / count(inner); sqrt(sum(abs2, rec[inner] .- m) / count(inner)) / m)
println("interior CoV: surface = ", round(cov(recA); digits = 3),
        "   emission+DOI = ", round(cov(recB); digits = 3))

npzwrite(joinpath(@__DIR__, "compare_norm_results.npz"), Dict(
    "radii" => radii, "radius_mm" => R,
    "prof_surface" => profile(recA), "prof_emission" => profile(recB)))
println("wrote compare_norm_results.npz")
