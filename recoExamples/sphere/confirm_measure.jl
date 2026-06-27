# Confirmation: is the empirical/geometric ~9% tilt efficiency, or just the LOR
# measure? Repeat the empirical-sensitivity pipeline with IDEAL events (sphere
# emission, pure geometry, efficiency = 1) instead of the real MC trues. If the
# ~9% ratio survives with efficiency absent from the inputs, it's the measure.
# Overlays the ideal ratio against the real-events ratio (from empirical_sens).
#
#   julia -t auto --project=recoExamples recoExamples/sphere/confirm_measure.jl

using RecoExamples
using RecoCrysp
using Random
using NPZ
import TOML

cfg = TOML.parsefile(joinpath(@__DIR__, "case_a.toml"))
const BACKEND = lowercase(get(get(cfg, "backend", Dict()), "device", "cpu"))
to_dev = BACKEND == "metal" ? (@eval using Metal; x -> MtlArray(x)) : identity

vs  = ntuple(_ -> Float32(cfg["grid"]["voxsize"]), 3)
n   = Tuple(Int.(cfg["grid"]["n"]))
org = ntuple(i -> -(n[i] - 1) / 2 * vs[i], 3)
R   = Float32(cfg["phantom"]["radius_mm"])
nsens = Int(cfg["sens"]["n_sample_lors"])
seed  = Int(cfg["sens"]["seed"])
afov  = Float32(cfg["scanner"]["afov_mm"])
r_det = Float32(cfg["scanner"]["sample_radius_mm"])

# --- IDEAL events: sphere emission, geometry only, no efficiency ----------------
println("generating $nsens ideal events ..."); flush(stdout)
ixs, ixe = ideal_sphere_lors(nsens; sphere_R = R, r_det = r_det,
                             halflength = afov / 2, rng = MersenneTwister(seed))
ixs = to_dev(ixs); ixe = to_dev(ixe)
x_sphere = to_dev(uniform_sphere(n, org, vs; radius = R, value = 1.0))
expected = joseph3d_fwd(ixs, ixe, x_sphere, org, vs)
w = ifelse.(expected .> 10.0f0, 1.0f0 ./ expected, 0.0f0)
sens_ideal = Array(sensitivity_image(ixs, ixe, n, org, vs; weights = w))

# --- geometric surface-pairs sensitivity ----------------------------------------
sc = ContinuousPET(diameter = 2 * r_det, afov = afov)
gxs, gxe = to_dev.(sample_lors(sc, nsens; rng = MersenneTwister(seed)))
sens_geom = Array(sensitivity_image(gxs, gxe, n, org, vs))

# --- radial profile of the ideal/geometric ratio --------------------------------
cx = Float32[org[1] + (i - 1) * vs[1] for i in 1:n[1]]
cy = Float32[org[2] + (j - 1) * vs[2] for j in 1:n[2]]
cz = Float32[org[3] + (k - 1) * vs[3] for k in 1:n[3]]
rr = Float32[sqrt(cx[i]^2 + cy[j]^2 + cz[k]^2) for i in 1:n[1], j in 1:n[2], k in 1:n[3]]
nb = ceil(Int, (R + 5) / vs[1])
radii = Float32[(b - 0.5) * vs[1] for b in 1:nb]
function profile(img)
    p = zeros(Float64, nb); cnt = zeros(Int, nb)
    for idx in eachindex(rr)
        b = floor(Int, rr[idx] / vs[1]) + 1
        b <= nb || continue
        p[b] += img[idx]; cnt[b] += 1
    end
    return p ./ max.(cnt, 1)
end
ratio = sens_ideal ./ ifelse.(sens_geom .> 0, sens_geom, Inf32)
pr = profile(ratio)
nin = floor(Int, (R - 2vs[1]) / vs[1])
pr ./= (sum(pr[1:nin]) / nin)                                  # normalize to interior mean

# carry over the real-events ratio from the previous diagnostic, if present
real_ratio = Float32[]
emp = joinpath(@__DIR__, "empirical_sens_results.npz")
isfile(emp) && (real_ratio = npzread(emp)["prof_ratio"])

npzwrite(joinpath(@__DIR__, "confirm_measure_results.npz"), Dict(
    "radii" => radii, "radius_mm" => R,
    "ratio_ideal" => Float32.(pr), "ratio_real" => real_ratio))
println("wrote confirm_measure_results.npz")
