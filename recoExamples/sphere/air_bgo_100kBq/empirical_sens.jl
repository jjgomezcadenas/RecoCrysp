# Diagnostic: is the case-(a) ~8% radial tilt real detector efficiency, or not?
# Build TWO sensitivities and compare their radial SHAPES (no reconstruction, so
# no circularity):
#   geometric : Aᵀ(1) over surface-sampled LORs  -- assumes efficiency = 1
#   empirical : back-project the true events weighted by 1/(chord through the
#               known sphere) -- this divides the source out and leaves Aᵀ(efficiency)
# Their ratio vs radius IS the efficiency shape the geometry assumes away. If it
# is ~flat, the 8% is not efficiency; if it carries ~8% radial structure, it is.
#
#   julia -t auto --project=recoExamples recoExamples/sphere/empirical_sens.jl
# Writes empirical_sens_results.npz (read by empirical_sens_plot.py).

using RecoExamples
using RecoCrysp
using Random
using NPZ
import TOML

cfg = TOML.parsefile(joinpath(@__DIR__, "config.toml"))
const BACKEND = lowercase(get(get(cfg, "backend", Dict()), "device", "cpu"))
if BACKEND == "metal"
    @eval using Metal
    to_dev(x) = MtlArray(x)
else
    to_dev(x) = x
end

c      = read_coincidences(cfg["data"]["lors"])
xs, xe = to_dev.(endpoints(c, is_true(c)))
vs  = ntuple(_ -> Float32(cfg["grid"]["voxsize"]), 3)
n   = Tuple(Int.(cfg["grid"]["n"]))
org = ntuple(i -> -(n[i] - 1) / 2 * vs[i], 3)
R   = Float32(cfg["phantom"]["radius_mm"])

# --- empirical: weight each true event by 1 / chord-through-the-known-sphere ----
x_sphere = to_dev(uniform_sphere(n, org, vs; radius = R, value = 1.0))
expected = joseph3d_fwd(xs, xe, x_sphere, org, vs)          # chord (mm) per event
thr = 10.0f0                                                # drop grazing LORs (short chord)
w   = ifelse.(expected .> thr, 1.0f0 ./ expected, 0.0f0)
kept = Int(sum(Array(w) .> 0))
println("kept $kept / $(length(expected)) true events (chord > $thr mm)")
sens_emp = Array(sensitivity_image(xs, xe, n, org, vs; weights = w))

# --- geometric: Aᵀ(1) over surface-sampled LORs --------------------------------
sc = ContinuousPET(diameter = 2 * Float32(cfg["scanner"]["sample_radius_mm"]),
                   afov = Float32(cfg["scanner"]["afov_mm"]))
nsens = Int(cfg["sens"]["n_sample_lors"])
gxs, gxe = to_dev.(sample_lors(sc, nsens; rng = MersenneTwister(Int(cfg["sens"]["seed"]))))
sens_geom = Array(sensitivity_image(gxs, gxe, n, org, vs))

# --- radial profiles + their ratio (the efficiency shape) ----------------------
cx = Float32[org[1] + (i - 1) * vs[1] for i in 1:n[1]]
cy = Float32[org[2] + (j - 1) * vs[2] for j in 1:n[2]]
cz = Float32[org[3] + (k - 1) * vs[3] for k in 1:n[3]]
rr = Float32[sqrt(cx[i]^2 + cy[j]^2 + cz[k]^2) for i in 1:n[1], j in 1:n[2], k in 1:n[3]]
nb    = ceil(Int, (R + 5) / vs[1])
inner = rr .< (R - 2 * vs[1])
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
pe = profile(sens_emp); pg = profile(sens_geom)
ratio = sens_emp ./ ifelse.(sens_geom .> 0, sens_geom, Inf32)
pr = profile(ratio)
# normalize each profile to its interior mean so flat -> 1
norm_in(p) = (m = sum(p[1:floor(Int, (R - 2vs[1]) / vs[1])]) / floor(Int, (R - 2vs[1]) / vs[1]); p ./ m)

npzwrite(joinpath(@__DIR__, "empirical_sens_results.npz"), Dict(
    "radii" => radii, "radius_mm" => R,
    "prof_emp" => Float32.(norm_in(pe)),
    "prof_geom" => Float32.(norm_in(pg)),
    "prof_ratio" => Float32.(norm_in(pr))))
println("wrote empirical_sens_results.npz")
