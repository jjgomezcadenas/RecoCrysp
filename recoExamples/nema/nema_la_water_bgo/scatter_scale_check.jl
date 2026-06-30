# Diagnostic: is the scatter contamination on the right SCALE for the listmode model?
#
# The listmode Poisson model is  pred_i = mult_i * fwd(x)_i + contamination_i.
# For the scatter correction to matter, contamination_i must be commensurate with
# the forward term mult_i*fwd(x)_i -- at ~20% scatter the ratio mean(contam)/
# mean(mult*fwd) should be ~0.2. If it is orders of magnitude smaller, the
# contamination (normalized to sum = n_scat over EVENTS, a binned-sinogram
# convention) is mis-scaled for a per-event listmode contamination, which would
# explain why the correction is negligible and indifferent to its own structure.
#
#   julia -t auto --project=recoExamples recoExamples/nema/nema_la_water_bgo/scatter_scale_check.jl

using RecoExamples
using RecoCrysp
using Random
using NPZ
import TOML

cfg = TOML.parsefile(joinpath(@__DIR__, "config.toml"))
c = read_coincidences(cfg["data"]["lors"])
tmask = is_true(c); smask = is_scatter(c); rmask = is_random(c)
n_true = count(tmask); n_scat = count(smask); n_rand = count(rmask); n_prompt = length(c)

vs    = ntuple(_ -> Float32(cfg["grid"]["voxsize"]), 3)
n     = Tuple(Int.(cfg["grid"]["n"]))
org   = ntuple(i -> -(n[i] - 1) / 2 * vs[i], 3)
mu_mm = Float32(cfg["attenuation"]["mu_per_cm"]) / 10
R_body = Float32(NEMA_BODY_R_MM); hz_body = Float32(NEMA_BODY_HALF_MM)
a_all = attenuation_factors(c.xstart, c.xend; R = R_body, mu = mu_mm, half_z = hz_body)

# cached AC sensitivity (built by compare_methods.jl)
cachef = joinpath(@__DIR__, "out", "_cache_sens.npz")
isfile(cachef) || error("run compare_methods.jl first to build $cachef")
nsens = Int(cfg["sens"]["n_sample_lors"])
base = Float32.(npzread(cachef)["base"])
sens_prompt = base .* Float32(n_prompt / nsens)
x0 = Float32.(base .> 0)

# a quick activity estimate for the forward term (uncorrected MLEM is enough to set the scale)
m_unc = ListmodePoissonModel(c.xstart, c.xend, sens_prompt; img_origin = org, voxsize = vs, mult = a_all)
x = mlem(m_unc, x0; niter = 20)

# forward term mult*fwd(x) at each prompt event
fwd = joseph3d_fwd(c.xstart, c.xend, x, org, vs)
fterm = a_all .* fwd

# the scatter contamination (3-coord baseline model, normalized to sum = n_scat)
s_r, z_m, dz = lor_sinogram_coords(c.xstart, c.xend)
g = cfg["scatter"]
scat = background_estimate(s_r, z_m, dz, smask;
    n_sr = Int(g["n_sr"]), n_zm = Int(g["n_zm"]), n_dz = Int(g["n_dz"]),
    span_sr = (0.0f0, Float32(g["sr_max_mm"])),
    span_zm = (-Float32(g["zm_max_mm"]), Float32(g["zm_max_mm"])),
    span_dz = (-Float32(g["dz_max_mm"]), Float32(g["dz_max_mm"])),
    smooth = (Float64(g["smooth_sr"]), Float64(g["smooth_zm"]), Float64(g["smooth_dz"])),
    total = Float64(n_scat))

pct(v, p) = sort(v)[clamp(round(Int, p / 100 * length(v)), 1, length(v))]
mf = sum(Float64.(fterm)) / length(fterm)
ms = sum(Float64.(scat)) / length(scat)
println("--- scatter-scale check (nema_la_water_bgo) ---")
println("events: n_prompt=$n_prompt  n_scat=$n_scat  scatter fraction=$(round(n_scat/n_prompt; digits=3))")
println("forward term mult*fwd(x):  mean=$(round(mf; sigdigits=4))  " *
        "median=$(round(pct(fterm,50); sigdigits=4))  p90=$(round(pct(fterm,90); sigdigits=4))")
println("scatter contamination:     mean=$(round(ms; sigdigits=4))  " *
        "median=$(round(pct(scat,50); sigdigits=4))  p90=$(round(pct(scat,90); sigdigits=4))")
println("RATIO mean(contam)/mean(forward) = $(round(ms/mf; sigdigits=4))   " *
        "(expected ~$(round(n_scat/n_prompt; digits=3)) if correctly scaled)")
println("=> if the ratio is ~100x below the scatter fraction, the contamination is mis-scaled.")
