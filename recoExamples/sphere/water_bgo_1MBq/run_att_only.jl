# Water, attenuation only. Reconstruct the uniform water sphere from TRUE
# coincidences (truth==0) two ways:
#   noac : no attenuation correction (mult=1, sens=Aᵀ(1))   -> attenuated image
#   ac   : attenuation correction    (mult=a, sens=Aᵀ(a))   -> recovers the source
# where a_i = exp(-mu * chord_i) is the analytic survival factor of LOR i through
# the uniform sphere (sphere_chord). Trues only, so the narrow-beam mu is exactly
# the right value to correct with (scatter is removed by the flag, handled later
# in run_att_scatter.jl). Carries the air ±8% LOR-measure tilt as the baseline.
#
#   julia -t auto --project=recoExamples recoExamples/sphere/water_bgo_1MBq/run_att_only.jl
# Writes water_bgo_1MBq_att_only.npz (read by att_only_plot.py).

using RecoExamples
using RecoCrysp
using Random
using NPZ
import TOML

cfg = TOML.parsefile(joinpath(@__DIR__, "config.toml"))
to_dev = lowercase(get(get(cfg, "backend", Dict()), "device", "cpu")) == "metal" ?
         (@eval using Metal; x -> MtlArray(x)) : identity

c = read_coincidences(cfg["data"]["lors"])
tmask = is_true(c)
n_ev = count(tmask)
println("prompts $(length(c)), trues $n_ev " *
        "($(round(100n_ev/length(c); digits=2))%)"); flush(stdout)

vs    = ntuple(_ -> Float32(cfg["grid"]["voxsize"]), 3)
n     = Tuple(Int.(cfg["grid"]["n"]))
org   = ntuple(i -> -(n[i] - 1) / 2 * vs[i], 3)
R     = Float32(cfg["phantom"]["radius_mm"])
niter = Int(cfg["recon"]["niter"])
mu_mm = Float32(cfg["attenuation"]["mu_per_cm"]) / 10           # cm^-1 -> mm^-1
println("mu = $(cfg["attenuation"]["mu_per_cm"]) /cm  ->  exp(-mu*2R) = " *
        "$(round(exp(-mu_mm * 2R); digits=3)) at the sphere centre"); flush(stdout)

# per-event attenuation survival a_i (CPU coords, then to device)
xs_t, xe_t = c.xstart[:, tmask], c.xend[:, tmask]
a_ev = attenuation_factors(xs_t, xe_t; R = R, mu = mu_mm)
println("event attenuation: mean a = $(round(sum(a_ev)/length(a_ev); digits=3)), " *
        "min $(round(minimum(a_ev); digits=3))"); flush(stdout)

# geometric sensitivity LORs (surface sampling, same measure as the air runs);
# weight each by its own chord-attenuation for the AC sensitivity Aᵀ(a)
sc = ContinuousPET(diameter = 2 * Float32(cfg["scanner"]["sample_radius_mm"]),
                   afov = Float32(cfg["scanner"]["afov_mm"]))
nsens = Int(cfg["sens"]["n_sample_lors"])
gxs_c, gxe_c = sample_lors(sc, nsens; rng = MersenneTwister(Int(cfg["sens"]["seed"])))
a_sens = attenuation_factors(gxs_c, gxe_c; R = R, mu = mu_mm)
gxs, gxe = to_dev(gxs_c), to_dev(gxe_c)
sens_noac = sensitivity_image(gxs, gxe, n, org, vs; scale = n_ev / nsens)
sens_ac   = sensitivity_image(gxs, gxe, n, org, vs; weights = to_dev(a_sens),
                              scale = n_ev / nsens)
x0 = Float32.(sens_ac .> 0)

xs_d, xe_d = to_dev(xs_t), to_dev(xe_t)
reco(sens; mult = nothing) = Array(mlem(
    ListmodePoissonModel(xs_d, xe_d, sens; img_origin = org, voxsize = vs,
                         mult = mult === nothing ? nothing : to_dev(mult)),
    x0; niter = niter))

rec_noac = reco(sens_noac)
rec_ac   = reco(sens_ac; mult = a_ev)
println("reconstructed noac / ac"); flush(stdout)

# radial profiles, each normalized to its own sphere-interior mean
cx = Float32[org[1] + (i - 1) * vs[1] for i in 1:n[1]]
cy = Float32[org[2] + (j - 1) * vs[2] for j in 1:n[2]]
cz = Float32[org[3] + (k - 1) * vs[3] for k in 1:n[3]]
rr = Float32[sqrt(cx[i]^2 + cy[j]^2 + cz[k]^2) for i in 1:n[1], j in 1:n[2], k in 1:n[3]]
nb = ceil(Int, maximum(cx) / vs[1])
radii = Float32[(b - 0.5) * vs[1] for b in 1:nb]
nin = floor(Int, (R - 2vs[1]) / vs[1])
function profile(img)
    p = zeros(Float64, nb); cnt = zeros(Int, nb)
    for idx in eachindex(rr)
        b = floor(Int, rr[idx] / vs[1]) + 1
        b <= nb || continue
        p[b] += img[idx]; cnt[b] += 1
    end
    p ./= max.(cnt, 1)
    return p ./ (sum(p[1:nin]) / nin)
end

kz = n[3] ÷ 2 + 1
npzwrite(joinpath(@__DIR__, "water_bgo_1MBq_att_only.npz"), Dict(
    "radii" => radii, "radius_mm" => R, "mu_per_cm" => Float32(cfg["attenuation"]["mu_per_cm"]),
    "prof_noac" => Float32.(profile(rec_noac)),
    "prof_ac" => Float32.(profile(rec_ac)),
    "slice_noac" => rec_noac[:, :, kz], "slice_ac" => rec_ac[:, :, kz],
    "extent" => Float32((n[1] - 1) / 2 * vs[1])))
println("wrote water_bgo_1MBq_att_only.npz")
