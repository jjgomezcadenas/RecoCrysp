# NEMA low-activity WATER (full physics: attenuation + 20.5% scatter + 3.2% randoms)
# -- the contrast-phantom payoff where the corrections are VISIBLE. Reconstruct three
# ways, ALL attenuation-corrected (mult = a = exp(-μ·cylinder_chord) through the water
# body; sens = Aᵀ(a)):
#   gold    : true coincidences (truth==0)                         -> target
#   uncorr  : all prompts, AC only, NO scatter/randoms correction  -> contrast washed
#   corr    : all prompts, AC + contamination = scatter + randoms  -> recovered
# Scatter (truth==1) and randoms (truth==2) models are the singles-free smoothed
# sinogram fractions (background_estimate). Recon = De Pierro quadratic-smoothness prior
# (penalized_mlem): monotone, stable under attenuation where OSL Huber diverges.
#
#   julia -t auto --project=recoExamples recoExamples/nema/nema_la_water_bgo/run.jl
# Writes out/nema_la_water_bgo.npz (read by plot.py).

using RecoExamples
using RecoCrysp
using Random
using NPZ
import TOML

cfg = TOML.parsefile(joinpath(@__DIR__, "config.toml"))
to_dev = lowercase(get(get(cfg, "backend", Dict()), "device", "cpu")) == "metal" ?
         (@eval using Metal; x -> MtlArray(x)) : identity

c = read_coincidences(cfg["data"]["lors"])
tmask = is_true(c); smask = is_scatter(c); rmask = is_random(c)
n_true = count(tmask); n_scat = count(smask); n_rand = count(rmask); n_prompt = length(c)
println("prompts $n_prompt: trues $n_true, scatter $n_scat " *
        "($(round(100n_scat/n_prompt; digits=1))%), randoms $n_rand " *
        "($(round(100n_rand/n_prompt; digits=1))%)"); flush(stdout)

vs    = ntuple(_ -> Float32(cfg["grid"]["voxsize"]), 3)
n     = Tuple(Int.(cfg["grid"]["n"]))
org   = ntuple(i -> -(n[i] - 1) / 2 * vs[i], 3)
niter = Int(cfg["recon"]["niter"])
mu_mm = Float32(cfg["attenuation"]["mu_per_cm"]) / 10           # cm^-1 -> mm^-1
R_body = Float32(NEMA_BODY_R_MM); hz_body = Float32(NEMA_BODY_HALF_MM)
println("mu = $(cfg["attenuation"]["mu_per_cm"]) /cm; body cylinder R=$R_body |z|<=$hz_body mm"); flush(stdout)

# per-event attenuation survival a = exp(-mu * cylinder_chord) through the water body
a_t   = attenuation_factors(c.xstart[:, tmask], c.xend[:, tmask]; R = R_body, mu = mu_mm, half_z = hz_body)
a_all = attenuation_factors(c.xstart, c.xend; R = R_body, mu = mu_mm, half_z = hz_body)
println("event attenuation (trues): mean a = $(round(sum(a_t)/length(a_t); digits=3)), " *
        "min $(round(minimum(a_t); digits=3))"); flush(stdout)

# attenuation-weighted sensitivity Aᵀ(a) over the sampled geometric LORs
sc = ContinuousPET(diameter = 2 * Float32(cfg["scanner"]["sample_radius_mm"]),
                   afov = Float32(cfg["scanner"]["afov_mm"]))
nsens = Int(cfg["sens"]["n_sample_lors"])
gxs, gxe = sample_lors(sc, nsens; rng = MersenneTwister(Int(cfg["sens"]["seed"])))
a_sens = attenuation_factors(gxs, gxe; R = R_body, mu = mu_mm, half_z = hz_body)
base = sensitivity_image(to_dev(gxs), to_dev(gxe), n, org, vs; weights = to_dev(a_sens))   # Aᵀ(a)
sens_gold   = base .* Float32(n_true / nsens)
sens_prompt = base .* Float32(n_prompt / nsens)
x0 = Float32.(base .> 0)
println("sensitivity Aᵀ(a) built over $nsens LORs"); flush(stdout)

# scatter + randoms models: smoothed local fractions in sinogram coords, per prompt
s_r, z_m, dz = lor_sinogram_coords(c.xstart, c.xend)
function bg_model(blk, mask, total)
    g = cfg[blk]
    background_estimate(s_r, z_m, dz, mask;
        n_sr = Int(g["n_sr"]), n_zm = Int(g["n_zm"]), n_dz = Int(g["n_dz"]),
        span_sr = (0.0f0, Float32(g["sr_max_mm"])),
        span_zm = (-Float32(g["zm_max_mm"]), Float32(g["zm_max_mm"])),
        span_dz = (-Float32(g["dz_max_mm"]), Float32(g["dz_max_mm"])),
        smooth = (Float64(g["smooth_sr"]), Float64(g["smooth_zm"]), Float64(g["smooth_dz"])),
        total = Float64(total))
end
s_est = bg_model("scatter", smask, n_scat)
r_est = bg_model("randoms", rmask, n_rand)
contam = s_est .+ r_est
println("scatter model sum $(round(sum(s_est)))/$n_scat, randoms model sum " *
        "$(round(sum(r_est)))/$n_rand"); flush(stdout)

# recon method: De Pierro quadratic (default), OSL Huber, or MLEM + post-filter
method = lowercase(get(cfg["recon"], "method", "quadratic"))
if method == "quadratic"
    prior = QuadraticSmoothnessPrior(Float32(cfg["recon"]["quad_beta"]))
    runrec(model) = penalized_mlem(model, x0, prior; niter = niter); post = identity
    rtag = "De Pierro quad β=$(cfg["recon"]["quad_beta"])"
elseif method == "huber"
    prior = HuberPrior(Float32(cfg["recon"]["huber_beta"]), Float32(cfg["recon"]["huber_delta"]))
    runrec(model) = osl_mlem(model, x0, prior; niter = niter); post = identity
    rtag = "Huber β=$(cfg["recon"]["huber_beta"]) δ=$(cfg["recon"]["huber_delta"])"
else
    fwhm = Float64(get(get(cfg, "postfilter", Dict()), "fwhm_mm", 0.0))
    runrec(model) = mlem(model, x0; niter = niter); post = img -> gaussian_postfilter(img, fwhm, vs)
    rtag = "MLEM + $(fwhm)mm post-filter"
end
reco(xs, xe, sens, mult; contam = nothing) = post(Array(runrec(
    ListmodePoissonModel(to_dev(xs), to_dev(xe), sens; img_origin = org, voxsize = vs,
        mult = to_dev(mult),
        contamination = contam === nothing ? nothing : to_dev(contam)))))

rec_gold   = reco(c.xstart[:, tmask], c.xend[:, tmask], sens_gold, a_t)
rec_uncorr = reco(c.xstart, c.xend, sens_prompt, a_all)
rec_corr   = reco(c.xstart, c.xend, sens_prompt, a_all; contam = contam)
println("reconstructed gold / uncorr / corr  [$rtag]"); flush(stdout)

# ROIs: per-sphere VOI means + central background mean, for each recon
smasks = nema_sphere_masks(n, org, vs; shrink_mm = Float64(cfg["roi"]["sphere_shrink_mm"]))
bgmask = nema_background_mask(n, org, vs; r_max_mm = Float64(cfg["roi"]["bg_r_max_mm"]),
                              z_half_mm = Float64(cfg["roi"]["bg_z_half_mm"]))
diam = Float64[d for (d, _) in smasks]
ratio = Float64(NEMA_HOT_RATIO)
roimean(img, m) = sum(img[m]) / count(m)
covbg(img) = (v = Float64[img[i] for i in eachindex(img) if bgmask[i]];
              mu = sum(v) / length(v); sqrt(sum((x - mu)^2 for x in v) / length(v)) / mu)
crc(img) = Float64[(roimean(img, mk) / roimean(img, bgmask) - 1) / (ratio - 1) * 100 for (_, mk) in smasks]
crc_g, crc_u, crc_c = crc(rec_gold), crc(rec_uncorr), crc(rec_corr)
for (lbl, rc, im) in (("gold", crc_g, rec_gold), ("uncorr", crc_u, rec_uncorr), ("corr", crc_c, rec_corr))
    println("  $lbl: bg $(round(roimean(im, bgmask); sigdigits=3)) CoV $(round(covbg(im); digits=3))  " *
            "CRC $(round(rc[1]))%..$(round(rc[end]))%"); flush(stdout)
end

kz = n[3] ÷ 2 + 1
mkpath(joinpath(@__DIR__, "out"))
npzwrite(joinpath(@__DIR__, "out", "nema_la_water_bgo.npz"), Dict(
    "diam_mm" => diam, "hot_ratio" => ratio,
    "crc_gold" => crc_g, "crc_uncorr" => crc_u, "crc_corr" => crc_c,
    "cov_gold" => covbg(rec_gold), "cov_uncorr" => covbg(rec_uncorr), "cov_corr" => covbg(rec_corr),
    "slice_gold" => rec_gold[:, :, kz], "slice_uncorr" => rec_uncorr[:, :, kz],
    "slice_corr" => rec_corr[:, :, kz],
    "extent_xy" => Float32((n[1] - 1) / 2 * vs[1])))
println("wrote out/nema_la_water_bgo.npz")
