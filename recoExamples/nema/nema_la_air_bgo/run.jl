# NEMA low-activity (vacuum, 3.07 MBq, 2.9% randoms) -- contrast phantom, randoms
# study. Vacuum => n=1, mult=1; reconstruct three ways:
#   gold    : true coincidences (truth==0)               -> target
#   uncorr  : all prompts (trues+randoms), no correction -> contrast washed by randoms
#   corr    : all prompts, contamination = randoms model (truth==2, sinogram)
# Measures the per-sphere contrast-recovery coefficient (CRC) for each. The randoms
# model uses the flag, no singles (background_estimate, src/background.jl).
#
#   julia -t auto --project=recoExamples recoExamples/nema/nema_la_air_bgo/run.jl
# Writes out/nema_la_air_bgo.npz (read by plot.py).

using RecoExamples
using RecoCrysp
using Random
using NPZ
import TOML

cfg = TOML.parsefile(joinpath(@__DIR__, "config.toml"))
to_dev = lowercase(get(get(cfg, "backend", Dict()), "device", "cpu")) == "metal" ?
         (@eval using Metal; x -> MtlArray(x)) : identity

c = read_coincidences(cfg["data"]["lors"])
tmask = is_true(c); rmask = is_random(c)
n_true = count(tmask); n_rand = count(rmask); n_prompt = length(c)
println("prompts $n_prompt: trues $n_true, randoms $n_rand " *
        "($(round(100n_rand/n_prompt; digits=2))%)"); flush(stdout)

vs    = ntuple(_ -> Float32(cfg["grid"]["voxsize"]), 3)
n     = Tuple(Int.(cfg["grid"]["n"]))
org   = ntuple(i -> -(n[i] - 1) / 2 * vs[i], 3)
niter = Int(cfg["recon"]["niter"])

# geometric sensitivity (vacuum: weights = 1); scale per event count
sc = ContinuousPET(diameter = 2 * Float32(cfg["scanner"]["sample_radius_mm"]),
                   afov = Float32(cfg["scanner"]["afov_mm"]))
nsens = Int(cfg["sens"]["n_sample_lors"])
gxs, gxe = sample_lors(sc, nsens; rng = MersenneTwister(Int(cfg["sens"]["seed"])))
base = sensitivity_image(to_dev(gxs), to_dev(gxe), n, org, vs)        # Aᵀ(1)
sens_gold   = base .* Float32(n_true / nsens)
sens_prompt = base .* Float32(n_prompt / nsens)
x0 = Float32.(base .> 0)

# randoms model: smoothed local randoms fraction in sinogram coords, sum = n_rand
s_r, z_m, dz = lor_sinogram_coords(c.xstart, c.xend)
rg = cfg["randoms"]
r_est = background_estimate(s_r, z_m, dz, rmask;
            n_sr = Int(rg["n_sr"]), n_zm = Int(rg["n_zm"]), n_dz = Int(rg["n_dz"]),
            span_sr = (0.0f0, Float32(rg["sr_max_mm"])),
            span_zm = (-Float32(rg["zm_max_mm"]), Float32(rg["zm_max_mm"])),
            span_dz = (-Float32(rg["dz_max_mm"]), Float32(rg["dz_max_mm"])),
            smooth = (Float64(rg["smooth_sr"]), Float64(rg["smooth_zm"]), Float64(rg["smooth_dz"])),
            total = Float64(n_rand))
println("randoms model: sum = $(round(sum(r_est); digits=0)) (target $n_rand)"); flush(stdout)

allx, allxe = to_dev(c.xstart), to_dev(c.xend)
# Reconstruction method: "huber" (edge-preserving OSL MAP-EM, the winner of the
# frontier comparison) or "mlem" (+ Gaussian post-filter). Both go through one
# `reco` that builds the model (carrying the randoms contamination) and runs it.
method = lowercase(get(cfg["recon"], "method", "huber"))
if method == "huber"
    prior = HuberPrior(Float32(cfg["recon"]["huber_beta"]), Float32(cfg["recon"]["huber_delta"]))
    runrec(model) = osl_mlem(model, x0, prior; niter = niter)
    post = identity
    tag = "Huber β=$(cfg["recon"]["huber_beta"]) δ=$(cfg["recon"]["huber_delta"])"
else
    fwhm = Float64(get(get(cfg, "postfilter", Dict()), "fwhm_mm", 0.0))
    runrec(model) = mlem(model, x0; niter = niter)
    post = img -> gaussian_postfilter(img, fwhm, vs)
    tag = "MLEM + $(fwhm)mm post-filter"
end
reco(xs, xe, sens; contam = nothing) = post(Array(runrec(
    ListmodePoissonModel(xs, xe, sens; img_origin = org, voxsize = vs,
        contamination = contam === nothing ? nothing : to_dev(contam)))))
rec_gold   = reco(to_dev(c.xstart[:, tmask]), to_dev(c.xend[:, tmask]), sens_gold)
rec_uncorr = reco(allx, allxe, sens_prompt)
rec_corr   = reco(allx, allxe, sens_prompt; contam = r_est)
println("reconstructed gold / uncorr / corr  ($tag)"); flush(stdout)

# ROIs: per-sphere VOI means + central background mean for each recon
smasks = nema_sphere_masks(n, org, vs; shrink_mm = Float64(cfg["roi"]["sphere_shrink_mm"]))
bgmask = nema_background_mask(n, org, vs; r_max_mm = Float64(cfg["roi"]["bg_r_max_mm"]),
                              z_half_mm = Float64(cfg["roi"]["bg_z_half_mm"]))
diam = Float64[d for (d, _) in smasks]
roimean(img, m) = sum(img[m]) / count(m)
sph_g = Float64[roimean(rec_gold, m) for (_, m) in smasks]
sph_u = Float64[roimean(rec_uncorr, m) for (_, m) in smasks]
sph_c = Float64[roimean(rec_corr, m) for (_, m) in smasks]
bg_g, bg_u, bg_c = roimean(rec_gold, bgmask), roimean(rec_uncorr, bgmask), roimean(rec_corr, bgmask)

kz = n[3] ÷ 2 + 1
mkpath(joinpath(@__DIR__, "out"))
npzwrite(joinpath(@__DIR__, "out", "nema_la_air_bgo.npz"), Dict(
    "diam_mm" => diam, "hot_ratio" => Float64(NEMA_HOT_RATIO),
    "sph_gold" => sph_g, "sph_uncorr" => sph_u, "sph_corr" => sph_c,
    "bg_gold" => bg_g, "bg_uncorr" => bg_u, "bg_corr" => bg_c,
    "slice_gold" => rec_gold[:, :, kz], "slice_uncorr" => rec_uncorr[:, :, kz],
    "slice_corr" => rec_corr[:, :, kz],
    "extent_xy" => Float32((n[1] - 1) / 2 * vs[1])))
println("wrote out/nema_la_air_bgo.npz")
