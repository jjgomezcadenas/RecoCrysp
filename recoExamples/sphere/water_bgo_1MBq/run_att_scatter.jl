# Water, attenuation + scatter. On top of the validated AC (run_att_only.jl), add
# the scatter background as the contamination term. Randoms (~1%) are dropped via
# the flag so the only contamination studied is scatter (SS+MS lumped, truth==1).
# Reconstruct three ways, all attenuation-corrected:
#   gold    : true coincidences only (truth==0)             -> target
#   uncorr  : trues+scatter, no scatter correction          -> biased
#   corr    : trues+scatter, contamination = scatter model  -> should -> gold
# The scatter model s_i = S~/P~ is the smoothed local scatter fraction in
# sinogram coordinates (background_estimate), calibrated to the flagged scatter count.
#
#   julia -t auto --project=recoExamples recoExamples/sphere/water_bgo_1MBq/run_att_scatter.jl
# Writes water_bgo_1MBq_att_scatter.npz (read by att_scatter_plot.py).

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
smask = is_scatter(c)
nrmask = tmask .| smask                       # trues+scatter (drop randoms)
n_true = count(tmask); n_scat = count(smask); n_nr = count(nrmask)
println("prompts $(length(c)): trues $n_true, scatter $n_scat " *
        "($(round(100n_scat/n_nr; digits=1))% of trues+scatter), randoms dropped " *
        "$(count(is_random(c)))"); flush(stdout)

vs    = ntuple(_ -> Float32(cfg["grid"]["voxsize"]), 3)
n     = Tuple(Int.(cfg["grid"]["n"]))
org   = ntuple(i -> -(n[i] - 1) / 2 * vs[i], 3)
R     = Float32(cfg["phantom"]["radius_mm"])
niter = Int(cfg["recon"]["niter"])
mu_mm = Float32(cfg["attenuation"]["mu_per_cm"]) / 10

# event subsets (CPU coords)
xs_t, xe_t   = c.xstart[:, tmask], c.xend[:, tmask]
xs_nr, xe_nr = c.xstart[:, nrmask], c.xend[:, nrmask]
scat_in_nr = is_scatter(c)[nrmask]            # which trues+scatter events are scatter

# attenuation survival per event
a_t  = attenuation_factors(xs_t, xe_t; R = R, mu = mu_mm)
a_nr = attenuation_factors(xs_nr, xe_nr; R = R, mu = mu_mm)

# scatter model: smoothed local scatter fraction in sinogram coords, summing to n_scat
s_r, z_m, dz = lor_sinogram_coords(xs_nr, xe_nr)
sg = cfg["scatter"]
s_est = background_estimate(s_r, z_m, dz, scat_in_nr;
                         n_sr = Int(sg["n_sr"]), n_zm = Int(sg["n_zm"]), n_dz = Int(sg["n_dz"]),
                         span_sr = (0.0f0, Float32(sg["sr_max_mm"])),
                         span_zm = (-Float32(sg["zm_max_mm"]), Float32(sg["zm_max_mm"])),
                         span_dz = (-Float32(sg["dz_max_mm"]), Float32(sg["dz_max_mm"])),
                         smooth = (Float64(sg["smooth_sr"]), Float64(sg["smooth_zm"]),
                                   Float64(sg["smooth_dz"])),
                         total = Float64(n_scat))
println("scatter model: sum = $(round(sum(s_est); digits=0)) (target $n_scat), " *
        "mean/event = $(round(sum(s_est)/length(s_est); sigdigits=3))"); flush(stdout)

# AC sensitivity Aᵀ(a): one weighted backprojection, scaled per event count
sc = ContinuousPET(diameter = 2 * Float32(cfg["scanner"]["sample_radius_mm"]),
                   afov = Float32(cfg["scanner"]["afov_mm"]))
nsens = Int(cfg["sens"]["n_sample_lors"])
gxs_c, gxe_c = sample_lors(sc, nsens; rng = MersenneTwister(Int(cfg["sens"]["seed"])))
a_sens = attenuation_factors(gxs_c, gxe_c; R = R, mu = mu_mm)
base = sensitivity_image(to_dev(gxs_c), to_dev(gxe_c), n, org, vs; weights = to_dev(a_sens))
sens_gold = base .* Float32(n_true / nsens)
sens_nr   = base .* Float32(n_nr / nsens)
x0 = Float32.(base .> 0)

# attenuation case: OSL Huber diverges here (Aᵀ(a) small at centre), so use the
# monotone De Pierro quadratic-smoothness prior (penalized_mlem). "huber"/"mlem"
# branches kept for the vacuum/contrast and unregularized paths.
method = lowercase(get(cfg["recon"], "method", "quadratic"))
if method == "quadratic"
    prior = QuadraticSmoothnessPrior(Float32(cfg["recon"]["quad_beta"]))
    runrec(model) = penalized_mlem(model, x0, prior; niter = niter); post = identity
elseif method == "huber"
    prior = HuberPrior(Float32(cfg["recon"]["huber_beta"]), Float32(cfg["recon"]["huber_delta"]))
    runrec(model) = osl_mlem(model, x0, prior; niter = niter); post = identity
else
    fwhm = Float64(get(get(cfg, "postfilter", Dict()), "fwhm_mm", 0.0))
    runrec(model) = mlem(model, x0; niter = niter); post = img -> gaussian_postfilter(img, fwhm, vs)
end
reco(xs, xe, sens, mult; contam = nothing) = post(Array(runrec(
    ListmodePoissonModel(to_dev(xs), to_dev(xe), sens; img_origin = org, voxsize = vs,
                         mult = to_dev(mult),
                         contamination = contam === nothing ? nothing : to_dev(contam)))))

rec_gold   = reco(xs_t, xe_t, sens_gold, a_t)
rec_uncorr = reco(xs_nr, xe_nr, sens_nr, a_nr)
rec_corr   = reco(xs_nr, xe_nr, sens_nr, a_nr; contam = s_est)
println("reconstructed gold / uncorr / corr"); flush(stdout)

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
mkpath(joinpath(@__DIR__, "out"))
npzwrite(joinpath(@__DIR__, "out", "water_bgo_1MBq_att_scatter.npz"), Dict(
    "radii" => radii, "radius_mm" => R,
    "prof_gold" => Float32.(profile(rec_gold)),
    "prof_uncorr" => Float32.(profile(rec_uncorr)),
    "prof_corr" => Float32.(profile(rec_corr)),
    "slice_gold" => rec_gold[:, :, kz],
    "slice_uncorr" => rec_uncorr[:, :, kz], "slice_corr" => rec_corr[:, :, kz],
    "extent" => Float32((n[1] - 1) / 2 * vs[1])))
println("wrote water_bgo_1MBq_att_scatter.npz")
