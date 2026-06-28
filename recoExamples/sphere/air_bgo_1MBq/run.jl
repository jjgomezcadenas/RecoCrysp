# Case (b): randoms. Reconstruct the high-activity vacuum sphere three ways:
#   gold    : true coincidences only (truth==0), no contamination
#   uncorr  : all prompts (trues+randoms), no correction  -> biased
#   corr    : all prompts, contamination = sinogram randoms model
# Vacuum, so mult = 1 and the only physics is the randoms background. The randoms
# model is the smoothed local randoms fraction in sinogram coords (truth==2 flag,
# background_estimate) — singles-free, the same machinery as the NEMA study —
# calibrated so its total equals the number of flagged random coincidences. Locked
# recipe (500 M sens, 2.5 mm grid, Huber).
#
#   julia -t auto --project=recoExamples recoExamples/sphere/air_bgo_1MBq/run.jl
# Writes air_bgo_1MBq.npz (read by plot.py).

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
n_rand = count(is_random(c))
println("prompts $(length(c)), trues $(count(tmask)), randoms $n_rand " *
        "($(round(100n_rand/length(c); digits=2))%)"); flush(stdout)

vs  = ntuple(_ -> Float32(cfg["grid"]["voxsize"]), 3)
n   = Tuple(Int.(cfg["grid"]["n"]))
org = ntuple(i -> -(n[i] - 1) / 2 * vs[i], 3)
R   = Float32(cfg["phantom"]["radius_mm"])
niter = Int(cfg["recon"]["niter"])

# shared geometric sensitivity (vacuum: weights = 1); scale to the prompt count
sc = ContinuousPET(diameter = 2 * Float32(cfg["scanner"]["sample_radius_mm"]),
                   afov = Float32(cfg["scanner"]["afov_mm"]))
nsens = Int(cfg["sens"]["n_sample_lors"])
gxs, gxe = to_dev.(sample_lors(sc, nsens; rng = MersenneTwister(Int(cfg["sens"]["seed"]))))
sens = sensitivity_image(gxs, gxe, n, org, vs; scale = length(c) / nsens)
x0 = Float32.(sens .> 0)

# randoms model per PROMPT event: smoothed local randoms fraction in sinogram
# coords (truth==2), singles-free, calibrated to the flagged random total.
rmask = is_random(c)
s_r, z_m, dz = lor_sinogram_coords(c.xstart, c.xend)
rg = cfg["randoms"]
r = background_estimate(s_r, z_m, dz, rmask;
        n_sr = Int(rg["n_sr"]), n_zm = Int(rg["n_zm"]), n_dz = Int(rg["n_dz"]),
        span_sr = (0.0f0, Float32(rg["sr_max_mm"])),
        span_zm = (-Float32(rg["zm_max_mm"]), Float32(rg["zm_max_mm"])),
        span_dz = (-Float32(rg["dz_max_mm"]), Float32(rg["dz_max_mm"])),
        smooth = (Float64(rg["smooth_sr"]), Float64(rg["smooth_zm"]), Float64(rg["smooth_dz"])),
        total = Float64(n_rand))
println("randoms model: sum = $(round(sum(r); digits=0)) (target $n_rand), " *
        "mean/event = $(round(sum(r)/length(r); sigdigits=2))"); flush(stdout)

# locked recipe: edge-preserving Huber (OSL MAP-EM), NEMA values; "mlem" branch keeps
# the unregularized + post-filter path.
method = lowercase(get(cfg["recon"], "method", "huber"))
if method == "huber"
    prior = HuberPrior(Float32(cfg["recon"]["huber_beta"]), Float32(cfg["recon"]["huber_delta"]))
    runrec(model) = osl_mlem(model, x0, prior; niter = niter); post = identity
else
    fwhm = Float64(get(get(cfg, "postfilter", Dict()), "fwhm_mm", 0.0))
    runrec(model) = mlem(model, x0; niter = niter); post = img -> gaussian_postfilter(img, fwhm, vs)
end
allx, allxe = to_dev(c.xstart), to_dev(c.xend)
reco(xs, xe; contam = nothing) = post(Array(runrec(
    ListmodePoissonModel(xs, xe, sens; img_origin = org, voxsize = vs,
                         contamination = contam === nothing ? nothing : to_dev(contam)))))

rec_gold   = reco(to_dev(c.xstart[:, tmask]), to_dev(c.xend[:, tmask]))
rec_uncorr = reco(allx, allxe)
rec_corr   = reco(allx, allxe; contam = r)
println("reconstructed gold / uncorr / corr [$method]"); flush(stdout)

# radial profiles, each normalized to the sphere-interior mean
cx = Float32[org[1] + (i - 1) * vs[1] for i in 1:n[1]]
cy = Float32[org[2] + (j - 1) * vs[2] for j in 1:n[2]]
cz = Float32[org[3] + (k - 1) * vs[3] for k in 1:n[3]]
rr = Float32[sqrt(cx[i]^2 + cy[j]^2 + cz[k]^2) for i in 1:n[1], j in 1:n[2], k in 1:n[3]]
nb = ceil(Int, (maximum(cx)) / vs[1])
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
    return p ./ (sum(p[1:nin]) / nin)         # normalize to interior mean
end

kz = n[3] ÷ 2 + 1
mkpath(joinpath(@__DIR__, "out"))
npzwrite(joinpath(@__DIR__, "out", "air_bgo_1MBq.npz"), Dict(
    "radii" => radii, "radius_mm" => R,
    "prof_gold" => Float32.(profile(rec_gold)),
    "prof_uncorr" => Float32.(profile(rec_uncorr)),
    "prof_corr" => Float32.(profile(rec_corr)),
    "slice_gold" => rec_gold[:, :, kz],
    "slice_uncorr" => rec_uncorr[:, :, kz], "slice_corr" => rec_corr[:, :, kz],
    "extent" => Float32((n[1] - 1) / 2 * vs[1])))
println("wrote air_bgo_1MBq.npz")
