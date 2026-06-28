# Conditioning test: does the quadratic SMOOTHNESS prior BOUND the noise as
# iterations grow? Reconstruct NEMA gold with penalized_mlem(QuadraticSmoothnessPrior(beta))
# for a range of beta (beta=0 == MLEM), recording background CoV and per-sphere CRC
# at EVERY iteration. MLEM's CoV grows linearly without bound; if the regularized
# CoV plateaus while CRC is preserved, the noise was conditioning-limited and the
# regularizer recovers a clean FULL-RESOLUTION image (no post-filter resolution trade).
#
#   julia -t auto --project=recoExamples recoExamples/nema/nema_la_air_bgo/scan_penalized.jl
# Writes out/nema_la_air_bgo_penalized_scan.npz (read by scan_penalized_plot.py).

using RecoExamples
using RecoCrysp
using Random
using NPZ
import TOML

cfg = TOML.parsefile(joinpath(@__DIR__, "config.toml"))
to_dev = lowercase(get(get(cfg, "backend", Dict()), "device", "cpu")) == "metal" ?
         (@eval using Metal; x -> MtlArray(x)) : identity

c = read_coincidences(cfg["data"]["lors"])
tmask = is_true(c); n_true = count(tmask)
vs  = ntuple(_ -> Float32(cfg["grid"]["voxsize"]), 3)
n   = Tuple(Int.(cfg["grid"]["n"]))
org = ntuple(i -> -(n[i] - 1) / 2 * vs[i], 3)
nsens = Int(cfg["sens"]["n_sample_lors"])
NITER = 32
betas = Float64.(get(get(cfg, "penalized", Dict()), "betas", [0.0, 100.0, 300.0, 1000.0]))

sc = ContinuousPET(diameter = 2 * Float32(cfg["scanner"]["sample_radius_mm"]),
                   afov = Float32(cfg["scanner"]["afov_mm"]))
gxs, gxe = sample_lors(sc, nsens; rng = MersenneTwister(Int(cfg["sens"]["seed"])))
base = sensitivity_image(to_dev(gxs), to_dev(gxe), n, org, vs)
sens_gold = base .* Float32(n_true / nsens)
x0 = Float32.(base .> 0)
model = ListmodePoissonModel(to_dev(c.xstart[:, tmask]), to_dev(c.xend[:, tmask]), sens_gold;
                             img_origin = org, voxsize = vs)
println("sens background-scale mean ~ used to gauge beta; sweeping betas = $betas"); flush(stdout)

bgmask = nema_background_mask(n, org, vs; r_max_mm = 25.0, z_half_mm = 10.0)
smasks = nema_sphere_masks(n, org, vs)
diam = Float64[d for (d, _) in smasks]
ratio = Float64(NEMA_HOT_RATIO)
roimean(img, m) = (s = 0.0; cnt = 0; @inbounds for i in eachindex(img); m[i] && (s += img[i]; cnt += 1); end; s / cnt)
function cov_bg(img)
    v = Float64[img[i] for i in eachindex(img) if bgmask[i]]
    m = sum(v) / length(v); sqrt(sum((x - m)^2 for x in v) / length(v)) / m
end

cov_mat = zeros(length(betas), NITER)
crc_arr = zeros(length(betas), NITER, length(diam))
for (bi, b) in enumerate(betas)
    prior = b == 0 ? NoPrior() : QuadraticSmoothnessPrior(Float32(b))
    cb = (it, x) -> begin
        xi = Array(x); mn = roimean(xi, bgmask)
        cov_mat[bi, it] = cov_bg(xi)
        for (si, (_, mk)) in enumerate(smasks)
            crc_arr[bi, it, si] = (roimean(xi, mk) / mn - 1) / (ratio - 1) * 100
        end
    end
    penalized_mlem(model, x0, prior; niter = NITER, callback = cb)
    println("beta $(lpad(round(Int,b),4))  CoV(it1/it16/it32) " *
            "$(round(cov_mat[bi,1];digits=2))/$(round(cov_mat[bi,16];digits=2))/$(round(cov_mat[bi,end];digits=2))  " *
            "CRC10(it32) $(round(crc_arr[bi,end,end];digits=0))%"); flush(stdout)
end

mkpath(joinpath(@__DIR__, "out"))
npzwrite(joinpath(@__DIR__, "out", "nema_la_air_bgo_penalized_scan.npz"), Dict(
    "betas" => betas, "iters" => collect(1:NITER), "diam_mm" => diam,
    "cov" => cov_mat, "crc" => crc_arr))
println("wrote out/nema_la_air_bgo_penalized_scan.npz")
