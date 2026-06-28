# Reconstruct NEMA gold three ways and put them side by side: MLEM + Gaussian
# post-filter, quadratic-smoothness prior (penalized_mlem), and Huber prior
# (osl_mlem). Parameters are chosen to land at a comparable background noise so
# the per-sphere CRC is the comparison. Records central slices, background CoV and
# per-sphere CRC for each.
#
#   julia -t auto --project=recoExamples recoExamples/nema/nema_la_air_bgo/compare_methods.jl
# Writes out/nema_la_air_bgo_methods.npz (read by compare_methods_plot.py).

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
niter = 30

sc = ContinuousPET(diameter = 2 * Float32(cfg["scanner"]["sample_radius_mm"]),
                   afov = Float32(cfg["scanner"]["afov_mm"]))
gxs, gxe = sample_lors(sc, nsens; rng = MersenneTwister(Int(cfg["sens"]["seed"])))
base = sensitivity_image(to_dev(gxs), to_dev(gxe), n, org, vs)
sens_gold = base .* Float32(n_true / nsens)
x0 = Float32.(base .> 0)
model = ListmodePoissonModel(to_dev(c.xstart[:, tmask]), to_dev(c.xend[:, tmask]), sens_gold;
                             img_origin = org, voxsize = vs)

# three reconstructions, parameters chosen for comparable background noise (~CoV 0.4)
gauss = gaussian_postfilter(Array(mlem(model, x0; niter = niter)), 7.0, vs)
quad  = Array(penalized_mlem(model, x0, QuadraticSmoothnessPrior(1000.0f0); niter = niter))
hub   = Array(osl_mlem(model, x0, HuberPrior(1000.0f0, 0.05f0); niter = niter))
println("reconstructed gauss / quadratic / huber (gold, niter $niter)"); flush(stdout)

bgmask = nema_background_mask(n, org, vs; r_max_mm = 25.0, z_half_mm = 10.0)
smasks = nema_sphere_masks(n, org, vs)
diam = Float64[d for (d, _) in smasks]
ratio = Float64(NEMA_HOT_RATIO)
roimean(img, m) = (s = 0.0; cnt = 0; @inbounds for i in eachindex(img); m[i] && (s += img[i]; cnt += 1); end; s / cnt)
function cov_bg(img)
    v = Float64[img[i] for i in eachindex(img) if bgmask[i]]
    m = sum(v) / length(v); sqrt(sum((x - m)^2 for x in v) / length(v)) / m
end
crc(img) = (mn = roimean(img, bgmask); Float64[(roimean(img, mk) / mn - 1) / (ratio - 1) * 100 for (_, mk) in smasks])

kz = n[3] ÷ 2 + 1
mkpath(joinpath(@__DIR__, "out"))
npzwrite(joinpath(@__DIR__, "out", "nema_la_air_bgo_methods.npz"), Dict(
    "diam_mm" => diam,
    "cov" => [cov_bg(gauss), cov_bg(quad), cov_bg(hub)],
    "crc_gauss" => crc(gauss), "crc_quad" => crc(quad), "crc_huber" => crc(hub),
    "slice_gauss" => gauss[:, :, kz], "slice_quad" => quad[:, :, kz], "slice_huber" => hub[:, :, kz],
    "extent_xy" => Float32((n[1] - 1) / 2 * vs[1])))
println("wrote out/nema_la_air_bgo_methods.npz")
for (lab, img) in (("gauss", gauss), ("quad", quad), ("huber", hub))
    println("  $lab: CoV $(round(cov_bg(img); digits=2))  CRC10 $(round(crc(img)[end]; digits=0))%")
end
