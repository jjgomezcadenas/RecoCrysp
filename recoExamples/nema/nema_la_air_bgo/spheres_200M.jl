# Reconstruct NEMA gold with the sensitivity sampled at nsens = 200M (the clean-
# background point: bg CoV ~0.10), to show the spheres once the sens Monte-Carlo
# noise is sampled down. MLEM niter=20 + 6mm post-filter, voxel 2.5mm.
#
#   julia -t auto --project=recoExamples recoExamples/nema/nema_la_air_bgo/spheres_200M.jl
# Writes out/nema_la_air_bgo_spheres_200M.npz (read by spheres_200M_plot.py).

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
niter = 20; fwhm = 6.0; nsens = 200_000_000

sc = ContinuousPET(diameter = 2 * Float32(cfg["scanner"]["sample_radius_mm"]),
                   afov = Float32(cfg["scanner"]["afov_mm"]))
gxs, gxe = to_dev.(sample_lors(sc, nsens; rng = MersenneTwister(Int(cfg["sens"]["seed"]))))
base = sensitivity_image(gxs, gxe, n, org, vs)
sens = base .* Float32(n_true / nsens)
x0 = Float32.(base .> 0)
rec = gaussian_postfilter(Array(mlem(ListmodePoissonModel(
    to_dev(c.xstart[:, tmask]), to_dev(c.xend[:, tmask]), sens;
    img_origin = org, voxsize = vs), x0; niter = niter)), fwhm, vs)
println("reconstructed gold at nsens=200M"); flush(stdout)

bgmask = nema_background_mask(n, org, vs; r_max_mm = 25.0, z_half_mm = 10.0)
smasks = nema_sphere_masks(n, org, vs)
diam = Float64[d for (d, _) in smasks]
ratio = Float64(NEMA_HOT_RATIO)
roimean(img, m) = (s = 0.0; cnt = 0; @inbounds for i in eachindex(img); m[i] && (s += img[i]; cnt += 1); end; s / cnt)
mn = roimean(rec, bgmask)
crc = Float64[(roimean(rec, mk) / mn - 1) / (ratio - 1) * 100 for (_, mk) in smasks]
v = Float64[rec[i] for i in eachindex(rec) if bgmask[i]]; mu = sum(v) / length(v)
cov = sqrt(sum((x - mu)^2 for x in v) / length(v)) / mu

kz = n[3] ÷ 2 + 1
mkpath(joinpath(@__DIR__, "out"))
npzwrite(joinpath(@__DIR__, "out", "nema_la_air_bgo_spheres_200M.npz"), Dict(
    "slice" => rec[:, :, kz], "diam_mm" => diam, "crc" => crc, "cov" => cov, "nsens" => nsens,
    "extent_xy" => Float32((n[1] - 1) / 2 * vs[1])))
println("wrote out/nema_la_air_bgo_spheres_200M.npz  (bg CoV $(round(cov;digits=3)))")
