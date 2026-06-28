# Where does the background noise come from? Test the sensitivity-sampling
# hypothesis: the sens image Aᵀ(1) is a Monte-Carlo estimate from a FIXED nsens
# LORs, so it carries ~1/sqrt(hits) sampling noise that sits in the denominator of
# every MLEM update and gets amplified. Vary nsens at everything else fixed (voxel,
# niter, post-filter, events) and measure the background CoV. If CoV drops with
# nsens, the sens sampling is a (cheap-to-fix) source; if flat, the noise is the
# intrinsic reconstruction conditioning.
#
#   julia -t auto --project=recoExamples recoExamples/nema/nema_la_air_bgo/scan_nsens.jl
# Writes out/nema_la_air_bgo_nsens_scan.npz (read by scan_nsens_plot.py).

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
xs_t, xe_t = to_dev(c.xstart[:, tmask]), to_dev(c.xend[:, tmask])
vs  = ntuple(_ -> Float32(cfg["grid"]["voxsize"]), 3)
n   = Tuple(Int.(cfg["grid"]["n"]))
org = ntuple(i -> -(n[i] - 1) / 2 * vs[i], 3)
niter = 20; fwhm = 6.0
nsens_list = [20_000_000, 50_000_000, 100_000_000, 200_000_000]

sc = ContinuousPET(diameter = 2 * Float32(cfg["scanner"]["sample_radius_mm"]),
                   afov = Float32(cfg["scanner"]["afov_mm"]))
bgmask = nema_background_mask(n, org, vs; r_max_mm = 25.0, z_half_mm = 10.0)
smasks = nema_sphere_masks(n, org, vs)
diam = Float64[d for (d, _) in smasks]
ratio = Float64(NEMA_HOT_RATIO)
roimean(img, m) = (s = 0.0; cnt = 0; @inbounds for i in eachindex(img); m[i] && (s += img[i]; cnt += 1); end; s / cnt)
covbg(img) = (v = Float64[img[i] for i in eachindex(img) if bgmask[i]];
              mu = sum(v) / length(v); sqrt(sum((x - mu)^2 for x in v) / length(v)) / mu)

covs = Float64[]; senscovs = Float64[]; crc_arr = Vector{Float64}[]
for ns in nsens_list
    gxs, gxe = to_dev.(sample_lors(sc, ns; rng = MersenneTwister(Int(cfg["sens"]["seed"]))))
    base = sensitivity_image(gxs, gxe, n, org, vs)
    sens = base .* Float32(n_true / ns)
    x0 = Float32.(base .> 0)
    rec = gaussian_postfilter(Array(mlem(ListmodePoissonModel(xs_t, xe_t, sens;
        img_origin = org, voxsize = vs), x0; niter = niter)), fwhm, vs)
    crc = Float64[(roimean(rec, mk) / roimean(rec, bgmask) - 1) / (ratio - 1) * 100 for (_, mk) in smasks]
    push!(covs, covbg(rec)); push!(senscovs, covbg(Array(base))); push!(crc_arr, crc)
    println("nsens $(lpad(ns,10))  bg CoV $(round(covs[end];digits=3))  " *
            "sens CoV $(round(senscovs[end];digits=3))  CRC10 $(round(crc[end];digits=0))%"); flush(stdout)
end

mkpath(joinpath(@__DIR__, "out"))
npzwrite(joinpath(@__DIR__, "out", "nema_la_air_bgo_nsens_scan.npz"), Dict(
    "nsens" => nsens_list, "cov" => covs, "sens_cov" => senscovs,
    "diam_mm" => diam, "crc" => reduce(vcat, [reshape(x, 1, :) for x in crc_arr]),
    "niter" => niter, "fwhm_mm" => fwhm))
println("wrote out/nema_la_air_bgo_nsens_scan.npz")
