# Diagnostic: the post-filter smoothness/contrast tradeoff. Reconstruct the GOLD
# (trues) once at the config niter, then apply a range of Gaussian post-filter
# widths to the SAME image and measure background CoV (noise) and per-sphere CRC
# (contrast) at each. Heavier filter -> smoother background but lower small-sphere
# CRC; this picks the FWHM for a target smoothness at known contrast cost.
#
#   julia -t auto --project=recoExamples recoExamples/nema/nema_la_air_bgo/scan_fwhm.jl
# Writes out/nema_la_air_bgo_fwhm_scan.npz (read by scan_fwhm_plot.py).

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
niter = Int(cfg["recon"]["niter"])
nsens = Int(cfg["sens"]["n_sample_lors"])

sc = ContinuousPET(diameter = 2 * Float32(cfg["scanner"]["sample_radius_mm"]),
                   afov = Float32(cfg["scanner"]["afov_mm"]))
gxs, gxe = sample_lors(sc, nsens; rng = MersenneTwister(Int(cfg["sens"]["seed"])))
base = sensitivity_image(to_dev(gxs), to_dev(gxe), n, org, vs)
sens_gold = base .* Float32(n_true / nsens)
x0 = Float32.(base .> 0)
rec = Array(mlem(ListmodePoissonModel(to_dev(c.xstart[:, tmask]), to_dev(c.xend[:, tmask]),
        sens_gold; img_origin = org, voxsize = vs), x0; niter = niter))
println("reconstructed gold (niter $niter), sweeping post-filter FWHM"); flush(stdout)

bgmask = nema_background_mask(n, org, vs; r_max_mm = 25.0, z_half_mm = 10.0)
smasks = nema_sphere_masks(n, org, vs)
diam = Float64[d for (d, _) in smasks]
ratio = Float64(NEMA_HOT_RATIO)
roimean(img, m) = (s = 0.0; cnt = 0; @inbounds for i in eachindex(img); m[i] && (s += img[i]; cnt += 1); end; s / cnt)
function cov_bg(img)
    v = Float64[img[i] for i in eachindex(img) if bgmask[i]]
    m = sum(v) / length(v); sqrt(sum((x - m)^2 for x in v) / length(v)) / m, m
end

fwhms = collect(0.0:2.0:12.0)
covs = Float64[]; crcs = Vector{Float64}[]
for f in fwhms
    img = gaussian_postfilter(rec, f, vs)
    cv, mn = cov_bg(img)
    crc = Float64[(roimean(img, m) / mn - 1) / (ratio - 1) * 100 for (_, m) in smasks]
    push!(covs, cv); push!(crcs, crc)
    println("FWHM $(lpad(round(Int,f),2)) mm   bg CoV $(round(cv; digits=3))   " *
            "CRC(37/10mm) $(round(crc[1]; digits=0))/$(round(crc[end]; digits=0))%"); flush(stdout)
end

crc_mat = reduce(vcat, [reshape(c, 1, :) for c in crcs])
mkpath(joinpath(@__DIR__, "out"))
npzwrite(joinpath(@__DIR__, "out", "nema_la_air_bgo_fwhm_scan.npz"), Dict(
    "fwhm_mm" => fwhms, "cov" => covs, "diam_mm" => diam, "crc" => crc_mat, "niter" => niter))
println("wrote out/nema_la_air_bgo_fwhm_scan.npz")
