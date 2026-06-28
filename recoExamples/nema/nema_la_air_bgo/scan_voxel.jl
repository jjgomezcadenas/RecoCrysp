# Noise vs voxel size: reconstruct NEMA gold across a range of voxel sizes at
# FIXED physical regularization (MLEM niter + fixed-mm Gaussian post-filter, so the
# effective resolution is matched), and measure background CoV and per-sphere CRC.
# Finer voxels represent the ~3.5mm resolution better (CRC of small spheres should
# rise) but get fewer sens hits/voxel (sens noisier) -- both reported. The sens is
# sampled from a fixed nsens LORs; sens-CoV is printed to expose that confound.
#
#   julia -t auto --project=recoExamples recoExamples/nema/nema_la_air_bgo/scan_voxel.jl
# Writes out/nema_la_air_bgo_voxel_scan.npz (read by scan_voxel_plot.py).

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
nsens = Int(cfg["sens"]["n_sample_lors"])
niter = 20; fwhm = 6.0                            # fixed physical regularization
voxels = [1.5f0, 2.0f0, 2.5f0, 3.0f0, 4.0f0]
HALFXY = 112.5; HALFZ = 90.0                      # physical FOV half-extents (mm)

sc = ContinuousPET(diameter = 2 * Float32(cfg["scanner"]["sample_radius_mm"]),
                   afov = Float32(cfg["scanner"]["afov_mm"]))
gxs, gxe = to_dev.(sample_lors(sc, nsens; rng = MersenneTwister(Int(cfg["sens"]["seed"]))))
ratio = Float64(NEMA_HOT_RATIO)
roimean(img, m) = (s = 0.0; cnt = 0; @inbounds for i in eachindex(img); m[i] && (s += img[i]; cnt += 1); end; s / cnt)
covbg(img, m) = (v = Float64[img[i] for i in eachindex(img) if m[i]];
                 mu = sum(v) / length(v); sqrt(sum((x - mu)^2 for x in v) / length(v)) / mu)

diam = Float64[d for (d, _) in nema_sphere_masks((3, 3, 3), (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))]
covs = Float64[]; senscovs = Float64[]; crc_arr = Vector{Float64}[]
for vsz in voxels
    vs = (vsz, vsz, vsz)
    n = (2 * round(Int, HALFXY / vsz) + 1, 2 * round(Int, HALFXY / vsz) + 1, 2 * round(Int, HALFZ / vsz) + 1)
    org = ntuple(i -> -(n[i] - 1) / 2 * vs[i], 3)
    base = sensitivity_image(gxs, gxe, n, org, vs)
    sens = base .* Float32(n_true / nsens)
    x0 = Float32.(base .> 0)
    rec = gaussian_postfilter(Array(mlem(ListmodePoissonModel(xs_t, xe_t, sens;
        img_origin = org, voxsize = vs), x0; niter = niter)), Float64(vsz) > 0 ? fwhm : fwhm, vs)
    bgmask = nema_background_mask(n, org, vs; r_max_mm = 25.0, z_half_mm = 10.0)
    smasks = nema_sphere_masks(n, org, vs)
    crc = Float64[(roimean(rec, mk) / roimean(rec, bgmask) - 1) / (ratio - 1) * 100 for (_, mk) in smasks]
    sc_cov = covbg(Array(base), nema_background_mask(n, org, vs; r_max_mm = 25.0, z_half_mm = 10.0))
    push!(covs, covbg(rec, bgmask)); push!(senscovs, sc_cov); push!(crc_arr, crc)
    println("voxel $(vsz)mm  grid $(n)  bg CoV $(round(covs[end];digits=3))  " *
            "sens CoV $(round(sc_cov;digits=3))  CRC(37/10) $(round(crc[1];digits=0))/$(round(crc[end];digits=0))%"); flush(stdout)
end

mkpath(joinpath(@__DIR__, "out"))
npzwrite(joinpath(@__DIR__, "out", "nema_la_air_bgo_voxel_scan.npz"), Dict(
    "voxel_mm" => Float64.(voxels), "cov" => covs, "sens_cov" => senscovs,
    "diam_mm" => diam, "crc" => reduce(vcat, [reshape(x, 1, :) for x in crc_arr]),
    "niter" => niter, "fwhm_mm" => fwhm))
println("wrote out/nema_la_air_bgo_voxel_scan.npz")
