# Noise vs counts: is the residual background noise (after regularization)
# statistics-limited? Reconstruct NEMA gold at FIXED regularization (MLEM niter +
# fixed Gaussian post-filter, so the effective resolution is fixed) while
# subsampling the true coincidences to fractions of N. Measure the residual
# background CoV and per-sphere CRC at each N. If CoV scales as 1/sqrt(N) the
# floor is Poisson statistics (more events is the lever); a flatter slope would
# point elsewhere. CRC should stay ~constant (resolution fixed by the filter).
#
#   julia -t auto --project=recoExamples recoExamples/nema/nema_la_air_bgo/scan_counts.jl
# Writes out/nema_la_air_bgo_counts_scan.npz (read by scan_counts_plot.py).

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
tidx = findall(tmask); ntot = length(tidx)
perm = randperm(MersenneTwister(7), ntot)        # shuffle for unbiased subsets
vs  = ntuple(_ -> Float32(cfg["grid"]["voxsize"]), 3)
n   = Tuple(Int.(cfg["grid"]["n"]))
org = ntuple(i -> -(n[i] - 1) / 2 * vs[i], 3)
nsens = Int(cfg["sens"]["n_sample_lors"])
niter = 20; fwhm = 6.0                           # FIXED regularization across all N
fractions = [0.125, 0.25, 0.5, 1.0]

sc = ContinuousPET(diameter = 2 * Float32(cfg["scanner"]["sample_radius_mm"]),
                   afov = Float32(cfg["scanner"]["afov_mm"]))
gxs, gxe = sample_lors(sc, nsens; rng = MersenneTwister(Int(cfg["sens"]["seed"])))
base = sensitivity_image(to_dev(gxs), to_dev(gxe), n, org, vs)
x0 = Float32.(base .> 0)

bgmask = nema_background_mask(n, org, vs; r_max_mm = 25.0, z_half_mm = 10.0)
smasks = nema_sphere_masks(n, org, vs)
diam = Float64[d for (d, _) in smasks]
ratio = Float64(NEMA_HOT_RATIO)
roimean(img, m) = (s = 0.0; cnt = 0; @inbounds for i in eachindex(img); m[i] && (s += img[i]; cnt += 1); end; s / cnt)
function cov_bg(img)
    v = Float64[img[i] for i in eachindex(img) if bgmask[i]]
    m = sum(v) / length(v); sqrt(sum((x - m)^2 for x in v) / length(v)) / m
end

Ns = Int[]; covs = Float64[]; crc_arr = Vector{Float64}[]
for f in fractions
    ns = round(Int, f * ntot)
    idx = tidx[perm[1:ns]]
    sens = base .* Float32(ns / nsens)
    rec = gaussian_postfilter(Array(mlem(ListmodePoissonModel(
        to_dev(c.xstart[:, idx]), to_dev(c.xend[:, idx]), sens;
        img_origin = org, voxsize = vs), x0; niter = niter)), fwhm, vs)
    cv = cov_bg(rec)
    crc = Float64[(roimean(rec, mk) / roimean(rec, bgmask) - 1) / (ratio - 1) * 100 for (_, mk) in smasks]
    push!(Ns, ns); push!(covs, cv); push!(crc_arr, crc)
    println("N $(lpad(ns,9))  ($(round(100f;digits=1))%)  bg CoV $(round(cv;digits=3))  " *
            "CRC(37/10) $(round(crc[1];digits=0))/$(round(crc[end];digits=0))%"); flush(stdout)
end

mkpath(joinpath(@__DIR__, "out"))
npzwrite(joinpath(@__DIR__, "out", "nema_la_air_bgo_counts_scan.npz"), Dict(
    "N" => Ns, "cov" => covs, "diam_mm" => diam,
    "crc" => reduce(vcat, [reshape(c, 1, :) for c in crc_arr]),
    "niter" => niter, "fwhm_mm" => fwhm))
println("wrote out/nema_la_air_bgo_counts_scan.npz")
