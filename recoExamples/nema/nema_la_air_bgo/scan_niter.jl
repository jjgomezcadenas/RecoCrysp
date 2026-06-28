# Diagnostic: is the NEMA background noise (a) MLEM iteration-amplification, (b) a
# noisy sensitivity image, or (c) a bug? Reconstruct the GOLD (trues) recording the
# background CoV at EVERY iteration, and first report the sensitivity image's own
# background CoV. Reading:
#   - sens CoV high            -> the sensitivity is the problem (undersampled / bug)
#   - recon CoV ~Poisson at it=1, climbing -> classic MLEM amplification (fix: stop/filter)
#   - recon CoV huge already at it=1        -> something else is wrong (bug)
#
#   julia -t auto --project=recoExamples recoExamples/nema/nema_la_air_bgo/scan_niter.jl
# Writes out/nema_la_air_bgo_niter_scan.npz.

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

sc = ContinuousPET(diameter = 2 * Float32(cfg["scanner"]["sample_radius_mm"]),
                   afov = Float32(cfg["scanner"]["afov_mm"]))
gxs, gxe = sample_lors(sc, nsens; rng = MersenneTwister(Int(cfg["sens"]["seed"])))
base = sensitivity_image(to_dev(gxs), to_dev(gxe), n, org, vs)
sens_gold = base .* Float32(n_true / nsens)
x0 = Float32.(base .> 0)

# uniform background ROI (central disk, sphere-free): r < 25 mm, |z| < 10 mm
bgmask = nema_background_mask(n, org, vs; r_max_mm = 25.0, z_half_mm = 10.0)
smasks = nema_sphere_masks(n, org, vs)              # (diam, mask) per hot sphere
diam = Float64[d for (d, _) in smasks]
ratio = Float64(NEMA_HOT_RATIO)
roimean(img, m) = (s = 0.0; cnt = 0; @inbounds for i in eachindex(img); m[i] && (s += img[i]; cnt += 1); end; s / cnt)
function cov_bg(img)
    v = Float64[img[i] for i in eachindex(img) if bgmask[i]]
    m = sum(v) / length(v)
    s = sqrt(sum((x - m)^2 for x in v) / length(v))
    return s / m, m
end

scov, smean = cov_bg(Array(base))
println("sensitivity image: background mean $(round(smean; sigdigits=3)), CoV $(round(scov; digits=3))")
flush(stdout)

# per-iteration: background CoV (noise) AND per-sphere CRC (contrast). The two
# curves cross-diagnose where to stop: CRC climbs as contrast converges, CoV
# climbs as noise grows; the knee is the unregularized stopping point.
iters = Int[]; covs = Float64[]; means = Float64[]
crcs = Vector{Float64}[]                             # one entry/iter: CRC per sphere (%)
model = ListmodePoissonModel(to_dev(c.xstart[:, tmask]), to_dev(c.xend[:, tmask]), sens_gold;
                             img_origin = org, voxsize = vs)
cb = (it, x) -> begin
    xi = Array(x)
    cv, mn = cov_bg(xi)
    crc = Float64[(roimean(xi, m) / mn - 1) / (ratio - 1) * 100 for (_, m) in smasks]
    push!(iters, it); push!(covs, cv); push!(means, mn); push!(crcs, crc)
    println("iter $(lpad(it,2))  bg CoV $(round(cv; digits=3))  CRC(37/10mm) " *
            "$(round(crc[1]; digits=0))/$(round(crc[end]; digits=0))%"); flush(stdout)
end
mlem(model, x0; niter = NITER, callback = cb)

crc_mat = reduce(vcat, [reshape(c, 1, :) for c in crcs])   # (niter, n_spheres)
mkpath(joinpath(@__DIR__, "out"))
npzwrite(joinpath(@__DIR__, "out", "nema_la_air_bgo_niter_scan.npz"), Dict(
    "iters" => iters, "cov" => covs, "mean" => means,
    "diam_mm" => diam, "crc" => crc_mat,
    "sens_cov" => scov, "sens_mean" => smean))
println("wrote out/nema_la_air_bgo_niter_scan.npz")
