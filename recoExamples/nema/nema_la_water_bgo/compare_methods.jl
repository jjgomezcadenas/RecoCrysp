# Systematic reconstruction-method comparison on the water NEMA phantom. The
# expensive, method-INDEPENDENT work (load 18M events, build the 500M attenuation-
# weighted sensitivity Aᵀ(a), the scatter+randoms models) is done ONCE; the AC
# sensitivity images are cached to out/_cache_sens.npz so later invocations are
# cheap. Then each [[variant]] in the config (or the tags passed as ARGS) is
# reconstructed gold/uncorr/corr on that same data and written to its own
# out/nema_la_water_bgo_<tag>.npz -- nothing overwrites anything else.
#
#   julia -t auto --project=recoExamples recoExamples/nema/nema_la_water_bgo/compare_methods.jl [tag ...]
#     no args -> run every variant in the config; else only the named tags.
# Read by plot.py <tag> (per-variant figure) and compare_plot.py (overlay).

using RecoExamples
using RecoCrysp
using Random
using NPZ
import TOML

cfg = TOML.parsefile(joinpath(@__DIR__, "config.toml"))
to_dev = lowercase(get(get(cfg, "backend", Dict()), "device", "cpu")) == "metal" ?
         (@eval using Metal; x -> MtlArray(x)) : identity

variants = cfg["variant"]
want = isempty(ARGS) ? String[v["tag"] for v in variants] : ARGS
run_set = [v for v in variants if v["tag"] in want]
isempty(run_set) && error("no matching variants for $(want); config has " *
                          "$(join([v["tag"] for v in variants], ", "))")
println("variants to run: $(join([v["tag"] for v in run_set], ", "))"); flush(stdout)

# ---- data + classification ----------------------------------------------------
c = read_coincidences(cfg["data"]["lors"])
tmask = is_true(c); smask = is_scatter(c); rmask = is_random(c)
n_true = count(tmask); n_scat = count(smask); n_rand = count(rmask); n_prompt = length(c)

vs    = ntuple(_ -> Float32(cfg["grid"]["voxsize"]), 3)
n     = Tuple(Int.(cfg["grid"]["n"]))
org   = ntuple(i -> -(n[i] - 1) / 2 * vs[i], 3)
mu_mm = Float32(cfg["attenuation"]["mu_per_cm"]) / 10
R_body = Float32(NEMA_BODY_R_MM); hz_body = Float32(NEMA_BODY_HALF_MM)

a_t   = attenuation_factors(c.xstart[:, tmask], c.xend[:, tmask]; R = R_body, mu = mu_mm, half_z = hz_body)
a_all = attenuation_factors(c.xstart, c.xend; R = R_body, mu = mu_mm, half_z = hz_body)

# ---- AC sensitivity Aᵀ(a): cached (the only expensive, method-independent step) -
mkpath(joinpath(@__DIR__, "out"))
cachef = joinpath(@__DIR__, "out", "_cache_sens.npz")
nsens = Int(cfg["sens"]["n_sample_lors"])
if isfile(cachef)
    base = to_dev(Float32.(npzread(cachef)["base"]))
    println("loaded cached sensitivity ($cachef)"); flush(stdout)
else
    sc = ContinuousPET(diameter = 2 * Float32(cfg["scanner"]["sample_radius_mm"]),
                       afov = Float32(cfg["scanner"]["afov_mm"]))
    gxs, gxe = sample_lors(sc, nsens; rng = MersenneTwister(Int(cfg["sens"]["seed"])))
    a_sens = attenuation_factors(gxs, gxe; R = R_body, mu = mu_mm, half_z = hz_body)
    base = sensitivity_image(to_dev(gxs), to_dev(gxe), n, org, vs; weights = to_dev(a_sens))
    npzwrite(cachef, Dict("base" => Array(base)))
    println("built + cached sensitivity Aᵀ(a) over $nsens LORs"); flush(stdout)
end
sens_gold   = base .* Float32(n_true / nsens)
sens_prompt = base .* Float32(n_prompt / nsens)
x0 = Float32.(base .> 0)

# ---- scatter + randoms contamination (method-independent) ----------------------
# Two parametrizations: the 3-coord (s_r, z_m, dz) model (phi-collapsed; OK for a
# centred phantom) and the 4-coord (s, phi, z_m, dz) model (signed s + azimuth;
# needed to localize scatter under the OFF-centre NEMA spheres). A block selects
# the 4-coord path with use_phi = true in its config; otherwise the 3-coord path.
s_r, z_m, dz = lor_sinogram_coords(c.xstart, c.xend)
s4, phi4, zm4, dz4 = lor_sinogram_coords4(c.xstart, c.xend)
# Listmode contamination SCALE: the additive term must be commensurate with the
# forward term mult*fwd(x) (intensity scale), not the fraction scale. With
# intensity_scale, contamination_i = fraction_i * (mult*fwd(x_uncorr))_i, where the
# uncorrected reconstruction supplies the per-LOR total-intensity proxy (it fits all
# prompts). Without it the term sits ~12x too small (the scatter-scale-check finding)
# and the correction is negligible. Off -> legacy sum=total fraction normalization.
isc = get(cfg["scatter"], "intensity_scale", false)
fterm = if isc
    xu = mlem(ListmodePoissonModel(to_dev(c.xstart), to_dev(c.xend), sens_prompt;
              img_origin = org, voxsize = vs, mult = to_dev(a_all)), x0; niter = 20)
    a_all .* Array(joseph3d_fwd(to_dev(c.xstart), to_dev(c.xend), xu, org, vs))
else
    nothing
end
function bg_model(blk, mask, total)
    g = cfg[blk]
    total = isc ? nothing : total          # intensity-scaled: keep the bare fraction
    if get(g, "use_phi", false)
        background_estimate4(s4, phi4, zm4, dz4, mask;
            n_s = Int(g["n_sr"]), n_phi = Int(g["n_phi"]),
            n_zm = Int(g["n_zm"]), n_dz = Int(g["n_dz"]),
            span_s = (-Float32(g["sr_max_mm"]), Float32(g["sr_max_mm"])),  # SIGNED radius
            span_phi = (0.0f0, Float32(pi)),
            span_zm = (-Float32(g["zm_max_mm"]), Float32(g["zm_max_mm"])),
            span_dz = (-Float32(g["dz_max_mm"]), Float32(g["dz_max_mm"])),
            smooth = (Float64(g["smooth_sr"]), Float64(g["smooth_phi"]),
                      Float64(g["smooth_zm"]), Float64(g["smooth_dz"])),
            total = total === nothing ? nothing : Float64(total), verbose = true)
    else
        background_estimate(s_r, z_m, dz, mask;
            n_sr = Int(g["n_sr"]), n_zm = Int(g["n_zm"]), n_dz = Int(g["n_dz"]),
            span_sr = (0.0f0, Float32(g["sr_max_mm"])),
            span_zm = (-Float32(g["zm_max_mm"]), Float32(g["zm_max_mm"])),
            span_dz = (-Float32(g["dz_max_mm"]), Float32(g["dz_max_mm"])),
            smooth = (Float64(g["smooth_sr"]), Float64(g["smooth_zm"]), Float64(g["smooth_dz"])),
            total = total === nothing ? nothing : Float64(total))
    end
end
frac_s = bg_model("scatter", smask, n_scat)
frac_r = bg_model("randoms", rmask, n_rand)
sscale = Float32(get(cfg["scatter"], "scale", 1.0))    # heuristic multiplier on the scatter term
frac = sscale .* frac_s .+ frac_r
contam = isc ? frac .* fterm : frac    # intensity-scaled vs legacy fraction
# output-tag suffix so a variant never overwrites another corrections setting
ssuf = sscale == 1.0f0 ? "" : "_sc" * replace(string(sscale), "." => "")
psuf = (get(cfg["scatter"], "use_phi", false) ? "_phi" : "") * (isc ? "_isc" : "") * ssuf

# ---- ROIs --------------------------------------------------------------------
smasks = nema_sphere_masks(n, org, vs; shrink_mm = Float64(cfg["roi"]["sphere_shrink_mm"]))
bgmask = nema_background_mask(n, org, vs; r_max_mm = Float64(cfg["roi"]["bg_r_max_mm"]),
                              z_half_mm = Float64(cfg["roi"]["bg_z_half_mm"]))
diam = Float64[d for (d, _) in smasks]
ratio = Float64(NEMA_HOT_RATIO)
roimean(img, m) = sum(img[m]) / count(m)
covbg(img) = (v = Float64[img[i] for i in eachindex(img) if bgmask[i]];
              mu = sum(v) / length(v); sqrt(sum((x - mu)^2 for x in v) / length(v)) / mu)
crc(img) = Float64[(roimean(img, mk) / roimean(img, bgmask) - 1) / (ratio - 1) * 100 for (_, mk) in smasks]
bv(img) = Float64[v for (_, v) in nema_background_variability(img, n, org, vs)]   # NEMA BV%, per sphere size

# ---- per-variant recon: (runrec, post) from the method + params ----------------
default_niter = Int(cfg["recon"]["niter"])
# Build an edge-preserving Prior + its label from a variant's "prior" family
# (rdp/huber/logcosh). Shared by the OSL and BSREM branches.
function build_prior(v, fam)
    if fam == "huber"
        return HuberPrior(Float32(v["beta"]), Float32(v["delta"])),
               "Huber β=$(v["beta"]) δ=$(v["delta"])"
    elseif fam == "rdp"
        return RelativeDifferencePrior(Float32(v["beta"]), Float32(get(v, "gamma", 2.0)),
                                       Float32(get(v, "epsilon", 0.01))),
               "RDP β=$(v["beta"]) γ=$(get(v,"gamma",2.0)) ε=$(get(v,"epsilon",0.01))"
    elseif fam == "logcosh"
        return LogcoshPrior(Float32(v["beta"]), Float32(v["scalar"])),
               "Logcosh β=$(v["beta"]) s=$(v["scalar"])"
    else
        error("unknown prior family $fam for variant $(v["tag"])")
    end
end

# returns (runrec, post, label, clamp_ref). clamp_ref is a Ref holding the mean
# OSL denominator-clamp fraction (diagnostic) for OSL methods, or nothing.
function make_recon(v)
    nit = Int(get(v, "niter", default_niter))
    m = lowercase(v["method"])
    if m == "mlem"
        fwhm = Float64(get(v, "fwhm_mm", 0.0))
        return (model -> mlem(model, x0; niter = nit)), (img -> gaussian_postfilter(img, fwhm, vs)),
               "MLEM + $(fwhm)mm", nothing
    elseif m == "quadratic"
        pr = QuadraticSmoothnessPrior(Float32(v["beta"]))
        return (model -> penalized_mlem(model, x0, pr; niter = nit)), identity,
               "De Pierro quad β=$(v["beta"])", nothing
    elseif m in ("huber", "rdp", "logcosh")
        cf = Ref(0.0)
        pr, lab = build_prior(v, m)
        return (model -> osl_mlem(model, x0, pr; niter = nit, clamp_frac = cf)), identity, lab, cf
    elseif m == "bsrem"
        fam = lowercase(get(v, "prior", "rdp"))
        pr, plab = build_prior(v, fam)
        rel = Float32(get(v, "relax", 1.0)); rg = Float32(get(v, "relax_gamma", 0.1))
        return (model -> bsrem(model, x0, pr; niter = nit, relax = rel, relax_gamma = rg)),
               identity, "BSREM[$plab] relax=$rel γ=$rg", nothing
    else
        error("unknown method $(v["method"]) for variant $(v["tag"])")
    end
end

kz = n[3] ÷ 2 + 1
for v in run_set
    runrec, post, label, clampref = make_recon(v)
    rc(xs, xe, sens, mult; cont = nothing) = post(Array(runrec(
        ListmodePoissonModel(to_dev(xs), to_dev(xe), sens; img_origin = org, voxsize = vs,
            mult = to_dev(mult), contamination = cont === nothing ? nothing : to_dev(cont)))))
    g  = rc(c.xstart[:, tmask], c.xend[:, tmask], sens_gold, a_t)
    u  = rc(c.xstart, c.xend, sens_prompt, a_all)
    cr = rc(c.xstart, c.xend, sens_prompt, a_all; cont = contam)   # clampref now holds corr's value
    crc_g, crc_u, crc_c = crc(g), crc(u), crc(cr)
    bv_g, bv_u, bv_c = bv(g), bv(u), bv(cr)
    cfrac = clampref === nothing ? NaN : clampref[]
    npzwrite(joinpath(@__DIR__, "out", "nema_la_water_bgo_$(v["tag"])$(psuf).npz"), Dict(
        "diam_mm" => diam, "hot_ratio" => ratio,            # tag/label are in the filename
        "crc_gold" => crc_g, "crc_uncorr" => crc_u, "crc_corr" => crc_c,
        "bv_gold" => bv_g, "bv_uncorr" => bv_u, "bv_corr" => bv_c,   # NEMA background variability %
        "cov_gold" => covbg(g), "cov_uncorr" => covbg(u), "cov_corr" => covbg(cr),
        "clamp_frac" => cfrac,
        "slice_gold" => g[:, :, kz], "slice_uncorr" => u[:, :, kz], "slice_corr" => cr[:, :, kz],
        "extent_xy" => Float32((n[1] - 1) / 2 * vs[1])))
    cftxt = isnan(cfrac) ? "" : "  clamp $(round(100cfrac; digits=1))%"
    println("[$(v["tag"])$psuf] $label : corr CRC $(round(crc_c[1]))%..$(round(crc_c[end]))%  " *
            "BV $(round(bv_c[1];digits=1))%..$(round(bv_c[end];digits=1))%  " *
            "(CoV $(round(covbg(cr);digits=3)))$cftxt"); flush(stdout)
end
println("done: $(join([v["tag"] for v in run_set], ", "))")
