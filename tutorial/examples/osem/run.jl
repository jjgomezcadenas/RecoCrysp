# OSEM example (tutorial): MLEM vs ordered-subsets EM (OSEM) on a SINGLE sharp
# dataset. The physics is held fixed at what the resolution example already
# validated -- a Derenzo phantom in a water cylinder, sharp data (G = 1),
# attenuation-corrected -- and only the SOLVER changes. We reconstruct the one
# dataset with MLEM (niter iterations) and with OSEM at several subset counts,
# tracking, after every image update, the negative log-likelihood, the error
# against the truth, and the cumulative SOLVE time (metric evaluation excluded).
#
# The story (Part I, Acceleration and Noise):
#   - vs image-update count, OSEM and MLEM overlay (one OSEM update ~ one MLEM
#     iteration), and the error is U-shaped (semi-convergence);
#   - vs wall-clock, OSEM-M reaches the same image ~M times sooner, because one
#     OSEM epoch (M updates) costs about one MLEM iteration.
#
#   julia -t auto --project=tutorial/examples/osem tutorial/examples/osem/run.jl
#
# Writes osem_results.npz next to this file (read by osem_plot.py).

using RecoCrysp
using Random
using Distributions
using NPZ
import TOML

cfg = TOML.parsefile(joinpath(@__DIR__, "osem.toml"))

n  = Tuple(Int.(cfg["grid"]["n"]))
vs = ntuple(_ -> Float32(cfg["grid"]["voxsize"]), 3)
org = ntuple(i -> -(n[i] - 1) / 2 * vs[i], 3)

# --- phantom: Derenzo activity (hot rods) and water-cylinder attenuation map ----
dz = cfg["derenzo"]
x_true = derenzo(n, org, vs; radius = dz["radius"], length = dz["length"],
                 rod_diameters = Float32.(dz["rod_diameters"]),
                 spacing = dz["spacing"], value = 1.0)
mumap = uniform_cylinder(n, org, vs; radius = dz["radius"], length = dz["length"],
                         value = Float32(cfg["phantom"]["mu_water"]))

# --- geometry, attenuation, sensitivity, and ONE sharp dataset ------------------
sc    = ContinuousPET(diameter = cfg["scanner"]["diameter"], afov = cfg["scanner"]["afov"])
nlors = Int(cfg["sim"]["n_lors"])
seed  = Int(cfg["sim"]["seed"])
total = Float32(cfg["sim"]["total_counts"])
niter = Int(cfg["recon"]["niter"])
subsets = Int.(cfg["recon"]["subsets"])
m_show  = Int(cfg["recon"]["m_show"])

xs, xe = sample_lors(sc, nlors; rng = MersenneTwister(seed))
a    = exp.(-joseph3d_fwd(xs, xe, mumap, org, vs))           # attenuation a = exp(-A mu)
sens = sensitivity_image(xs, xe, n, org, vs; weights = a)
x0   = Float32.(sens .> 0)

ybar   = a .* joseph3d_fwd(xs, xe, x_true, org, vs)          # sharp: G = 1
scale  = total / sum(ybar)
counts = Float32[rand(MersenneTwister(seed + 1), Poisson(Float64(l))) for l in scale .* ybar]
model  = ListmodePoissonModel(xs, xe, sens; img_origin = org, voxsize = vs,
                              counts = counts, mult = a)

# --- metrics (evaluated OUTSIDE the timed sections) ------------------------------
# The error is measured against whatever the reconstruction can recover: the
# sharp truth for sharp data (act 1), the resolution-limited image G*x for
# smeared data (act 2). Both are normalized by the mean activity over the rods.
hot  = x_true .> 0
supp = mumap .> 0
gap  = supp .& .!hot
rmean(im) = sum(im[hot]) / max(count(hot), 1)
gmean(im) = sum(im[gap]) / max(count(gap), 1)
contrast(im) = rmean(im) / gmean(im)
function relerr(im, target)
    u = im ./ rmean(im)
    t = target ./ rmean(target)
    sqrt(sum(abs2, u[supp] .- t[supp])) / sqrt(sum(abs2, t[supp]))
end

# --- tracked solvers (em_update is public; we time the SOLVE only) ---------------
# MLEM: one full-data update per step. OSEM: one epoch = one update per subset
# model. After each image update we record (#updates, cumulative solve time,
# full-data NLL, error vs the recoverable target, rod/gap contrast).
# Each tracker also keeps `xbest`, the iterate with the lowest error against the
# target — the early-stopping image, which differs from the final iterate once
# the error turns over.
function track_mlem(m, x0, niter, target)
    x = copy(x0); xbest = copy(x0); rbest = Inf; t = 0.0
    upd = Float64[]; tim = Float64[]; nll = Float64[]; rer = Float64[]; con = Float64[]
    for k in 1:niter
        t += @elapsed (x = em_update(m, x))
        r = relerr(x, target)
        r < rbest && (rbest = r; xbest = copy(x))
        push!(upd, k); push!(tim, t)
        push!(nll, neg_log_likelihood(m, x)); push!(rer, r); push!(con, contrast(x))
    end
    return x, (; upd, tim, nll, rer, con, xbest)
end

function track_osem(models, full, x0, nepochs, M, target)
    x = copy(x0); xbest = copy(x0); rbest = Inf; t = 0.0
    upd = Float64[]; tim = Float64[]; nll = Float64[]; rer = Float64[]; con = Float64[]
    for e in 1:nepochs
        t += @elapsed for m in models           # M updates, ~one MLEM iteration of work
            x = em_update(m, x)
        end
        r = relerr(x, target)
        r < rbest && (rbest = r; xbest = copy(x))
        push!(upd, e * M); push!(tim, t)         # cumulative image-update count
        push!(nll, neg_log_likelihood(full, x)); push!(rer, r); push!(con, contrast(x))
    end
    return x, (; upd, tim, nll, rer, con, xbest)
end

# warm up the kernels so the first timed iteration is not a compile -------------
em_update(model, x0)

# === ACT 1: sharp data — acceleration (MLEM vs OSEM at several subset counts) ===
rec_mlem, h_mlem = track_mlem(model, x0, niter, x_true)

osem_hist = Dict{Int,Any}()
rec_show  = rec_mlem
for M in subsets
    models = subset_models(xs, xe, org, vs, n, M; counts = counts, mult = a)
    em_update(models[1], x0)                      # warm up subset kernel
    recM, hM = track_osem(models, model, x0, niter ÷ M, M, x_true)
    osem_hist[M] = hM
    M == m_show && (global rec_show = recM)
end

# === ACT 2: smeared data — semi-convergence and early stopping =================
# Same geometry, attenuation and sensitivity; only the activity is blurred by the
# library resolution operator G before projection. The recoverable target is now
# G*x_true (band-limited), so MLEM recovers it in a few iterations and then has
# only noise left to fit — the error against G*x turns over (early stopping). We
# reconstruct with MLEM and OSEM(m_show); OSEM reaches the minimum M times sooner.
fwhm   = Float32(cfg["resolution"]["fwhm"])
total2 = Float32(cfg["sim"]["smeared_counts"])           # fewer counts than act 1
x_blur = gaussian_blur(x_true, fwhm, vs)                 # G * x_true
ybar2  = a .* joseph3d_fwd(xs, xe, x_blur, org, vs)
scale2 = total2 / sum(ybar2)
counts2 = Float32[rand(MersenneTwister(seed + 2), Poisson(Float64(l))) for l in scale2 .* ybar2]
model2  = ListmodePoissonModel(xs, xe, sens; img_origin = org, voxsize = vs,
                               counts = counts2, mult = a)

rec_mlem2, h_mlem2 = track_mlem(model2, x0, niter, x_blur)
models2 = subset_models(xs, xe, org, vs, n, m_show; counts = counts2, mult = a)
rec_osem2, h_osem2 = track_osem(models2, model2, x0, niter ÷ m_show, m_show, x_blur)

# --- quantitative summary -------------------------------------------------------
# Report, per solver, the best error against the recoverable target and the SOLVE
# time taken to reach it; the speed-up is the MLEM time over the OSEM time to the
# same-quality image. For the smeared act the best error is an interior MINIMUM
# (semi-convergence) — running past it gets worse.
argmin_(v) = findmin(v)[2]
report(tag, h) = (i = argmin_(h.rer);
    println(rpad(tag, 9), " best rel.err = ", round(h.rer[i]; digits = 3),
            " at ", Int(h.upd[i]), " / ", Int(h.upd[end]), " updates (",
            round(h.tim[i]; digits = 1), " s),  contrast = ", round(h.con[i]; digits = 1)); h.tim[i])

println("ACT 1 — sharp data (acceleration):")
t_mlem = report("MLEM", h_mlem)
for M in subsets
    tM = report("OSEM-$M", osem_hist[M])
    println(rpad("", 9), "  speed-up to its best image vs MLEM: ", round(t_mlem / tM; digits = 1), "x")
end

println("ACT 2 — smeared data (semi-convergence vs G*x):")
i2 = argmin_(h_mlem2.rer)
println("  MLEM   min rel.err = ", round(h_mlem2.rer[i2]; digits = 3), " at ", Int(h_mlem2.upd[i2]),
        " updates; final (", Int(h_mlem2.upd[end]), ") = ", round(h_mlem2.rer[end]; digits = 3),
        "  -> turnover ", h_mlem2.rer[end] > h_mlem2.rer[i2] ? "PRESENT" : "absent")
j2 = argmin_(h_osem2.rer)
tm2 = h_mlem2.tim[i2]; to2 = h_osem2.tim[j2]
println("  OSEM-$m_show min rel.err = ", round(h_osem2.rer[j2]; digits = 3), " at ", Int(h_osem2.upd[j2]),
        " updates; reaches its min ", round(tm2 / to2; digits = 1), "x sooner than MLEM")

# --- dump central transverse slices + convergence histories ---------------------
# act 2 images: the recoverable G*x, and the reconstructions at their early-stop
# minimum and at the (worse) final iteration, to show the cost of going too far.
kz = n[3] ÷ 2 + 1
out = Dict{String,Any}(
    # --- act 1 (sharp / acceleration) ---
    "slice_truth" => x_true[:, :, kz],
    "slice_mlem"  => rec_mlem[:, :, kz],
    "slice_osem"  => rec_show[:, :, kz],
    "extent"  => Float32((n[1] - 1) / 2 * vs[1]),
    "subsets" => collect(subsets),
    "m_show"  => m_show,
    "mlem_upd" => h_mlem.upd, "mlem_tim" => h_mlem.tim,
    "mlem_nll" => h_mlem.nll, "mlem_rer" => h_mlem.rer, "mlem_con" => h_mlem.con,
    # --- act 2 (smeared / early stopping) ---
    "fwhm"            => fwhm,
    "slice_blur"      => x_blur[:, :, kz],
    "slice_mlem2_min" => h_mlem2.xbest[:, :, kz],   # early-stop iterate (min error)
    "slice_mlem2_end" => rec_mlem2[:, :, kz],       # final iterate (worse: overfit)
    "mlem2_upd" => h_mlem2.upd, "mlem2_tim" => h_mlem2.tim, "mlem2_rer" => h_mlem2.rer,
    "osem2_upd" => h_osem2.upd, "osem2_tim" => h_osem2.tim, "osem2_rer" => h_osem2.rer,
)
for M in subsets
    h = osem_hist[M]
    out["osem$(M)_upd"] = h.upd; out["osem$(M)_tim"] = h.tim
    out["osem$(M)_nll"] = h.nll; out["osem$(M)_rer"] = h.rer; out["osem$(M)_con"] = h.con
end
npzwrite(joinpath(@__DIR__, "osem_results.npz"), out)
println("wrote osem_results.npz")
