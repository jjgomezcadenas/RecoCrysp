# Case (a): reconstruct a uniform sphere from PTCRYSP Monte-Carlo listmode using
# the library's analytic geometric sensitivity. In VACUUM (the intended dataset)
# there is no attenuation and no scatter, so reconstructing the true coincidences
# with n=1 should yield a FLAT sphere — the test of the normalization and of the
# analytic/MC geometry match. The reconstruction code is the generic listmode
# MLEM; only the data and the sensitivity come from the MC side.
#
#   julia -t auto --project=recoExamples recoExamples/sphere/run_case_a.jl
#
# Writes case_a_results.npz next to this file (read by case_a_plot.py).

using RecoExamples
using RecoCrysp
using Random
using NPZ
import TOML

cfg  = TOML.parsefile(joinpath(@__DIR__, "case_a.toml"))
lors = cfg["data"]["lors"]
isfile(lors) || error("listmode file not found:\n  $lors\n" *
                      "Generate the vacuum sphere first (case (a) needs vacuum/air).")

# --- backend: cpu or metal. The reader stays CPU-side; only the endpoint and
# sensitivity arrays move to the device, and the (backend-agnostic) projectors
# dispatch on the array type. `to_dev` moves an array; `sync_dev` blocks for
# honest per-iteration GPU timing (a no-op on CPU).
const BACKEND = lowercase(get(get(cfg, "backend", Dict()), "device", "cpu"))
if BACKEND == "metal"
    @eval using Metal
    to_dev(x)  = MtlArray(x)
    sync_dev() = Metal.synchronize()
else
    to_dev(x)  = x
    sync_dev() = nothing
end
println("backend = ", BACKEND); flush(stdout)

# --- load and select the true coincidences (vacuum has no scatter; drop randoms)
t0 = time()
c = read_coincidences(lors)
mask   = is_true(c)
xs, xe = to_dev.(endpoints(c, mask))
nev    = size(xs, 2)
println("loaded $(length(c)) coincidences; using $nev true ",
        "($(round(time() - t0; digits = 1)) s)"); flush(stdout)

# --- grid centred on the sphere -------------------------------------------------
vs  = ntuple(_ -> Float32(cfg["grid"]["voxsize"]), 3)
n   = Tuple(Int.(cfg["grid"]["n"]))
org = ntuple(i -> -(n[i] - 1) / 2 * vs[i], 3)
R   = Float32(cfg["phantom"]["radius_mm"])
x_true = uniform_sphere(n, org, vs; radius = R, value = 1.0)

# --- analytic geometric sensitivity over INDEPENDENT sampled LORs ---------------
# vacuum: weights = 1 (n = 1, a = 1). The event and sensitivity LOR sets differ,
# so rescale the sensitivity to the event count (the decoupled `scale` path).
t0 = time()
sc = ContinuousPET(diameter = 2 * Float32(cfg["scanner"]["sample_radius_mm"]),
                   afov = Float32(cfg["scanner"]["afov_mm"]))
nsens    = Int(cfg["sens"]["n_sample_lors"])
gxs, gxe = to_dev.(sample_lors(sc, nsens; rng = MersenneTwister(Int(cfg["sens"]["seed"]))))
sens     = sensitivity_image(gxs, gxe, n, org, vs; scale = nev / nsens)
sync_dev()
x0       = Float32.(sens .> 0)
println("sensitivity built over $nsens LORs ($(round(time() - t0; digits = 1)) s)"); flush(stdout)

# --- reconstruct (generic listmode MLEM; counts = 1, mult = 1) ------------------
# per-iteration timing so the run's progress is visible (and gives the per-iter
# cost for the CPU-vs-Metal comparison); sync_dev makes GPU timing honest.
tprev = Ref(time())
function progress(k, x)
    sync_dev()
    now = time()
    println("  iter $k   $(round(now - tprev[]; digits = 2)) s"); flush(stdout)
    tprev[] = now
end
model = ListmodePoissonModel(xs, xe, sens; img_origin = org, voxsize = vs)
tmlem = time()
rec   = Array(mlem(model, x0; niter = Int(cfg["recon"]["niter"]), callback = progress))
println("MLEM done ($(round(time() - tmlem; digits = 1)) s)"); flush(stdout)

# --- flatness diagnostic: normalized mean activity vs radius inside the sphere --
cx = Float32[org[1] + (i - 1) * vs[1] for i in 1:n[1]]
cy = Float32[org[2] + (j - 1) * vs[2] for j in 1:n[2]]
cz = Float32[org[3] + (k - 1) * vs[3] for k in 1:n[3]]
rr = Float32[sqrt(cx[i]^2 + cy[j]^2 + cz[k]^2) for i in 1:n[1], j in 1:n[2], k in 1:n[3]]
inner = rr .< (R - 2 * vs[1])              # well inside, away from the edge
recn  = rec ./ (sum(rec[inner]) / count(inner))   # normalize to interior mean
println("interior (r < R-2vox): mean = 1.0 by constr., CoV = ",
        round(sqrt(sum(abs2, recn[inner] .- 1) / count(inner)); digits = 3))

# radial profile of the normalized reconstruction (bins of width one voxel)
nb = ceil(Int, (R + 5) / vs[1])
prof = zeros(Float64, nb); cnt = zeros(Int, nb)
for idx in eachindex(rr)
    b = floor(Int, rr[idx] / vs[1]) + 1
    b <= nb || continue
    prof[b] += recn[idx]; cnt[b] += 1
end
prof ./= max.(cnt, 1)
radii = Float32[(b - 0.5) * vs[1] for b in 1:nb]

# --- dump central slices + the radial profile -----------------------------------
kz = n[3] ÷ 2 + 1
npzwrite(joinpath(@__DIR__, "case_a_results.npz"), Dict(
    "slice_true"  => x_true[:, :, kz],
    "slice_rec"   => recn[:, :, kz],
    "slice_rec_xz" => recn[:, n[2] ÷ 2 + 1, :],
    "radii"       => radii,
    "radial_prof" => Float32.(prof),
    "radius_mm"   => R,
    "extent"      => Float32((n[1] - 1) / 2 * vs[1]),
))
println("wrote case_a_results.npz")
