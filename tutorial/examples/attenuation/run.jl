# Attenuation example (tutorial §6.1): a uniform-activity water sphere, with NO
# detector resolution (G = 1). We simulate attenuated, Poisson-noisy listmode
# data and reconstruct it twice — with and without attenuation correction —
# to show the classic cupping artifact and its removal.
#
#   julia --project=tutorial/examples/attenuation tutorial/examples/attenuation/run.jl
#
# Writes results.npz next to this file (read by plot.py).

using RecoCrysp
using Random
using Distributions
using NPZ
import TOML

cfg = TOML.parsefile(joinpath(@__DIR__, "attenuation.toml"))

n  = Tuple(Int.(cfg["grid"]["n"]))
vs = ntuple(_ -> Float32(cfg["grid"]["voxsize"]), 3)
org = ntuple(i -> -(n[i] - 1) / 2 * vs[i], 3)          # grid centred on the origin

Rsph    = Float32(cfg["phantom"]["sphere_radius"])
act     = Float32(cfg["phantom"]["activity"])
muwater = Float32(cfg["phantom"]["mu_water"])

# --- phantom: activity x_true and attenuation map mu, both the same sphere -----
xc = Float32[org[1] + (i - 1) * vs[1] for i in 1:n[1]]
yc = Float32[org[2] + (j - 1) * vs[2] for j in 1:n[2]]
zc = Float32[org[3] + (k - 1) * vs[3] for k in 1:n[3]]

x_true = zeros(Float32, n)
mumap  = zeros(Float32, n)
inside = falses(n)
for k in 1:n[3], j in 1:n[2], i in 1:n[1]
    if xc[i]^2 + yc[j]^2 + zc[k]^2 <= Rsph^2
        x_true[i, j, k] = act
        mumap[i, j, k]  = muwater
        inside[i, j, k] = true
    end
end

# --- geometry: sample LORs on the continuous cylinder --------------------------
sc = ContinuousPET(diameter = cfg["scanner"]["diameter"], afov = cfg["scanner"]["afov"])
nlors = Int(cfg["sim"]["n_lors"])
seed  = Int(cfg["sim"]["seed"])
xs, xe = sample_lors(sc, nlors; rng = MersenneTwister(seed))

# --- forward model & simulation (G = 1) ---------------------------------------
# attenuation factor a = exp(-A mu): project the mu-map and exponentiate
a = exp.(-joseph3d_fwd(xs, xe, mumap, org, vs))
# expected counts ybar = a .* (A x_true); scale to the requested total, draw Poisson
ybar  = a .* joseph3d_fwd(xs, xe, x_true, org, vs)
scale = Float32(cfg["sim"]["total_counts"]) / sum(ybar)
lam   = scale .* ybar
rng   = MersenneTwister(seed + 1)
counts = Float32[rand(rng, Poisson(Float64(l))) for l in lam]
x_true_s = scale .* x_true                       # truth at the same count scale

# --- reconstruction: with vs without attenuation correction --------------------
# Sensitivity = A^T over the SAME LOR set as the events (scale = 1). Here the
# sampled LORs ARE the acquisition geometry: we build the data by projecting the
# phantom onto these LORs and binning Poisson counts onto them — a
# propagation-based simulation — so the events and the sensitivity share the set
# and their sampling noise cancels in the MLEM update. (A true Monte-Carlo
# dataset would emit individual events on independent LORs; there the sensitivity
# is a separate, denser A^T(a) sample passed with scale = n_events / n_sens.
# See sensitivity_image.)
niter = Int(cfg["recon"]["niter"])

sens_ac = sensitivity_image(xs, xe, n, org, vs; weights = a)     # A^T(a)
model_ac = ListmodePoissonModel(xs, xe, sens_ac;
                                img_origin = org, voxsize = vs, counts = counts, mult = a)

sens_no = sensitivity_image(xs, xe, n, org, vs)                  # A^T(1)
model_no = ListmodePoissonModel(xs, xe, sens_no;
                                img_origin = org, voxsize = vs, counts = counts)  # mult = 1

x0 = Float32.(sens_ac .> 0)                       # uniform start inside the FOV
x_ac = mlem(model_ac, x0; niter = niter)
x_no = mlem(model_no, x0; niter = niter)

relerr(u, v) = sqrt(sum(abs2, u .- v)) / sqrt(sum(abs2, v))
println("rel. error inside sphere:  with AC = ", round(relerr(x_ac[inside], x_true_s[inside]); digits = 3),
        " | without AC = ", round(relerr(x_no[inside], x_true_s[inside]); digits = 3))

# interior noise: coefficient of variation in the flat core (avoids the edge)
core = falses(n)
for k in 1:n[3], j in 1:n[2], i in 1:n[1]
    xc[i]^2 + yc[j]^2 + zc[k]^2 <= (0.6f0 * Rsph)^2 && (core[i, j, k] = true)
end
cv(v) = (m = sum(v) / length(v); sqrt(sum(abs2, v .- m) / length(v)) / m)
println("interior CV (noise, with AC):  ", round(cv(x_ac[core]); digits = 3),
        " over ", niter, " iters")

# --- dump central slice and profile for plotting -------------------------------
kz = n[3] ÷ 2 + 1
jy = n[2] ÷ 2 + 1
npzwrite(joinpath(@__DIR__, "attenuation_results.npz"), Dict(
    "slice_true" => x_true_s[:, :, kz], "slice_ac" => x_ac[:, :, kz], "slice_no" => x_no[:, :, kz],
    "prof_true"  => x_true_s[:, jy, kz], "prof_ac" => x_ac[:, jy, kz], "prof_no" => x_no[:, jy, kz],
    "x_mm" => xc, "voxsize" => Float32(vs[1]), "sphere_radius" => Rsph,
))
println("wrote attenuation_results.npz")
