# Resolution example (tutorial): a Derenzo rod phantom in a water cylinder, with
# attenuation correction. We reconstruct TWO datasets that differ only by the
# detector resolution, to separate the reconstruction from the resolution:
#   (1) a sharp dataset (G = 1) -> MLEM recovers the sharp rods (validation);
#   (2) a smeared dataset (G = fwhm) -> MLEM recovers the blurred image G*x.
# Geometry, attenuation and sensitivity are shared; only the simulated activity
# and the resulting counts differ. Resolution is injected at simulation time via
# the library operator G (gaussian_blur); both reconstructions use plain A.
#
#   julia -t auto --project=tutorial/examples/resolution tutorial/examples/resolution/run.jl
#
# Writes resolution_results.npz next to this file (read by resolution_plot.py).

using RecoCrysp
using Random
using Distributions
using NPZ
import TOML

cfg = TOML.parsefile(joinpath(@__DIR__, "resolution.toml"))

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

# the recoverable image when resolution is on: G * x_true (library operator G)
fwhm = Float32(cfg["resolution"]["fwhm"])
x_blur = gaussian_blur(x_true, fwhm, vs)

# --- shared geometry, attenuation and sensitivity -------------------------------
sc = ContinuousPET(diameter = cfg["scanner"]["diameter"], afov = cfg["scanner"]["afov"])
nlors = Int(cfg["sim"]["n_lors"])
seed  = Int(cfg["sim"]["seed"])
niter = Int(cfg["recon"]["niter"])
total = Float32(cfg["sim"]["total_counts"])

xs, xe = sample_lors(sc, nlors; rng = MersenneTwister(seed))
a    = exp.(-joseph3d_fwd(xs, xe, mumap, org, vs))           # attenuation a = exp(-A mu)
sens = sensitivity_image(xs, xe, n, org, vs; weights = a)     # shared (depends on a, not activity)
x0   = Float32.(sens .> 0)

# simulate y ~ Poisson(total-scaled a .* A*activity) and reconstruct (plain A, AC)
function simulate_reconstruct(activity, seed_off)
    ybar  = a .* joseph3d_fwd(xs, xe, activity, org, vs)
    scale = total / sum(ybar)
    rng   = MersenneTwister(seed + seed_off)
    counts = Float32[rand(rng, Poisson(Float64(l))) for l in scale .* ybar]
    model  = ListmodePoissonModel(xs, xe, sens; img_origin = org, voxsize = vs,
                                  counts = counts, mult = a)
    return mlem(model, x0; niter = niter)
end

rec_sharp = simulate_reconstruct(x_true, 1)   # G = 1 data
rec_blur  = simulate_reconstruct(x_blur, 2)   # G = fwhm data

# --- quantitative summary -------------------------------------------------------
# Each reconstruction is compared (normalized over the support) to what it should
# recover: the sharp truth for the sharp data, G*x_true for the smeared data.
hot  = x_true .> 0
supp = mumap .> 0
gap  = supp .& .!hot
relerr(u, v) = sqrt(sum(abs2, u .- v)) / sqrt(sum(abs2, v))
rmean(im) = sum(im[hot]) / max(count(hot), 1)
gmean(im) = sum(im[gap]) / max(count(gap), 1)
nrm(im) = im ./ rmean(im)
println("MLEM from sharp   vs truth :  rel.err = ", round(relerr(nrm(rec_sharp)[supp], nrm(x_true)[supp]); digits = 3),
        "   contrast rod/gap = ", round(rmean(rec_sharp) / gmean(rec_sharp); digits = 1))
println("MLEM from smeared vs G*x   :  rel.err = ", round(relerr(nrm(rec_blur)[supp], nrm(x_blur)[supp]); digits = 3),
        "   contrast rod/gap = ", round(rmean(rec_blur) / gmean(rec_blur); digits = 1))
println("recoverable contrast: truth = Inf,  G*x = ", round(rmean(x_blur) / gmean(x_blur); digits = 1))

# --- dump central transverse slices (the Derenzo pattern lives in x-y) -----------
kz = n[3] ÷ 2 + 1
npzwrite(joinpath(@__DIR__, "resolution_results.npz"), Dict(
    "slice_true"      => x_true[:, :, kz],
    "slice_blur"      => x_blur[:, :, kz],
    "slice_rec_sharp" => rec_sharp[:, :, kz],
    "slice_rec_blur"  => rec_blur[:, :, kz],
    "extent" => Float32((n[1] - 1) / 2 * vs[1]),
))
println("wrote resolution_results.npz")
