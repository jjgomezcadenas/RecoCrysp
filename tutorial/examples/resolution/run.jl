# Resolution example (tutorial): a Derenzo rod phantom in a water cylinder, with
# attenuation correction and the CRYSP detector resolution. We inject resolution
# at simulation time by blurring the activity with the Gaussian operator G, then
# reconstruct with plain A (attenuation-corrected). The reconstruction converges
# towards G*x_true, so it resolves the coarse rod sectors and merges the fine
# ones — a direct picture of the 3.5 mm limit.
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

# --- resolution: blur the activity with the Gaussian operator G (library) -------
fwhm = Float32(cfg["resolution"]["fwhm"])
x_blur = gaussian_blur(x_true, fwhm, vs)

# --- geometry + simulation: y ~ Poisson(a .* A*(G*x_true)) ----------------------
sc = ContinuousPET(diameter = cfg["scanner"]["diameter"], afov = cfg["scanner"]["afov"])
nlors = Int(cfg["sim"]["n_lors"])
seed  = Int(cfg["sim"]["seed"])
xs, xe = sample_lors(sc, nlors; rng = MersenneTwister(seed))

a    = exp.(-joseph3d_fwd(xs, xe, mumap, org, vs))         # attenuation a = exp(-A mu)
ybar = a .* joseph3d_fwd(xs, xe, x_blur, org, vs)          # G already applied
scale = Float32(cfg["sim"]["total_counts"]) / sum(ybar)
rng = MersenneTwister(seed + 1)
counts = Float32[rand(rng, Poisson(Float64(l))) for l in scale .* ybar]
x_blur_s = scale .* x_blur                                 # the recoverable image, scaled

# --- attenuation-corrected reconstruction (matched sensitivity, plain A) --------
sens  = sensitivity_image(xs, xe, n, org, vs; weights = a)
model = ListmodePoissonModel(xs, xe, sens; img_origin = org, voxsize = vs,
                             counts = counts, mult = a)
x0 = Float32.(sens .> 0)
x_rec = mlem(model, x0; niter = Int(cfg["recon"]["niter"]))

# --- dump the central transverse slice (the Derenzo pattern lives in x-y) -------
kz = n[3] ÷ 2 + 1
npzwrite(joinpath(@__DIR__, "resolution_results.npz"), Dict(
    "slice_true" => x_true[:, :, kz],
    "slice_blur" => x_blur_s[:, :, kz],
    "slice_rec"  => x_rec[:, :, kz],
    "extent" => Float32((n[1] - 1) / 2 * vs[1]),
    "rod_diameters" => Float32.(dz["rod_diameters"]),
))
println("wrote resolution_results.npz")
