# Diagnostic: dump the scatter/prompt sinogram histograms (raw + smoothed) that
# run_att_scatter.jl builds its scatter model from, for inspection. Same data,
# masks, coordinates and [scatter] binning as the run -- only the histograms are
# written out (no reconstruction). CPU only.
#
#   julia -t auto --project=recoExamples recoExamples/sphere/water_bgo_1MBq/dump_sinogram.jl
# Writes water_bgo_1MBq_sinogram.npz (read by sinogram_plot.py).

using RecoExamples
using NPZ
import TOML

cfg = TOML.parsefile(joinpath(@__DIR__, "config.toml"))
c = read_coincidences(cfg["data"]["lors"])
nr = is_true(c) .| is_scatter(c)                 # trues+scatter (drop randoms), as in the run
xs, xe = c.xstart[:, nr], c.xend[:, nr]
scat = is_scatter(c)[nr]
println("trues+scatter $(count(nr)), of which scatter $(count(scat)) " *
        "($(round(100count(scat)/count(nr); digits=1))%)"); flush(stdout)

s_r, z_m, dz = lor_sinogram_coords(xs, xe)
sg = cfg["scatter"]
span_sr = (0.0f0, Float32(sg["sr_max_mm"]))
span_zm = (-Float32(sg["zm_max_mm"]), Float32(sg["zm_max_mm"]))
span_dz = (-Float32(sg["dz_max_mm"]), Float32(sg["dz_max_mm"]))
smooth  = (Float64(sg["smooth_sr"]), Float64(sg["smooth_zm"]), Float64(sg["smooth_dz"]))
S, P, Ssm, Psm, spr, spz, spd = scatter_sinograms(s_r, z_m, dz, scat;
    n_sr = Int(sg["n_sr"]), n_zm = Int(sg["n_zm"]), n_dz = Int(sg["n_dz"]),
    span_sr = span_sr, span_zm = span_zm, span_dz = span_dz, smooth = smooth)

npzwrite(joinpath(@__DIR__, "water_bgo_1MBq_sinogram.npz"), Dict(
    "S" => S, "P" => P, "Ssm" => Ssm, "Psm" => Psm,
    "span_sr" => collect(Float32.(spr)), "span_zm" => collect(Float32.(spz)),
    "span_dz" => collect(Float32.(spd)),
    "radius_mm" => Float32(cfg["phantom"]["radius_mm"]),
    "smooth" => Float32[smooth[1], smooth[2], smooth[3]]))
println("wrote water_bgo_1MBq_sinogram.npz")
