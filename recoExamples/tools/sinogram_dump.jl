# Dump the background/prompt sinograms (raw + smoothed) that a contamination model
# is built from, for inspection (src/background.jl). Generic and class-agnostic:
# works for scatter or randoms, any phantom -- only the per-event class flag and
# the binning differ, and both come from the scenario config.
#
#   julia -t auto --project=recoExamples recoExamples/tools/sinogram_dump.jl <config.toml> <class>
#     <class> = scatter | randoms
#
# Reads [data].lors, the [<class>] binning block, [phantom].radius_mm (edge marker),
# and `tag`. Prompt population = trues + the chosen background class (the other
# contamination is dropped). Writes <config_dir>/out/<tag>_sinogram.npz, read by
# sinogram_plot.py (which writes the figure to the sibling figures/).

using RecoExamples
using NPZ
import TOML

length(ARGS) == 2 || error("usage: sinogram_dump.jl <config.toml> <class:scatter|randoms>")
cfgpath, class = ARGS[1], ARGS[2]
class in ("scatter", "randoms") || error("class must be 'scatter' or 'randoms'")
cfg = TOML.parsefile(cfgpath)

c = read_coincidences(cfg["data"]["lors"])
bg_all = class == "scatter" ? is_scatter(c) : is_random(c)
keep = is_true(c) .| bg_all                       # trues + this background class
xs, xe = c.xstart[:, keep], c.xend[:, keep]
bg = bg_all[keep]
println("$class: prompts(trues+$class) $(count(keep)), background $(count(bg)) " *
        "($(round(100count(bg)/count(keep); digits=1))%)"); flush(stdout)

s_r, z_m, dz = lor_sinogram_coords(xs, xe)
sg = cfg[class]
span_sr = (0.0f0, Float32(sg["sr_max_mm"]))
span_zm = (-Float32(sg["zm_max_mm"]), Float32(sg["zm_max_mm"]))
span_dz = (-Float32(sg["dz_max_mm"]), Float32(sg["dz_max_mm"]))
smooth  = (Float64(sg["smooth_sr"]), Float64(sg["smooth_zm"]), Float64(sg["smooth_dz"]))
S, P, Ssm, Psm, spr, spz, spd = background_sinograms(s_r, z_m, dz, bg;
    n_sr = Int(sg["n_sr"]), n_zm = Int(sg["n_zm"]), n_dz = Int(sg["n_dz"]),
    span_sr = span_sr, span_zm = span_zm, span_dz = span_dz, smooth = smooth)

tag = cfg["tag"]
edge = haskey(cfg, "phantom") ? Float32(cfg["phantom"]["radius_mm"]) : 0.0f0
outdir = joinpath(dirname(abspath(cfgpath)), "out")
mkpath(outdir)
out = joinpath(outdir, "$(tag)_sinogram.npz")
npzwrite(out, Dict(
    "S" => S, "P" => P, "Ssm" => Ssm, "Psm" => Psm,
    "span_sr" => collect(Float32.(spr)), "span_zm" => collect(Float32.(spz)),
    "span_dz" => collect(Float32.(spd)),
    "radius_mm" => edge,
    "smooth" => Float32[smooth[1], smooth[2], smooth[3]]))
println("wrote $out  (class=$class)")
