# Julia port of parallelproj benchmarks/00_pet_sinogram_nontof.py
#
# Non-TOF sinogram projections of a GE DMI-like scanner (34-gon, 16 crystals
# per side, 36 rings): times forward and back projection of one 34-subset
# (8 views, ~4.3M LORs) for all 6 sinogram memory orders and all 3 symmetry
# axes, on CPU and (if functional) Metal. Results go to benchmark/results/.
#
# Run (first time: see benchmark/throughput.jl header to set up the env):
#   julia -t auto --project=benchmark benchmark/sinogram_nontof.jl [num_runs] [num_subsets]

using Printf
using Statistics
using RecoCrysp
import Metal

num_runs = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5
num_subsets = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 34

# image properties (match the python benchmark)
num_trans = 215
num_ax = 71
voxel_size = (2.78f0, 2.78f0, 2.78f0)

# scanner properties
num_rings = 36
ring_positions = Float32.(5.31556 .* (0:(num_rings - 1)) .+ ((0:(num_rings - 1)) .÷ 9) .* 2.8)
ring_positions .-= 0.5f0 * maximum(ring_positions)

sinogram_orders = (:PVR, :PRV, :VPR, :VRP, :RPV, :RVP)
symmetry_axes = (1, 2, 3)

results_dir = joinpath(@__DIR__, "results")
mkpath(results_dir)

function run_mode(to_dev, mode_label, csv_rows)
    println("\n=== $mode_label ===")
    for sym_axis in symmetry_axes
        scanner = RegularPolygonPETScannerGeometry(;
            radius = 0.5 * (744.1 + 2 * 8.51),
            num_sides = 34,
            num_lor_endpoints_per_side = 16,
            lor_spacing = 4.03125,
            ring_positions = ring_positions,
            symmetry_axis = sym_axis,
        )

        # box-like test image
        img_shape = [num_trans, num_trans, num_trans]
        img_shape[sym_axis] = num_ax
        n0, n1, n2 = img_shape
        img = zeros(Float32, n0, n1, n2)
        sl = Any[(n0 ÷ 4 + 1):(3 * n0 ÷ 4), (n1 ÷ 4 + 1):(3 * n1 ÷ 4), (n2 ÷ 4 + 1):(3 * n2 ÷ 4)]
        sl[sym_axis] = 1:img_shape[sym_axis]
        img[sl[1], sl[2], sl[3]] .= 1.0f0

        img_origin = (-(Float32.(img_shape) ./ 2) .+ 0.5f0) .* voxel_size

        d_img = to_dev(img)

        for order in sinogram_orders
            desc = RegularPolygonPETLORDescriptor(
                scanner; radial_trim = 65, sinogram_order = order,
            )
            views = 1:num_subsets:desc.num_views
            xstart, xend = get_lor_coordinates(desc; views = views)
            nlors = size(xstart, 2)

            d_xs, d_xe = to_dev(xstart), to_dev(xend)
            d_ones = to_dev(ones(Float32, nlors))
            d_proj = to_dev(zeros(Float32, nlors))
            d_bimg = to_dev(zeros(Float32, n0, n1, n2))

            tf = Float64[]
            tb = Float64[]
            for ir in 0:num_runs  # run 0 = warm-up / compile
                t0 = @elapsed joseph3d_fwd!(d_proj, d_xs, d_xe, d_img, img_origin, voxel_size)
                t1 = @elapsed begin
                    fill!(d_bimg, 0.0f0)
                    joseph3d_back!(d_bimg, d_xs, d_xe, d_ones, img_origin, voxel_size)
                end
                if ir > 0
                    push!(tf, t0)
                    push!(tb, t1)
                    push!(csv_rows,
                          "$order,$sym_axis,$ir,$t0,$t1,nontof_sinogram,$mode_label,$num_subsets")
                end
            end
            @printf("%-10s axis %d %-4s (%7d LORs): fwd %6.3f ± %5.3f s, back %6.3f ± %5.3f s\n",
                    mode_label, sym_axis, order, nlors,
                    mean(tf), std(tf), mean(tb), std(tb))
        end
    end
    return nothing
end

header = "sinogram order,symmetry axis,run,t forward (s),t back (s),data,mode,num_subsets"

csv_rows = String[]
run_mode(identity, "CPU", csv_rows)
if Metal.functional()
    run_mode(Metal.MtlArray, "Metal", csv_rows)
else
    println("Metal not functional — GPU pass skipped")
end

output_file = joinpath(
    results_dir, "nontof_sinogram__julia__numruns_$(num_runs)__numsubsets_$(num_subsets).csv",
)
open(output_file, "w") do io
    println(io, header)
    foreach(r -> println(io, r), csv_rows)
end
println("\nresults written to $output_file")
