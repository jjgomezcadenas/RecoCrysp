# Julia port of parallelproj benchmarks/02_pet_lm_nontof.py
#
# Non-TOF listmode projections of a GE DMI-like scanner. The python original
# reads a real 10M-event listmode file (zenodo.org/records/8404015); since
# that file is not assumed here, events are SYNTHESIZED as uniformly random
# detector pairs, which preserves the benchmark's scattered-memory-access
# character (random LORs, random image access) but not the exact spatial
# distribution of a real scan. Pass --presort to sort events by in-ring index
# difference (the python benchmark's cache-friendliness experiment).
#
# Run (first time: see benchmark/throughput.jl header to set up the env):
#   julia -t auto --project=benchmark benchmark/lm_nontof.jl [num_events] [num_runs] [--presort]

using Printf
using Random
using Statistics
using RecoCrysp
import Metal

pos_args = filter(a -> !startswith(a, "--"), ARGS)
num_events = length(pos_args) >= 1 ? parse(Int, pos_args[1]) : 10_000_000
num_runs = length(pos_args) >= 2 ? parse(Int, pos_args[2]) : 5
presort = "--presort" in ARGS

# image properties (match the python benchmark)
num_trans = 215
num_ax = 71
voxel_size = (2.78f0, 2.78f0, 2.78f0)

# scanner properties
num_rings = 36
ring_positions = Float32.(5.31556 .* (0:(num_rings - 1)) .+ ((0:(num_rings - 1)) .÷ 9) .* 2.8)
ring_positions .-= 0.5f0 * maximum(ring_positions)

symmetry_axes = (1, 2, 3)
data_str = presort ? "nontof_listmode_presorted" : "nontof_listmode"

results_dir = joinpath(@__DIR__, "results")
mkpath(results_dir)

# synthetic listmode events: uniformly random detector pairs
rng = MersenneTwister(0)
nper = 34 * 16
ring1 = rand(rng, 1:num_rings, num_events)
idx1 = rand(rng, 1:nper, num_events)
ring2 = rand(rng, 1:num_rings, num_events)
idx2 = rand(rng, 1:nper, num_events)

if presort
    println("pre-sorting events by in-ring index difference")
    p = sortperm(idx1 .- idx2)
    ring1, idx1, ring2, idx2 = ring1[p], idx1[p], ring2[p], idx2[p]
end

function run_mode(to_dev, mode_label, csv_rows)
    println("\n=== $mode_label ($(num_events ÷ 1_000_000)e6 synthetic events) ===")
    for sym_axis in symmetry_axes
        img_shape = [num_trans, num_trans, num_trans]
        img_shape[sym_axis] = num_ax
        n0, n1, n2 = img_shape
        img_origin = (-(Float32.(img_shape) ./ 2) .+ 0.5f0) .* voxel_size
        img = ones(Float32, n0, n1, n2)

        scanner = RegularPolygonPETScannerGeometry(;
            radius = 0.5 * (744.1 + 2 * 8.51),
            num_sides = 34,
            num_lor_endpoints_per_side = 16,
            lor_spacing = 4.03125,
            ring_positions = ring_positions,
            symmetry_axis = sym_axis,
        )

        xstart = get_lor_endpoints(scanner, ring1, idx1)
        xend = get_lor_endpoints(scanner, ring2, idx2)

        d_img = to_dev(img)
        d_xs, d_xe = to_dev(xstart), to_dev(xend)
        d_y = to_dev(ones(Float32, num_events))
        d_proj = to_dev(zeros(Float32, num_events))
        d_bimg = to_dev(zeros(Float32, n0, n1, n2))

        tf = Float64[]
        tb = Float64[]
        for ir in 0:num_runs  # run 0 = warm-up / compile
            t0 = @elapsed joseph3d_fwd!(d_proj, d_xs, d_xe, d_img, img_origin, voxel_size)
            t1 = @elapsed begin
                fill!(d_bimg, 0.0f0)
                joseph3d_back!(d_bimg, d_xs, d_xe, d_y, img_origin, voxel_size)
            end
            if ir > 0
                push!(tf, t0)
                push!(tb, t1)
                push!(csv_rows, "$sym_axis,$ir,$t0,$t1,$data_str,$mode_label,$num_events")
            end
        end
        @printf("%-10s axis %d: fwd %6.3f ± %5.3f s (%6.1f Mev/s), back %6.3f ± %5.3f s (%6.1f Mev/s)\n",
                mode_label, sym_axis, mean(tf), std(tf), num_events / mean(tf) / 1.0e6,
                mean(tb), std(tb), num_events / mean(tb) / 1.0e6)
    end
    return nothing
end

header = "symmetry axis,run,t forward (s),t back (s),data,mode,num_events"

csv_rows = String[]
run_mode(identity, "CPU", csv_rows)
if Metal.functional()
    run_mode(Metal.MtlArray, "Metal", csv_rows)
else
    println("Metal not functional — GPU pass skipped")
end

output_file = joinpath(
    results_dir, "$(data_str)__julia__numruns_$(num_runs)__numevents_$(num_events).csv",
)
open(output_file, "w") do io
    println(io, header)
    foreach(r -> println(io, r), csv_rows)
end
println("\nresults written to $output_file")
