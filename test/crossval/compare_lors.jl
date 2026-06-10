# Cross-validate the Julia sinogram LOR coordinates against the Python
# parallelproj reference produced by gen_reference_lors.py.
#
# For each (symmetry_axis, sinogram_order) case we rebuild the identical
# scanner geometry in Julia and compare get_lor_coordinates() element-wise to
# the reference. Coordinate components map directly (Julia row c <-> Python
# column c-1); the only index translation is the symmetry axis (Python 0/1/2
# -> Julia 1/2/3).
#
# Run (first time: instantiate, see header of compare command below):
#   julia --project=test/crossval test/crossval/compare_lors.jl

using NPZ
using Printf
using JosephProjectors

ref = npzread(joinpath(@__DIR__, "reference_lors.npz"))

radius = Float64(ref["radius"])
num_sides = Int(ref["num_sides"])
nlps = Int(ref["num_lor_endpoints_per_side"])
lor_spacing = Float64(ref["lor_spacing"])
num_rings = Int(ref["num_rings"])
radial_trim = Int(ref["radial_trim"])
ring_positions = Float32.(ref["ring_positions"])

sinogram_orders = ("PVR", "PRV", "VPR", "VRP", "RPV", "RVP")

println("Cross-validation: Julia get_lor_coordinates vs Python parallelproj\n")
@printf("%-6s %-5s %8s %14s %14s\n", "sym", "order", "nlors", "max abs diff", "result")

all_ok = true
for pysym in 0:2
    scanner = RegularPolygonPETScannerGeometry(;
        radius = radius,
        num_sides = num_sides,
        num_lor_endpoints_per_side = nlps,
        lor_spacing = lor_spacing,
        ring_positions = ring_positions,
        symmetry_axis = pysym + 1,           # 0-based -> 1-based
    )
    for order in sinogram_orders
        desc = RegularPolygonPETLORDescriptor(
            scanner; radial_trim = radial_trim, sinogram_order = Symbol(order),
        )
        xs, xe = get_lor_coordinates(desc)   # (3, nlors)

        key = "sym$(pysym)_$(order)"
        rxs = ref["$(key)__xstart"]          # (nlors, 3)
        rxe = ref["$(key)__xend"]
        nlors = size(xs, 2)

        # compare: Julia xs[:, k] vs Python rxs[k, :]
        dmax = 0.0
        for k in 1:nlors, c in 1:3
            dmax = max(dmax, abs(Float64(xs[c, k]) - rxs[k, c]))
            dmax = max(dmax, abs(Float64(xe[c, k]) - rxe[k, c]))
        end
        ok = dmax < 1.0e-3      # float32 geometry; mm-scale coords ~ up to ~400
        global all_ok &= ok
        @printf("%-6d %-5s %8d %14.2e %14s\n",
                pysym, order, nlors, dmax, ok ? "PASS" : "FAIL")
    end
end

println()
if all_ok
    println("All cases match the Python parallelproj reference.")
else
    println("MISMATCH detected — see FAIL rows above.")
    exit(1)
end
