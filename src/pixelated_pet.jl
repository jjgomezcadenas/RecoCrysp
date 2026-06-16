# Geometry for PIXELATED (discrete-crystal) PET: a regular-polygon ring scanner
# and its sinogram LOR descriptor. Ported from parallelproj's pet_scanners.py
# (RegularPolygonPETScannerGeometry) and pet_lors.py (RegularPolygonPETLORDescriptor
# with span = 1, END_FIRST zig-zag ordering). The continuous-cylinder model for
# monolithic detectors lives in monolithic_pet.jl.
#
# Conventions: Julia is 1-based — `symmetry_axis` is 1, 2 or 3, ring and
# in-ring crystal indices start at 1. The geometry is built on the host;
# move the returned LOR coordinate matrices to the GPU with e.g. MtlArray.

"""
    RegularPolygonPETScannerGeometry(; radius, num_sides,
        num_lor_endpoints_per_side, lor_spacing, ring_positions, symmetry_axis)

Cylindrical PET scanner made of `length(ring_positions)` stacked regular-polygon
rings with `num_sides` flat sides and `num_lor_endpoints_per_side` crystals per
side (spaced `lor_spacing` mm apart, centered on each side). `radius` is the
distance from the scanner center to a detector face, `ring_positions` the axial
coordinate of each ring, and `symmetry_axis` (1, 2 or 3) the axial direction.

All LOR endpoint world coordinates are precomputed in `all_lor_endpoints`
(a `(3, num_rings * num_lor_endpoints_per_ring)` matrix); endpoint `k` of ring
`r` is column `(r - 1) * num_lor_endpoints_per_ring + k`.
"""
struct RegularPolygonPETScannerGeometry
    radius::Float32
    num_sides::Int
    num_lor_endpoints_per_side::Int
    lor_spacing::Float32
    ring_positions::Vector{Float32}
    symmetry_axis::Int
    all_lor_endpoints::Matrix{Float32}
end

function RegularPolygonPETScannerGeometry(;
    radius::Real,
    num_sides::Integer,
    num_lor_endpoints_per_side::Integer,
    lor_spacing::Real,
    ring_positions::AbstractVector{<:Real},
    symmetry_axis::Integer,
)
    symmetry_axis in (1, 2, 3) || throw(ArgumentError("symmetry_axis must be 1, 2 or 3"))
    # transverse axis pair per symmetry axis (matches parallelproj's 0-based
    # mapping 0 -> (2,1), 1 -> (0,2), 2 -> (1,0))
    ax0, ax1 = symmetry_axis == 1 ? (3, 2) : symmetry_axis == 2 ? (1, 3) : (2, 1)

    N = Int(num_lor_endpoints_per_side)
    nper = Int(num_sides) * N
    nrings = length(ring_positions)
    rad = Float32(radius)
    spacing = Float32(lor_spacing)

    pts = zeros(Float32, 3, nper * nrings)
    for ring in 0:(nrings - 1), k in 0:(nper - 1)
        side = k ÷ N
        within = k % N
        pos = spacing * (Float32(within) - (N - 1) / 2.0f0)
        phi = 2.0f0 * Float32(pi) * side / num_sides
        col = ring * nper + k + 1
        pts[ax0, col] = cos(phi) * rad - sin(phi) * pos
        pts[ax1, col] = sin(phi) * rad + cos(phi) * pos
        pts[symmetry_axis, col] = Float32(ring_positions[ring + 1])
    end

    return RegularPolygonPETScannerGeometry(
        rad, Int(num_sides), N, spacing, Float32.(ring_positions), Int(symmetry_axis), pts,
    )
end

num_rings(s::RegularPolygonPETScannerGeometry) = length(s.ring_positions)
num_lor_endpoints_per_ring(s::RegularPolygonPETScannerGeometry) =
    s.num_sides * s.num_lor_endpoints_per_side

"""
    get_lor_endpoints(scanner, rings, idx_in_rings) -> Matrix{Float32}

World coordinates (one column per endpoint, shape `(3, n)`) of the crystals
with 1-based ring numbers `rings` and 1-based in-ring indices `idx_in_rings`.
"""
function get_lor_endpoints(
    s::RegularPolygonPETScannerGeometry,
    rings::AbstractVector{<:Integer},
    idx_in_rings::AbstractVector{<:Integer},
)
    nper = num_lor_endpoints_per_ring(s)
    cols = (rings .- 1) .* nper .+ idx_in_rings
    return s.all_lor_endpoints[:, cols]
end

"""
    RegularPolygonPETLORDescriptor(scanner; radial_trim = 3, sinogram_order = :RVP)

Span-1 sinogram LOR descriptor for a [`RegularPolygonPETScannerGeometry`](@ref)
with unconstrained ring difference (every ring pair is one sinogram plane,
ordered `rd = 0, +1, -1, +2, -2, …` and by ring sum within each `rd`, as in
parallelproj's `Michelogram`). The in-ring detector pairs follow parallelproj's
`END_FIRST` zig-zag convention; `radial_trim` drops that many radial bins on
each side. `sinogram_order` (one of `:PVR, :PRV, :VPR, :VRP, :RPV, :RVP` for
Plane/View/Radial) sets the memory order of the LORs returned by
[`get_lor_coordinates`](@ref).
"""
struct RegularPolygonPETLORDescriptor
    scanner::RegularPolygonPETScannerGeometry
    radial_trim::Int
    sinogram_order::Symbol
    num_rad::Int
    num_views::Int
    num_planes::Int
    start_plane_ring::Vector{Int32}    # 1-based ring index per plane
    end_plane_ring::Vector{Int32}
    start_in_ring_index::Matrix{Int32} # (num_views, num_rad), 1-based crystal index
    end_in_ring_index::Matrix{Int32}
end

function RegularPolygonPETLORDescriptor(
    scanner::RegularPolygonPETScannerGeometry;
    radial_trim::Integer = 3,
    sinogram_order::Symbol = :RVP,
)
    sinogram_order in (:PVR, :PRV, :VPR, :VRP, :RPV, :RVP) ||
        throw(ArgumentError("invalid sinogram_order $sinogram_order"))

    n = num_lor_endpoints_per_ring(scanner)
    nrings = num_rings(scanner)
    trim = Int(radial_trim)
    nrad = n - 1 - 2 * trim
    nviews = n ÷ 2

    # planes: rd = 0, +1, -1, ... with ring sum ascending within each rd
    start_plane = Int32[]
    end_plane = Int32[]
    rds = Int[0]
    for k in 1:(nrings - 1)
        push!(rds, k)
        push!(rds, -k)
    end
    for rd in rds
        for s in max(0, -rd):min(nrings - 1, nrings - 1 - rd)
            push!(start_plane, s + 1)
            push!(end_plane, s + rd + 1)
        end
    end
    nplanes = length(start_plane)

    # in-ring crystal pairs (END_FIRST zig-zag), computed 0-based as in the
    # reference, then shifted to 1-based
    m = 2 * (n ÷ 2)
    start_seq = [j ÷ 2 for j in 0:(m - 2)]
    end_seq = vcat([-1], [-((t + 4) ÷ 2) for t in 0:(m - 3)])
    rad_range = (trim + 1):(m - 1 - trim)

    start_in_ring = Matrix{Int32}(undef, nviews, nrad)
    end_in_ring = Matrix{Int32}(undef, nviews, nrad)
    for view in 0:(nviews - 1)
        for (jr, j) in enumerate(rad_range)
            sidx = start_seq[j] - view
            eidx = end_seq[j] - view
            sidx < 0 && (sidx += n)
            eidx < 0 && (eidx += n)
            start_in_ring[view + 1, jr] = Int32(sidx + 1)
            end_in_ring[view + 1, jr] = Int32(eidx + 1)
        end
    end

    return RegularPolygonPETLORDescriptor(
        scanner, trim, sinogram_order, nrad, nviews, nplanes,
        start_plane, end_plane, start_in_ring, end_in_ring,
    )
end

"""
    get_lor_coordinates(desc; views = 1:desc.num_views) -> (xstart, xend)

LOR start/end world coordinates for the selected sinogram views, as
`(3, nlors)` `Float32` matrices with `nlors = num_planes * length(views) *
num_rad`. The column (memory) order is the row-major linearization of the
sinogram axes in `desc.sinogram_order` — e.g. `:PVR` lays LORs out with the
radial index fastest and the plane index slowest, exactly as the corresponding
parallelproj benchmark does.
"""
function get_lor_coordinates(
    desc::RegularPolygonPETLORDescriptor;
    views::AbstractVector{<:Integer} = 1:desc.num_views,
)
    sc = desc.scanner
    sa = sc.symmetry_axis
    nv, nr, npl = length(views), desc.num_rad, desc.num_planes

    # transverse endpoint coordinates of the selected views (ring independent)
    sxy = Array{Float32}(undef, 3, nv, nr)
    exy = Array{Float32}(undef, 3, nv, nr)
    for (jv, view) in enumerate(views), jr in 1:nr
        ks = desc.start_in_ring_index[view, jr]
        ke = desc.end_in_ring_index[view, jr]
        for c in 1:3
            sxy[c, jv, jr] = sc.all_lor_endpoints[c, ks]
            exy[c, jv, jr] = sc.all_lor_endpoints[c, ke]
        end
    end

    name = String(desc.sinogram_order)
    pP = findfirst('P', name)
    pV = findfirst('V', name)
    pR = findfirst('R', name)
    sizes = zeros(Int, 3)
    sizes[pP], sizes[pV], sizes[pR] = npl, nv, nr

    nlors = npl * nv * nr
    xstart = Array{Float32}(undef, 3, nlors)
    xend = Array{Float32}(undef, 3, nlors)

    col = 0
    idx = zeros(Int, 3)
    @inbounds for i1 in 1:sizes[1], i2 in 1:sizes[2], i3 in 1:sizes[3]
        idx[1], idx[2], idx[3] = i1, i2, i3
        p, v, r = idx[pP], idx[pV], idx[pR]
        col += 1
        zs = sc.ring_positions[desc.start_plane_ring[p]]
        ze = sc.ring_positions[desc.end_plane_ring[p]]
        for c in 1:3
            xstart[c, col] = sxy[c, v, r]
            xend[c, col] = exy[c, v, r]
        end
        xstart[sa, col] = zs
        xend[sa, col] = ze
    end

    return xstart, xend
end
