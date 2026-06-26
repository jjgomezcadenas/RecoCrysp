# Read PTCRYSP detector-level coincidence listmode (lors_det.h5) into the
# endpoint arrays and per-event class labels that RecoCrysp reconstruction needs.
# Reads the on-disk schema directly with HDF5.jl — no dependency on PTCryspMC.
#
# Schema (one row per coincidence): endpoints x1..z2, true origin x0..z0, and
# energies e1/e2 are Int16 stored in 0.1 mm / 0.1 keV (scale factors in the file
# attributes); `truth` is 0=true, 1=scatter, 2=random; `nscat1`,`nscat2` are the
# per-gamma phantom-scatter counts. Class = (truth, nscat1+nscat2):
#   true   truth==0   |  single  truth==1 & sum==1
#   random truth==2   |  multiple truth==1 & sum>=2     (randoms keyed on truth first)

using HDF5

"""
    MCCoincidences

PTCRYSP coincidence listmode in reconstruction-ready form. `xstart`/`xend` are the
two gamma detector hits and `origin` the true annihilation point, each `(3, N)`
world-frame mm (z axial). `truth` is 0=true / 1=scatter / 2=random; `nscat` is the
per-LOR scatter order `nscat1+nscat2`; `elem1`/`elem2` are the `(iz, iphi)`
detector-element indices of each hit; `energy` is `(2, N)` keV.
"""
struct MCCoincidences
    xstart::Matrix{Float32}
    xend::Matrix{Float32}
    origin::Matrix{Float32}
    truth::Vector{Int8}
    nscat::Vector{Int16}
    elem1::Matrix{Int16}
    elem2::Matrix{Int16}
    energy::Matrix{Float32}
end

Base.length(c::MCCoincidences) = length(c.truth)

# (3, N) Float32 from three on-disk columns, scaled by `s`.
function _xyz(a, b, c, s)
    N = length(a)
    m = Matrix{Float32}(undef, 3, N)
    @inbounds for i in 1:N
        m[1, i] = Float32(a[i]) * s
        m[2, i] = Float32(b[i]) * s
        m[3, i] = Float32(c[i]) * s
    end
    return m
end

_rows(a, b) = permutedims(hcat(a, b))   # (2, N) from two N-vectors

"""
    read_coincidences(path) -> MCCoincidences

Load a PTCRYSP `lors_det.h5` coincidence file. Integer positions/energies are
rescaled to mm / keV using the file's `xyz_scale_mm` and `e_scale_keV` attributes.
"""
function read_coincidences(path::AbstractString)
    h5open(path, "r") do f
        xs = Float32(read_attribute(f, "xyz_scale_mm"))
        es = Float32(read_attribute(f, "e_scale_keV"))
        col(n) = read(f[n])
        xstart = _xyz(col("x1_mm"), col("y1_mm"), col("z1_mm"), xs)
        xend   = _xyz(col("x2_mm"), col("y2_mm"), col("z2_mm"), xs)
        origin = _xyz(col("x0_mm"), col("y0_mm"), col("z0_mm"), xs)
        truth  = Int8.(col("truth"))
        nscat  = Int16.(col("nscat1")) .+ Int16.(col("nscat2"))
        elem1  = _rows(Int16.(col("iz1")), Int16.(col("iphi1")))
        elem2  = _rows(Int16.(col("iz2")), Int16.(col("iphi2")))
        energy = _rows(Float32.(col("e1_keV")) .* es, Float32.(col("e2_keV")) .* es)
        return MCCoincidences(xstart, xend, origin, truth, nscat, elem1, elem2, energy)
    end
end

# --- class masks (keep randoms keyed on `truth` first) --------------------------
is_true(c::MCCoincidences)             = c.truth .== Int8(0)
is_scatter(c::MCCoincidences)          = c.truth .== Int8(1)          # single + multiple
is_random(c::MCCoincidences)           = c.truth .== Int8(2)
is_single_scatter(c::MCCoincidences)   = is_scatter(c) .& (c.nscat .== Int16(1))
is_multiple_scatter(c::MCCoincidences) = is_scatter(c) .& (c.nscat .>= Int16(2))

"""
    endpoints(c[, mask]) -> (xstart, xend)

The `(3, M)` endpoint matrices for reconstruction, optionally restricted to the
events selected by the boolean `mask` (e.g. `is_true(c)`).
"""
endpoints(c::MCCoincidences) = (c.xstart, c.xend)
endpoints(c::MCCoincidences, mask::AbstractVector{Bool}) =
    (c.xstart[:, mask], c.xend[:, mask])
