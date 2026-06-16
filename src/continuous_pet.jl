# Geometry for CONTINUOUS (monolithic-detector) PET: the detector is modeled as
# a continuous cylindrical surface rather than discrete crystals, so a line of
# response may have endpoints anywhere on it. This represents a monolithic
# scintillator whose interaction position (x, y, z) is estimated continuously
# (e.g. by a CNN): the spatial resolution is set by the detector + network and is
# injected at simulation time (a Gaussian smear / phantom blur), not by any
# crystal pitch. The discrete pixelated-crystal model is in pixelated_pet.jl.
#
# Convention: the cylinder axis is z (axis 3); world coordinates are in mm.

"""
    ContinuousPET(; diameter, afov)   # or radius = ...

Continuous cylindrical PET scanner — a detector surface of the given `diameter`
(mm) and axial field of view `afov` (mm), cylinder axis along z. Endpoints of a
line of response may lie anywhere on the surface (no discrete crystals), so this
models a monolithic detector with continuously estimated interaction positions.
Pass `radius` in place of `diameter` if preferred.
"""
struct ContinuousPET
    radius::Float32
    afov::Float32
end

function ContinuousPET(; diameter = nothing, radius = nothing, afov)
    (diameter === nothing) == (radius === nothing) &&
        throw(ArgumentError("give exactly one of `diameter` or `radius`"))
    r = radius === nothing ? Float32(diameter) / 2 : Float32(radius)
    return ContinuousPET(r, Float32(afov))
end

"""
    sample_lors(scanner::ContinuousPET, nlors; rng = Random.default_rng())

Sample `nlors` random LORs as chords of the detector cylinder: each endpoint is
an independent uniform point on the lateral surface (azimuth `φ ∈ [0, 2π)`,
axial `z ∈ [-afov/2, afov/2)`). Returns the start and end world coordinates as
two `(3, nlors)` `Float32` matrices, ready for the projectors.
"""
function sample_lors(
    scanner::ContinuousPET, nlors::Integer; rng::AbstractRNG = Random.default_rng(),
)
    R = scanner.radius
    halfz = scanner.afov / 2
    xstart = Array{Float32}(undef, 3, nlors)
    xend = Array{Float32}(undef, 3, nlors)
    twopi = 2.0f0 * Float32(pi)
    @inbounds for j in 1:nlors
        φs = twopi * rand(rng, Float32)
        φe = twopi * rand(rng, Float32)
        xstart[1, j] = R * cos(φs)
        xstart[2, j] = R * sin(φs)
        xstart[3, j] = (2.0f0 * rand(rng, Float32) - 1.0f0) * halfz
        xend[1, j] = R * cos(φe)
        xend[2, j] = R * sin(φe)
        xend[3, j] = (2.0f0 * rand(rng, Float32) - 1.0f0) * halfz
    end
    return xstart, xend
end
