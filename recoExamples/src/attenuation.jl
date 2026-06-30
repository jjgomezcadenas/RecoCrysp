# Analytic attenuation for a uniform sphere of activity in a uniform attenuator.
#
# In PET the attenuation of a coincidence depends only on the TOTAL path the two
# 511 keV photons travel through the attenuator -- i.e. the chord the LOR cuts
# through the medium -- not on where along the line the annihilation happened.
# For a uniform sphere of radius R centred at the origin that chord is the
# line-sphere intersection length, and the per-LOR survival factor is
#   a = exp(-mu * chord).
# This is exact for the att-only study (uniform water sphere), no voxel mu-map.

"""
    sphere_chord(x1, x2, R) -> Float32

Length of the intersection of the (infinite) line through `x1`,`x2` with a sphere
of radius `R` centred at the origin; 0 if the line misses. Points are length-3
indexable (mm).
"""
function sphere_chord(x1, x2, R)
    dx = x2[1] - x1[1]; dy = x2[2] - x1[2]; dz = x2[3] - x1[3]
    a = dx * dx + dy * dy + dz * dz
    a < 1.0f-12 && return 0.0f0
    b = 2 * (x1[1] * dx + x1[2] * dy + x1[3] * dz)
    c = x1[1] * x1[1] + x1[2] * x1[2] + x1[3] * x1[3] - R * R
    disc = b * b - 4 * a * c
    disc <= 0 && return 0.0f0
    return Float32(sqrt(disc) / sqrt(a))            # |t2-t1|*|d| = sqrt(disc)/|d|
end

"""
    cylinder_chord(x1, x2, R, half_z) -> Float32

Length of the intersection of the (infinite) line through `x1`,`x2` with a finite
cylinder of radius `R` about the z-axis, capped to `|z| <= half_z`; 0 if the line
misses. The exact attenuation chord for a uniform-water-cylinder body (e.g. the
NEMA phantom): the line's radial interval (`x²+y² <= R²`) intersected with the
axial slab. Points are length-3 indexable (mm).
"""
function cylinder_chord(x1, x2, R, half_z)
    dx = x2[1] - x1[1]; dy = x2[2] - x1[2]; dz = x2[3] - x1[3]
    d2 = dx * dx + dy * dy + dz * dz
    d2 < 1.0f-12 && return 0.0f0
    # radial interval [tr_lo, tr_hi] where x(t)²+y(t)² <= R²
    a = dx * dx + dy * dy
    tr_lo = -Inf32; tr_hi = Inf32
    if a < 1.0f-12                                   # line ∥ z-axis: radius fixed
        (x1[1] * x1[1] + x1[2] * x1[2] > R * R) && return 0.0f0
    else
        b = 2 * (x1[1] * dx + x1[2] * dy)
        c = x1[1] * x1[1] + x1[2] * x1[2] - R * R
        disc = b * b - 4 * a * c
        disc <= 0 && return 0.0f0
        sq = sqrt(disc)
        tr_lo = (-b - sq) / (2 * a); tr_hi = (-b + sq) / (2 * a)
    end
    # axial interval [tz_lo, tz_hi] where |z(t)| <= half_z
    tz_lo = -Inf32; tz_hi = Inf32
    if abs(dz) < 1.0f-12                             # line ⊥ z-axis: z fixed
        (abs(x1[3]) > half_z) && return 0.0f0
    else
        ta = (-half_z - x1[3]) / dz; tb = (half_z - x1[3]) / dz
        tz_lo = min(ta, tb); tz_hi = max(ta, tb)
    end
    tlo = max(tr_lo, tz_lo); thi = min(tr_hi, tz_hi)
    thi <= tlo && return 0.0f0
    return Float32((thi - tlo) * sqrt(d2))
end

"""
    attenuation_factors(xs, xe; R, mu, half_z = nothing) -> Vector{Float32}

Per-LOR survival factor `a = exp(-mu * chord)` for endpoint columns of the `(3,N)`
matrices `xs`,`xe`. `mu` is the linear attenuation coefficient in the coordinate
units (mm^-1). The body is a uniform **sphere** of radius `R` (default), or a
finite **cylinder** of radius `R` capped at `|z| <= half_z` when `half_z` is given
(the NEMA water body). Compute on the CPU coordinate arrays, then move to the device.
"""
function attenuation_factors(xs, xe; R, mu, half_z = nothing)
    N = size(xs, 2)
    a = Vector{Float32}(undef, N)
    if half_z === nothing
        @inbounds for i in 1:N
            L = sphere_chord((xs[1, i], xs[2, i], xs[3, i]),
                             (xe[1, i], xe[2, i], xe[3, i]), R)
            a[i] = exp(-mu * L)
        end
    else
        hz = Float32(half_z)
        @inbounds for i in 1:N
            L = cylinder_chord((xs[1, i], xs[2, i], xs[3, i]),
                               (xe[1, i], xe[2, i], xe[3, i]), R, hz)
            a[i] = exp(-mu * L)
        end
    end
    return a
end
