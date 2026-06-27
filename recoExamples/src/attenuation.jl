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
    attenuation_factors(xs, xe; R, mu) -> Vector{Float32}

Per-LOR survival factor `a = exp(-mu * chord)` for endpoint columns of the `(3,N)`
matrices `xs`,`xe` through a uniform sphere of radius `R`. `mu` is the linear
attenuation coefficient in the same length units as the coordinates (mm^-1).
Compute on the CPU coordinate arrays, then move to the device.
"""
function attenuation_factors(xs, xe; R, mu)
    N = size(xs, 2)
    a = Vector{Float32}(undef, N)
    @inbounds for i in 1:N
        L = sphere_chord((xs[1, i], xs[2, i], xs[3, i]),
                         (xe[1, i], xe[2, i], xe[3, i]), R)
        a[i] = exp(-mu * L)
    end
    return a
end
