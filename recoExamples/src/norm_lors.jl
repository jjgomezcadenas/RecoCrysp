# Emission-matched geometric LOR sampler for the sensitivity image, with DOI.
#
# Instead of drawing uniform endpoint pairs on the detector surface (sample_lors),
# this generates LORs the way the scanner physically makes them: an isotropic
# emission from a point in the FOV, both 511 keV photons traced to the detector
# cylinder, each interacting at a depth drawn from the crystal's exponential
# absorption along its own path. Two consequences vs surface sampling:
#   - the line measure matches the MC's emission->detection process, and
#   - endpoints spread through the crystal depth (DOI), at a single radius no more.
# Back-projecting these LORs gives a sensitivity with the correct geometric radial
# shape -- up to the angle-dependent CONTAINMENT efficiency, which is material
# physics and not accessible from geometry (that needs an MC normalization scan).

using Random

# Smallest positive ray parameter t with |(px,py) + t (dx,dy)| = R (forward exit
# of a ray starting inside the cylinder); NaN if the ray is axial/misses.
function _ray_cyl_t(px, py, dx, dy, R)
    a = dx * dx + dy * dy
    a < 1.0f-12 && return NaN32
    b = 2 * (px * dx + py * dy)
    c = px * px + py * py - R * R
    disc = b * b - 4 * a * c
    disc < 0 && return NaN32
    return (-b + sqrt(disc)) / (2a)
end

# Interaction depth along the path: exponential with mean `L`, truncated at the
# crystal path length `maxd` (longer for oblique incidence).
_doi_depth(L, maxd, rng) = maxd <= 0 ? 0.0 :
    -L * log(1 - rand(rng) * (1 - exp(-maxd / L)))

# DOI depth for an endpoint on the inner cylinder, travelling along the (unit)
# lateral direction (dx,dy): exp-truncated at the along-track distance to r_out.
function _doi_along(px, py, dx, dy, r_out, L, rng)
    t = _ray_cyl_t(px, py, dx, dy, r_out)
    (isnan(t) || t <= 0) && return 0.0
    return _doi_depth(L, t, rng)
end

"""
    ideal_sphere_lors(n; sphere_R, r_det, halflength, rng) -> (xstart, xend)

Generate `n` IDEAL coincidences from a uniform sphere of radius `sphere_R`:
emit isotropically from a uniform point in the sphere, accept if both photons
geometrically reach the detector cylinder `r_det` within ±`halflength`. No energy
window, no containment, no efficiency — pure geometry. Used to check whether the
empirical-sensitivity radial structure survives with efficiency absent from the
inputs (→ it's the LOR measure, not efficiency).
"""
function ideal_sphere_lors(n; sphere_R, r_det, halflength, rng = Random.default_rng())
    xs = Matrix{Float32}(undef, 3, n)
    xe = Matrix{Float32}(undef, 3, n)
    got = 0
    while got < n
        rad = sphere_R * cbrt(rand(rng))                 # uniform point in sphere
        c0 = 2 * rand(rng) - 1; s0 = sqrt(1 - c0 * c0); a0 = 2π * rand(rng)
        px = rad * s0 * cos(a0); py = rad * s0 * sin(a0); pz = rad * c0
        c1 = 2 * rand(rng) - 1; s1 = sqrt(1 - c1 * c1); a1 = 2π * rand(rng)  # isotropic emission
        dx = s1 * cos(a1); dy = s1 * sin(a1); dz = c1
        ok = true; q1 = (0.0f0, 0.0f0, 0.0f0); q2 = (0.0f0, 0.0f0, 0.0f0)
        for sgn in (1, -1)
            ux, uy, uz = sgn * dx, sgn * dy, sgn * dz
            t = _ray_cyl_t(px, py, ux, uy, r_det)
            (isnan(t) || t <= 0) && (ok = false; break)
            qz = pz + t * uz
            abs(qz) > halflength && (ok = false; break)
            q = (Float32(px + t * ux), Float32(py + t * uy), Float32(qz))
            sgn == 1 ? (q1 = q) : (q2 = q)
        end
        ok || continue
        got += 1
        xs[1, got], xs[2, got], xs[3, got] = q1
        xe[1, got], xe[2, got], xe[3, got] = q2
    end
    return xs, xe
end

"""
    surface_doi_lors(n; r_inner, wall, halflength, att_length_mm, rng) -> (xstart, xend)

Sample `n` LORs as uniform detector endpoint pairs on the inner cylinder
`r_inner` (the source-free enumeration), then displace each endpoint along the
line to its crystal interaction depth (DOI): depth ~ exponential(`att_length_mm`)
truncated at the along-track path to `r_inner+wall`. So oblique LORs interact
deeper (longer crystal chord), automatically. For [`sensitivity_image`](@ref).
"""
function surface_doi_lors(n; r_inner, wall, halflength, att_length_mm,
                          rng = Random.default_rng())
    xs = Matrix{Float32}(undef, 3, n)
    xe = Matrix{Float32}(undef, 3, n)
    r_out = r_inner + wall
    for k in 1:n
        z1 = halflength * (2 * rand(rng) - 1); φ1 = 2π * rand(rng)
        z2 = halflength * (2 * rand(rng) - 1); φ2 = 2π * rand(rng)
        a1x = r_inner * cos(φ1); a1y = r_inner * sin(φ1)
        a2x = r_inner * cos(φ2); a2y = r_inner * sin(φ2)
        ux = a1x - a2x; uy = a1y - a2y; uz = z1 - z2
        nu = sqrt(ux * ux + uy * uy + uz * uz); ux /= nu; uy /= nu; uz /= nu
        s1 = _doi_along(a1x, a1y,  ux,  uy, r_out, att_length_mm, rng)   # endpoint 1 along +u
        s2 = _doi_along(a2x, a2y, -ux, -uy, r_out, att_length_mm, rng)   # endpoint 2 along -u
        xs[1, k] = Float32(a1x + s1 * ux); xs[2, k] = Float32(a1y + s1 * uy); xs[3, k] = Float32(z1 + s1 * uz)
        xe[1, k] = Float32(a2x - s2 * ux); xe[2, k] = Float32(a2y - s2 * uy); xe[3, k] = Float32(z2 - s2 * uz)
    end
    return xs, xe
end

"""
    emission_sens_lors(n; r_inner, wall, halflength, fov_radius, fov_halflength,
                       att_length_mm, rng) -> (xstart, xend)

Sample `n` detectable LORs by isotropic emission from the FOV with DOI, as `(3,n)`
Float32 endpoint matrices for [`sensitivity_image`](@ref). `r_inner`/`wall` are the
crystal inner radius and thickness, `halflength` the axial half-AFOV, `fov_*` the
emission region, `att_length_mm` the crystal absorption length at 511 keV.
"""
function emission_sens_lors(n; r_inner, wall, halflength, fov_radius, fov_halflength,
                            att_length_mm, rng = Random.default_rng())
    xs = Matrix{Float32}(undef, 3, n)
    xe = Matrix{Float32}(undef, 3, n)
    r_out = r_inner + wall
    got = 0
    while got < n
        # emission point uniform in the FOV cylinder
        rr = fov_radius * sqrt(rand(rng)); th = 2π * rand(rng)
        px = rr * cos(th); py = rr * sin(th); pz = fov_halflength * (2 * rand(rng) - 1)
        # isotropic direction
        ct = 2 * rand(rng) - 1; st = sqrt(max(0.0, 1 - ct * ct)); ph = 2π * rand(rng)
        dx = st * cos(ph); dy = st * sin(ph); dz = ct
        ok = true
        e = ((0.0f0, 0.0f0, 0.0f0), (0.0f0, 0.0f0, 0.0f0))
        for s in (1, -1)
            ux, uy, uz = s * dx, s * dy, s * dz
            t_in  = _ray_cyl_t(px, py, ux, uy, r_inner)
            t_out = _ray_cyl_t(px, py, ux, uy, r_out)
            (isnan(t_in) || t_in <= 0) && (ok = false; break)
            ez = pz + t_in * uz
            abs(ez) > halflength && (ok = false; break)          # misses the AFOV
            d = _doi_depth(att_length_mm, t_out - t_in, rng)     # DOI along the path
            q = (Float32(px + (t_in + d) * ux), Float32(py + (t_in + d) * uy),
                 Float32(pz + (t_in + d) * uz))
            e = s == 1 ? (q, e[2]) : (e[1], q)
        end
        ok || continue
        got += 1
        xs[1, got], xs[2, got], xs[3, got] = e[1]
        xe[1, got], xe[2, got], xe[3, got] = e[2]
    end
    return xs, xe
end
