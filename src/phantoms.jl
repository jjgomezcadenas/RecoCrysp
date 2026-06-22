# Digital phantoms rasterized onto a voxel grid: a uniform sphere, a uniform
# cylinder, and a Derenzo rod pattern. Each returns a Float32 image on the grid
# defined by (img_shape, img_origin, voxsize) — world coordinates in mm, the
# cylinder/Derenzo axis along z (axis 3). Built on the host; move to the GPU with
# the array constructor if needed.

# voxel-centre world coordinate vectors for the grid
function _grid_axes(shape, origin, voxsize)
    vs = voxsize isa Number ? ntuple(_ -> Float32(voxsize), 3) : NTuple{3,Float32}(voxsize)
    org = NTuple{3,Float32}(origin)
    n = NTuple{3,Int}(shape)
    ax = ntuple(d -> Float32[org[d] + (i - 1) * vs[d] for i in 1:n[d]], 3)
    return n, vs, org, ax
end

# index range of axis `d` whose coordinate falls in [lo, hi]
_axis_range(lo, hi, org_d, vs_d, n_d) =
    max(1, ceil(Int, (lo - org_d) / vs_d + 1)):min(n_d, floor(Int, (hi - org_d) / vs_d + 1))

"""
    uniform_sphere(shape, origin, voxsize; radius, value = 1, center = (0,0,0)) -> img

`Float32` image with `value` inside a sphere of `radius` (mm) centred at `center`
(mm), zero outside.
"""
function uniform_sphere(shape, origin, voxsize; radius, value = 1, center = (0, 0, 0))
    n, _, _, ax = _grid_axes(shape, origin, voxsize)
    x, y, z = ax
    c = NTuple{3,Float32}(center)
    img = zeros(Float32, n)
    R2 = Float32(radius)^2
    v = Float32(value)
    @inbounds for k in 1:n[3], j in 1:n[2], i in 1:n[1]
        ((x[i] - c[1])^2 + (y[j] - c[2])^2 + (z[k] - c[3])^2 <= R2) && (img[i, j, k] = v)
    end
    return img
end

"""
    uniform_cylinder(shape, origin, voxsize; radius, length, value = 1, center = (0,0,0)) -> img

`Float32` image with `value` inside a cylinder of transverse `radius` and axial
extent `length` (mm) about `center`, axis along z, zero outside.
"""
function uniform_cylinder(shape, origin, voxsize; radius, length, value = 1, center = (0, 0, 0))
    n, _, _, ax = _grid_axes(shape, origin, voxsize)
    x, y, z = ax
    c = NTuple{3,Float32}(center)
    img = zeros(Float32, n)
    R2 = Float32(radius)^2
    hz = Float32(length) / 2
    v = Float32(value)
    @inbounds for k in 1:n[3], j in 1:n[2], i in 1:n[1]
        if abs(z[k] - c[3]) <= hz && (x[i] - c[1])^2 + (y[j] - c[2])^2 <= R2
            img[i, j, k] = v
        end
    end
    return img
end

"""
    derenzo(shape, origin, voxsize; radius, length, rod_diameters,
            spacing = 2.0, value = 1, center = (0,0,0)) -> img

Derenzo phantom: `length(rod_diameters)` angular sectors of hot rods inside a
cylinder of transverse `radius` and axial extent `length` (mm), axis along z.
Sector `s` is filled with rods of diameter `rod_diameters[s]` (mm) on a
triangular lattice with centre-to-centre pitch `spacing × diameter`. Rods carry
`value`; everything else is zero.
"""
function derenzo(shape, origin, voxsize; radius, length, rod_diameters,
                 spacing = 2.0, value = 1, center = (0, 0, 0))
    n, _, org, ax = _grid_axes(shape, origin, voxsize)
    vs = voxsize isa Number ? ntuple(_ -> Float32(voxsize), 3) : NTuple{3,Float32}(voxsize)
    x, y, z = ax
    c = NTuple{3,Float32}(center)
    img = zeros(Float32, n)
    v = Float32(value)
    Rcyl = Float32(radius)
    hz = Float32(length) / 2
    kz = _axis_range(c[3] - hz, c[3] + hz, org[3], vs[3], n[3])   # axial extent (shared)

    nsec = size(rod_diameters, 1)
    half = Float32(pi) / nsec                                     # half sector angle
    for s in 1:nsec
        d = Float32(rod_diameters[s])
        rr = d / 2
        p = Float32(spacing) * d                                  # lattice pitch
        φ0 = 2.0f0 * Float32(pi) * (s - 1) / nsec + half          # sector bisector
        cu, su = cos(φ0), sin(φ0)
        rowstep = p * sqrt(3.0f0) / 2
        rmin = 2 * p                                              # central gap
        row = 0
        while true
            u = rmin + row * rowstep
            u > Rcyl - rr && break
            halfw = u * tan(half) - rr                            # wedge half-width
            off = (row % 2) * p / 2                               # triangular offset
            row += 1
            halfw < 0 && continue
            m = floor(Int, (halfw + abs(off)) / p) + 1
            for col in -m:m
                w = col * p + off
                abs(w) <= halfw || continue
                cx = c[1] + u * cu - w * su
                cy = c[2] + u * su + w * cu
                (cx^2 + cy^2 > (Rcyl - rr)^2) && continue
                ii = _axis_range(cx - rr, cx + rr, org[1], vs[1], n[1])
                jj = _axis_range(cy - rr, cy + rr, org[2], vs[2], n[2])
                rr2 = rr^2
                @inbounds for k in kz, j in jj, i in ii
                    ((x[i] - cx)^2 + (y[j] - cy)^2 <= rr2) && (img[i, j, k] = v)
                end
            end
        end
    end
    return img
end
