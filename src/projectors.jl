# Joseph 3D matched forward/back projectors: the line-integral operator A
# (forward) and its exact matched adjoint Aᵀ (back), as a single
# KernelAbstractions kernel source running on CPU threads, Apple Metal, CUDA, ...
# All arithmetic is Float32 (Metal has no Float64). Ported from libparallelproj
# (KUL-recon-lab, Apache-2.0); see docs/tex/joseph3d_note.tex for the formal note.
#
# Joseph's method evaluates pᵢ = ∫ f̃ dℓ along LOR i (with
# f̃ the trilinear interpolation of the voxel image) by:
#   1. slab-method ray/cube intersection to find the principal axis `dir` (the
#      axis with the largest direction cosine) and the voxel-plane range to walk;
#   2. stepping plane-by-plane along `dir`, bilinearly interpolating in each
#      perpendicular plane (a gather for forward, an atomic scatter for back);
#   3. scaling the plane sum by the path-length correction cf = Δ_dir / cos θ_dir.
# The back kernel applies the exact transpose of these weights, so the pair
# satisfies ⟨Ax, y⟩ = ⟨x, Aᵀy⟩ to machine precision — the property MLEM/OSEM
# convergence relies on.
#
# Conventions (matching parallelproj):
#   - image: (n0, n1, n2) AbstractArray{Float32,3}; voxel (0,0,0) is centered at
#     world coordinates `img_origin`; voxel sizes are `voxsize`
#   - LOR endpoints: (3, nlors) matrices in world coordinates
#   - forward: proj[i] = line integral of the trilinearly interpolated image
#   - back: exact matched adjoint, accumulated into the image with atomics

# fminf/fmaxf semantics (ignore NaN operands) — Base.min/max propagate NaN, which
# breaks the IEEE-754 slab method when a ray is parallel to a slab (0 * Inf = NaN).
@inline ieee_min(a::Float32, b::Float32) = ifelse(isnan(a), b, ifelse(isnan(b), a, min(a, b)))
@inline ieee_max(a::Float32, b::Float32) = ifelse(isnan(a), b, ifelse(isnan(b), a, max(a, b)))

# branch-based tuple access for a runtime axis index (avoids dynamic tuple
# indexing inside GPU kernels)
@inline select3(t, k::Int32) = k == Int32(1) ? t[1] : (k == Int32(2) ? t[2] : t[3])

"""
    ray_cube_intersection_joseph(xs, xe, org, voxsize, n)

Slab-method ray/image-cube intersection with the Joseph-specific outputs:
`(direction, correction, start_plane, end_plane)` where `direction` is the
principal axis (1, 2 or 3), `correction = voxsize[dir] / cos[dir]`, and
`start_plane:end_plane` are the 0-based voxel-plane indices to traverse.
If the ray misses the cube, `start_plane == end_plane == -1`.
"""
@inline function ray_cube_intersection_joseph(
    xs::NTuple{3,Float32},
    xe::NTuple{3,Float32},
    org::NTuple{3,Float32},
    voxsize::NTuple{3,Float32},
    n::NTuple{3,Int32},
)
    dr = (xe[1] - xs[1], xe[2] - xs[2], xe[3] - xs[3])

    tmin = 0.0f0
    tmax = 1.0f0

    bmin = org[1] - 0.5f0 * voxsize[1]
    bmax = bmin + n[1] * voxsize[1]
    invd = 1.0f0 / dr[1]
    t1 = (bmin - xs[1]) * invd
    t2 = (bmax - xs[1]) * invd
    tmin = ieee_max(tmin, ieee_min(t1, t2))
    tmax = ieee_min(tmax, ieee_max(t1, t2))

    bmin = org[2] - 0.5f0 * voxsize[2]
    bmax = bmin + n[2] * voxsize[2]
    invd = 1.0f0 / dr[2]
    t1 = (bmin - xs[2]) * invd
    t2 = (bmax - xs[2]) * invd
    tmin = ieee_max(tmin, ieee_min(t1, t2))
    tmax = ieee_min(tmax, ieee_max(t1, t2))

    bmin = org[3] - 0.5f0 * voxsize[3]
    bmax = bmin + n[3] * voxsize[3]
    invd = 1.0f0 / dr[3]
    t1 = (bmin - xs[3]) * invd
    t2 = (bmax - xs[3]) * invd
    tmin = ieee_max(tmin, ieee_min(t1, t2))
    tmax = ieee_min(tmax, ieee_max(t1, t2))

    if tmax < tmin
        return Int32(1), 1.0f0, Int32(-1), Int32(-1)
    end

    c0 = dr[1] * dr[1]
    c1 = dr[2] * dr[2]
    c2 = dr[3] * dr[3]
    ssum = c0 + c1 + c2
    c0 /= ssum
    c1 /= ssum
    c2 /= ssum

    dir = Int32(1)
    if c1 >= c0 && c1 >= c2
        dir = Int32(2)
    elseif c2 >= c0 && c2 >= c1
        dir = Int32(3)
    end
    csd = select3((c0, c1, c2), dir)
    correction = select3(voxsize, dir) / sqrt(csd)

    xsd = select3(xs, dir)
    drd = select3(dr, dir)
    orgd = select3(org, dir)
    vsd = select3(voxsize, dir)
    f1 = (xsd + tmin * drd - orgd) / vsd
    f2 = (xsd + tmax * drd - orgd) / vsd

    start_plane = Int32(-1)
    end_plane = Int32(-1)
    # if the integer parts differ, at least one full voxel plane is crossed
    if unsafe_trunc(Int32, f1) != unsafe_trunc(Int32, f2)
        if f1 > f2
            f1, f2 = f2, f1
        end
        start_plane = unsafe_trunc(Int32, floor(f1)) + Int32(1)
        end_plane = unsafe_trunc(Int32, floor(f2))
    end

    return dir, correction, start_plane, end_plane
end

# 0-based sample with zero padding outside the image (column-major layout)
@inline function sample3(img, n::NTuple{3,Int32}, i0::Int32, i1::Int32, i2::Int32)
    if i0 < Int32(0) || i0 >= n[1] || i1 < Int32(0) || i1 >= n[2] || i2 < Int32(0) || i2 >= n[3]
        return 0.0f0
    end
    @inbounds return img[(i2 * n[2] + i1) * n[1] + i0 + Int32(1)]
end

# 0-based atomic scatter, ignoring out-of-image targets
@inline function inject3!(img, n::NTuple{3,Int32}, i0::Int32, i1::Int32, i2::Int32, v::Float32)
    if i0 < Int32(0) || i0 >= n[1] || i1 < Int32(0) || i1 >= n[2] || i2 < Int32(0) || i2 >= n[3]
        return nothing
    end
    @inbounds @atomic img[(i2 * n[2] + i1) * n[1] + i0 + Int32(1)] += v
    return nothing
end

@inline function bilinear_interp_fixed0(img, n, i0::Int32, f1::Float32, f2::Float32)
    j1 = unsafe_trunc(Int32, floor(f1))
    j2 = unsafe_trunc(Int32, floor(f2))
    w1 = f1 - j1
    w2 = f2 - j2
    v00 = sample3(img, n, i0, j1, j2)
    v10 = sample3(img, n, i0, j1 + Int32(1), j2)
    v01 = sample3(img, n, i0, j1, j2 + Int32(1))
    v11 = sample3(img, n, i0, j1 + Int32(1), j2 + Int32(1))
    return v00 * (1 - w1) * (1 - w2) + v10 * w1 * (1 - w2) + v01 * (1 - w1) * w2 + v11 * w1 * w2
end

@inline function bilinear_interp_fixed1(img, n, f0::Float32, i1::Int32, f2::Float32)
    j0 = unsafe_trunc(Int32, floor(f0))
    j2 = unsafe_trunc(Int32, floor(f2))
    w0 = f0 - j0
    w2 = f2 - j2
    v00 = sample3(img, n, j0, i1, j2)
    v10 = sample3(img, n, j0 + Int32(1), i1, j2)
    v01 = sample3(img, n, j0, i1, j2 + Int32(1))
    v11 = sample3(img, n, j0 + Int32(1), i1, j2 + Int32(1))
    return v00 * (1 - w0) * (1 - w2) + v10 * w0 * (1 - w2) + v01 * (1 - w0) * w2 + v11 * w0 * w2
end

@inline function bilinear_interp_fixed2(img, n, f0::Float32, f1::Float32, i2::Int32)
    j0 = unsafe_trunc(Int32, floor(f0))
    j1 = unsafe_trunc(Int32, floor(f1))
    w0 = f0 - j0
    w1 = f1 - j1
    v00 = sample3(img, n, j0, j1, i2)
    v10 = sample3(img, n, j0 + Int32(1), j1, i2)
    v01 = sample3(img, n, j0, j1 + Int32(1), i2)
    v11 = sample3(img, n, j0 + Int32(1), j1 + Int32(1), i2)
    return v00 * (1 - w0) * (1 - w1) + v10 * w0 * (1 - w1) + v01 * (1 - w0) * w1 + v11 * w0 * w1
end

@inline function bilinear_interp_adj_fixed0!(img, n, i0::Int32, f1::Float32, f2::Float32, val::Float32)
    j1 = unsafe_trunc(Int32, floor(f1))
    j2 = unsafe_trunc(Int32, floor(f2))
    w1 = f1 - j1
    w2 = f2 - j2
    inject3!(img, n, i0, j1, j2, val * (1 - w1) * (1 - w2))
    inject3!(img, n, i0, j1 + Int32(1), j2, val * w1 * (1 - w2))
    inject3!(img, n, i0, j1, j2 + Int32(1), val * (1 - w1) * w2)
    inject3!(img, n, i0, j1 + Int32(1), j2 + Int32(1), val * w1 * w2)
    return nothing
end

@inline function bilinear_interp_adj_fixed1!(img, n, f0::Float32, i1::Int32, f2::Float32, val::Float32)
    j0 = unsafe_trunc(Int32, floor(f0))
    j2 = unsafe_trunc(Int32, floor(f2))
    w0 = f0 - j0
    w2 = f2 - j2
    inject3!(img, n, j0, i1, j2, val * (1 - w0) * (1 - w2))
    inject3!(img, n, j0 + Int32(1), i1, j2, val * w0 * (1 - w2))
    inject3!(img, n, j0, i1, j2 + Int32(1), val * (1 - w0) * w2)
    inject3!(img, n, j0 + Int32(1), i1, j2 + Int32(1), val * w0 * w2)
    return nothing
end

@inline function bilinear_interp_adj_fixed2!(img, n, f0::Float32, f1::Float32, i2::Int32, val::Float32)
    j0 = unsafe_trunc(Int32, floor(f0))
    j1 = unsafe_trunc(Int32, floor(f1))
    w0 = f0 - j0
    w1 = f1 - j1
    inject3!(img, n, j0, j1, i2, val * (1 - w0) * (1 - w1))
    inject3!(img, n, j0 + Int32(1), j1, i2, val * w0 * (1 - w1))
    inject3!(img, n, j0, j1 + Int32(1), i2, val * (1 - w0) * w1)
    inject3!(img, n, j0 + Int32(1), j1 + Int32(1), i2, val * w0 * w1)
    return nothing
end

@kernel function fwd_kernel!(proj, @Const(xstart), @Const(xend), @Const(img), org, voxsize, n)
    i = @index(Global)
    @inbounds begin
        xs = (xstart[1, i], xstart[2, i], xstart[3, i])
        xe = (xend[1, i], xend[2, i], xend[3, i])

        dir, cf, istart, iend = ray_cube_intersection_joseph(xs, xe, org, voxsize, n)

        acc = 0.0f0
        if istart != Int32(-1)
            d0 = xe[1] - xs[1]
            d1 = xe[2] - xs[2]
            d2 = xe[3] - xs[3]
            if dir == Int32(1)
                a1 = (d1 * voxsize[1]) / (voxsize[2] * d0)
                b1 = (xs[2] - org[2] + d1 * (org[1] - xs[1]) / d0) / voxsize[2]
                a2 = (d2 * voxsize[1]) / (voxsize[3] * d0)
                b2 = (xs[3] - org[3] + d2 * (org[1] - xs[1]) / d0) / voxsize[3]
                f1 = istart * a1 + b1
                f2 = istart * a2 + b2
                for i0 in istart:iend
                    acc += bilinear_interp_fixed0(img, n, i0, f1, f2)
                    f1 += a1
                    f2 += a2
                end
            elseif dir == Int32(2)
                a0 = (d0 * voxsize[2]) / (voxsize[1] * d1)
                b0 = (xs[1] - org[1] + d0 * (org[2] - xs[2]) / d1) / voxsize[1]
                a2 = (d2 * voxsize[2]) / (voxsize[3] * d1)
                b2 = (xs[3] - org[3] + d2 * (org[2] - xs[2]) / d1) / voxsize[3]
                f0 = istart * a0 + b0
                f2 = istart * a2 + b2
                for i1 in istart:iend
                    acc += bilinear_interp_fixed1(img, n, f0, i1, f2)
                    f0 += a0
                    f2 += a2
                end
            else
                a0 = (d0 * voxsize[3]) / (voxsize[1] * d2)
                b0 = (xs[1] - org[1] + d0 * (org[3] - xs[3]) / d2) / voxsize[1]
                a1 = (d1 * voxsize[3]) / (voxsize[2] * d2)
                b1 = (xs[2] - org[2] + d1 * (org[3] - xs[3]) / d2) / voxsize[2]
                f0 = istart * a0 + b0
                f1 = istart * a1 + b1
                for i2 in istart:iend
                    acc += bilinear_interp_fixed2(img, n, f0, f1, i2)
                    f0 += a0
                    f1 += a1
                end
            end
        end
        proj[i] = cf * acc
    end
end

@kernel function back_kernel!(img, @Const(xstart), @Const(xend), @Const(proj), org, voxsize, n)
    i = @index(Global)
    @inbounds begin
        p = proj[i]
        if p != 0.0f0
            xs = (xstart[1, i], xstart[2, i], xstart[3, i])
            xe = (xend[1, i], xend[2, i], xend[3, i])

            dir, cf, istart, iend = ray_cube_intersection_joseph(xs, xe, org, voxsize, n)

            if istart != Int32(-1)
                val = cf * p
                d0 = xe[1] - xs[1]
                d1 = xe[2] - xs[2]
                d2 = xe[3] - xs[3]
                if dir == Int32(1)
                    a1 = (d1 * voxsize[1]) / (voxsize[2] * d0)
                    b1 = (xs[2] - org[2] + d1 * (org[1] - xs[1]) / d0) / voxsize[2]
                    a2 = (d2 * voxsize[1]) / (voxsize[3] * d0)
                    b2 = (xs[3] - org[3] + d2 * (org[1] - xs[1]) / d0) / voxsize[3]
                    f1 = istart * a1 + b1
                    f2 = istart * a2 + b2
                    for i0 in istart:iend
                        bilinear_interp_adj_fixed0!(img, n, i0, f1, f2, val)
                        f1 += a1
                        f2 += a2
                    end
                elseif dir == Int32(2)
                    a0 = (d0 * voxsize[2]) / (voxsize[1] * d1)
                    b0 = (xs[1] - org[1] + d0 * (org[2] - xs[2]) / d1) / voxsize[1]
                    a2 = (d2 * voxsize[2]) / (voxsize[3] * d1)
                    b2 = (xs[3] - org[3] + d2 * (org[2] - xs[2]) / d1) / voxsize[3]
                    f0 = istart * a0 + b0
                    f2 = istart * a2 + b2
                    for i1 in istart:iend
                        bilinear_interp_adj_fixed1!(img, n, f0, i1, f2, val)
                        f0 += a0
                        f2 += a2
                    end
                else
                    a0 = (d0 * voxsize[3]) / (voxsize[1] * d2)
                    b0 = (xs[1] - org[1] + d0 * (org[3] - xs[3]) / d2) / voxsize[1]
                    a1 = (d1 * voxsize[3]) / (voxsize[2] * d2)
                    b1 = (xs[2] - org[2] + d1 * (org[3] - xs[3]) / d2) / voxsize[2]
                    f0 = istart * a0 + b0
                    f1 = istart * a1 + b1
                    for i2 in istart:iend
                        bilinear_interp_adj_fixed2!(img, n, f0, f1, i2, val)
                        f0 += a0
                        f1 += a1
                    end
                end
            end
        end
    end
end

"""
    joseph3d_fwd!(proj, xstart, xend, img, img_origin, voxsize)

Joseph 3D forward projection of `img` along the LORs with endpoints given by
the columns of `xstart` and `xend` (size `(3, nlors)`), writing the line
integrals into `proj` (length `nlors`). Runs on the backend of `img`
(CPU, Metal, CUDA, ...). All arrays must live on the same device.
"""
function joseph3d_fwd!(
    proj::AbstractVector{Float32},
    xstart::AbstractMatrix{Float32},
    xend::AbstractMatrix{Float32},
    img::AbstractArray{Float32,3},
    img_origin,
    voxsize,
)
    nlors = size(xstart, 2)
    @assert size(xstart, 1) == 3 && size(xend) == size(xstart) && length(proj) == nlors
    backend = KA.get_backend(img)
    fwd_kernel!(backend)(
        proj, xstart, xend, img,
        NTuple{3,Float32}(img_origin), NTuple{3,Float32}(voxsize), Int32.(size(img));
        ndrange = nlors,
    )
    KA.synchronize(backend)
    return proj
end

"""
    joseph3d_fwd(xstart, xend, img, img_origin, voxsize) -> proj

Allocating variant of [`joseph3d_fwd!`](@ref).
"""
function joseph3d_fwd(xstart, xend, img, img_origin, voxsize)
    proj = similar(img, Float32, size(xstart, 2))
    return joseph3d_fwd!(proj, xstart, xend, img, img_origin, voxsize)
end

"""
    joseph3d_back!(img, xstart, xend, proj, img_origin, voxsize)

Matched adjoint of [`joseph3d_fwd!`](@ref): accumulates `proj`-weighted ray
contributions into `img` (not zeroed here — zero it first if needed).
"""
function joseph3d_back!(
    img::AbstractArray{Float32,3},
    xstart::AbstractMatrix{Float32},
    xend::AbstractMatrix{Float32},
    proj::AbstractVector{Float32},
    img_origin,
    voxsize,
)
    nlors = size(xstart, 2)
    @assert size(xstart, 1) == 3 && size(xend) == size(xstart) && length(proj) == nlors
    backend = KA.get_backend(img)
    back_kernel!(backend)(
        img, xstart, xend, proj,
        NTuple{3,Float32}(img_origin), NTuple{3,Float32}(voxsize), Int32.(size(img));
        ndrange = nlors,
    )
    KA.synchronize(backend)
    return img
end

"""
    joseph3d_back(xstart, xend, proj, img_shape, img_origin, voxsize) -> img

Allocating variant of [`joseph3d_back!`](@ref); the image starts at zero and is
allocated on the device of `proj`.
"""
function joseph3d_back(xstart, xend, proj, img_shape, img_origin, voxsize)
    img = similar(proj, Float32, img_shape)
    fill!(img, 0.0f0)
    return joseph3d_back!(img, xstart, xend, proj, img_origin, voxsize)
end
