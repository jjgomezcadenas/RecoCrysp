# NEMA NU-2 Image-Quality phantom geometry (geometry_nema.json + the source
# regions): a body cylinder R=100mm, half-length 90mm, with six hot spheres
# (Ø 10/13/17/22/28/37 mm) on a 57.2mm-radius ring at z=0, hot:background = 4:1.
# Dimensions in mm. Provides the ground-truth map and the sphere / background ROIs
# for contrast-recovery analysis. General (shared by nema_la and nema).

struct NemaSphere
    diam_mm::Float64
    r_mm::Float64
    pos::NTuple{3,Float64}
end

const NEMA_BODY_R_MM    = 100.0
const NEMA_BODY_HALF_MM = 90.0
const NEMA_HOT_RATIO    = 4.0          # sphere : background activity concentration
const NEMA_SPHERES = (
    NemaSphere(37.0, 18.5, (  57.2,   0.0, 0.0)),
    NemaSphere(28.0, 14.0, (  28.6,  49.5, 0.0)),
    NemaSphere(22.0, 11.0, ( -28.6,  49.5, 0.0)),
    NemaSphere(17.0,  8.5, ( -57.2,   0.0, 0.0)),
    NemaSphere(13.0,  6.5, ( -28.6, -49.5, 0.0)),
    NemaSphere(10.0,  5.0, (  28.6, -49.5, 0.0)),
)

_nema_axis(org, vs, n, d) = Float64[org[d] + (i - 1) * vs[d] for i in 1:n[d]]

"""
    nema_sphere_masks(n, org, vs; shrink_mm=0) -> Vector{Tuple{Float64,BitArray{3}}}

For each NEMA hot sphere, a boolean mask of grid voxels within `r-shrink_mm` of the
sphere centre (`shrink_mm` to back off the partial-volume rim). Returns
`(diameter_mm, mask)` pairs, largest sphere first.
"""
function nema_sphere_masks(n, org, vs; shrink_mm = 0.0)
    xs = _nema_axis(org, vs, n, 1); ys = _nema_axis(org, vs, n, 2); zs = _nema_axis(org, vs, n, 3)
    out = Tuple{Float64,BitArray{3}}[]
    for s in NEMA_SPHERES
        rr2 = max(s.r_mm - shrink_mm, 0.0)^2
        m = falses(n...)
        @inbounds for k in 1:n[3], j in 1:n[2], i in 1:n[1]
            m[i, j, k] = (xs[i] - s.pos[1])^2 + (ys[j] - s.pos[2])^2 + (zs[k] - s.pos[3])^2 <= rr2
        end
        push!(out, (s.diam_mm, m))
    end
    return out
end

"""
    nema_background_mask(n, org, vs; r_max_mm=30, z_half_mm=10) -> BitArray{3}

Central, sphere-free background ROI: voxels with transverse radius < `r_max_mm`
(the spheres sit on a 57.2mm ring, so the centre is clean background) and
|z| < `z_half_mm`.
"""
function nema_background_mask(n, org, vs; r_max_mm = 30.0, z_half_mm = 10.0)
    xs = _nema_axis(org, vs, n, 1); ys = _nema_axis(org, vs, n, 2); zs = _nema_axis(org, vs, n, 3)
    m = falses(n...)
    @inbounds for k in 1:n[3], j in 1:n[2], i in 1:n[1]
        m[i, j, k] = (xs[i]^2 + ys[j]^2 <= r_max_mm^2) && (abs(zs[k]) <= z_half_mm)
    end
    return m
end

"""
    nema_background_variability(img, n, org, vs; z_off, edge_mm, clear_mm)
        -> Vector{Tuple{Float64,Float64}}

NEMA NU-2 background variability per sphere size (the standard noise metric, *not*
per-voxel CoV): for each sphere, lay non-overlapping background ROIs of that
sphere's diameter on the central slice and at the axial offsets `z_off` (mm), each
fully inside the body (≥ `edge_mm` from the wall) and clear of every sphere
(≥ `clear_mm`), and report `BV = 100·std(ROI means)/mean(ROI means)`. Returns
`(diam_mm, BV%)`, largest sphere first. Because the ROI is the sphere's size, BV
falls with sphere diameter (ROI averaging) -- the literature behaviour. Placement
is a regular non-overlapping grid (an approximation to NEMA's hand-placed 12/slice).
"""
function nema_background_variability(img, n, org, vs;
        z_off = (-20.0, -10.0, 0.0, 10.0, 20.0), edge_mm = 8.0, clear_mm = 12.0)
    xs = _nema_axis(org, vs, n, 1); ys = _nema_axis(org, vs, n, 2); zs = _nema_axis(org, vs, n, 3)
    out = Tuple{Float64,Float64}[]
    for s in NEMA_SPHERES
        rd = s.diam_mm / 2; step = 2 * rd
        means = Float64[]
        for zoff in z_off
            (zoff < zs[1] || zoff > zs[end]) && continue
            k = argmin(abs.(zs .- zoff))
            cy = -NEMA_BODY_R_MM + rd
            while cy <= NEMA_BODY_R_MM - rd
                cx = -NEMA_BODY_R_MM + rd
                while cx <= NEMA_BODY_R_MM - rd
                    inbody = hypot(cx, cy) <= NEMA_BODY_R_MM - edge_mm - rd
                    clear = all((cx - sp.pos[1])^2 + (cy - sp.pos[2])^2 >= (sp.r_mm + rd + clear_mm)^2
                                for sp in NEMA_SPHERES)
                    if inbody && clear
                        ssum = 0.0; cnt = 0
                        @inbounds for j in 1:n[2], i in 1:n[1]
                            if (xs[i] - cx)^2 + (ys[j] - cy)^2 <= rd^2
                                ssum += img[i, j, k]; cnt += 1
                            end
                        end
                        cnt > 0 && push!(means, ssum / cnt)
                    end
                    cx += step
                end
                cy += step
            end
        end
        if length(means) >= 2
            mu = sum(means) / length(means)
            sd = sqrt(sum((m - mu)^2 for m in means) / length(means))
            push!(out, (s.diam_mm, 100 * sd / mu))
        else
            push!(out, (s.diam_mm, NaN))
        end
    end
    return out
end

"""
    nema_true_image(n, org, vs) -> Array{Float32,3}

Ground-truth activity in relative units: 1 in the body cylinder, `NEMA_HOT_RATIO`
in the hot spheres. For a reference panel / sanity check.
"""
function nema_true_image(n, org, vs)
    xs = _nema_axis(org, vs, n, 1); ys = _nema_axis(org, vs, n, 2); zs = _nema_axis(org, vs, n, 3)
    img = zeros(Float32, n...)
    @inbounds for k in 1:n[3], j in 1:n[2], i in 1:n[1]
        inbody = (xs[i]^2 + ys[j]^2 <= NEMA_BODY_R_MM^2) && (abs(zs[k]) <= NEMA_BODY_HALF_MM)
        img[i, j, k] = inbody ? 1.0f0 : 0.0f0
    end
    for s in NEMA_SPHERES
        @inbounds for k in 1:n[3], j in 1:n[2], i in 1:n[1]
            if (xs[i] - s.pos[1])^2 + (ys[j] - s.pos[2])^2 + (zs[k] - s.pos[3])^2 <= s.r_mm^2
                img[i, j, k] = Float32(NEMA_HOT_RATIO)
            end
        end
    end
    return img
end
