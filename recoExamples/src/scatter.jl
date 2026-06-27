# Scatter contamination estimate from the ground-truth flags.
#
# The flagged scatter events (truth==1) are samples of the scatter distribution.
# We build a SMOOTH scatter model from them (like a single-scatter simulation
# produces a smooth sinogram), not an event-by-event subtraction. Each LOR is
# reduced to sinogram coordinates (radial offset s_r, axial midpoint z_m,
# obliquity dz; azimuth drops out for a centred phantom), scatter and prompts are
# histogrammed and smoothed there, and the per-event background is the smoothed
# local scatter fraction
#   s_i = S~_b(i) / P~_b(i),
# which sums to the scatter total by construction. Feeds `contamination` in
# ListmodePoissonModel, same per-event convention as the randoms estimate.

"""
    lor_sinogram_coords(xs, xe) -> (s_r, z_m, dz)

Reduced sinogram coordinates for the `(3,N)` endpoint columns: `s_r` the
perpendicular distance of each LOR from the z-axis (transverse radial offset),
`z_m = (z1+z2)/2` the axial midpoint, `dz = z2-z1` the obliquity. All `Float32`
vectors of length N (CPU).
"""
function lor_sinogram_coords(xs, xe)
    N = size(xs, 2)
    s_r = Vector{Float32}(undef, N); z_m = similar(s_r); dz = similar(s_r)
    @inbounds for i in 1:N
        x1 = xs[1, i]; y1 = xs[2, i]; z1 = xs[3, i]
        x2 = xe[1, i]; y2 = xe[2, i]; z2 = xe[3, i]
        dxy = hypot(x2 - x1, y2 - y1)
        s_r[i] = dxy < 1.0f-6 ? 0.0f0 : abs(x1 * y2 - x2 * y1) / dxy
        z_m[i] = 0.5f0 * (z1 + z2)
        dz[i]  = z2 - z1
    end
    return s_r, z_m, dz
end

# 1D Gaussian kernel (radius 3 sigma), normalized.
function _gauss_kernel(sigma)
    sigma <= 0 && return [1.0]
    r = max(1, ceil(Int, 3 * sigma))
    k = [exp(-(i^2) / (2 * sigma^2)) for i in -r:r]
    return k ./ sum(k)
end

# Separable Gaussian smoothing of a 3D array (reflect at the borders).
function _smooth3(A, sigma)
    k = _gauss_kernel(sigma)
    length(k) == 1 && return A
    r = (length(k) - 1) ÷ 2
    sz = size(A)
    B = copy(A)
    for d in 1:3
        C = similar(B)
        n = sz[d]
        for I in CartesianIndices(B)
            acc = 0.0
            base = I[d]
            for (m, w) in enumerate(k)
                j = clamp(base + (m - 1 - r), 1, n)      # reflect/clamp at edges
                Jt = ntuple(t -> t == d ? j : I[t], 3)
                acc += w * B[CartesianIndex(Jt)]
            end
            C[I] = acc
        end
        B = C
    end
    return B
end

_binidx(v, lo, hi, n) = clamp(floor(Int, (v - lo) / (hi - lo) * n) + 1, 1, n)

"""
    scatter_estimate(s_r, z_m, dz, scat_mask; n_sr, n_zm, n_dz, smooth, total=nothing)
        -> Vector{Float32}

Per-event scatter background `s_i` for the prompt LORs whose sinogram coordinates
are `s_r,z_m,dz` (length N), `scat_mask` flagging which of them are scatter. Both
the scatter subset and all prompts are histogrammed on an `n_sr×n_zm×n_dz` grid
spanning the data range, Gaussian-smoothed (`smooth`, in bins), and `s_i` is the
smoothed local scatter fraction `S~/P~` at each event's bin. If `total` is given
(e.g. the flagged scatter count) the result is rescaled to sum to it.
"""
function scatter_estimate(s_r, z_m, dz, scat_mask;
                          n_sr, n_zm, n_dz, smooth, total = nothing)
    lo_r, hi_r = extrema(s_r); lo_z, hi_z = extrema(z_m); lo_d, hi_d = extrema(dz)
    hi_r = hi_r > lo_r ? hi_r : lo_r + 1; hi_z = hi_z > lo_z ? hi_z : lo_z + 1
    hi_d = hi_d > lo_d ? hi_d : lo_d + 1
    S = zeros(Float64, n_sr, n_zm, n_dz); P = zeros(Float64, n_sr, n_zm, n_dz)
    @inbounds for i in eachindex(s_r)
        bi = _binidx(s_r[i], lo_r, hi_r, n_sr)
        bj = _binidx(z_m[i], lo_z, hi_z, n_zm)
        bk = _binidx(dz[i], lo_d, hi_d, n_dz)
        P[bi, bj, bk] += 1
        scat_mask[i] && (S[bi, bj, bk] += 1)
    end
    S = _smooth3(S, smooth); P = _smooth3(P, smooth)
    s = Vector{Float32}(undef, length(s_r))
    @inbounds for i in eachindex(s_r)
        bi = _binidx(s_r[i], lo_r, hi_r, n_sr)
        bj = _binidx(z_m[i], lo_z, hi_z, n_zm)
        bk = _binidx(dz[i], lo_d, hi_d, n_dz)
        s[i] = P[bi, bj, bk] > 0 ? Float32(S[bi, bj, bk] / P[bi, bj, bk]) : 0.0f0
    end
    if total !== nothing
        sm = sum(s); sm > 0 && (s .*= Float32(total / sm))
    end
    return s
end
