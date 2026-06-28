# Smooth background (contamination) estimate from the ground-truth flags.
#
# A flagged background class -- scatter (truth==1) OR randoms (truth==2) -- gives
# samples of that background's distribution. We build a SMOOTH model from them
# (like a single-scatter simulation produces a smooth sinogram), not an
# event-by-event subtraction. Each LOR is reduced to sinogram coordinates (radial
# offset s_r, axial midpoint z_m, obliquity dz; azimuth drops out for a centred
# phantom), the background subset and all prompts are histogrammed and smoothed
# there, and the per-event background is the smoothed local background fraction
#   b_i = B~_b(i) / P~_b(i),
# which sums to the background total by construction. Feeds `contamination` in
# ListmodePoissonModel. Class-agnostic: pass the scatter mask for scatter, the
# randoms mask for randoms.

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

# Separable Gaussian smoothing of a 3D array (clamp at borders). `sigmas` is the
# per-axis width in bins; an axis with sigma <= 0 is left UNSMOOTHED -- used to
# preserve a genuinely sharp physical edge along that axis (here s_r at s_r = R).
function _smooth3(A, sigmas)
    sz = size(A)
    B = copy(A)
    for d in 1:3
        sigmas[d] <= 0 && continue
        k = _gauss_kernel(sigmas[d])
        length(k) == 1 && continue
        r = (length(k) - 1) ÷ 2
        C = similar(B); n = sz[d]
        for I in CartesianIndices(B)
            acc = 0.0; base = I[d]
            for (m, w) in enumerate(k)
                j = clamp(base + (m - 1 - r), 1, n)
                Jt = ntuple(t -> t == d ? j : I[t], 3)
                acc += w * B[CartesianIndex(Jt)]
            end
            C[I] = acc
        end
        B = C
    end
    return B
end

"""
    gaussian_postfilter(img, fwhm_mm, voxsize) -> Array

Separable 3D Gaussian post-filter of full-width-half-maximum `fwhm_mm`, the
standard noise control for an unregularized MLEM/OSEM reconstruction (iterate to
converge contrast, then smooth away the high-frequency speckle). Per-axis sigma in
voxels is `fwhm_mm / (2.3548 * voxsize[d])`; `fwhm_mm <= 0` returns `img` unchanged.
"""
gaussian_postfilter(img, fwhm_mm, voxsize) = fwhm_mm <= 0 ? img :
    _smooth3(img, ntuple(d -> fwhm_mm / (2.3548 * Float64(voxsize[d])), 3))

_binidx(v, lo, hi, n) = clamp(floor(Int, (v - lo) / (hi - lo) * n) + 1, 1, n)

# Histogram the background subset and all prompts on the sinogram grid over the
# given per-axis `(lo,hi)` spans (out-of-range events clamp into the edge bins --
# the far tails are nearly empty). Shared by background_estimate and
# background_sinograms so they always bin identically.
function _sino_hist(s_r, z_m, dz, bg_mask; n_sr, n_zm, n_dz, span_sr, span_zm, span_dz)
    lo_r, hi_r = span_sr; lo_z, hi_z = span_zm; lo_d, hi_d = span_dz
    S = zeros(Float64, n_sr, n_zm, n_dz); P = zeros(Float64, n_sr, n_zm, n_dz)
    @inbounds for i in eachindex(s_r)
        bi = _binidx(s_r[i], lo_r, hi_r, n_sr)
        bj = _binidx(z_m[i], lo_z, hi_z, n_zm)
        bk = _binidx(dz[i], lo_d, hi_d, n_dz)
        P[bi, bj, bk] += 1
        bg_mask[i] && (S[bi, bj, bk] += 1)
    end
    return S, P, span_sr, span_zm, span_dz
end

"""
    background_estimate(s_r, z_m, dz, bg_mask; n_sr, n_zm, n_dz,
                     span_sr, span_zm, span_dz, smooth, total=nothing) -> Vector{Float32}

Per-event background `b_i` for the prompt LORs whose sinogram coordinates are
`s_r,z_m,dz` (length N), `bg_mask` flagging which of them belong to the background
class (scatter or randoms). Both the background subset and all prompts are
histogrammed on an `n_sr×n_zm×n_dz` grid over the per-axis `(lo,hi)` spans,
smoothed with the per-axis widths `smooth` (3-tuple, in bins; an axis with 0 is
left sharp -- use 0 for `s_r` to keep a background edge at `s_r=R`), and `b_i` is
the smoothed local background fraction `B~/P~` at each event's bin. If `total` is
given the result is rescaled to sum to it.
"""
function background_estimate(s_r, z_m, dz, bg_mask;
                          n_sr, n_zm, n_dz, span_sr, span_zm, span_dz,
                          smooth, total = nothing)
    S, P = _sino_hist(s_r, z_m, dz, bg_mask; n_sr = n_sr, n_zm = n_zm, n_dz = n_dz,
                      span_sr = span_sr, span_zm = span_zm, span_dz = span_dz)
    S = _smooth3(S, smooth); P = _smooth3(P, smooth)
    lo_r, hi_r = span_sr; lo_z, hi_z = span_zm; lo_d, hi_d = span_dz
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

"""
    background_sinograms(s_r, z_m, dz, bg_mask; n_sr, n_zm, n_dz,
                      span_sr, span_zm, span_dz, smooth)
        -> (S, P, S_smooth, P_smooth, span_sr, span_zm, span_dz)

The background and prompt sinogram histograms (3D, `n_sr×n_zm×n_dz`) and their
smoothed versions, plus each axis span. Same binning/smoothing as
[`background_estimate`](@ref); the local background fraction the model applies is
`B_smooth ./ P_smooth`. For inspecting the background model.
"""
function background_sinograms(s_r, z_m, dz, bg_mask; n_sr, n_zm, n_dz,
                           span_sr, span_zm, span_dz, smooth)
    S, P = _sino_hist(s_r, z_m, dz, bg_mask; n_sr = n_sr, n_zm = n_zm, n_dz = n_dz,
                      span_sr = span_sr, span_zm = span_zm, span_dz = span_dz)
    return S, P, _smooth3(S, smooth), _smooth3(P, smooth), span_sr, span_zm, span_dz
end
