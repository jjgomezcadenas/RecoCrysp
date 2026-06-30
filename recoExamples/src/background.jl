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

# Separable Gaussian smoothing of an N-D array. `sigmas` is the per-axis width in
# bins; an axis with sigma <= 0 is left UNSMOOTHED (preserve a genuinely sharp
# physical edge, e.g. s_r at s_r = R). `circular` (per-axis bool, default all
# false) wraps an axis at its borders instead of clamping -- required for a
# periodic coordinate such as the azimuthal view phi. Works for any ndims(A).
function _smoothnd(A, sigmas; circular = ntuple(_ -> false, ndims(A)))
    nd = ndims(A); sz = size(A)
    B = copy(A)
    for d in 1:nd
        sigmas[d] <= 0 && continue
        k = _gauss_kernel(sigmas[d])
        length(k) == 1 && continue
        r = (length(k) - 1) ÷ 2
        C = similar(B); n = sz[d]; wrap = circular[d]
        for I in CartesianIndices(B)
            acc = 0.0; base = I[d]
            for (m, w) in enumerate(k)
                off = base + (m - 1 - r)
                j = wrap ? mod(off - 1, n) + 1 : clamp(off, 1, n)
                Jt = ntuple(t -> t == d ? j : I[t], nd)
                acc += w * B[CartesianIndex(Jt)]
            end
            C[I] = acc
        end
        B = C
    end
    return B
end

# 3-D convenience wrapper (clamp at all borders) -- preserves the original API.
_smooth3(A, sigmas) = _smoothnd(A, sigmas)

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

# ============================= 4-coordinate path ==============================
# The 3-coordinate model above DROPS the azimuthal view phi (valid only for a
# centred, azimuthally symmetric phantom). For an off-centre phantom (e.g. the
# NEMA spheres) the scatter is localized in phi, and collapsing over phi replaces
# the scatter under each sphere with the azimuthal average at that radius -- so
# the contamination is right in total but mislocalized, and subtracting it barely
# restores contrast. The functions below add phi (and a SIGNED radial coordinate,
# required so that opposite-side LORs do not fold together) for a full
# (s, phi, z_m, dz) sinogram, filled here from the truth scatter flags.

"""
    lor_sinogram_coords4(xs, xe) -> (s, phi, z_m, dz)

Full reduced sinogram coordinates for the `(3,N)` endpoint columns: the SIGNED
transverse radial offset `s = (x1·y2 - x2·y1)/|d_xy|`, the azimuthal view
`phi = atan(dy, dx)` folded to `[0, pi)` (negating `s` on fold, so each line has a
unique `(s, phi)`), the axial midpoint `z_m`, and the obliquity `dz = z2 - z1`.
Unlike [`lor_sinogram_coords`](@ref) (unsigned `s_r`, no `phi`), this localizes a
LOR in the transverse plane and is the parametrization to use for an off-centre
phantom. `phi` is periodic on `[0, pi)`; smooth it circularly.
"""
function lor_sinogram_coords4(xs, xe)
    N = size(xs, 2)
    s = Vector{Float32}(undef, N); phi = similar(s); z_m = similar(s); dz = similar(s)
    @inbounds for i in 1:N
        x1 = xs[1, i]; y1 = xs[2, i]; z1 = xs[3, i]
        x2 = xe[1, i]; y2 = xe[2, i]; z2 = xe[3, i]
        dx = x2 - x1; dy = y2 - y1
        dxy = hypot(dx, dy)
        si = dxy < 1.0f-6 ? 0.0f0 : (x1 * y2 - x2 * y1) / dxy   # signed radial offset
        ph = atan(dy, dx)                                       # view direction in (-pi, pi]
        if ph < 0.0f0; ph += Float32(pi); si = -si; end        # fold to [0, pi), flip sign
        ph >= Float32(pi) && (ph -= Float32(pi))               # guard the pi edge
        s[i] = si; phi[i] = ph; z_m[i] = 0.5f0 * (z1 + z2); dz[i] = z2 - z1
    end
    return s, phi, z_m, dz
end

# 4-D histogram of the background subset (S) and all prompts (P), shared by
# background_estimate4 so both bin identically.
function _sino_hist4(s, phi, z_m, dz, bg_mask; n_s, n_phi, n_zm, n_dz,
                     span_s, span_phi, span_zm, span_dz)
    lo_s, hi_s = span_s; lo_p, hi_p = span_phi; lo_z, hi_z = span_zm; lo_d, hi_d = span_dz
    S = zeros(Float64, n_s, n_phi, n_zm, n_dz); P = zeros(Float64, n_s, n_phi, n_zm, n_dz)
    @inbounds for i in eachindex(s)
        bi = _binidx(s[i],   lo_s, hi_s, n_s)
        bp = _binidx(phi[i], lo_p, hi_p, n_phi)
        bj = _binidx(z_m[i], lo_z, hi_z, n_zm)
        bk = _binidx(dz[i],  lo_d, hi_d, n_dz)
        P[bi, bp, bj, bk] += 1
        bg_mask[i] && (S[bi, bp, bj, bk] += 1)
    end
    return S, P
end

"""
    background_estimate4(s, phi, z_m, dz, bg_mask; n_s, n_phi, n_zm, n_dz,
        span_s, span_phi, span_zm, span_dz, smooth, total=nothing, verbose=false)
        -> Vector{Float32}

The φ-aware counterpart of [`background_estimate`](@ref): per-event background `b_i`
from a full `(s, phi, z_m, dz)` sinogram (signed `s`). `smooth` is a 4-tuple of
per-axis Gaussian widths in bins; the `phi` axis is smoothed CIRCULARLY (periodic),
the others clamp. With `n_phi = 1` and `span_phi = (0, pi)` this reduces to the
3-coordinate model on signed `s`. If `total` is given the result is rescaled to sum
to it. `verbose` prints the scatter-count occupancy (min/median per occupied bin) so
the φ binning can be checked against starvation on the real data.
"""
function background_estimate4(s, phi, z_m, dz, bg_mask;
                              n_s, n_phi, n_zm, n_dz,
                              span_s, span_phi, span_zm, span_dz,
                              smooth, total = nothing, verbose = false)
    S, P = _sino_hist4(s, phi, z_m, dz, bg_mask; n_s = n_s, n_phi = n_phi,
                       n_zm = n_zm, n_dz = n_dz, span_s = span_s, span_phi = span_phi,
                       span_zm = span_zm, span_dz = span_dz)
    if verbose
        occ = filter(>(0.0), vec(S)); med = isempty(occ) ? 0.0 : sort(occ)[cld(length(occ), 2)]
        println("  [bg4] scatter bins: $(length(occ))/$(length(S)) occupied, " *
                "counts/occupied-bin min $(isempty(occ) ? 0 : Int(minimum(occ))) " *
                "median $(round(med; digits=1))"); flush(stdout)
    end
    circ = (false, true, false, false)               # phi (axis 2) is periodic
    S = _smoothnd(S, smooth; circular = circ); P = _smoothnd(P, smooth; circular = circ)
    lo_s, hi_s = span_s; lo_p, hi_p = span_phi; lo_z, hi_z = span_zm; lo_d, hi_d = span_dz
    out = Vector{Float32}(undef, length(s))
    @inbounds for i in eachindex(s)
        bi = _binidx(s[i],   lo_s, hi_s, n_s)
        bp = _binidx(phi[i], lo_p, hi_p, n_phi)
        bj = _binidx(z_m[i], lo_z, hi_z, n_zm)
        bk = _binidx(dz[i],  lo_d, hi_d, n_dz)
        out[i] = P[bi, bp, bj, bk] > 0 ? Float32(S[bi, bp, bj, bk] / P[bi, bp, bj, bk]) : 0.0f0
    end
    if total !== nothing
        sm = sum(out); sm > 0 && (out .*= Float32(total / sm))
    end
    return out
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
