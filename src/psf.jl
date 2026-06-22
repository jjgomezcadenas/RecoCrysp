# Image-space Gaussian resolution operator G (the detector point-spread of the
# data model, Sec. resolution). A separable, isotropic Gaussian blur of the
# activity image; with a symmetric kernel and zero-padded boundaries it is
# self-adjoint, so it is the forward G in A·G. Host Float32 arrays — a one-time
# simulation-side operation, not in the reconstruction loop.

const _FWHM_TO_SIGMA = 1.0f0 / (2.0f0 * sqrt(2.0f0 * log(2.0f0)))   # 1/2.3548

# 1-D Gaussian convolution along axis `d`, zero-padded.
function _gauss1d(img::Array{Float32,3}, σ::Float32, d::Int)
    σ <= 0.0f0 && return img
    r = max(1, ceil(Int, 3σ))
    k = Float32[exp(-(i * i) / (2σ * σ)) for i in -r:r]
    k ./= sum(k)
    n = size(img)
    out = zeros(Float32, n)
    @inbounds for I in CartesianIndices(img)
        acc = 0.0f0
        base = I[d]
        for (kk, off) in enumerate(-r:r)
            j = base + off
            if 1 <= j <= n[d]
                J = CartesianIndex(ntuple(t -> t == d ? j : I[t], 3))
                acc += k[kk] * img[J]
            end
        end
        out[I] = acc
    end
    return out
end

"""
    gaussian_blur(img, fwhm, voxsize) -> blurred

Blur the 3-D image `img` with an isotropic Gaussian point-spread of full width at
half maximum `fwhm` (mm); `voxsize` is the voxel size in mm (scalar or 3-tuple).
Implemented as a separable convolution along each axis with a normalized 1-D
Gaussian (zero-padded), so `sum(img)` is preserved for sources away from the
border. The operator is self-adjoint, making it the forward resolution operator
`G` of the data model (`A·G`).
"""
function gaussian_blur(img::AbstractArray{Float32,3}, fwhm::Real, voxsize)
    vs = voxsize isa Number ? ntuple(_ -> Float32(voxsize), 3) : NTuple{3,Float32}(voxsize)
    σ = ntuple(d -> Float32(fwhm) * _FWHM_TO_SIGMA / vs[d], 3)
    out = Array{Float32,3}(img)
    for d in 1:3
        out = _gauss1d(out, σ[d], d)
    end
    return out
end
