# Penalized-likelihood MLEM (MAP-EM) via De Pierro's surrogate.
#
# Minimizes the negative Poisson log-likelihood plus a QUADRATIC prior R(x), in
# closed form, preserving non-negativity:
#
#     minimize  L(x) + R(x),   L = Σ_i ȳ_i - counts_i log ȳ_i,   ȳ = n·(Ax)+s,
#
# Every iteration uses the ordinary EM numerator t = x ⊙ Aᵀ(n·counts/pred) and the
# sensitivity, then the De Pierro update
#
#     x⁺ = 2t / ( √(B² + 4·G·t) + B ),
#
# where the prior only sets (B, G). At β = 0 this is exactly MLEM (B = sens, G = 0
# ⇒ x⁺ = t/sens). Two quadratic priors are provided; an edge-preserving prior
# (Huber/LogCosh) is NOT quadratic and needs a gradient-based solver instead --
# the De Pierro closed form requires a quadratic penalty.

abstract type Prior end

"""No prior: `penalized_mlem` with `NoPrior` is plain MLEM."""
struct NoPrior <: Prior end

"""
    QuadraticIntensityPrior(β, z)

Quadratic intensity prior `R(x) = (β/2)‖x - z‖²` pulling the image toward the
reference `z` (an array on the same backend/shape as the image). De Pierro's
example prior; weak/biasing -- mainly a baseline.
"""
struct QuadraticIntensityPrior{T} <: Prior
    beta::Float32
    z::T
end

"""
    QuadraticSmoothnessPrior(β)

Quadratic pairwise smoothness prior `R(x) = (β/2) Σ_j Σ_{k∈N_j} (x_j - x_k)²` over
the 6-connected (face) neighbourhood with unit weights -- the noise-controlling
prior. The neighbour sums are computed by a KernelAbstractions kernel (CPU/GPU).
"""
struct QuadraticSmoothnessPrior <: Prior
    beta::Float32
end

# --- 6-neighbour sums via a KA kernel (mirrors the projector kernel idiom) ------
# For each voxel: nsum = Σ_{k∈N_j} x_k and wsum = Σ_{k∈N_j} 1 over in-bounds face
# neighbours (so boundary voxels count 3..6 neighbours, weight 0 outside).
@kernel function _neigh6_kernel!(nsum, wsum, @Const(x), n)
    I = @index(Global, Cartesian)
    i, j, k = I[1], I[2], I[3]
    s = 0.0f0; w = 0.0f0
    @inbounds begin
        if i > 1;    s += x[i-1, j, k]; w += 1.0f0; end
        if i < n[1]; s += x[i+1, j, k]; w += 1.0f0; end
        if j > 1;    s += x[i, j-1, k]; w += 1.0f0; end
        if j < n[2]; s += x[i, j+1, k]; w += 1.0f0; end
        if k > 1;    s += x[i, j, k-1]; w += 1.0f0; end
        if k < n[3]; s += x[i, j, k+1]; w += 1.0f0; end
        nsum[I] = s
        wsum[I] = w
    end
end

function _neighbour_sums(x)
    nsum = similar(x); wsum = similar(x)
    backend = KA.get_backend(x)
    _neigh6_kernel!(backend)(nsum, wsum, x, Int32.(size(x)); ndrange = size(x))
    KA.synchronize(backend)
    return nsum, wsum
end

# --- (B, G) for the De Pierro update; all reduce to (sens, 0) ⇒ MLEM ------------
_prior_BG(::NoPrior, x, sens) = (sens, 0.0f0)

_prior_BG(p::QuadraticIntensityPrior, x, sens) = (sens .- p.beta .* p.z, p.beta)

function _prior_BG(p::QuadraticSmoothnessPrior, x, sens)
    nsum, wsum = _neighbour_sums(x)
    # c_j = β Σ_k (x_j + x_k) = β (wsum·x + nsum);  γ_j = 2β Σ_k 1 = 2β·wsum
    B = sens .- p.beta .* (wsum .* x .+ nsum)
    G = (2.0f0 * p.beta) .* wsum
    return B, G
end

"""
    penalized_mlem(model, x0, prior = NoPrior(); niter = 50, callback = nothing)

Penalized-likelihood MLEM (De Pierro). Runs `niter` iterations from `x0` under the
quadratic `prior`; `callback(k, x)` runs after each iteration if given. With
`NoPrior` it is identical to [`mlem`](@ref). Voxels outside the FOV (`sens == 0`)
are left unchanged.
"""
function penalized_mlem(m::ListmodePoissonModel, x0, prior::Prior = NoPrior();
                        niter::Integer = 50, callback = nothing)
    x = copy(x0)
    sens = m.sensitivity
    for k in 1:niter
        t, _ = _em_numerator(m, x)
        B, G = _prior_BG(prior, x, sens)
        denom = sqrt.(B .^ 2 .+ 4.0f0 .* G .* t) .+ B
        x = ifelse.(sens .> 0.0f0, 2.0f0 .* t ./ denom, x)
        x = max.(x, 0.0f0)
        callback === nothing || callback(k, x)
    end
    return x
end
