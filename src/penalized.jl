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

# ============================ edge-preserving priors ============================
# Non-quadratic priors have NO De Pierro closed form; we optimize L+R by the
# One-Step-Late (OSL, Green 1990) MAP-EM update
#     x⁺ = t / (sens + ∇R(x)),
# evaluating the penalty gradient at the current iterate. β=0 ⇒ MLEM. OSL is the
# standard MLEM-structured edge-preserving scheme; it is not provably monotone
# (the penalty gradient is one-step-late), so we rely on β=0≡MLEM and empirics.

"""
    HuberPrior(β, δ)

Edge-preserving Huber prior on 6-neighbour differences: penalty
`β Σ_j Σ_{k∈N_j} ρ_δ(x_j-x_k)` with Huber `ρ_δ` (quadratic for |t|≤δ, linear
beyond). Its gradient uses the clamped influence `ψ_δ(t)=clamp(t,-δ,δ)`: below δ
it smooths like the quadratic prior, above δ (an edge) the influence saturates so
edges are preserved. Use with [`osl_mlem`](@ref). Large δ ⇒ quadratic smoothness.
"""
struct HuberPrior <: Prior
    beta::Float32
    delta::Float32
end

# Σ_{k∈N_j} clamp(x_j - x_k, -δ, δ) over in-bounds 6-neighbours (the Huber influence).
@kernel function _huber_grad_kernel!(g, @Const(x), n, delta)
    I = @index(Global, Cartesian)
    i, j, k = I[1], I[2], I[3]
    xc = x[I]; s = 0.0f0
    @inbounds begin
        if i > 1;    s += clamp(xc - x[i-1, j, k], -delta, delta); end
        if i < n[1]; s += clamp(xc - x[i+1, j, k], -delta, delta); end
        if j > 1;    s += clamp(xc - x[i, j-1, k], -delta, delta); end
        if j < n[2]; s += clamp(xc - x[i, j+1, k], -delta, delta); end
        if k > 1;    s += clamp(xc - x[i, j, k-1], -delta, delta); end
        if k < n[3]; s += clamp(xc - x[i, j, k+1], -delta, delta); end
        g[I] = s
    end
end

# penalty gradient ∇R(x) (includes β); for OSL.
function penalty_gradient(p::HuberPrior, x)
    g = similar(x)
    backend = KA.get_backend(x)
    _huber_grad_kernel!(backend)(g, x, Int32.(size(x)), p.delta; ndrange = size(x))
    KA.synchronize(backend)
    return p.beta .* g
end

"""
    LogcoshPrior(β, scalar)

Edge-preserving Logcosh prior on 6-neighbour differences: penalty
`β Σ_j Σ_{k∈N_j} (1/s²)·log cosh(s·(x_j-x_k))`, a smooth (C^∞) analogue of Huber
with sharpness `s`. Its influence is the soft clamp `(1/s)·tanh(s·t)`: quadratic
for `|t|≪1/s`, saturating to `±1/s` across an edge (so `s` plays Huber's `1/δ`
role). Use with [`osl_mlem`](@ref). Mirrors STIR's `LogcoshPrior` (and, being a
parabolic-surrogate prior there, is the natural prior for a future OSSPS).
"""
struct LogcoshPrior <: Prior
    beta::Float32
    scalar::Float32
end

# Σ_{k∈N_j} (1/s)·tanh(s·(x_j - x_k)) over in-bounds 6-neighbours.
@kernel function _logcosh_grad_kernel!(g, @Const(x), n, s)
    I = @index(Global, Cartesian)
    i, j, k = I[1], I[2], I[3]
    xc = x[I]; acc = 0.0f0; inv_s = 1.0f0 / s
    @inbounds begin
        if i > 1;    acc += inv_s * tanh(s * (xc - x[i-1, j, k])); end
        if i < n[1]; acc += inv_s * tanh(s * (xc - x[i+1, j, k])); end
        if j > 1;    acc += inv_s * tanh(s * (xc - x[i, j-1, k])); end
        if j < n[2]; acc += inv_s * tanh(s * (xc - x[i, j+1, k])); end
        if k > 1;    acc += inv_s * tanh(s * (xc - x[i, j, k-1])); end
        if k < n[3]; acc += inv_s * tanh(s * (xc - x[i, j, k+1])); end
        g[I] = acc
    end
end

function penalty_gradient(p::LogcoshPrior, x)
    g = similar(x)
    backend = KA.get_backend(x)
    _logcosh_grad_kernel!(backend)(g, x, Int32.(size(x)), p.scalar; ndrange = size(x))
    KA.synchronize(backend)
    return p.beta .* g
end

"""
    RelativeDifferencePrior(β, γ, ε)

Relative Difference Prior (Nuyts et al. 2002; the GE Q.Clear penalty),
edge-preserving on 6-neighbour differences: penalty
`β Σ_j Σ_{k∈N_j} (x_j-x_k)² / (x_j+x_k+γ|x_j-x_k|+ε)`. The intensity-relative
denominator makes the penalty \emph{scale with the local activity}, so one `(γ,ε)`
adapts across image levels --- unlike Huber's absolute `δ`, which must be retuned
when the image scale changes. `γ` sets edge sharpness (default 2); `ε>0`
regularizes the origin (small but nonzero for gradient methods). Per-pair gradient
`(x-y)(γ|x-y|+x+3y+2ε)/(x+y+γ|x-y|+ε)²`. Use with [`osl_mlem`](@ref). Mirrors
STIR's `RelativeDifferencePrior` (gradient-based only; no parabolic surrogate).
"""
struct RelativeDifferencePrior <: Prior
    beta::Float32
    gamma::Float32
    epsilon::Float32
end

# RDP per-pair derivative ∂/∂x_j of (x_j-x_k)²/(x_j+x_k+γ|x_j-x_k|+ε); scalar, GPU-safe.
@inline function _rdp_deriv(xj, xk, gamma, eps)
    d = xj - xk; ad = abs(d)
    den = xj + xk + gamma * ad + eps
    return den > 0.0f0 ? d * (gamma * ad + xj + 3.0f0 * xk + 2.0f0 * eps) / (den * den) : 0.0f0
end

@kernel function _rdp_grad_kernel!(g, @Const(x), n, gamma, eps)
    I = @index(Global, Cartesian)
    i, j, k = I[1], I[2], I[3]
    xc = x[I]; acc = 0.0f0
    @inbounds begin
        if i > 1;    acc += _rdp_deriv(xc, x[i-1, j, k], gamma, eps); end
        if i < n[1]; acc += _rdp_deriv(xc, x[i+1, j, k], gamma, eps); end
        if j > 1;    acc += _rdp_deriv(xc, x[i, j-1, k], gamma, eps); end
        if j < n[2]; acc += _rdp_deriv(xc, x[i, j+1, k], gamma, eps); end
        if k > 1;    acc += _rdp_deriv(xc, x[i, j, k-1], gamma, eps); end
        if k < n[3]; acc += _rdp_deriv(xc, x[i, j, k+1], gamma, eps); end
        g[I] = acc
    end
end

function penalty_gradient(p::RelativeDifferencePrior, x)
    g = similar(x)
    backend = KA.get_backend(x)
    _rdp_grad_kernel!(backend)(g, x, Int32.(size(x)), p.gamma, p.epsilon; ndrange = size(x))
    KA.synchronize(backend)
    return p.beta .* g
end

"""
    osl_mlem(model, x0, prior; niter = 50, denom_clamp = 10f0, clamp_frac = nothing, callback = nothing)

One-Step-Late MAP-EM for an edge-preserving (non-quadratic) `prior` exposing
`penalty_gradient`. Update `x⁺ = t / (sens + ∇R(x))`, clamped non-negative.

Because the prior gradient is taken one step late, an unsafeguarded denominator
`sens + ∇R` can collapse, or go negative, where `sens` is small --- under
attenuation `sens = Aᵀ(a)` is small at the centre, so a voxel's denominator
vanishes and the update runs away (the divergence seen on contrast+attenuation
phantoms). Following STIR's `OSMAPOSL`, the denominator is clamped to
`[sens/denom_clamp, sens·denom_clamp]` --- i.e. the one-step-late correction factor
is held in `[1/denom_clamp, denom_clamp]` --- which keeps OSL stable without moving
its fixed point (at convergence `∇R` is small and the clamp is inactive). Set
`denom_clamp` larger to loosen the guard, or `Inf` to recover the raw OSL update.
Pass a `Ref` as `clamp_frac` to read back the mean fraction of in-FOV voxels whose
denominator was clamped (a diagnostic for how hard the guard is working).
β=0 is [`mlem`](@ref); for quadratic priors prefer [`penalized_mlem`](@ref) (monotone).
"""
function osl_mlem(m::ListmodePoissonModel, x0, prior::Prior; niter::Integer = 50,
                  denom_clamp::Real = 10.0f0, clamp_frac = nothing, callback = nothing)
    x = copy(x0)
    sens = m.sensitivity
    c = Float32(denom_clamp)
    lo = sens ./ c; hi = sens .* c                       # per-voxel denominator bounds
    nfov = clamp_frac === nothing ? 1 : count(>(0.0f0), sens)
    acc = 0.0                                            # mean fraction of FOV voxels clamped
    for k in 1:niter
        t, _ = _em_numerator(m, x)
        raw = sens .+ penalty_gradient(prior, x)
        denom = clamp.(raw, lo, hi)
        if clamp_frac !== nothing
            acc += Float64(sum((sens .> 0.0f0) .& ((raw .< lo) .| (raw .> hi)))) / nfov
        end
        x = ifelse.(sens .> 0.0f0, t ./ max.(denom, 1.0f-20), x)
        x = max.(x, 0.0f0)
        callback === nothing || callback(k, x)
    end
    clamp_frac === nothing || (clamp_frac[] = acc / niter)
    return x
end

"""
    bsrem(model, x0, prior; niter = 50, relax = 1f0, relax_gamma = 0.1f0, callback = nothing)

Block Sequential Regularized EM (Ahn & Fessler 2003) — the convergent relaxed
preconditioned-gradient MAP algorithm behind GE's Q.Clear. One iteration is

    x⁺ = x + relaxₙ · (x / sens) · [ Aᵀ(mult·counts/pred) − sens − ∇R(x) ],   clamped ≥ 0,

with the relaxation schedule `relaxₙ = relax / (1 + relax_gamma·(n-1))` (n = 1-based
iteration). The bracket is the gradient of the MAP objective `L(x) − R(x)` (data EM
gradient `Aᵀ(mult·counts/pred) − sens` minus the penalty gradient `∇R`, which already
includes β); the preconditioner is the EM preconditioner `x/sens`.

Unlike [`osl_mlem`](@ref), the prior enters the *numerator* (as a true gradient term),
not the EM denominator, and the decreasing relaxation makes the iteration converge to
the MAP optimum of `L − R` rather than running away — so it is stable under attenuation
(small central `sens`) where OSL diverges, with no denominator clamp. At `relax = 1`,
`relax_gamma = 0` and `NoPrior` it reproduces [`mlem`](@ref) exactly (the bracket is then
`x·Aᵀ(mult·counts/pred)/sens − x`, i.e. the MLEM step minus `x`). Works for any `prior`
exposing [`penalty_gradient`](@ref) (RDP/Huber/Logcosh) as well as `NoPrior`. Voxels
outside the FOV (`sens == 0`) are left unchanged.
"""
function bsrem(m::ListmodePoissonModel, x0, prior::Prior = NoPrior();
               niter::Integer = 50, relax::Real = 1.0f0, relax_gamma::Real = 0.1f0,
               callback = nothing)
    x = copy(x0)
    sens = m.sensitivity
    r0 = Float32(relax); rg = Float32(relax_gamma)
    infov = sens .> 0.0f0
    for k in 1:niter
        relk = r0 / (1.0f0 + rg * Float32(k - 1))
        grad = _map_gradient(m, x, prior)            # ∇(L − R) ascent direction
        precond = ifelse.(infov, x ./ sens, 0.0f0)   # EM preconditioner x/sens
        x = max.(x .+ relk .* precond .* grad, 0.0f0)
        callback === nothing || callback(k, x)
    end
    return x
end

# Ascent gradient of the MAP objective L − R:  Aᵀ(mult·counts/pred) − sens − ∇R(x).
# `_gradient` returns the descent gradient of the negative log-likelihood,
# `sens − Aᵀ(mult·counts/pred)`, so its negation is the data EM ascent term (valid
# even where x = 0, since it is computed by direct backprojection, not t/x).
function _map_gradient(m::ListmodePoissonModel, x, prior::Prior)
    return .-_gradient(m, x) .- penalty_gradient(prior, x)
end

penalty_gradient(::NoPrior, x) = zero(x)

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
