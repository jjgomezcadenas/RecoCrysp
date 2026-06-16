# Listmode Poisson reconstruction building blocks (MLEM / OSEM), mirroring the
# design of parallelproj.functions.NegPoissonLogLListmode and the em_update
# helper from its examples.
#
# The forward model is A_LM (project the image along each LOR/event), supplied
# by the RecoCrysp Joseph projectors, with a per-LOR multiplicative correction
# factor n (detector normalization × geometric × attenuation) and additive
# background s:
#
#     pred_e = n_e · (A_LM x)_e + s_e .
#
# The objective is the negative Poisson log-likelihood in image space,
#
#     f(x) = <sens, x> - sum_e counts_e * log(pred_e) ,
#
# with gradient   grad f(x) = sens - A_LM^T( n · counts / pred )
# and the MLEM step written as a preconditioned gradient descent
#
#     x_{k+1} = x_k - (x_k / sens) ⊙ grad f(x_k)
#             = (x_k / sens) ⊙ A_LM^T( n · counts / pred ).
#
# `sens` is the sensitivity image A_full^T(n) — backprojection of the per-LOR
# normalization factors (all-ones when no normalization) over ALL geometric
# LORs, NOT just the detected events. It is decoupled from the event list,
# exactly as in parallelproj, so the data term only projects over events.
#
# `counts` is per-LOR: 1 per event for raw listmode, an integer multiplicity for
# compressed listmode, or expected counts (n·A x_true) for a deterministic
# noise-free test. `mult` is the per-LOR factor n (default all-ones). All
# arithmetic is Float32; arrays must share one backend.

"""
    sensitivity_image(xstart, xend, img_shape, img_origin, voxsize; weights = nothing)

Sensitivity image `Aᵀ·w` — backprojection of the per-LOR weights `w` (default
all ones) over the supplied LOR list. Pass all geometric LORs (and, for a real
scanner, `weights =` the multiplicative normalization factors) to obtain the
MLEM/OSEM preconditioner.
"""
function sensitivity_image(xstart, xend, img_shape, img_origin, voxsize; weights = nothing)
    if weights === nothing
        weights = similar(xstart, Float32, size(xstart, 2))
        fill!(weights, 1.0f0)
    end
    return joseph3d_back(
        xstart, xend, weights, img_shape,
        NTuple{3,Float32}(img_origin), NTuple{3,Float32}(voxsize),
    )
end

"""
    ListmodePoissonModel(xstart, xend, sensitivity; img_origin, voxsize,
                         counts = nothing, contamination = nothing, mult = nothing)

Negative Poisson log-likelihood model for a list of LORs/events with endpoints
in the columns of `xstart`/`xend` (`(3, nlor)`). `sensitivity` is the
precomputed sensitivity image `Aᵀ(n)` (see [`sensitivity_image`](@ref); pass
`weights = n` there). `counts` defaults to all ones (raw listmode);
`contamination` (additive background `s`, must keep predicted counts positive)
defaults to zeros; `mult` is the per-LOR multiplicative normalization factor `n`
(detector efficiency × geometric × attenuation), defaulting to all ones. All
arrays must live on the same backend as `sensitivity`.
"""
struct ListmodePoissonModel{V,C,S}
    xstart::V
    xend::V
    counts::C
    contamination::C
    mult::C
    sensitivity::S
    img_origin::NTuple{3,Float32}
    voxsize::NTuple{3,Float32}
    img_shape::NTuple{3,Int}
end

function ListmodePoissonModel(
    xstart, xend, sensitivity;
    img_origin, voxsize, counts = nothing, contamination = nothing, mult = nothing,
)
    nlor = size(xstart, 2)
    if counts === nothing
        counts = similar(xstart, Float32, nlor)
        fill!(counts, 1.0f0)
    end
    if contamination === nothing
        contamination = similar(xstart, Float32, nlor)
        fill!(contamination, 0.0f0)
    end
    if mult === nothing
        mult = similar(xstart, Float32, nlor)
        fill!(mult, 1.0f0)
    end
    return ListmodePoissonModel(
        xstart, xend, counts, contamination, mult, sensitivity,
        NTuple{3,Float32}(img_origin), NTuple{3,Float32}(voxsize), size(sensitivity),
    )
end

"""
    predicted(model, x)

Per-LOR predicted counts `n · (A_LM x) + s`.
"""
predicted(m::ListmodePoissonModel, x) =
    m.mult .* joseph3d_fwd(m.xstart, m.xend, x, m.img_origin, m.voxsize) .+ m.contamination

# gradient of the negative Poisson log-likelihood w.r.t. the image (internal)
function _gradient(m::ListmodePoissonModel, x)
    pred = predicted(m, x)
    # n·counts/pred, with the convention 0/0 = 0 for unmeasured LORs
    ratio = ifelse.(m.counts .> 0.0f0, m.mult .* m.counts ./ pred, 0.0f0)
    bp = joseph3d_back(m.xstart, m.xend, ratio, m.img_shape, m.img_origin, m.voxsize)
    return m.sensitivity .- bp
end

"""
    neg_log_likelihood(model, x)

Negative Poisson log-likelihood `⟨sens, x⟩ - Σ counts·log(pred)` (up to an
additive constant). Monotonically non-increasing under [`mlem`](@ref).
"""
function neg_log_likelihood(m::ListmodePoissonModel, x)
    pred = predicted(m, x)
    # clamp before log: unmeasured LORs (counts == 0) may carry a tiny negative
    # predicted value from Float32 rounding; those terms are masked out anyway.
    logpred = log.(max.(pred, 1.0f-20))
    data = ifelse.(m.counts .> 0.0f0, m.counts .* logpred, 0.0f0)
    return Float64(sum(m.sensitivity .* x)) - Float64(sum(data))
end

"""
    em_update(model, x) -> x_new

One MLEM iteration as a preconditioned gradient step
`x - (x / sens) ⊙ ∇f(x)`. Voxels with zero sensitivity (outside the FOV) are
left unchanged.
"""
function em_update(m::ListmodePoissonModel, x)
    g = _gradient(m, x)
    precond = ifelse.(m.sensitivity .> 0.0f0, x ./ m.sensitivity, 0.0f0)
    # project onto the non-negative orthant: the subtractive preconditioned-
    # gradient form equals the multiplicative MLEM update in exact arithmetic,
    # but Float32 cancellation can leave rounding-level negatives.
    return max.(x .- precond .* g, 0.0f0)
end

"""
    mlem(model, x0; niter = 50, callback = nothing)

Run `niter` MLEM iterations from `x0`. `callback(k, x)` is invoked after each
iteration if given. Returns the final image.
"""
function mlem(m::ListmodePoissonModel, x0; niter::Integer = 50, callback = nothing)
    x = copy(x0)
    for k in 1:niter
        x = em_update(m, x)
        callback === nothing || callback(k, x)
    end
    return x
end

"""
    osem(models, x0; nepochs = 10, callback = nothing)

Ordered-subsets EM: one epoch applies an [`em_update`](@ref) for each subset
model in `models` (each carrying its own event subset and subset sensitivity).
`callback(epoch, x)` runs after each full epoch.
"""
function osem(
    models::AbstractVector{<:ListmodePoissonModel}, x0;
    nepochs::Integer = 10, callback = nothing,
)
    x = copy(x0)
    for e in 1:nepochs
        for m in models
            x = em_update(m, x)
        end
        callback === nothing || callback(e, x)
    end
    return x
end

"""
    subset_models(xstart, xend, img_origin, voxsize, img_shape, nsub;
                  counts = nothing, mult = nothing)

Split a LOR list into `nsub` interleaved subsets (subset `k` = every `nsub`-th
LOR starting at `k`) and build a [`ListmodePoissonModel`](@ref) for each, with
its own exact subset sensitivity `A_subᵀ(n_sub)`. Returns the vector of models
for [`osem`](@ref).
"""
function subset_models(
    xstart, xend, img_origin, voxsize, img_shape, nsub::Integer;
    counts = nothing, mult = nothing,
)
    models = ListmodePoissonModel[]
    nlor = size(xstart, 2)
    for k in 1:nsub
        idx = k:nsub:nlor
        sxs = xstart[:, idx]
        sxe = xend[:, idx]
        sc = counts === nothing ? nothing : counts[idx]
        sm = mult === nothing ? nothing : mult[idx]
        ssens = sensitivity_image(sxs, sxe, img_shape, img_origin, voxsize; weights = sm)
        push!(models, ListmodePoissonModel(
            sxs, sxe, ssens;
            img_origin = img_origin, voxsize = voxsize, counts = sc, mult = sm,
        ))
    end
    return models
end
