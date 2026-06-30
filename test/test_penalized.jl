# Verification of penalized_mlem (De Pierro). On a small synthetic listmode
# problem: (1) NoPrior reproduces mlem; (2) the De Pierro update monotonically
# decreases the penalized cost L + R for both quadratic priors (the MM guarantee
# -- a strong check that (B,G) are correct); (3) the smoothness prior reduces
# background variance vs unregularized at matched iterations.
#
#   julia --project=recoExamples test/test_penalized.jl

using RecoCrysp
using Random, Statistics, Test

rng = MersenneTwister(0)
n = (16, 16, 16); vs = (4.0f0, 4.0f0, 4.0f0)
org = ntuple(i -> -(n[i] - 1) / 2 * vs[i], 3)

# hot sphere (radius 16 mm) in a warm background -> structure + a flat region
ax = [Float32[org[d] + (i - 1) * vs[d] for i in 1:n[d]] for d in 1:3]
phantom = fill(0.5f0, n...)
for k in 1:n[3], j in 1:n[2], i in 1:n[1]
    (ax[1][i]^2 + ax[2][j]^2 + ax[3][k]^2 <= 16.0f0^2) && (phantom[i, j, k] = 2.0f0)
end

sc = ContinuousPET(diameter = 300.0f0, afov = 64.0f0)
xs, xe = sample_lors(sc, 40000; rng = rng)
ybar = joseph3d_fwd(xs, xe, phantom, org, vs)

# Poisson data (Knuth; means are small here)
function poissrand(rng, lam)
    lam <= 0 && return 0.0f0
    L = exp(-lam); k = 0; p = 1.0
    while true
        k += 1; p *= rand(rng)
        p <= L && return Float32(k - 1)
    end
end
counts = Float32[poissrand(rng, y) for y in ybar]

sens = sensitivity_image(xs, xe, n, org, vs)                 # Aᵀ(1), same LOR set
m = ListmodePoissonModel(xs, xe, sens; img_origin = org, voxsize = vs, counts = counts)
x0 = Float32.(sens .> 0)

# explicit penalty terms (CPU brute force) for the cost-monotonicity check
Rintensity(x, β, z) = 0.5 * β * sum((x .- z) .^ 2)
function Rsmooth(x, β)
    S = 0.0
    @inbounds for k in 1:n[3], j in 1:n[2], i in 1:n[1]
        v = x[i, j, k]
        i > 1    && (S += (v - x[i-1, j, k])^2); i < n[1] && (S += (v - x[i+1, j, k])^2)
        j > 1    && (S += (v - x[i, j-1, k])^2); j < n[2] && (S += (v - x[i, j+1, k])^2)
        k > 1    && (S += (v - x[i, j, k-1])^2); k < n[3] && (S += (v - x[i, j, k+1])^2)
    end
    return 0.5 * β * S
end

@testset "penalized_mlem (De Pierro)" begin
    # (1) NoPrior == mlem
    a = penalized_mlem(m, x0, NoPrior(); niter = 40)
    b = mlem(m, x0; niter = 40)
    @test maximum(abs.(a .- b)) <= 1.0f-4 * maximum(b)

    # (2a) intensity prior: penalized cost monotonically non-increasing
    β = 0.5f0; z = fill(0.5f0, n...)
    costs = Float64[]
    penalized_mlem(m, x0, QuadraticIntensityPrior(β, z); niter = 30,
        callback = (k, x) -> push!(costs, neg_log_likelihood(m, x) + Rintensity(x, β, z)))
    @test all(diff(costs) .<= 1e-3 * abs.(costs[1:end-1]))   # non-increasing (tol for FP)

    # (2b) smoothness prior: penalized cost monotonically non-increasing
    βs = 0.2f0
    costs2 = Float64[]
    penalized_mlem(m, x0, QuadraticSmoothnessPrior(βs); niter = 30,
        callback = (k, x) -> push!(costs2, neg_log_likelihood(m, x) + Rsmooth(x, βs)))
    @test all(diff(costs2) .<= 1e-3 * abs.(costs2[1:end-1]))

    # (3) smoothness reduces background variance vs unregularized (same niter)
    raw  = penalized_mlem(m, x0, NoPrior(); niter = 30)
    smo  = penalized_mlem(m, x0, QuadraticSmoothnessPrior(βs); niter = 30)
    bg = [ax[1][i]^2 + ax[2][j]^2 + ax[3][k]^2 > 24.0f0^2 && sens[i, j, k] > 0
          for i in 1:n[1], j in 1:n[2], k in 1:n[3]]
    @test std(smo[bg]) < std(raw[bg])
    println("  cost(intensity) $(round(costs[1];sigdigits=5)) -> $(round(costs[end];sigdigits=5))")
    println("  cost(smooth)    $(round(costs2[1];sigdigits=5)) -> $(round(costs2[end];sigdigits=5))")
    println("  bg std  raw $(round(std(raw[bg]);sigdigits=3))  smooth $(round(std(smo[bg]);sigdigits=3))")
end

@testset "osl_mlem (Huber, OSL)" begin
    bg = [ax[1][i]^2 + ax[2][j]^2 + ax[3][k]^2 > 24.0f0^2 && sens[i, j, k] > 0
          for i in 1:n[1], j in 1:n[2], k in 1:n[3]]
    # (1) beta=0 == mlem
    a = osl_mlem(m, x0, HuberPrior(0.0f0, 0.3f0); niter = 40)
    b = mlem(m, x0; niter = 40)
    @test maximum(abs.(a .- b)) <= 1.0f-4 * maximum(b)
    # (2) Huber reduces background variance vs unregularized at matched niter
    raw = mlem(m, x0; niter = 30)
    hub = osl_mlem(m, x0, HuberPrior(0.2f0, 0.3f0); niter = 30)
    @test std(hub[bg]) < std(raw[bg])
    println("  bg std  raw $(round(std(raw[bg]);sigdigits=3))  huber $(round(std(hub[bg]);sigdigits=3))")
end

@testset "bsrem (relaxed preconditioned MAP)" begin
    bg = [ax[1][i]^2 + ax[2][j]^2 + ax[3][k]^2 > 24.0f0^2 && sens[i, j, k] > 0
          for i in 1:n[1], j in 1:n[2], k in 1:n[3]]
    # (1) NoPrior, relax=1, relax_gamma=0 == mlem (the bracket is the MLEM step minus x)
    a = bsrem(m, x0, NoPrior(); niter = 40, relax = 1.0f0, relax_gamma = 0.0f0)
    b = mlem(m, x0; niter = 40)
    @test maximum(abs.(a .- b)) <= 1.0f-4 * maximum(b)
    # (2) RDP (Q.Clear penalty) reduces background variance vs unregularized at matched niter
    raw = mlem(m, x0; niter = 30)
    rdp = bsrem(m, x0, RelativeDifferencePrior(0.3f0, 2.0f0, 1.0f-2); niter = 30,
                relax = 1.0f0, relax_gamma = 0.1f0)
    @test std(rdp[bg]) < std(raw[bg])
    @test all(isfinite, rdp) && minimum(rdp) >= 0.0f0      # stable, non-negative (no clamp needed)
    println("  bg std  raw $(round(std(raw[bg]);sigdigits=3))  bsrem-rdp $(round(std(rdp[bg]);sigdigits=3))")
end

@testset "osl_mlem (Logcosh / RDP edge priors)" begin
    bg = [ax[1][i]^2 + ax[2][j]^2 + ax[3][k]^2 > 24.0f0^2 && sens[i, j, k] > 0
          for i in 1:n[1], j in 1:n[2], k in 1:n[3]]
    raw = mlem(m, x0; niter = 30)
    # (1) beta=0 == mlem for both priors (the clamp leaves denom = sens at beta=0)
    @test maximum(abs.(osl_mlem(m, x0, LogcoshPrior(0.0f0, 3.0f0); niter = 40) .- mlem(m, x0; niter = 40))) <= 1.0f-4 * maximum(raw)
    @test maximum(abs.(osl_mlem(m, x0, RelativeDifferencePrior(0.0f0, 2.0f0, 1.0f-2); niter = 40) .- mlem(m, x0; niter = 40))) <= 1.0f-4 * maximum(raw)
    # (2) each reduces background variance vs unregularized at matched niter
    lc  = osl_mlem(m, x0, LogcoshPrior(0.2f0, 3.0f0); niter = 30)
    rdp = osl_mlem(m, x0, RelativeDifferencePrior(0.3f0, 2.0f0, 1.0f-2); niter = 30)
    @test std(lc[bg])  < std(raw[bg])
    @test std(rdp[bg]) < std(raw[bg])
    println("  bg std  raw $(round(std(raw[bg]);sigdigits=3))  logcosh $(round(std(lc[bg]);sigdigits=3))  rdp $(round(std(rdp[bg]);sigdigits=3))")
end
