using Test
using Random
using LinearAlgebra
using Distributions
using RecoCrysp

import Metal

# random chords of a sphere with the given radius (columns = LORs)
function make_lors(rng, nlors, radius)
    xs = randn(rng, Float32, 3, nlors)
    xe = randn(rng, Float32, 3, nlors)
    xs .*= radius ./ sqrt.(sum(abs2, xs; dims = 1))
    xe .*= radius ./ sqrt.(sum(abs2, xe; dims = 1))
    return xs, xe
end

# voxel-(0,0,0) center such that the image box is centered at the world origin
centered_origin(n, vs) = @. -(n - 1) * vs / 2

relerr(a, b) = sqrt(sum(abs2, a .- b)) / sqrt(sum(abs2, b))

# Build a noise-free listmode reconstruction problem on backend `to_dev`:
# a full-ring scanner, all geometric LORs as the event list, counts = the exact
# forward projection of a two-level box phantom. Returns device arrays for the
# model plus host copies of the phantom and FOV mask for comparison.
function build_recon_problem(to_dev)
    nrings = 7
    ringpos = Float32.(4 .* (0:(nrings - 1)))
    ringpos .-= 0.5f0 * maximum(ringpos)
    scanner = RegularPolygonPETScannerGeometry(;
        radius = 100.0, num_sides = 48, num_lor_endpoints_per_side = 1,
        lor_spacing = 4.0, ring_positions = ringpos, symmetry_axis = 3,
    )
    desc = RegularPolygonPETLORDescriptor(scanner; radial_trim = 4, sinogram_order = :RVP)
    xs_h, xe_h = get_lor_coordinates(desc)

    ish = (41, 41, nrings)
    vs = (4.0f0, 4.0f0, 4.0f0)
    org = ntuple(i -> -(ish[i] - 1) / 2 * vs[i], 3)

    x_true = zeros(Float32, ish)
    x_true[14:28, 14:28, 2:6] .= 1.0f0
    x_true[19:23, 19:23, 3:5] .= 2.0f0

    xs = to_dev(xs_h)
    xe = to_dev(xe_h)
    xt = to_dev(x_true)
    y = joseph3d_fwd(xs, xe, xt, org, vs)             # noise-free expected counts
    sens = sensitivity_image(xs, xe, ish, org, vs)
    model = ListmodePoissonModel(xs, xe, sens; img_origin = org, voxsize = vs, counts = y)

    fov = Array(sens) .> 0
    x0 = to_dev(Float32.(fov))                        # uniform inside the FOV
    return (; model, x_true, fov, x0, xt, org, vs, ish, xs, xe, y)
end

"""
Maximum relative error of axis-aligned rays through a uniform unit image,
whose exact line integral is `n * voxsize`.
"""
function analytic_max_rel(to_dev)
    n = (32, 32, 32)
    vs = (2.0f0, 2.0f0, 2.0f0)
    org = centered_origin(n, vs)
    img = ones(Float32, n)

    nrays = 100
    rng = MersenneTwister(1)
    xs = zeros(Float32, 3, 3 * nrays)
    xe = zeros(Float32, 3, 3 * nrays)
    for ax in 1:3, k in 1:nrays
        col = (ax - 1) * nrays + k
        o = 20.0f0 .* (rand(rng, Float32, 3) .- 0.5f0)  # stay well inside the box
        o[ax] = 0.0f0
        xs[:, col] = o
        xe[:, col] = o
        xs[ax, col] = -150.0f0
        xe[ax, col] = 150.0f0
    end

    proj = Array(joseph3d_fwd(to_dev(xs), to_dev(xe), to_dev(img), org, vs))
    expected = n[1] * vs[1]
    return maximum(abs.(proj .- expected)) / expected
end

"""
Relative deviation of the adjointness identity `<Ax, y> == <x, A'y>` for a
random image, random LORs and random projection weights.
"""
function adjointness_rel(to_dev)
    rng = MersenneTwister(2)
    n = (32, 32, 32)
    vs = (2.0f0, 2.0f0, 2.0f0)
    org = centered_origin(n, vs)

    nlors = 50_000
    x = rand(rng, Float32, n)
    xs, xe = make_lors(rng, nlors, 80.0f0)
    y = rand(rng, Float32, nlors)

    Ax = Array(joseph3d_fwd(to_dev(xs), to_dev(xe), to_dev(x), org, vs))
    Aty = Array(joseph3d_back(to_dev(xs), to_dev(xe), to_dev(y), n, org, vs))

    lhs = dot(Float64.(Ax), Float64.(y))
    rhs = dot(Float64.(vec(x)), Float64.(vec(Aty)))
    return abs(lhs - rhs) / abs(lhs)
end

@testset "RecoCrysp" begin
    @testset "CPU" begin
        @test analytic_max_rel(identity) < 1.0f-3
        @test adjointness_rel(identity) < 1.0e-3
    end

    @testset "rays missing the image" begin
        n = (16, 16, 16)
        vs = (2.0f0, 2.0f0, 2.0f0)
        org = centered_origin(n, vs)
        img = ones(Float32, n)
        # parallel to the box, far outside
        xs = Float32[-100.0 100.0; 100.0 100.0; 0.0 0.0]
        xe = Float32[100.0 100.0; 100.0 -100.0; 0.0 0.0]
        proj = joseph3d_fwd(xs, xe, img, org, vs)
        @test all(proj .== 0.0f0)
        bimg = joseph3d_back(xs, xe, Float32[1.0, 1.0], size(img), org, vs)
        @test all(bimg .== 0.0f0)
    end

    @testset "PET scanner geometry" begin
        nr = 4
        ring_pos = Float32.(2.0 .* (0:(nr - 1)))
        ring_pos .-= 0.5f0 * maximum(ring_pos)
        scanner = RegularPolygonPETScannerGeometry(;
            radius = 100.0, num_sides = 8, num_lor_endpoints_per_side = 4,
            lor_spacing = 3.0, ring_positions = ring_pos, symmetry_axis = 3,
        )
        nper = 8 * 4
        @test size(scanner.all_lor_endpoints) == (3, nper * nr)

        # transverse distance of every endpoint: sqrt(R^2 + pos^2) with
        # pos in {±1.5, ±4.5} for 4 crystals spaced 3 mm about the side center
        trans = scanner.all_lor_endpoints[[1, 2], :]
        d = vec(sqrt.(sum(abs2, trans; dims = 1)))
        @test all(minimum(d) .≈ sqrt(100.0f0^2 + 1.5f0^2))
        @test all(maximum(d) .≈ sqrt(100.0f0^2 + 4.5f0^2))

        # axial coordinate equals the ring position
        for r in 1:nr
            pts = get_lor_endpoints(scanner, fill(r, 3), [1, 2, nper])
            @test all(pts[3, :] .== ring_pos[r])
        end

        desc = RegularPolygonPETLORDescriptor(scanner; radial_trim = 3,
                                              sinogram_order = :PVR)
        @test desc.num_views == nper ÷ 2
        @test desc.num_rad == nper - 1 - 2 * 3
        @test desc.num_planes == nr^2

        xs, xe = get_lor_coordinates(desc)
        nlors = desc.num_planes * desc.num_views * desc.num_rad
        @test size(xs) == (3, nlors) && size(xe) == (3, nlors)

        # every plane's axial coordinates are a valid ring-position pair
        @test all(in(ring_pos), xs[3, :]) && all(in(ring_pos), xe[3, :])

        # different sinogram orders contain the same LORs, just reordered
        xs2, xe2 = get_lor_coordinates(
            RegularPolygonPETLORDescriptor(scanner; radial_trim = 3,
                                           sinogram_order = :RVP),
        )
        lorset(a, b) = sort!([(a[:, i]..., b[:, i]...) for i in axes(a, 2)])
        @test lorset(xs, xe) == lorset(xs2, xe2)
    end

    @testset "continuous PET geometry" begin
        sc = ContinuousPET(diameter = 774, afov = 1024)     # CRYSP dimensions
        @test sc.radius == 387.0f0
        @test sc.afov == 1024.0f0
        @test ContinuousPET(radius = 387, afov = 1024).radius == sc.radius
        @test_throws ArgumentError ContinuousPET(diameter = 774, radius = 387, afov = 1024)
        @test_throws ArgumentError ContinuousPET(afov = 1024)

        nlors = 100_000
        xs, xe = sample_lors(sc, nlors; rng = MersenneTwister(1))
        @test size(xs) == (3, nlors) && size(xe) == (3, nlors)

        # every endpoint lies on the cylinder surface (transverse radius == R)
        rad(m) = sqrt.(m[1, :] .^ 2 .+ m[2, :] .^ 2)
        @test all(isapprox.(rad(xs), sc.radius; rtol = 1.0f-4))
        @test all(isapprox.(rad(xe), sc.radius; rtol = 1.0f-4))

        # axial coordinate stays within the AFOV
        @test maximum(abs.(xs[3, :])) <= sc.afov / 2
        @test maximum(abs.(xe[3, :])) <= sc.afov / 2

        # seeded sampling is reproducible
        xs2, _ = sample_lors(sc, nlors; rng = MersenneTwister(1))
        @test xs == xs2

        # the sampled LORs drive the forward projector and cross the FOV
        n = (32, 32, 32)
        vs = (4.0f0, 4.0f0, 4.0f0)
        org = centered_origin(n, vs)
        proj = joseph3d_fwd(xs, xe, ones(Float32, n), org, vs)
        @test length(proj) == nlors && any(proj .> 0.0f0)
    end

    @testset "continuous sensitivity normalization (scale)" begin
        # sens = scale * A^T(w) must stay on the EVENT per-LOR scale: computed over
        # a denser LOR set (n_sens) with scale = n_events/n_sens, its magnitude is
        # unchanged in expectation but its Monte-Carlo noise is lower.
        sc = ContinuousPET(diameter = 200.0, afov = 60.0)
        ish = (31, 31, 11)
        vs = (4.0f0, 4.0f0, 4.0f0)
        org = ntuple(i -> -(ish[i] - 1) / 2 * vs[i], 3)

        n_ev = 1_000_000
        xs1, xe1 = sample_lors(sc, 250_000; rng = MersenneTwister(20))
        xs2, xe2 = sample_lors(sc, 1_000_000; rng = MersenneTwister(21))
        s1 = sensitivity_image(xs1, xe1, ish, org, vs; scale = n_ev / size(xs1, 2))
        s2 = sensitivity_image(xs2, xe2, ish, org, vs; scale = n_ev / size(xs2, 2))

        cx = Float32[org[1] + (i - 1) * vs[1] for i in 1:ish[1]]
        cy = Float32[org[2] + (j - 1) * vs[2] for j in 1:ish[2]]
        mask = falses(ish)
        for k in 1:ish[3], j in 1:ish[2], i in 1:ish[1]
            cx[i]^2 + cy[j]^2 <= 40.0f0^2 && (mask[i, j, k] = true)
        end
        mn(v) = sum(v) / length(v)
        cv(v) = (m = mn(v); sqrt(sum(abs2, v .- m) / length(v)) / m)
        m1, m2 = mn(s1[mask]), mn(s2[mask])
        @test abs(m1 - m2) / m2 < 0.05           # same scale, independent of n_sens
        @test cv(s2[mask]) < cv(s1[mask])        # denser sens is smoother

        # default scale = 1 is unchanged (backward compatible)
        @test sensitivity_image(xs1, xe1, ish, org, vs) ==
              sensitivity_image(xs1, xe1, ish, org, vs; scale = 1)
    end

    @testset "phantoms" begin
        n = (61, 61, 61)            # ±60 mm: large enough to contain the phantoms
        vs = (2.0f0, 2.0f0, 2.0f0)
        org = ntuple(i -> -(n[i] - 1) / 2 * vs[i], 3)
        vvox = prod(vs)

        sph = uniform_sphere(n, org, vs; radius = 40.0, value = 2.0)
        @test all(v -> v == 0.0f0 || v == 2.0f0, sph)
        @test isapprox(count(>(0), sph) * vvox, 4 / 3 * pi * 40.0^3; rtol = 0.05)

        cyl = uniform_cylinder(n, org, vs; radius = 40.0, length = 30.0)
        @test isapprox(count(>(0), cyl) * vvox, pi * 40.0^2 * 30.0; rtol = 0.05)

        # derenzo rods: nonzero, valued, inside the cylinder and the axial extent
        der = derenzo(n, org, vs; radius = 50.0, length = 30.0,
                      rod_diameters = [4.0, 5.0, 6.0, 8.0, 10.0, 12.0], value = 1.0)
        @test count(>(0), der) > 0
        x = Float32[org[1] + (i - 1) * vs[1] for i in 1:n[1]]
        y = Float32[org[2] + (i - 1) * vs[2] for i in 1:n[2]]
        z = Float32[org[3] + (i - 1) * vs[3] for i in 1:n[3]]
        inside = true
        @inbounds for k in 1:n[3], j in 1:n[2], i in 1:n[1]
            if der[i, j, k] > 0
                (x[i]^2 + y[j]^2 <= 50.0^2 + 1 && abs(z[k]) <= 15.0 + 1.0e-3) || (inside = false)
            end
        end
        @test inside
    end

    @testset "gaussian_blur (resolution operator G)" begin
        n = (41, 41, 41)
        vs = (1.5f0, 1.5f0, 1.5f0)
        fwhm = 6.0
        pt = zeros(Float32, n); pt[21, 21, 21] = 1.0f0
        g = gaussian_blur(pt, fwhm, vs)
        @test isapprox(sum(g), 1.0f0; atol = 1.0f-3)                    # counts preserved

        # FWHM recovered from the central line (G of a delta is a Gaussian)
        line = g[:, 21, 21]
        xs = Float32[(i - 21) * vs[1] for i in 1:n[1]]
        mu = sum(xs .* line) / sum(line)
        sig = sqrt(sum(line .* (xs .- mu) .^ 2) / sum(line))
        @test isapprox(2.3548f0 * sig, Float32(fwhm); rtol = 0.05)

        # self-adjoint: <Gx, y> == <x, Gy>
        rng = MersenneTwister(5)
        a = rand(rng, Float32, n); b = rand(rng, Float32, n)
        @test isapprox(sum(gaussian_blur(a, fwhm, vs) .* b),
                       sum(a .* gaussian_blur(b, fwhm, vs)); rtol = 1.0f-4)
    end

    @testset "reconstruction (noise-free, CPU)" begin
        p = build_recon_problem(identity)

        # the true image is an exact MLEM fixed point (contamination = 0)
        x1 = em_update(p.model, p.xt)
        @test relerr(Array(x1)[p.fov], p.x_true[p.fov]) < 1.0f-3

        # MLEM: monotone likelihood + convergence toward the phantom
        nlls = Float64[]
        x = mlem(p.model, p.x0; niter = 100,
                 callback = (k, xx) -> push!(nlls, neg_log_likelihood(p.model, xx)))
        @test maximum(diff(nlls)) <= 1.0e-4 * abs(nlls[1])   # non-increasing (tol)
        @test nlls[end] < nlls[1]
        @test relerr(Array(x)[p.fov], p.x_true[p.fov]) < 0.2

        # OSEM with 6 subsets reaches comparable quality in fewer epochs
        models = subset_models(p.xs, p.xe, p.org, p.vs, p.ish, 6; counts = p.y)
        xo = osem(models, p.x0; nepochs = 15)
        @test relerr(Array(xo)[p.fov], p.x_true[p.fov]) < 0.2
    end

    @testset "reconstruction (Poisson noise, CPU)" begin
        p = build_recon_problem(identity)

        # scale the phantom to a realistic total count level, then draw Poisson
        # counts (seeded). p.y holds the noise-free expected counts A·x_true.
        total_counts = 1.0f6
        scale = total_counts / sum(p.y)
        x_true = scale .* p.x_true
        lambda = scale .* p.y
        rng = MersenneTwister(7)
        counts = Float32[l > 0 ? rand(rng, Poisson(Float64(l))) : 0.0f0 for l in lambda]

        model = ListmodePoissonModel(p.xs, p.xe, p.model.sensitivity;
                                     img_origin = p.org, voxsize = p.vs, counts = counts)

        # MLEM on noisy data: the data log-likelihood still improves monotonically
        nlls = Float64[]
        x50 = mlem(model, p.x0; niter = 50,
                   callback = (k, xx) -> push!(nlls, neg_log_likelihood(model, xx)))
        @test maximum(diff(nlls)) <= 1.0e-4 * abs(nlls[1])

        # reconstruction stays non-negative and recovers the phantom at early stop
        @test all(x50 .>= 0.0f0)
        e50 = relerr(x50[p.fov], x_true[p.fov])
        @test e50 < 0.2

        # MLEM semi-convergence: many more iterations amplify noise (worse image)
        x200 = mlem(model, p.x0; niter = 200)
        @test relerr(x200[p.fov], x_true[p.fov]) > e50
    end

    @testset "normalization (sens = Aᵀ·norm)" begin
        p = build_recon_problem(identity)

        # per-LOR multiplicative factors n (e.g. crystal-efficiency pattern)
        rng = MersenneTwister(11)
        n = 0.4f0 .+ 0.6f0 .* rand(rng, Float32, size(p.xs, 2))
        counts = n .* p.y                       # normalized noise-free data, s = 0
        sens = sensitivity_image(p.xs, p.xe, p.ish, p.org, p.vs; weights = n)
        model = ListmodePoissonModel(p.xs, p.xe, sens;
                                     img_origin = p.org, voxsize = p.vs,
                                     counts = counts, mult = n)

        # x_true is still an exact MLEM fixed point of the normalized model
        @test relerr(em_update(model, p.xt)[p.fov], p.x_true[p.fov]) < 1.0f-3

        # and MLEM with correct normalization recovers the phantom
        x = mlem(model, p.x0; niter = 60)
        e_correct = relerr(x[p.fov], p.x_true[p.fov])
        @test e_correct < 0.2

        # OSEM with normalization (subset sensitivities use Aᵀ(n_subset))
        models = subset_models(p.xs, p.xe, p.org, p.vs, p.ish, 6; counts = counts, mult = n)
        xo = osem(models, p.x0; nepochs = 12)
        @test relerr(xo[p.fov], p.x_true[p.fov]) < 0.2

        # ignoring normalization on normalized data biases the reconstruction
        wrong = ListmodePoissonModel(
            p.xs, p.xe, sensitivity_image(p.xs, p.xe, p.ish, p.org, p.vs);
            img_origin = p.org, voxsize = p.vs, counts = counts,  # mult = 1
        )
        xw = mlem(wrong, p.x0; niter = 60)
        @test relerr(xw[p.fov], p.x_true[p.fov]) > e_correct
    end

    if Metal.functional()
        @testset "Metal" begin
            @test analytic_max_rel(Metal.MtlArray) < 1.0f-3
            @test adjointness_rel(Metal.MtlArray) < 1.0e-3
        end

        @testset "CPU vs Metal consistency" begin
            rng = MersenneTwister(4)
            n = (32, 32, 32)
            vs = (2.0f0, 2.0f0, 2.0f0)
            org = centered_origin(n, vs)
            nlors = 50_000
            x = rand(rng, Float32, n)
            xs, xe = make_lors(rng, nlors, 80.0f0)
            y = rand(rng, Float32, nlors)

            p_cpu = joseph3d_fwd(xs, xe, x, org, vs)
            b_cpu = joseph3d_back(xs, xe, y, n, org, vs)
            p_gpu = Array(joseph3d_fwd(Metal.MtlArray(xs), Metal.MtlArray(xe),
                                       Metal.MtlArray(x), org, vs))
            b_gpu = Array(joseph3d_back(Metal.MtlArray(xs), Metal.MtlArray(xe),
                                        Metal.MtlArray(y), n, org, vs))

            @test maximum(abs.(p_cpu .- p_gpu)) / maximum(abs.(p_cpu)) < 1.0f-4
            @test maximum(abs.(b_cpu .- b_gpu)) / maximum(abs.(b_cpu)) < 1.0f-3
        end

        @testset "reconstruction CPU vs Metal" begin
            pc = build_recon_problem(identity)
            pm = build_recon_problem(Metal.MtlArray)
            xc = mlem(pc.model, pc.x0; niter = 40)
            xm = mlem(pm.model, pm.x0; niter = 40)
            @test relerr(Array(xm)[pc.fov], Array(xc)[pc.fov]) < 1.0f-3
        end
    else
        @info "Metal not functional on this machine — GPU testsets skipped"
    end
end
