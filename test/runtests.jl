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
