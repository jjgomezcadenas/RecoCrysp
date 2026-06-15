using Test
using Random
using LinearAlgebra
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
    else
        @info "Metal not functional on this machine — GPU testsets skipped"
    end
end
