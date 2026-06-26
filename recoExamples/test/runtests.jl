using Test
using HDF5
using RecoExamples

# Write a tiny PTCRYSP-schema coincidence file with known rows, then check the
# reader rescales correctly and the class masks match the (truth, nscat) table.
function write_fixture(path)
    # five hand-built events: true, single, multiple, random(unscat), random(scat)
    truth  = Int8[0, 1, 1, 2, 2]
    nscat1 = Int8[0, 1, 1, 0, 1]
    nscat2 = Int8[0, 0, 2, 0, 0]
    h5open(path, "w") do f
        attributes(f)["xyz_scale_mm"] = 0.1
        attributes(f)["e_scale_keV"]  = 0.1
        f["truth"]  = truth
        f["nscat1"] = nscat1
        f["nscat2"] = nscat2
        # positions: store in 0.1 mm units, so value 3870 -> 387.0 mm
        for (n, v) in (("x1_mm", 3870), ("y1_mm", 0), ("z1_mm", -1000),
                       ("x2_mm", -3870), ("y2_mm", 0), ("z2_mm", 1000),
                       ("x0_mm", 50), ("y0_mm", -50), ("z0_mm", 0))
            f[n] = fill(Int16(v), 5)
        end
        for (n, v) in (("iz1", 3), ("iphi1", 10), ("iz2", 15), ("iphi2", 30))
            f[n] = fill(Int16(v), 5)
        end
        f["e1_keV"] = fill(Int16(5110), 5)   # -> 511.0 keV
        f["e2_keV"] = fill(Int16(5000), 5)
    end
end

@testset "MC listmode reader" begin
    mktempdir() do dir
        path = joinpath(dir, "fix.h5")
        write_fixture(path)
        c = read_coincidences(path)

        @test length(c) == 5
        @test size(c.xstart) == (3, 5)
        @test c.xstart[:, 1] ≈ Float32[387.0, 0.0, -100.0]      # rescaled by 0.1
        @test c.xend[:, 1]   ≈ Float32[-387.0, 0.0, 100.0]
        @test c.origin[:, 1] ≈ Float32[5.0, -5.0, 0.0]
        @test c.energy[:, 1] ≈ Float32[511.0, 500.0]
        @test c.elem1[:, 1] == Int16[3, 10]                     # (iz, iphi)

        # class masks against the known truth/nscat
        @test is_true(c)             == Bool[1, 0, 0, 0, 0]
        @test is_single_scatter(c)   == Bool[0, 1, 0, 0, 0]
        @test is_multiple_scatter(c) == Bool[0, 0, 1, 0, 0]
        @test is_random(c)           == Bool[0, 0, 0, 1, 1]     # keyed on truth, any nscat
        @test is_scatter(c)          == (is_single_scatter(c) .| is_multiple_scatter(c))

        # endpoints(mask) selects the right columns
        xs, xe = endpoints(c, is_random(c))
        @test size(xs, 2) == 2
    end
end

# Optional: if the real water file is present, sanity-check the class fractions.
const WATER = expanduser("~/Projects/PTCryspMC.jl/prod/sphere_water_csi/lors_det.h5")
if isfile(WATER)
    @testset "MC listmode reader — real water file" begin
        c = read_coincidences(WATER)
        n = length(c)
        @test n > 100_000
        ftrue = count(is_true(c)) / n
        frand = count(is_random(c)) / n
        @test 0.78 < ftrue < 0.86                # ~82% true
        @test frand < 0.01                       # randoms negligible
        @test count(is_single_scatter(c)) > count(is_multiple_scatter(c))
    end
end
