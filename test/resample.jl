writing_output = false
Random.seed!(42)

if VERSION < v"1.5"
    ver = "111"
elseif VERSION < v"1.7"
    ver = "150"
else
    # Julia 1.7 switched the default RNG from a single process-wide
    # MersenneTwister to a per-Task Xoshiro256++ (TaskLocalRNG), so seeded
    # draws no longer match the "150" reference data.
    ver = "1126"
end

weights = rand(400)
weights = weights ./ sum(weights)

test_sys_resample    = SMC.resample(weights, method = :systematic)
test_multi_resample  = SMC.resample(weights, method = :multinomial)
test_poly_resample   = SMC.resample(weights, method = :polyalgo)

display(@benchmark SMC.resample($weights, method = :systematic))
display(@benchmark SMC.resample($weights, method = :multinomial))
display(@benchmark SMC.resample($weights, method = :polyalgo))

saved_filename = string("reference/resample_version=", ver, ".jld2")
if writing_output 
    jldopen(saved_filename, true, true, true, IOStream) do file
        write(file, "sys", test_sys_resample)
        write(file, "multi", test_multi_resample)
        write(file, "poly", test_poly_resample)
    end
end

saved_sys_resample   = load(saved_filename, "sys")
saved_multi_resample = load(saved_filename, "multi")
saved_poly_resample  = load(saved_filename, "poly")

####################################################################

@testset "Resampling methods" begin
    @test test_sys_resample   == saved_sys_resample
    @test test_multi_resample == saved_multi_resample
    @test test_poly_resample  == saved_poly_resample
end
