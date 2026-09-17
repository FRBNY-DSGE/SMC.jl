writing_output = false
if !@isdefined(run_benchmarks); run_benchmarks = false; end
Random.seed!(42)   # plain (not @everywhere): single-process tests need the task-local RNG pinned

# Julia 1.7+ switched the default RNG to a per-Task Xoshiro256++, so the random weights
# and the seeded resampling draws differ from the "150" data — regenerate with the
# writing_output flag on under the target Julia.
if VERSION < v"1.5"
    ver = "111"
elseif VERSION < v"1.7"
    ver = "150"
else
    ver = "1126"
end

weights = rand(400)
weights = weights ./ sum(weights)

test_sys_resample    = SMC.resample(weights, method = :systematic)
test_multi_resample  = SMC.resample(weights, method = :multinomial)
test_poly_resample   = SMC.resample(weights, method = :polyalgo)

# Guard benchmarks: results above are already computed, so benchmarking (which consumes
# the RNG) can't perturb them, but keep it off by default.
if run_benchmarks
    @btime SMC.resample($weights, method = :systematic)
    @btime SMC.resample($weights, method = :multinomial)
    @btime SMC.resample($weights, method = :polyalgo)
end

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
