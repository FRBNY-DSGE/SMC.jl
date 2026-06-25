writing_output = false
@everywhere Random.seed!(42)

if VERSION < v"1.5"
    ver = "111"
elseif VERSION < v"1.7"
    ver = "150"
else
    ver = "170"
end

Random.seed!(42)
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
    isfile(saved_filename) && rm(saved_filename)
    JLD2.jldsave(saved_filename;
        sys   = test_sys_resample,
        multi = test_multi_resample,
        poly  = test_poly_resample)
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
