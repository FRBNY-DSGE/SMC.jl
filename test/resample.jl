writing_output = false
@everywhere Random.seed!(42)

if VERSION < v"1.5"
    ver = "111"
else 
    ver = "150"
end

weights = rand(400)
weights = weights ./ sum(weights)

test_sys_resample    = SMC.resample(weights, method = :systematic)
test_multi_resample  = SMC.resample(weights, method = :multinomial)
test_poly_resample   = SMC.resample(weights, method = :polyalgo)

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
