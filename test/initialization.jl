write_test_output = false

path = dirname(@__FILE__)

if VERSION < v"1.5"
    ver = "111"
else 
    ver = "150"
end

###################################################################
# Test: initial_draw!()
###################################################################
include("modelsetup.jl")

m = setup_linear_model()

# Read in generated data
data = h5read("reference/test_data.h5", "data")
X = h5read("reference/test_data.h5", "X")

save = normpath(joinpath(dirname(@__FILE__),"save"))
m <= Setting(:saveroot, save)

####################################################################
init_cloud = SMC.Cloud(length(m.parameters), get_setting(m,:n_particles))

@everywhere Random.seed!(42)
SMC.initial_draw!(loglik_fn, m.parameters, data, init_cloud)

if write_test_output
    JLD2.jldopen(string("reference/initial_draw_out_version=", ver, ".jld2"), "w") do file
        write(file, "cloud", init_cloud)
    end
end

saved_init_cloud = load(string("reference/initial_draw_out_version=", ver, ".jld2"), "cloud")

@testset "Initial draw" begin
    @test @test_matrix_approx_eq SMC.get_vals(init_cloud) SMC.get_vals(saved_init_cloud)
    @test @test_matrix_approx_eq SMC.get_loglh(init_cloud) SMC.get_loglh(saved_init_cloud)
end

###################################################################
# Test: one_draw()
###################################################################
draw = SMC.one_draw(loglik_fn, m.parameters, data)

if write_test_output
    JLD2.jldopen(string("reference/one_draw_out_version=", ver, ".jld2"), true, true, true, IOStream) do file
        file["draw"] = draw
    end
end

test_draw = JLD2.jldopen(string("reference/one_draw_out_version=", ver, ".jld2"), "r") do file
    file["draw"]
end

###################################################################
@testset "One draw" begin
    @test draw[1] == test_draw[1]
    @test draw[2] ≈ test_draw[2]
end

###################################################################
# Test: draw_likelihood()
###################################################################
draw_lik = SMC.draw_likelihood(loglik_fn, m.parameters, data, vec(draw[1]))

if write_test_output
    JLD2.jldopen(string("reference/draw_likelihood_out_version=", ver, ".jld2"), true, true, true, IOStream) do file
        file["draw_lik"] = draw_lik
    end
end
test_draw_lik = JLD2.jldopen(string("reference/draw_likelihood_out_version=", ver, ".jld2"), "r") do file
    file["draw_lik"]
end

###################################################################
@testset "Draw Likelihood" begin
    @test draw_lik[1][1] ≈ test_draw_lik[1][1]
    @test draw_lik[2] == test_draw_lik[2]
end

###################################################################
# Test: initialize_likelihoods!()
###################################################################
SMC.initialize_likelihoods!(loglik_fn, m.parameters, data, init_cloud)

if write_test_output
    JLD2.jldopen(string("reference/initialize_likelihood_out_version=", ver, ".jld2"), true, true, true, IOStream) do file
        file["init_lik_cloud"] = init_cloud
    end
end
test_init_cloud = JLD2.jldopen(string("reference/initialize_likelihood_out_version=", ver, ".jld2"), "r") do file
    file["init_lik_cloud"]
end

###################################################################
@testset "Initialize Likelihoods" begin
    @test @test_matrix_approx_eq SMC.get_vals(init_cloud) SMC.get_vals(test_init_cloud)
    @test @test_matrix_approx_eq SMC.get_loglh(init_cloud) SMC.get_loglh(test_init_cloud)
end


###################################################################
# Test: initialize_cloud_settings!()
###################################################################
SMC.initialize_cloud_settings!(init_cloud)

if write_test_output
    JLD2.jldopen(string("reference/initialize_cloud_settings_version=", ver, ".jld2"), true, true, true, IOStream) do file
        file["init_cloud"] = init_cloud
    end
end
test_init_cloud = JLD2.jldopen(string("reference/initialize_cloud_settings_version=", ver, ".jld2"), "r") do file
    file["init_cloud"]
end

###################################################################
@testset "Initialize Cloud" begin
    @test init_cloud.ESS         == test_init_cloud.ESS
    @test init_cloud.stage_index == test_init_cloud.stage_index
    @test init_cloud.n_Φ         == test_init_cloud.n_Φ
    @test init_cloud.resamples   == test_init_cloud.resamples
    @test init_cloud.c           == test_init_cloud.c
    @test init_cloud.total_sampling_time == test_init_cloud.total_sampling_time
    @test init_cloud.tempering_schedule  == test_init_cloud.tempering_schedule
end
