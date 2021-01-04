using ModelConstructors, HDF5, Random, JLD2, FileIO, SMC, Test
include("modelsetup.jl")

path = dirname(@__FILE__)
writing_output = false

if VERSION < v"1.5"
    ver = "111"
else
    ver = "150"
end

m = setup_linear_model()

save = normpath(joinpath(dirname(@__FILE__),"save"))
m <= Setting(:saveroot, save)
savepath = rawpath(m, "estimate", "smc_cloud.jld2")
particle_store_path = rawpath(m, "estimate", "smcsave.h5")

data = h5read("reference/test_data.h5", "data")

@everywhere Random.seed!(42)

println("Estimating Linear Model... (approx. 3 minutes)")

SMC.smc(loglik_fn, m.parameters, data, verbose = :none, use_fixed_schedule = true,
        parallel = true, n_Φ = 120, n_mh_steps = 1, resampling_method = :polyalgo,
        data_vintage = "200707", target = 0.25, savepath = savepath,
        particle_store_path = particle_store_path, α = .9, threshold_ratio = .5, smc_iteration = 0)

println("Estimation done!")

test_file = load(rawpath(m, "estimate", "smc_cloud.jld2"))
test_cloud  = test_file["cloud"]
test_w      = test_file["w"]
test_W      = test_file["W"]

if writing_output
    jldopen(string("reference/smc_cloud_fix=true_version=", ver, ".jld2"), true, true, true, IOStream) do file
        write(file, "cloud", test_cloud)
        write(file, "w", test_w)
        write(file, "W", test_W)
    end
end

saved_file  = load(string("reference/smc_cloud_fix=true_version=", ver, ".jld2"))
saved_cloud = saved_file["cloud"]
saved_w     = saved_file["w"]
saved_W     = saved_file["W"]

####################################################################
cloud_fields = fieldnames(typeof(test_cloud))
@testset "Linear Regression Parameter Estimates Are Close" begin
    # Posterior mean should be close to true parameters
    @test maximum(abs.(mean(SMC.get_vals(test_cloud), dims = 2) -
                       [1., 1., 1., 2., 2., 1., 3., 3., 1.])) < .5
end

@testset "ParticleCloud Fields: Linear" begin
    @test @test_matrix_approx_eq SMC.get_vals(test_cloud) SMC.get_vals(saved_cloud)
    @test @test_matrix_approx_eq SMC.get_loglh(test_cloud) SMC.get_loglh(saved_cloud)
    @test length(test_cloud.particles) == length(saved_cloud.particles)
    @test test_cloud.tempering_schedule == saved_cloud.tempering_schedule
    @test test_cloud.ESS ≈ saved_cloud.ESS
    @test test_cloud.stage_index == saved_cloud.stage_index
    @test test_cloud.n_Φ == saved_cloud.n_Φ
    @test test_cloud.resamples == saved_cloud.resamples
    @test test_cloud.c == saved_cloud.c
    @test test_cloud.accept == saved_cloud.accept
end

test_particle  = test_cloud.particles[1,:]
saved_particle = saved_cloud.particles[1,:]
N = length(test_particle)
@testset "Individual Particle Fields Post-SMC: Linear" begin
    @test test_particle[1:SMC.ind_para_end(N)] ≈ saved_particle[1:SMC.ind_para_end(N)]
    @test test_particle[SMC.ind_loglh(N)]      ≈ saved_particle[SMC.ind_loglh(N)]
    @test test_particle[SMC.ind_logpost(N)]    ≈ saved_particle[SMC.ind_logpost(N)]
    @test test_particle[SMC.ind_logprior(N)]   ≈ saved_particle[SMC.ind_logprior(N)]
    @test test_particle[SMC.ind_old_loglh(N)] == saved_particle[SMC.ind_old_loglh(N)]
    @test test_particle[SMC.ind_accept(N)]    == saved_particle[SMC.ind_accept(N)]
    @test test_particle[SMC.ind_weight(N)]     ≈ saved_particle[SMC.ind_weight(N)]
end

@testset "Weight Matrices: Linear" begin
    @test @test_matrix_approx_eq test_w saved_w
    @test @test_matrix_approx_eq test_W saved_W
end

# TODO: add a check here that the correct parameters are estimated

####################################################################
# Bridging Test
####################################################################
include("modelsetup.jl")
m = setup_linear_model()

save = normpath(joinpath(dirname(@__FILE__),"save"))
m <= Setting(:saveroot, save)

data = h5read("reference/test_data.h5", "data")

@everywhere Random.seed!(42)

# Estimate with 1st half of sample
m_old = deepcopy(m)

println("Estimating Linear Model on 1st half of sample... (approx. 2 minutes)")
SMC.smc(loglik_fn, m_old.parameters, data[:, 1:Int(floor(end/2))], verbose = :none,
        use_fixed_schedule = true, parallel = true,  n_Φ = 100, n_mh_steps = 1,
        resampling_method = :polyalgo, data_vintage = "000000", target = 0.25,
        savepath = savepath, particle_store_path = particle_store_path, α = .9,
        threshold_ratio = .5, smc_iteration = 0, n_parts = 1000)

m_new = deepcopy(m)

# Estimate with 2nd half of sample
m_new <= Setting(:data_vintage, "200708")
m_new <= Setting(:tempered_update_prior_weight, 0.5)
#m_new <= Setting(:tempered_update, true)
old_vint = "000000"
m_new <= Setting(:previous_data_vintage, old_vint)
loadpath = rawpath(m_old, "estimate", "smc_cloud.jld2")

old_cloud = load(loadpath, "cloud")

println("Estimating Linear Model using a bridge distribution... (approx. 2 minutes)")
SMC.smc(loglik_fn, m_new.parameters, data, old_data=data[:,1:Int(floor(end/2))], old_cloud=old_cloud,
        verbose = :none, use_fixed_schedule = true, parallel = true,
        n_Φ = 100, n_mh_steps = 1, resampling_method = :polyalgo, data_vintage = "200708",
        target = 0.25, savepath = savepath, particle_store_path = particle_store_path,
        α = .9, threshold_ratio = .5, smc_iteration = 0)

loadpath = rawpath(m_new, "estimate", "smc_cloud.jld2")
new_cloud = load(loadpath, "cloud")

@testset "Linear Regression Parameter Estimates Are Close" begin
    # Posterior mean should be close to true parameters
    @test maximum(abs.(mean(SMC.get_vals(test_cloud), dims = 2) - [1., 1., 1., 2., 2., 1., 3., 3., 1.])) < .5
end

# Clean output files up
rm(rawpath(m_new, "estimate", "smc_cloud.jld2"))
rm(rawpath(m_new, "estimate", "smcsave.h5"))
