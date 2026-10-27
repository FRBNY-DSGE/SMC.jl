using ModelConstructors, HDF5, Random, JLD2, FileIO, SMC, Test, Distributed
include("modelsetup.jl")

path = dirname(@__FILE__)
writing_output = false
if !@isdefined(run_benchmarks); run_benchmarks = false; end

# Exercise the parallel path only when separate worker processes are present.
# Load SMC on the workers before this file to test that path.
use_parallel = nprocs() > 1   # true iff at least one separate worker process exists
if use_parallel
    modelsetup_path = joinpath(@__DIR__, "modelsetup.jl")
    @everywhere workers() using Random
    @everywhere workers() include($modelsetup_path)
end

# Keep the version tag for optional fixture output.
if VERSION < v"1.5"
    ver = "111"
elseif VERSION < v"1.7"
    ver = "150"
else
    ver = "1126"
end

m = setup_linear_model()

save = normpath(joinpath(dirname(@__FILE__),"save"))
m <= Setting(:saveroot, save)
savepath = rawpath(m, "estimate", "smc_cloud.jld2")
particle_store_path = rawpath(m, "estimate", "smcsave.h5")

data = h5read("reference/test_data.h5", "data")
full_data = copy(data)

println("Estimating Linear Model... (approx. 8 minutes)")

# n_mh_steps = 5: fewer MH steps per tempering stage under-mix this 9-parameter problem
# (the weakest-identified equation, σ1, gets stuck — max|dev| ≈ 0.65 at 1 step, erratic up
# to ~48). 3 steps used to pass only because acceptance was summed over the steps (N3), so c
# adapted to a much larger scale; with acceptance per step, 3 steps got stuck at 1 of 12 seeds
# (seed 42: max|dev| 15.8) and 5 steps passed at all 12 (max|dev| ≤ 0.134).
SMC.smc(loglik_fn, m.parameters, data, verbose = :none, use_fixed_schedule = true,
        parallel = use_parallel, n_Φ = 120, n_mh_steps = 5, resampling_method = :polyalgo,
        data_vintage = "200707", target = 0.25, savepath = savepath,
        particle_store_path = particle_store_path, α = .9, threshold_ratio = .5,
        smc_iteration = 0, smc_seed = 42)

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

@testset "Linear Regression Parameter Estimates Are Close" begin
    # Posterior mean should be close to true parameters
    @test maximum(abs.(mean(SMC.get_vals(test_cloud), dims = 2) -
                       [1., 1., 1., 2., 2., 1., 3., 3., 1.])) < .5
end

@testset "ParticleCloud Fields: Linear" begin
    @test size(SMC.get_vals(test_cloud)) == (9, length(test_cloud))
    @test all(isfinite, SMC.get_vals(test_cloud))
    @test all(isfinite, SMC.get_loglh(test_cloud))
    @test all(isfinite, SMC.get_weights(test_cloud))
    @test sum(SMC.get_weights(test_cloud)) ≈ length(test_cloud)
    @test test_cloud.tempering_schedule[end] ≈ 1.0
end

@testset "Individual Particle Fields Post-SMC: Linear" begin
    particle = test_cloud.particles[1, :]
    N = length(particle)
    @test all(isfinite, particle[1:SMC.ind_para_end(N)])
    @test isfinite(particle[SMC.ind_loglh(N)])
    @test isfinite(particle[SMC.ind_logprior(N)])
    @test particle[SMC.ind_weight(N)] >= 0
end

@testset "Weight Matrices: Linear" begin
    @test all(isfinite, test_w)
    @test all(isfinite, test_W)
    @test all(test_w .>= 0)
    @test all(test_W .>= 0)
end


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
global parameters = :caller_parameters

println("Estimating Linear Model on 1st half of sample... (approx. 2 minutes)")
SMC.smc(loglik_fn, m_old.parameters, data[:, 1:Int(floor(end/2))], verbose = :none,
        use_fixed_schedule = true, parallel = use_parallel,  n_Φ = 100, n_mh_steps = 1,
        resampling_method = :polyalgo, data_vintage = "000000", target = 0.25,
        savepath = savepath, particle_store_path = particle_store_path, α = .9,
        threshold_ratio = .5, smc_iteration = 0, n_parts = 1000, smc_seed = 43)
@test data == full_data
@test parameters === :caller_parameters

m_new = deepcopy(m)

# Estimate with 2nd half of sample
m_new <= Setting(:data_vintage, "200708")
old_vint = "000000"
m_new <= Setting(:previous_data_vintage, old_vint)
loadpath = rawpath(m_old, "estimate", "smc_cloud.jld2")

old_result = load(loadpath)
old_cloud = old_result["cloud"]
old_w, old_W = old_result["w"], old_result["W"]
old_mdd = sum(log.(vec(sum(old_w[:, 2:end] .* old_W[:, 1:end-1] ./ length(old_cloud), dims = 1))))
old_data = data[:, 1:Int(floor(end/2))]

println("Estimating Linear Model using a bridge distribution... (approx. 2 minutes)")
SMC.smc(loglik_fn, m_new.parameters, data, old_data=old_data, old_cloud=old_cloud,
        verbose = :none, use_fixed_schedule = true, parallel = use_parallel,
        n_Φ = 100, n_mh_steps = 1, resampling_method = :polyalgo, data_vintage = "200708",
        target = 0.25, savepath = savepath, particle_store_path = particle_store_path,
        α = .9, threshold_ratio = .5, smc_iteration = 0,
        tempered_update_prior_weight = 0.5, log_prob_old_data = old_mdd,
        smc_seed = 44)
@test data == full_data
@test parameters === :caller_parameters

loadpath = rawpath(m_new, "estimate", "smc_cloud.jld2")
new_cloud = load(loadpath, "cloud")

@testset "Bridged Cloud Likelihoods and Weights" begin
    # A tempered update must retain the previous-data likelihood and store the
    # likelihood for the complete data in the current log-likelihood field.
    bridge_vals = SMC.get_vals(new_cloud; transpose = false)
    bridge_loglh = SMC.get_loglh(new_cloud)
    bridge_old_loglh = SMC.get_old_loglh(new_cloud)
    @test all(isapprox.(bridge_loglh,
                        [loglik_fn(view(bridge_vals, i, :), data) for i in axes(bridge_vals, 1)]))
    @test all(isapprox.(bridge_old_loglh,
                        [loglik_fn(view(bridge_vals, i, :), old_data) for i in axes(bridge_vals, 1)]))

    # Cloud weights are normalized to the particle count in SMC.jl.
    bridge_weights = SMC.get_weights(new_cloud)
    @test all(isfinite, bridge_weights)
    @test all(>=(0), bridge_weights)
    @test sum(bridge_weights) ≈ length(new_cloud)
    @test maximum(abs.(vec(mean(SMC.get_vals(new_cloud), dims = 2)) .-
                        [1., 1., 1., 2., 2., 1., 3., 3., 1.])) < .5
end

# Clean output files up
rm(rawpath(m_new, "estimate", "smc_cloud.jld2"))
rm(rawpath(m_new, "estimate", "smcsave.h5"))

# Benchmark last so re-running smc (which overwrites savepath) can't clobber the cloud the
# assertions above load from savepath.
if run_benchmarks
    @btime SMC.smc($loglik_fn, $m.parameters, $data, verbose = :none,
        use_fixed_schedule = true, parallel = $use_parallel, n_Φ = 120, n_mh_steps = 3,
        resampling_method = :polyalgo, data_vintage = "200707", target = 0.25,
        savepath = $savepath, particle_store_path = $particle_store_path,
        α = .9, threshold_ratio = .5, smc_iteration = 0) evals=1 samples=1
end
