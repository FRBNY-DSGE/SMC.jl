using ModelConstructors, HDF5, Random, JLD2, FileIO, SMC, Test
include("modelsetup.jl")

path = dirname(@__FILE__)
writing_output = false
if !@isdefined(run_benchmarks); run_benchmarks = false; end

# Keep the version tag for optional fixture output.
if VERSION < v"1.5"
    ver = "111"
elseif VERSION < v"1.7"
    ver = "150"
else
    ver = "1126"
end

m = setup_linear_model(; regime_switching = true)
m <= Setting(:regime_switching, true, true, "rs", "") # For file output purposes

save = normpath(joinpath(dirname(@__FILE__),"save"))
m <= Setting(:saveroot, save)
savepath = rawpath(m, "estimate", "smc_cloud.jld2")
particle_store_path = rawpath(m, "estimate", "smcsave.h5")

data = h5read("reference/test_data.h5", "rsdata")

println("Estimating Linear Model... (approx. 8 minutes)")

SMC.smc(rs_loglik_fn, m.parameters, data, verbose = :none,
        use_fixed_schedule = true, parallel = false,
        n_Φ = 120, n_mh_steps = 1, resampling_method = :polyalgo,
        data_vintage = "200707", target = 0.25, savepath = savepath,
        particle_store_path = particle_store_path, α = .9,
        threshold_ratio = .5, smc_iteration = 0,
        regime_switching = true, toggle = true, smc_seed = 42)

println("Estimation done!")

test_file = load(rawpath(m, "estimate", "smc_cloud.jld2"))
test_cloud  = test_file["cloud"]
test_w      = test_file["w"]
test_W      = test_file["W"]
mean_para = mean(SMC.get_vals(test_cloud), dims = 2)
true_para = [1., 1., 1., # α1, β1, σ1 (regime 1)
             2., 2., 1., # α2, β2, σ2 (regime 1)
             3., 3., 1., # α3, β3, σ3 (regime 1)
             1., 1.,     # α1 regimes = 2-3
             2., 3.,     # β1 regimes = 2-3
             2., 2.,     # α2 regimes = 2-3
             3., 4.,     # β2 regimes = 2-3
             3., 3.,     # α3 regimes = 2-3
             4., 5.]     # β3 regimes = 2-3

if writing_output
    jldopen(string("reference/smc_cloud_fix=true_rs=true_version=", ver, ".jld2"), true, true, true, IOStream) do file
        write(file, "cloud", test_cloud)
        write(file, "w", test_w)
        write(file, "W", test_W)
    end
end

@testset "Linear Regression Regime-Switching Parameter Estimates Are Close" begin
    @test maximum(abs.(mean_para - true_para)) < .5
end

@testset "ParticleCloud Fields: Linear" begin
    @test size(SMC.get_vals(test_cloud)) == (length(true_para), length(test_cloud))
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

# Clean output files up
rm(rawpath(m, "estimate", "smc_cloud.jld2"))
rm(rawpath(m, "estimate", "smcsave.h5"))

# Benchmark last so re-running smc (which overwrites savepath) can't clobber the cloud the
# assertions above load from savepath.
if run_benchmarks
    @btime SMC.smc($rs_loglik_fn, $m.parameters, $data, verbose = :none,
        use_fixed_schedule = true, parallel = false, n_Φ = 120, n_mh_steps = 1,
        resampling_method = :polyalgo, data_vintage = "200707", target = 0.25,
        savepath = $savepath, particle_store_path = $particle_store_path,
        α = .9, threshold_ratio = .5, smc_iteration = 0,
        regime_switching = true, toggle = true) evals=1 samples=1
end
