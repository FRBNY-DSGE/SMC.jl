using Test, Distributed, SMC, ModelConstructors, Distributions, JLD2, Random, LinearAlgebra
@everywhere using SMC, ModelConstructors, Distributions, LinearAlgebra
@everywhere include(joinpath($(@__DIR__), "param_blocks_coverage_workers.jl"))

# More coverage for fixed mutation blocks (param_blocks) and blockwise_c, beyond param_blocks.jl:
# 1. a regime-switching parameter's extra regime in its own block, end to end;
# 2. serial and 2-worker runs with blocks are bitwise identical;
# 3. the blocked sampler recovers a known Gaussian posterior;
# 4. a bridge (tempered update from an old cloud) with blocks recovers the new posterior.

# The likelihoods (Gaussian toy, its old-data version, the regime-switching toy) are in
# param_blocks_coverage_workers.jl, so workers added below can load them too.

function blocks_gauss_parameters()
    m = GenericModel()
    for k in 1:3
        m <= parameter(Symbol("x$k"), 0., (-20., 20.), (-20., 20.), Untransformed(), Normal(0, 2), fixed = false)
    end
    return m.parameters
end

function gauss_posterior(μ)
    P = GAUSS_Q + I / 4
    return P \ (GAUSS_Q * μ), inv(P)
end

function blocks_regime_parameters()
    m = GenericModel()
    m <= parameter(:x1, 0., (-10., 10.), (-10., 10.), Untransformed(), Normal(0, 1), fixed = false)
    m <= parameter(:x2, 0., (-10., 10.), (-10., 10.), Untransformed(), Normal(0, 1), fixed = false)
    m <= parameter(:a,  0., (-10., 10.), (-10., 10.), Untransformed(), Normal(0, 1), fixed = false)
    set_regime_val!(m[:a], 1, 0.)
    set_regime_val!(m[:a], 2, 0.)
    set_regime_prior!(m[:a], 1, Normal(0, 1))
    set_regime_prior!(m[:a], 2, Normal(0, 1))
    return m.parameters
end

# Weighted mean and covariance of the first n parameters of a saved cloud
function cloud_moments(cloud, n)
    W = SMC.get_weights(cloud)
    W = W ./ sum(W)
    X = cloud.particles[:, 1:n]
    μ = vec(sum(W .* X, dims = 1))
    D = X .- μ'
    return μ, D' * (W .* D)
end

mktempdir() do dir
    function run(name, lik, params; kwargs...)
        path = joinpath(dir, name * ".jld2")
        SMC.smc(lik, params, zeros(1, 1); verbose = :none, smc_seed = 7, α = 0.9,
                savepath = path, particle_store_path = replace(path, ".jld2" => ".h5"), kwargs...)
        return load(path)
    end

    @testset "Regime-switching parameter in its own block" begin
        # The extra regime of a is named :a_reg2 (smc_main.jl); its column is 4, after the base
        # parameters x1, x2, a. Posterior of a's regime 2: N(-0.3 * 10 / 11, 1 / 11).
        out = run("regime", blocks_regime_likelihood, blocks_regime_parameters();
                  regime_switching = true, n_parts = 400, n_Φ = 20, n_mh_steps = 3,
                  param_blocks = [[:a_reg2]], blockwise_c = true)
        h = out["stage_health"]
        @test out["param_blocks"] == [[:a_reg2], [:x1, :x2, :a]]
        @test h["block_labels"] == ["a_reg2", "rest"]
        @test size(h["c_blocks"]) == (2, length(h["stage"]))
        @test all(0 .<= h["accept_blocks"][:, 2:end] .<= 1)
        @test h["c_blocks"][1, end] != h["c_blocks"][2, end]
        cloud = out["cloud"]
        @test size(cloud.particles, 2) - 5 == 4             # x1, x2, a, a_reg2 (+5 bookkeeping columns)
        μ, Σ = cloud_moments(cloud, 4)
        # Each regime of a gets its own posterior: regime 1 near 0.2 * 10 / 11, regime 2 near -0.3 * 10 / 11
        @test abs(μ[3] - 0.2 * 10 / 11) < 0.1
        @test abs(μ[4] + 0.3 * 10 / 11) < 0.1
        @test abs(Σ[4, 4] - 1 / 11) < 0.04
        # Regime 2 really is mutated: many distinct values, not a few resampled copies
        @test length(unique(cloud.particles[:, 4])) > 100
    end

    @testset "Serial and 2 workers are bitwise identical with blocks" begin
        added = length(workers()) < 2 ?
            Distributed.addprocs(2 - length(workers()); exeflags = "--project=$(Base.active_project())") : Int[]
        try
            # Workers that existed at the top already have the likelihoods; load them on new ones
            # (remotecall_eval, since @everywhere with `using` is not allowed inside this closure)
            if !isempty(added)
                Distributed.remotecall_eval(Main, added, :(using SMC, ModelConstructors, Distributions, LinearAlgebra))
                worker_file = joinpath(@__DIR__, "param_blocks_coverage_workers.jl")
                Distributed.remotecall_eval(Main, added, :(include($worker_file)))
            end
            kw = (n_parts = 200, n_Φ = 12, n_mh_steps = 2, n_blocks = 2,
                  param_blocks = [[:x1]], blockwise_c = true)
            serial   = run("serial",   blocks_gauss_likelihood, blocks_gauss_parameters(); parallel = false, kw...)
            parallel = run("parallel", blocks_gauss_likelihood, blocks_gauss_parameters(); parallel = true,  kw...)
            @test serial["cloud"].particles == parallel["cloud"].particles
            @test serial["w"] == parallel["w"] && serial["W"] == parallel["W"]
            @test serial["stage_health"]["c_blocks"] == parallel["stage_health"]["c_blocks"]
            @test serial["stage_health"]["accept_blocks"] == parallel["stage_health"]["accept_blocks"]
        finally
            isempty(added) || Distributed.rmprocs(added)
        end
    end

    @testset "Blocked sampler recovers a known Gaussian posterior" begin
        μ_true, Σ_true = gauss_posterior(GAUSS_μ)
        sd = sqrt.(diag(Σ_true))
        for (name, kw) in (("gauss_rest",    NamedTuple()),
                           ("gauss_blocks",  (param_blocks = [[:x1]], blockwise_c = false)),
                           ("gauss_blockwc", (param_blocks = [[:x1]], blockwise_c = true)))
            out = run(name, blocks_gauss_likelihood, blocks_gauss_parameters();
                      n_parts = 2000, n_Φ = 40, n_mh_steps = 3, kw...)
            μ, Σ = cloud_moments(out["cloud"], 3)
            @test all(abs.(μ .- μ_true) .< 0.15 .* sd)                 # means within 0.15 posterior sd
            @test all(abs.(sqrt.(diag(Σ)) ./ sd .- 1) .< 0.15)          # sds within 15%
            ρ, ρ_true = Σ[1, 2] / sqrt(Σ[1, 1] * Σ[2, 2]), Σ_true[1, 2] / (sd[1] * sd[2])
            @test abs(ρ - ρ_true) < 0.05                                # the x1-x2 correlation survives
        end
    end

    @testset "Bridge from an old cloud with blocks" begin
        # Old cloud: SMC on the old likelihood. Bridge (prior weight 0) to the new likelihood:
        # the target ends at prior × new likelihood, the same posterior as the test above.
        old = run("old", blocks_gauss_old_likelihood, blocks_gauss_parameters();
                  n_parts = 2000, n_Φ = 40, n_mh_steps = 3)
        μ_old_true, _ = gauss_posterior(GAUSS_μ0)
        μ_old, _ = cloud_moments(old["cloud"], 3)
        @test all(abs.(μ_old .- μ_old_true) .< 0.1)
        μ_true, Σ_true = gauss_posterior(GAUSS_μ)
        sd = sqrt.(diag(Σ_true))
        out = run("bridge", blocks_gauss_likelihood, blocks_gauss_parameters();
                  old_data = zeros(1, 1), old_cloud = deepcopy(old["cloud"]),
                  old_loglikelihood = blocks_gauss_old_likelihood, tempered_update_prior_weight = 0.0,
                  n_parts = 2000, n_Φ = 40, n_mh_steps = 3, use_fixed_schedule = false,
                  tempering_target = 0.95, param_blocks = [[:x1]], blockwise_c = true)
        h = out["stage_health"]
        @test h["block_labels"] == ["x1", "rest"]
        @test all(0 .<= h["accept_blocks"][:, 2:end] .<= 1)
        @test out["cloud"].tempering_schedule[end] == 1.0
        μ, Σ = cloud_moments(out["cloud"], 3)
        @test all(abs.(μ .- μ_true) .< 0.15 .* sd)
        @test all(abs.(sqrt.(diag(Σ)) ./ sd .- 1) .< 0.15)
    end
end
