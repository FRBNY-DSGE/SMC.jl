using Test, Random, Distributions, ModelConstructors, SMC, JLD2

function adaptive_bridge_old_likelihood(parameters, data; new_model_params = false)
    return -1.0
end

function adaptive_bridge_new_likelihood(parameters, data)
    θ = parameters[1].value
    return θ > 0.0 ? -Inf : -0.5 * (θ + 0.5)^2
end

function adaptive_bridge_model()
    model = GenericModel()
    model <= parameter(:θ, 0.0, (-10.0, 10.0), (-10.0, 10.0),
                       Untransformed(), Normal(0.0, 1.0), fixed = false)
    return model
end

@testset "Adaptive bridge with impossible new-data particles" begin
    n = 256
    model = adaptive_bridge_model()
    data = zeros(1, 1)
    old_cloud = SMC.Cloud(1, n)
    SMC.initial_draw!(adaptive_bridge_old_likelihood, model.parameters, data,
                      old_cloud; smc_seed = 73)
    old_cloud.ESS[1] = n
    @test count(>(0.0), vec(SMC.get_vals(old_cloud))) > n ÷ 4

    loglh = [-Inf, 0.0, 0.0, 0.0]
    weights = ones(4)
    @test SMC.compute_ESS(loglh, weights, 0.0, 0.0) == 4.0
    @test SMC.compute_ESS(loglh, weights, 0.1, 0.0) == 3.0

    mktempdir() do dir
        function bridge_run(name, count, prior_share; supply_old_mdd = false)
            savepath = joinpath(dir, name * ".jld2")
            kwargs = supply_old_mdd ? (log_prob_old_data = -1.0,) : NamedTuple()
            SMC.smc(adaptive_bridge_new_likelihood, adaptive_bridge_model().parameters, data;
                    old_data = data, old_cloud = deepcopy(old_cloud),
                    old_loglikelihood = adaptive_bridge_old_likelihood,
                    n_parts = count, n_Φ = 12, n_mh_steps = 1,
                    use_fixed_schedule = false, tempering_target = 0.9,
                    threshold_ratio = 0.8, resampling_method = :multinomial,
                    tempered_update_prior_weight = prior_share,
                    smc_seed = 77, verbose = :none, savepath = savepath,
                    particle_store_path = joinpath(dir, name * ".h5"), kwargs...)
            return load(savepath)
        end

        @test_throws ArgumentError bridge_run("missing_old_mdd", n, 0.25)
        same = bridge_run("same", n, 0.0)
        resized = bridge_run("resized", n + 1, 0.0)
        leveled = bridge_run("leveled", n, 0.0; supply_old_mdd = true)
        resized_level = bridge_run("resized_level", n + 1, 0.0; supply_old_mdd = true)
        mixed = bridge_run("mixed", n, 0.25; supply_old_mdd = true)

        function log_mdd(run)
            w, W = run["w"], run["W"]
            count = size(w, 1)
            return sum(log.(vec(sum(w[:, 2:end] .* W[:, 1:end-1] ./ count, dims = 1))))
        end

        # Exact evidence for θ ~ N(0, 1), L(θ) = exp(-0.5(θ+0.5)^2) 1(θ ≤ 0).
        analytic = -0.5 * log(2.0) - 1 / 16 + log(cdf(Normal(), 1 / (2sqrt(2))))
        for run in (same, resized, leveled, resized_level, mixed)
            @test run["cloud"].tempering_schedule[end] == 1.0
            @test all(isfinite, run["w"])
            @test all(isfinite, run["W"])
        end
        @test abs(log_mdd(same) - (analytic + 1.0)) < 0.4
        @test abs(log_mdd(resized) - (analytic + 1.0)) < 0.4
        @test abs(log_mdd(leveled) - analytic) < 0.4
        @test abs(log_mdd(resized_level) - analytic) < 0.4
        @test abs(log_mdd(mixed) - analytic) < 0.4
        @test abs(log_mdd(same) - log_mdd(resized)) < 0.3
        @test abs(log_mdd(leveled) - log_mdd(resized_level)) < 0.3
        @test abs(log_mdd(leveled) - log_mdd(mixed)) < 0.3
    end
end
