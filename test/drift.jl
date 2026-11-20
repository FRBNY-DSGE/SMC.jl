using Test, Distributed, SMC, ModelConstructors, Distributions, JLD2, Random, LinearAlgebra, Statistics
@everywhere using SMC, ModelConstructors, Distributions, LinearAlgebra
@everywhere include(joinpath($(@__DIR__), "drift_workers.jl"))
include(joinpath(@__DIR__, "drift_fixture_configs.jl"))

# Mutation drift diagnostic (src/drift.jl) and the opt-in fixes: the ridge move, ridge on
# trigger and adaptive MH steps.
#   1. mutation_shift: its paired, cluster-robust SE is right (z ≈ N(0, 1) under invariance)
#   2. the ridge move's acceptance ratio satisfies detailed balance exactly
#   3. the ridge kernel and the block/ridge mixture leave a Gaussian target invariant
#      (an i.i.d. cloud stays π-distributed: KS and moments; a long chain: moments)
#   4. a well-mixed SMC run gives z ≈ N(0, 1); a crippled kernel on a ridge target gives a
#      large z, with the leading direction along the true ridge
#   5. every option off reproduces SMC main (95f0350) bitwise
#   6. with the fixes on: serial = 2 workers bitwise, and a resumed run = the uninterrupted one

# One-sample Kolmogorov-Smirnov p-value against a continuous distribution
function drift_ks_pvalue(x::AbstractVector, d::UnivariateDistribution)
    xs = sort(x); n = length(xs)
    F = cdf.(d, xs)
    D = maximum(max.((1:n) ./ n .- F, F .- (0:n-1) ./ n))
    return ccdf(Kolmogorov(), sqrt(n) * D)
end

# A particle vector (parameters, loglh, logprior, old loglh, accept, weight) at θ
function drift_particle(lik, params, θ)
    ModelConstructors.update!(params, θ)
    return vcat(θ, lik(params, zeros(1, 1)), ModelConstructors.prior(params), 0.0, 0.0, 1.0)
end

@testset "mutation_shift: paired cluster-robust SE" begin
    rng = Xoshiro(1)
    N, ρ = 4000, 0.9
    zs, ratio = Float64[], Float64[]
    for rep in 1:300
        pre  = randn(rng, N, 2)
        post = ρ .* pre .+ sqrt(1 - ρ^2) .* randn(rng, N, 2)     # AR(1): leaves N(0, I) invariant
        sh = mutation_shift(pre, post, ones(N); u = [1.0, 0.0])
        push!(zs, sh.proj / sh.proj_se)
        push!(ratio, sh.se[1] / sqrt(2 * (1 - ρ) / N))
    end
    @test abs(mean(zs)) < 0.2 && 0.85 < std(zs) < 1.15
    @test all(0.9 .< ratio .< 1.1)                                  # SE ≈ √(2(1 - ρ)/ESS)
    # Copies of one resampled particle form one cluster: with an independence kernel (post
    # drawn afresh from the target), naive per-particle SEs would be too small
    zc = Float64[]
    for rep in 1:300
        parents = randn(rng, N ÷ 4, 1)
        pre  = repeat(parents, inner = (4, 1))
        post = randn(rng, N, 1)
        sh = mutation_shift(pre, post, ones(N); u = [1.0])
        rep == 1 && @test sh.n_clusters == N ÷ 4
        push!(zc, sh.proj / sh.proj_se)
    end
    @test abs(mean(zc)) < 0.2 && 0.85 < std(zc) < 1.15
    # A systematic shift is detected; weights enter
    pre = randn(rng, N, 2); post = pre .+ [0.05 0.0]
    sh = mutation_shift(pre, post .+ 0.01 .* randn(rng, N, 2), ones(N); u = [1.0, 0.0])
    @test sh.proj / sh.proj_se > 20
    sh_w = mutation_shift(pre, post, vcat(ones(N - 1), 0.0); sd = ones(2))
    @test sh_w.delta ≈ [0.05, 0.0]
    # The log likelihood shift
    sh_l = mutation_shift(pre, post, ones(N); loglh_pre = zeros(N), loglh_post = fill(0.5, N))
    @test sh_l.dloglh ≈ 0.5
end

@testset "Ridge move: detailed balance of the acceptance ratio" begin
    rng = Xoshiro(2)
    S = Float64
    # log π_ϕ(θ) = ϕ loglh + logprior (+ the bridge term with old data); here a generic one
    loglh(θ) = -0.5 * dot(θ .- 1.0, θ .- 1.0) * 3.0
    logprior(θ) = -0.5 * dot(θ, θ) / 4.0
    oldlh(θ) = -0.5 * sum(abs2, θ .+ 0.5)
    e = normalize(randn(rng, 4))
    for ϕ in (0.3, 1.0), μshare in (0.0, 0.4), rep in 1:200
        θ  = randn(rng, 4)
        θ2 = θ .+ 0.7 * randn(rng) .* e                   # a ridge move: θ2 - θ ∥ e
        r12 = SMC.mh_acceptance_ratio(ϕ, loglh(θ2), loglh(θ), logprior(θ2), logprior(θ),
                                      oldlh(θ2), oldlh(θ), zero(S), zero(S), μshare, -1.3)
        r21 = SMC.mh_acceptance_ratio(ϕ, loglh(θ), loglh(θ2), logprior(θ), logprior(θ2),
                                      oldlh(θ), oldlh(θ2), zero(S), zero(S), μshare, -1.3)
        # log π_ϕ for the bridge target (1-ϕ) log bridge(old) + ϕ loglh + logprior
        bf(x) = SMC.bridge_log_factor(oldlh(x), μshare, -1.3)
        logπ(x) = ϕ * loglh(x) + (ϕ == 1.0 ? 0.0 : (1 - ϕ) * bf(x)) + logprior(x)
        # π(θ) q(θ→θ2) α(θ→θ2) = π(θ2) q(θ2→θ) α(θ2→θ), q symmetric
        lhs = logπ(θ)  + log(min(1.0, r12))
        rhs = logπ(θ2) + log(min(1.0, r21))
        @test isapprox(lhs, rhs; atol = 1e-10)
        @test isapprox(r12 * r21, 1.0; rtol = 1e-10)
    end
end

@testset "Ridge kernel and the block/ridge mixture leave π invariant" begin
    params = drift_parameters(3, 2.0)
    μt, Σt, _ = drift_gauss_truth(DRIFT_Q3, DRIFT_MU3, 2.0)
    sdt = sqrt.(diag(Σt))
    F = eigen(Symmetric(Σt))
    e = F.vectors[:, end]                                 # the long axis
    e2 = normalize([1.0, -2.0, 0.5])                      # and a direction that is not an eigenvector
    ridge_of(d, scale) = (direction = d, step = scale * sqrt(dot(d, Σt * d)), prob = 1.0)
    data = zeros(1, 1)
    # i.i.d. cloud from π, one mutation each: the mutated cloud is again i.i.d. from π
    rng = Xoshiro(3)
    N = 20000
    L = cholesky(Symmetric(Σt)).L
    cloud0 = [μt .+ L * randn(rng, 3) for _ in 1:N]
    for (name, blocks, ridge, nmh) in (("ridge only", Vector{Int}[], ridge_of(e, 2.0), 3),
                                        ("ridge only, non-eigen direction", Vector{Int}[], ridge_of(e2, 1.5), 3),
                                        ("mixture (block RW + ridge, p = 0.5)", [[1, 2, 3]],
                                         merge(ridge_of(e2, 1.5), (prob = 0.5,)), 2))
        Random.seed!(4)
        counts = zeros(Int, 2)
        post = map(cloud0) do θ
            p = SMC.mutation(drift_gauss3_loglh, params, data, drift_particle(drift_gauss3_loglh, params, θ),
                             μt, Σt, 3, blocks, blocks, 1.0; c = 0.6, α = 1.0, n_mh_steps = nmh,
                             ridge = ridge, ridge_counts = counts)
            p[1:3]
        end
        X = reduce(hcat, post)'
        @testset "$name" begin
            @test 0.1 < counts[2] / counts[1] < 0.95                 # the ridge moves do move
            for j in 1:3
                @test drift_ks_pvalue(X[:, j], Normal(μt[j], sdt[j])) > 1e-3
                @test abs(mean(X[:, j]) - μt[j]) < 4 * sdt[j] / sqrt(N)
            end
            pe = X * e2
            @test drift_ks_pvalue(pe, Normal(dot(e2, μt), sqrt(dot(e2, Σt * e2)))) > 1e-3
            @test abs(var(pe) / dot(e2, Σt * e2) - 1) < 4 * sqrt(2 / N)
            # the mutation shift of an exactly π-distributed cloud is noise
            sh = mutation_shift(reduce(hcat, cloud0)', X, ones(N); u = normalize(e2 .* sdt))
            @test abs(sh.proj / sh.proj_se) < 4
        end
    end
    # A long chain of the mixture: moments by batch means
    Random.seed!(5)
    p = drift_particle(drift_gauss3_loglh, params, μt)
    T, B = 40_000, 40
    chain = zeros(T, 3)
    for t in 1:T
        p = SMC.mutation(drift_gauss3_loglh, params, data, p, μt, Σt, 3, [[1], [2, 3]], [[1], [2, 3]], 1.0;
                         c = 0.8, α = 1.0, n_mh_steps = 1, ridge = merge(ridge_of(e2, 1.5), (prob = 0.5,)))
        chain[t, :] = p[1:3]
    end
    bm = [mean(chain[(b-1)*(T÷B)+1:b*(T÷B), j]) for b in 1:B, j in 1:3]
    se = vec(std(bm, dims = 1)) ./ sqrt(B)
    @test all(abs.(vec(mean(chain, dims = 1)) .- μt) .< 4.5 .* se)
    @test all(abs.(vec(var(chain, dims = 1)) ./ diag(Σt) .- 1) .< 0.1)
    # Without a ridge, mutation is unchanged (no extra random numbers): same draws as before
    p0 = drift_particle(drift_gauss3_loglh, params, μt)
    Random.seed!(6); a = SMC.mutation(drift_gauss3_loglh, params, data, copy(p0), μt, Σt, 3, [[1, 2, 3]], [[1, 2, 3]], 1.0; c = 0.5, n_mh_steps = 2)
    Random.seed!(6); b = SMC.mutation(drift_gauss3_loglh, params, data, copy(p0), μt, Σt, 3, [[1, 2, 3]], [[1, 2, 3]], 1.0; c = 0.5, n_mh_steps = 2, ridge = nothing)
    @test a == b
end

mktempdir() do dir
    function drift_run(name, lik, params; kwargs...)
        path = joinpath(dir, name * ".jld2")
        SMC.smc(lik, params, zeros(1, 1); verbose = :none, savepath = path,
                particle_store_path = replace(path, ".jld2" => ".h5"), kwargs...)
        return load(path)
    end

    @testset "Diagnostic: z ≈ N(0, 1) when well mixed" begin
        # A 3-dim Gaussian, many MH steps per stage, small tempering steps
        zs = Float64[]
        for seed in 1:3
            out = drift_run("mixed$seed", drift_gauss3_loglh, drift_parameters(3, 2.0); smc_seed = seed,
                            n_parts = 1000, n_mh_steps = 8, n_Φ = 60, λ = 2.0, use_fixed_schedule = true,
                            α = 1.0, c = 0.5)
            h = out["stage_health"]
            append!(zs, Base.filter(isfinite, h["drift_z"][15:end]))     # after the first stages
            @test length(h["drift_z"]) == length(h["stage"]) && size(h["drift_delta"], 2) == length(h["stage"])
        end
        @test abs(mean(zs)) < 0.35
        @test 0.7 < std(zs) < 1.4
        @test drift_ks_pvalue(zs, Normal()) > 1e-3
    end

    @testset "Diagnostic: a cloud lagging along the ridge" begin
        # An i.i.d. cloud from the posterior shifted 1.5 sd along its ridge (the long axis) and
        # mutated at ϕ = 1 by a slow block RW (it closes a few % of the gap per stage): z is
        # large along the ridge, the leading direction aligns with it, and loglh rises.
        params = drift_parameters(DRIFT_D, DRIFT_SDP)
        μt, Σt, _ = drift_gauss_truth(DRIFT_Q, DRIFT_MU, DRIFT_SDP)
        F = eigen(Symmetric(Σt)); w = F.vectors[:, end]; sw = sqrt(F.values[end])
        sdt = sqrt.(diag(Σt))
        v = w ./ sdt; v ./= norm(v)                           # the ridge in sd units
        N = 2000
        rng = Xoshiro(8)
        L = cholesky(Symmetric(Σt)).L
        X = reduce(hcat, [μt .+ 1.5 * sw .* w .+ L * randn(rng, DRIFT_D) for _ in 1:N])'
        P = reduce(hcat, [drift_particle(drift_ridge_loglh, params, X[i, :]) for i in 1:N])'
        ds = SMC.DriftSettings(collect(1:DRIFT_D), DRIFT_D; span = 5, lag = 2)
        st = SMC.new_drift_state(DRIFT_D, ds)
        blocks = [collect(1:DRIFT_D)]
        zs, coss = Float64[], Float64[]
        Random.seed!(9)
        for stage in 1:10
            pre = P[:, 1:DRIFT_D]
            u = SMC.leading_direction(st["g"])
            P = reduce(hcat, [SMC.mutation(drift_ridge_loglh, params, zeros(1, 1), P[i, :], μt, Σt, DRIFT_D,
                                            blocks, blocks, 1.0; c = 0.25, α = 1.0, n_mh_steps = 1) for i in 1:N])'
            post = P[:, 1:DRIFT_D]
            sh = mutation_shift(pre, post, ones(N); u = any(!iszero, u) ? u : nothing,
                                loglh_pre = zeros(N), loglh_post = zeros(N))
            z = sh.proj / sh.proj_se
            SMC.update_drift_state!(st, ds, z, sh.delta, NaN)
            push!(zs, z)
            push!(coss, abs(dot(sh.delta, v)) / norm(sh.delta))
        end
        @test all(isnan, zs[1:3]) && all(zs[4:end] .> 5)
        @test abs(dot(SMC.leading_direction(st["g"]), v)) > 0.9
        @test minimum(coss) > 0.8
        # the log likelihood shift is the direction-free version of the same signal
        pre = P[:, 1:DRIFT_D]; lpre = P[:, end-4]
        P2 = reduce(hcat, [SMC.mutation(drift_ridge_loglh, params, zeros(1, 1), P[i, :], μt, Σt, DRIFT_D,
                                         blocks, blocks, 1.0; c = 0.25, α = 1.0, n_mh_steps = 1) for i in 1:N])'
        sh = mutation_shift(pre, P2[:, 1:DRIFT_D], ones(N); loglh_pre = lpre, loglh_post = P2[:, end-4])
        @test sh.dloglh / sh.dloglh_se > 3
        # The alarm: fire after drift_trigger_stages (3) values of z > 3, calm after 3 with |z| < 2
        alarm(z) = SMC.update_drift_state!(st, ds, z, sh.delta, NaN)
        @test alarm(0.5) == 0 && alarm(10.0) == 0 && alarm(10.0) == 0
        @test alarm(10.0) == 1 && alarm(0.5) == 0 && alarm(0.1) == 0 && alarm(-0.3) == -1
    end

    @testset "Adaptive MH without a ridge move" begin
        ds = SMC.DriftSettings(collect(1:3), 3; trigger_z = 3.0, trigger_stages = 2,
                               calm_z = 2.0, adaptive_mh = true, n_mh_base = 1,
                               n_mh_max = 3)
        state = SMC.new_drift_state(3, ds)
        step(z) = SMC.update_drift_state!(state, ds, z, zeros(3), NaN)
        @test state["n_mh"] == 1 && !state["ridge_on"]
        @test step(4.0) == 0 && state["n_mh"] == 1
        @test step(4.0) == 1 && state["n_mh"] == 2
        @test step(4.0) == 1 && state["n_mh"] == 3
        @test step(4.0) == 1 && state["n_mh"] == 3
        @test step(0.0) == 0 && state["n_mh"] == 3
        @test step(0.0) == -1 && state["n_mh"] == 2
        @test step(0.0) == -1 && state["n_mh"] == 1
        @test !state["ridge_on"]

        out = drift_run("adaptive_only", drift_ridge_loglh,
                        drift_parameters(DRIFT_D, DRIFT_SDP);
                        smc_seed = 5, n_parts = 300, n_mh_steps = 1, n_mh_max = 3,
                        adaptive_mh = true, use_fixed_schedule = false,
                        tempering_target = 0.95, c = 0.1, α = 0.9,
                        drift_trigger_z = 0.0, drift_trigger_stages = 1,
                        drift_calm_z = 0.0)
        h = out["stage_health"]
        @test any(h["drift_alarm"] .== 1)
        @test maximum(h["n_mh"]) > 1 && maximum(h["n_mh"]) <= 3
        @test all(h["ridge_on"] .== 0) && all(h["ridge_tried"] .== 0)
        @test all(isnan, h["ridge_accept"])
        @test all(h["n_mh"][s + 1] == clamp(h["n_mh"][s] + h["drift_alarm"][s], 1, 3)
                  for s in 1:length(h["n_mh"])-1)
    end

    @testset "Every option off matches main and is bitwise across option spellings" begin
        ref = load(joinpath(@__DIR__, "reference", "drift_off_main.jld2"))
        @test ref["smc_commit"] != "unknown"
        # CI can use a different Julia patch release from the fixture's. BLAS and
        # floating-point reductions can differ at the last few bits across releases.
        # Compare the fixture elementwise with a tight tolerance; keep the two runs made
        # in this process exactly equal to verify that disabled options change nothing.
        fixture_close(a::Number, b::Number) =
            isequal(a, b) || isapprox(a, b; atol = 1e-10, rtol = 1e-10)
        fixture_close(a::AbstractArray, b::AbstractArray) =
            size(a) == size(b) && all(fixture_close(x, y) for (x, y) in zip(a, b))
        for (name, target, kw) in DRIFT_OFF_CONFIGS
            got = drift_off_run(dir, name, target; kw...)
            explicit = drift_off_run(dir, name * "_explicit", target; kw..., ridge_mode = :off,
                                     adaptive_mh = false, n_mh_max = kw.n_mh_steps, ridge_prob = 0.3,
                                     ridge_direction = :pc, drift_params = [:x1, :x2], drift_ewma_span = 4)
            for f in keys(ref["runs"][name])
                @test fixture_close(got[f], ref["runs"][name][f])
                @test isequal(got[f], explicit[f])
            end
        end
    end

    @testset "Fixes on: the ridge and adaptive MH steps work, serial = 2 workers, resume" begin
        kw = (smc_seed = 5, n_parts = 300, n_mh_steps = 1, use_fixed_schedule = false, tempering_target = 0.95,
              c = 0.1, α = 0.9, param_blocks = [[:x1], [:x2, :x3]], blockwise_c = true,
              ridge_mode = :trigger, adaptive_mh = true, n_mh_max = 3, drift_calm_z = 0.0,
              # trigger_z 0 and 1 stage: the alarm fires at the first positive z (machinery test)
              drift_trigger_z = 0.0, drift_trigger_stages = 1)
        base = drift_run("fix_serial", drift_ridge_loglh, drift_parameters(DRIFT_D, DRIFT_SDP); kw...,
                         save_intermediate = true, intermediate_stage_increment = 10)
        h = base["stage_health"]
        @test any(h["ridge_on"] .> 0) && any(h["n_mh"] .> 1)
        on = findall(>(0), h["ridge_on"])
        @test all(0 .<= h["ridge_accept"][on] .<= 1) && all(h["ridge_tried"][on] .> 0)
        @test all(isnan, h["ridge_accept"][setdiff(1:end, on)])
        @test base["cloud"].tempering_schedule[end] == 1.0
        @test maximum(h["n_mh"]) <= 3
        # resume from a checkpoint after the fixes have switched on
        ck = Base.filter(s -> s >= on[1] + 1, 10:10:length(h["stage"]) - 1)
        @test !isempty(ck)
        resumed = drift_run("fix_serial_resumed", drift_ridge_loglh, drift_parameters(DRIFT_D, DRIFT_SDP); kw...,
                            continue_intermediate = true, intermediate_stage_start = ck[1],
                            loadpath = joinpath(dir, "fix_serial_stage=$(ck[1]).jld2"))
        @test resumed["cloud"].particles == base["cloud"].particles
        @test resumed["w"] == base["w"] && resumed["W"] == base["W"]
        @test resumed["stage_health"]["n_mh"] == h["n_mh"]
        @test resumed["stage_health"]["ridge_c"] == h["ridge_c"]
        added = length(workers()) < 2 ?
            Distributed.addprocs(2 - length(workers()); exeflags = "--project=$(Base.active_project())") : Int[]
        try
            if !isempty(added)
                Distributed.remotecall_eval(Main, added, :(using SMC, ModelConstructors, Distributions, LinearAlgebra))
                Distributed.remotecall_eval(Main, added, :(include($(joinpath(@__DIR__, "drift_workers.jl")))))
            end
            par = drift_run("fix_parallel", drift_ridge_loglh, drift_parameters(DRIFT_D, DRIFT_SDP); kw..., parallel = true)
            @test par["cloud"].particles == base["cloud"].particles
            @test par["w"] == base["w"] && par["W"] == base["W"]
            @test par["stage_health"]["ridge_tried"] == h["ridge_tried"]
        finally
            isempty(added) || Distributed.rmprocs(added)
        end
        # The drift columns go to the sidecar CSV; the stage health CSV keeps its layout
        csv = readlines(joinpath(dir, "fix_serial_drift.csv"))
        @test startswith(csv[1], "stage,phi,drift_rms") && occursin("ridge_accept", csv[1]) && endswith(csv[1], "dir_top")
        @test length(csv) == length(h["stage"]) + 1
        @test !occursin("drift", readlines(joinpath(dir, "fix_serial_stage_health.csv"))[1])
        # Bad settings are refused before any likelihood is evaluated
        @test_throws ArgumentError drift_run("bad1", drift_gauss3_loglh, drift_parameters(3, 2.0);
                                             smc_seed = 1, n_parts = 50, n_Φ = 5, ridge_mode = :sometimes)
        @test_throws ArgumentError drift_run("bad2", drift_gauss3_loglh, drift_parameters(3, 2.0);
                                             smc_seed = 1, n_parts = 50, n_Φ = 5, n_mh_steps = 3, n_mh_max = 2)
        @test_throws ArgumentError drift_run("bad3", drift_gauss3_loglh, drift_parameters(3, 2.0);
                                             smc_seed = 1, n_parts = 50, n_Φ = 5, ridge_direction = [1.0, 0.0])
    end
end
