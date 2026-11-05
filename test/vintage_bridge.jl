using Test, SMC, ModelConstructors, Distributions, LinearAlgebra, Random, JLD2, FileIO, Statistics

# Vintage bridge (same model, same priors, same sample length; old_data = revised vintage,
# data = real-time vintage) on a conjugate linear-Gaussian model, where the posterior and the
# MDD are known exactly:
#   y = X θ + ε, ε ~ N(0, σ² I), θ ~ N(0, τ² I), X fixed.
# The old cloud is either a set of exact draws from p(θ | Ỹ) (no old-cloud MC error) or an
# SMC run on Ỹ. Checks:
#   1. identity bridge (Ỹ == Y as a separate matrix): incremental weights constant, one stage,
#      old_loglh == loglh exactly, level MDD == log_prob_old_data;
#   2. exactness: bridged posterior moments and MDD (μ = 0 with the true old MDD; μ = 0.5;
#      blocks + blockwise c) against the closed form, over 3 seeds;
#   3. μ = 0: an offset δ in log_prob_old_data shifts the MDD by exactly δ and leaves the
#      posterior unchanged;
#   4. the old likelihood only ever receives old_data, the new one only data (recorded by
#      object identity), with blocks + blockwise c, and with old_loglikelihood === loglikelihood;
#      final old_loglh / loglh columns equal fresh evaluations on Ỹ / Y;
#   5. resume mid-bridge from an intermediate save is bitwise equal to the uninterrupted run;
#   6. equal old/new vintage strings and equal sizes trigger no shortcut;
#   7. two known weaknesses, as @test_broken: the stale ESS of the old cloud at bridge start,
#      and the 0 * -Inf = NaN rejection at ϕ_n = 1 in the MH step.

const VB_T = 60
const VB_K = 3
const VB_σ = 1.0
const VB_τ = 1.0
const VB_X = let r = Xoshiro(2026); randn(r, VB_T, VB_K) end
const VB_θ0 = [0.5, -0.3, 0.8]
const VB_Y_RT = let r = Xoshiro(11); permutedims(VB_X * VB_θ0 .+ VB_σ .* randn(r, VB_T)) end  # 1 × T
const VB_Y_REV = let r = Xoshiro(12)                     # same length, last 30 periods revised
    y = copy(VB_Y_RT); y[1, 31:end] .+= 1.5 .* randn(r, 30); y
end

vb_ll(θ::AbstractVector, y::AbstractMatrix) =
    -0.5 * VB_T * log(2π * VB_σ^2) - 0.5 * sum(abs2, vec(y) .- VB_X * θ) / VB_σ^2
vb_vals(p::ParameterVector) = [x.value for x in p]

function vb_posterior(y)
    P = VB_X' * VB_X / VB_σ^2 + I / VB_τ^2
    Σ = Matrix(inv(Symmetric(P)))
    return Σ * (VB_X' * vec(y)) / VB_σ^2, (Σ + Σ') / 2
end
vb_logmdd(y) = logpdf(MvNormal(zeros(VB_T), Symmetric(VB_σ^2 * I + VB_τ^2 * VB_X * VB_X')), vec(y))

function vb_params()
    m = GenericModel()
    for k in 1:VB_K
        m <= parameter(Symbol("θ$k"), 0., (-50., 50.), (-50., 50.), Untransformed(), Normal(0, VB_τ),
                       fixed = false)
    end
    return m.parameters
end

# Call recorder: counts calls by which matrix object was passed
mutable struct VBRec
    new_rt::Int; new_other::Int; old_rev::Int; old_other::Int; any_rt::Int; any_rev::Int; any_other::Int
end
VBRec() = VBRec(0, 0, 0, 0, 0, 0, 0)
const VB_REC = Ref(VBRec())
const VB_DATA = Ref(VB_Y_RT)        # the matrix object passed as `data`
const VB_OLD  = Ref(VB_Y_REV)       # the matrix object passed as `old_data`

function vb_new_lik(p::ParameterVector, data::AbstractMatrix)
    data === VB_DATA[] ? (VB_REC[].new_rt += 1) : (VB_REC[].new_other += 1)
    return vb_ll(vb_vals(p), data)
end
function vb_old_lik(p::ParameterVector, data::AbstractMatrix; new_model_params::Bool = false)
    data === VB_OLD[] ? (VB_REC[].old_rev += 1) : (VB_REC[].old_other += 1)
    return vb_ll(vb_vals(p), data)
end
# one function for both (SMC's default old_loglikelihood === loglikelihood)
function vb_any_lik(p::ParameterVector, data::AbstractMatrix)
    if data === VB_DATA[]
        VB_REC[].any_rt += 1
    elseif data === VB_OLD[]
        VB_REC[].any_rev += 1
    else
        VB_REC[].any_other += 1
    end
    return vb_ll(vb_vals(p), data)
end

# Exact draws from p(θ | y) as an SMC cloud (uniform weights, loglh on y, old_loglh 0)
function vb_exact_cloud(y, N; seed)
    μ, Σ = vb_posterior(y)
    D = rand(Xoshiro(seed), MvNormal(μ, Symmetric(Σ)), N)
    pv = vb_params()
    c = SMC.Cloud(VB_K, N)
    for i in 1:N
        ModelConstructors.update!(pv, D[:, i])
        c.particles[i, 1:VB_K] = D[:, i]
        c.particles[i, VB_K + 1] = vb_ll(D[:, i], y)
        c.particles[i, VB_K + 2] = ModelConstructors.prior(pv)
        c.particles[i, VB_K + 3] = 0.0
        c.particles[i, VB_K + 4] = 0.25
        c.particles[i, VB_K + 5] = 1.0
    end
    c.ESS = [Float64(N)]
    c.tempering_schedule = [0.0, 1.0]
    c.c = 0.5
    return c
end

function vb_moments(cloud)
    W = SMC.get_weights(cloud); W = W ./ sum(W)
    X = cloud.particles[:, 1:VB_K]
    μ = vec(sum(W .* X, dims = 1)); D = X .- μ'
    return μ, D' * (W .* D)
end

const VB_N = 2000
vb_common(; kw...) = (verbose = :none, n_parts = VB_N, n_mh_steps = 2, c = 0.5, α = 0.9,
                      use_fixed_schedule = false, tempering_target = 0.95, n_Φ = 100, λ = 2.0,
                      data_vintage = "191231", kw...)

mktempdir() do dir
    function vb_run(name; lik = vb_new_lik, data = VB_DATA[], kw...)
        path = joinpath(dir, name * ".jld2")
        SMC.smc(lik, vb_params(), data; savepath = path,
                particle_store_path = replace(path, ".jld2" => ".h5"), vb_common(; kw...)...)
        return load(path)    # SMC.smc returns only the cloud; the file has w, W, health, MDDs
    end

    μ_rt, Σ_rt   = vb_posterior(VB_Y_RT)
    sd_rt        = sqrt.(diag(Σ_rt))
    lmdd_rt      = vb_logmdd(VB_Y_RT)
    lmdd_rev     = vb_logmdd(VB_Y_REV)
    μ_rev, _     = vb_posterior(VB_Y_REV)
    println("vintage bridge toy: posterior shift (rev → rt) in sd units ", round.((μ_rt .- μ_rev) ./ sd_rt, digits = 2),
            "; log MDD rt $(round(lmdd_rt, digits = 4)), rev $(round(lmdd_rev, digits = 4))")

    @testset "1. Identity bridge: old_data == data (separate matrix), exact old posterior" begin
        old_same = copy(VB_Y_RT)
        VB_OLD[] = old_same
        VB_REC[] = VBRec()
        old_cloud = vb_exact_cloud(VB_Y_RT, VB_N; seed = 1)
        out = vb_run("identity"; old_data = old_same, old_cloud = old_cloud,
                     old_loglikelihood = vb_old_lik, tempered_update_prior_weight = 0.0,
                     log_prob_old_data = lmdd_rt, smc_seed = 1)
        cloud = out["cloud"]
        @test cloud.tempering_schedule == [0.0, 1.0]           # one stage
        @test cloud.stage_index == 2
        w = out["w"][:, 2]
        @test maximum(abs.(log.(w) .- lmdd_rt)) < 1e-9         # every incremental weight = exp(Δϕ log p(Ỹ))
        @test out["stage_health"]["ess"][1] ≈ 1.0              # no weight dispersion
        @test abs(out["log_mdd_estimates"]["smc_level"] - lmdd_rt) < 1e-9
        @test SMC.get_old_loglh(cloud) == SMC.get_loglh(cloud) # old term evaluated on identical data
        μ, Σ = vb_moments(cloud)
        @test all(abs.(μ .- μ_rt) .< 0.1 .* sd_rt)
        @test all(abs.(sqrt.(diag(Σ)) ./ sd_rt .- 1) .< 0.1)
        r = VB_REC[]
        @test r.new_other == 0 && r.old_other == 0
        @test r.old_rev == r.new_rt - VB_N                      # one old call per MH proposal
        VB_OLD[] = VB_Y_REV
    end

    @testset "2. Exactness: bridged posterior and MDD vs closed form (3 seeds)" begin
        for seed in 1:3, (lab, kw) in (("mu0",      (tempered_update_prior_weight = 0.0,)),
                                        ("mu05",     (tempered_update_prior_weight = 0.5,)),
                                        ("mu0_blk",  (tempered_update_prior_weight = 0.0,
                                                      param_blocks = [[:θ1]], blockwise_c = true)),
                                        ("mu05_blk", (tempered_update_prior_weight = 0.5,
                                                      param_blocks = [[:θ1]], blockwise_c = true)))
            VB_REC[] = VBRec()
            out = vb_run("exact_$(lab)_$seed"; old_data = VB_Y_REV,
                         old_cloud = vb_exact_cloud(VB_Y_REV, VB_N; seed = 100 + seed),
                         old_loglikelihood = vb_old_lik, log_prob_old_data = lmdd_rev,
                         smc_seed = seed, kw...)
            cloud = out["cloud"]
            μ, Σ = vb_moments(cloud)
            lm = out["log_mdd_estimates"]["smc_level"]
            println("  $lab seed $seed: stages $(cloud.stage_index - 1), mean err/sd ",
                    round.((μ .- μ_rt) ./ sd_rt, digits = 3), ", sd ratio ",
                    round.(sqrt.(diag(Σ)) ./ sd_rt, digits = 3), ", MDD err ", round(lm - lmdd_rt, digits = 4))
            @test cloud.stage_index > 2                         # a real bridge, not one step
            @test all(abs.(μ .- μ_rt) .< 0.12 .* sd_rt)
            @test all(abs.(sqrt.(diag(Σ)) ./ sd_rt .- 1) .< 0.1)
            @test abs(lm - lmdd_rt) < 0.1
            r = VB_REC[]
            @test r.new_other == 0 && r.old_other == 0 && r.old_rev > 0
        end
        # from-scratch reference on the same data (same tolerances)
        out = vb_run("scratch"; smc_seed = 5)
        μ, Σ = vb_moments(out["cloud"])
        println("  from scratch: mean err/sd ", round.((μ .- μ_rt) ./ sd_rt, digits = 3),
                ", MDD err ", round(out["log_mdd_estimates"]["smc"] - lmdd_rt, digits = 4))
        @test all(abs.(μ .- μ_rt) .< 0.12 .* sd_rt)
        @test abs(out["log_mdd_estimates"]["smc"] - lmdd_rt) < 0.1
    end

    @testset "2b. Old cloud from an SMC run on the revised vintage" begin
        old = vb_run("old_smc"; data = VB_Y_REV, lik = vb_old_lik, smc_seed = 21)   # (data === VB_OLD)
        lm_old = old["log_mdd_estimates"]["smc"]
        println("  old SMC: final ESS entry $(old["cloud"].ESS[end]), weights ESS ",
                sum(SMC.get_weights(old["cloud"]))^2 / sum(abs2, SMC.get_weights(old["cloud"])),
                ", old MDD err ", round(lm_old - lmdd_rev, digits = 4))
        out = vb_run("bridge_from_smc"; old_data = VB_Y_REV, old_cloud = deepcopy(old["cloud"]),
                     old_loglikelihood = vb_old_lik, log_prob_old_data = lm_old, smc_seed = 22,
                     param_blocks = [[:θ1]], blockwise_c = true)
        μ, Σ = vb_moments(out["cloud"])
        @test all(abs.(μ .- μ_rt) .< 0.12 .* sd_rt)
        @test all(abs.(sqrt.(diag(Σ)) ./ sd_rt .- 1) .< 0.1)
        @test abs(out["log_mdd_estimates"]["smc_level"] - lmdd_rt) < 0.15
    end

    @testset "3. μ = 0: an error δ in log_prob_old_data shifts only the MDD, by δ" begin
        δ = 0.7
        oc = vb_exact_cloud(VB_Y_REV, VB_N; seed = 7)
        a = vb_run("off_a"; old_data = VB_Y_REV, old_cloud = deepcopy(oc), old_loglikelihood = vb_old_lik,
                   log_prob_old_data = lmdd_rev, smc_seed = 3)
        b = vb_run("off_b"; old_data = VB_Y_REV, old_cloud = deepcopy(oc), old_loglikelihood = vb_old_lik,
                   log_prob_old_data = lmdd_rev + δ, smc_seed = 3)
        @test a["cloud"].tempering_schedule ≈ b["cloud"].tempering_schedule atol = 1e-10
        @test maximum(abs.(a["cloud"].particles[:, 1:VB_K] .- b["cloud"].particles[:, 1:VB_K])) < 1e-8
        @test b["log_mdd_estimates"]["smc_level"] - a["log_mdd_estimates"]["smc_level"] ≈ δ atol = 1e-8
        # without log_prob_old_data there is no level for μ = 0
        c = vb_run("off_c"; old_data = VB_Y_REV, old_cloud = deepcopy(oc), old_loglikelihood = vb_old_lik,
                   smc_seed = 3)
        @test isnan(c["log_mdd_estimates"]["smc_level"])
        @test c["log_mdd_estimates"]["smc"] ≈ a["log_mdd_estimates"]["smc"] - lmdd_rev atol = 1e-8
    end

    @testset "4. Which data each likelihood gets; stored columns match fresh evaluations" begin
        for (lab, kw) in (("plain", NamedTuple()), ("blocks_bwc", (param_blocks = [[:θ1]], blockwise_c = true)),
                          ("blocks_bwc_mu05", (param_blocks = [[:θ2], [:θ1, :θ3]], blockwise_c = true,
                                               tempered_update_prior_weight = 0.5, n_parts = 1500)))
            VB_REC[] = VBRec()
            out = vb_run("data_$lab"; old_data = VB_Y_REV, old_cloud = vb_exact_cloud(VB_Y_REV, VB_N; seed = 9),
                         old_loglikelihood = vb_old_lik, log_prob_old_data = lmdd_rev, smc_seed = 4, kw...)
            r = VB_REC[]
            @test r.new_other == 0
            @test r.old_other == 0
            @test r.old_rev > 0
            cloud = out["cloud"]
            P = cloud.particles[:, 1:VB_K]
            @test maximum(abs.(SMC.get_old_loglh(cloud) .- [vb_ll(P[i, :], VB_Y_REV) for i in axes(P, 1)])) < 1e-9
            @test maximum(abs.(SMC.get_loglh(cloud) .- [vb_ll(P[i, :], VB_Y_RT) for i in axes(P, 1)])) < 1e-9
            @test SMC.get_old_loglh(cloud) != SMC.get_loglh(cloud)
        end
        # default old_loglikelihood (=== loglikelihood): the one function sees both matrices
        VB_REC[] = VBRec()
        out = vb_run("data_default"; lik = vb_any_lik, old_data = VB_Y_REV,
                     old_cloud = vb_exact_cloud(VB_Y_REV, VB_N; seed = 9), log_prob_old_data = lmdd_rev,
                     smc_seed = 4, param_blocks = [[:θ1]], blockwise_c = true)
        r = VB_REC[]
        @test r.any_other == 0
        @test r.any_rev == r.any_rt - VB_N                     # stage 0: N new-data calls; then pairs
        cloud = out["cloud"]
        P = cloud.particles[:, 1:VB_K]
        @test maximum(abs.(SMC.get_old_loglh(cloud) .- [vb_ll(P[i, :], VB_Y_REV) for i in axes(P, 1)])) < 1e-9
    end

    @testset "5. Resume mid-bridge reproduces the uninterrupted run bitwise" begin
        for (lab, kw) in (("plain", (tempered_update_prior_weight = 0.0,)),
                          ("blocks_mu05", (tempered_update_prior_weight = 0.5, param_blocks = [[:θ1]],
                                           blockwise_c = true)))
            oc = vb_exact_cloud(VB_Y_REV, VB_N; seed = 13)
            full = vb_run("resume_$lab"; old_data = VB_Y_REV, old_cloud = deepcopy(oc),
                          old_loglikelihood = vb_old_lik, log_prob_old_data = lmdd_rev, smc_seed = 8,
                          save_intermediate = true, intermediate_stage_increment = 1, kw...)
            n = full["cloud"].stage_index
            k = max(2, n ÷ 2)
            @test n >= 3
            path = joinpath(dir, "resume_$lab.jld2")
            ckpt = replace(path, ".jld2" => "_stage=$k.jld2")
            VB_REC[] = VBRec()
            SMC.smc(vb_new_lik, vb_params(), VB_Y_RT; savepath = joinpath(dir, "resumed_$lab.jld2"),
                    particle_store_path = joinpath(dir, "resumed_$lab.h5"),
                    vb_common(; old_data = VB_Y_REV, old_cloud = deepcopy(oc),
                              old_loglikelihood = vb_old_lik, log_prob_old_data = lmdd_rev,
                              continue_intermediate = true, intermediate_stage_start = k,
                              loadpath = ckpt, kw...)...)
            res = load(joinpath(dir, "resumed_$lab.jld2"))
            println("  resume $lab: $n stages, resumed from $k")
            @test res["cloud"].particles == full["cloud"].particles
            @test res["w"] == full["w"]
            @test res["W"] == full["W"]
            @test res["cloud"].tempering_schedule == full["cloud"].tempering_schedule
            @test res["log_mdd_estimates"]["smc"] == full["log_mdd_estimates"]["smc"]
            @test VB_REC[].old_other == 0 && VB_REC[].old_rev > 0   # resumed stages still use old_data
        end
    end

    @testset "6. Same size and same vintage string: no shortcut" begin
        oc = vb_exact_cloud(VB_Y_REV, VB_N; seed = 17)
        a = vb_run("vint_a"; old_data = VB_Y_REV, old_cloud = deepcopy(oc), old_loglikelihood = vb_old_lik,
                   log_prob_old_data = lmdd_rev, smc_seed = 6, old_vintage = "191231")  # == data_vintage
        b = vb_run("vint_b"; old_data = VB_Y_REV, old_cloud = deepcopy(oc), old_loglikelihood = vb_old_lik,
                   log_prob_old_data = lmdd_rev, smc_seed = 6, old_vintage = "000000")
        @test size(VB_Y_REV) == size(VB_Y_RT)
        @test a["cloud"].particles == b["cloud"].particles
        @test a["cloud"].stage_index > 2
    end

    @testset "7. Known weaknesses (7a fixed by B2; 7b documented as broken)" begin
        # 7a. B2. A μ = 0 bridge used to keep the old cloud's last ESS entry, which SMC records BEFORE
        # resampling. If the old run resampled at its last stage, ESS[end] < N/2 while the weights
        # are uniform, so the first bridge stage targets 0.95 * ESS[end] instead of 0.95 * N.
        oc = vb_exact_cloud(VB_Y_REV, VB_N; seed = 19)
        oc.ESS = [0.45 * VB_N]                     # as left by a final-stage resample
        out = vb_run("stale_ess"; old_data = VB_Y_REV, old_cloud = oc, old_loglikelihood = vb_old_lik,
                     log_prob_old_data = lmdd_rev, smc_seed = 9)
        ess1 = out["stage_health"]["ess"][1]
        println("  stale ESS: first-stage ESS/N = $(round(ess1, digits = 3)) (tempering target 0.95)")
        @test ess1 > 0.9                           # B2: was 0.428 (0.95 * 0.45)

        # 7b. At ϕ_n = 1 the target has no old-data term, but a proposal whose old likelihood is
        # -Inf gives (1 - ϕ_n) * (-Inf - finite) = NaN, so it is always rejected.
        pv = vb_params()
        oc = vb_exact_cloud(VB_Y_RT, 1; seed = 23)
        μ_, Σ_ = vb_posterior(VB_Y_RT)
        acc(oldlik) = sum(1:200) do s
            p = copy(oc.particles[1, :]); p[VB_K + 3] = vb_ll(p[1:VB_K], VB_Y_REV)
            p0 = copy(p)                                   # mutation updates p in place
            Random.seed!(s)
            q = SMC.mutation(vb_new_lik, pv, VB_Y_RT, p, μ_, Σ_, VB_K, [collect(1:VB_K)], [collect(1:VB_K)], 1.0;
                             c = 0.5, α = 0.9, old_data = VB_Y_REV, old_loglikelihood = oldlik,
                             tempered_update_prior_weight = 0.0, log_prob_old_data = lmdd_rev)
            q[1:VB_K] != p0[1:VB_K]
        end
        a_fin = acc((p, d; kw...) -> vb_ll(vb_vals(p), d))
        a_inf = acc((p, d; kw...) -> -Inf)
        println("  ϕ = 1 MH: accepted $a_fin / 200 with a finite old likelihood, $a_inf / 200 with -Inf")
        @test a_fin > 0
        @test_broken a_inf == a_fin
    end

    @testset "8. μ = 0 with a DSGE-sized log_prob_old_data (log likelihoods near -5600)" begin
        # Same toy, every log likelihood shifted by K (posterior unchanged, MDD shifted by K).
        # The incremental weight is exp(Δϕ (loglh - old_loglh + log_prob_old_data)): with
        # log_prob_old_data ≈ -5600 it underflows to 0 for every particle once Δϕ > ~0.13.
        K = -5600.0
        lik_K(p, d) = vb_new_lik(p, d) + K
        old_K(p, d; kw...) = vb_old_lik(p, d) + K
        function cloud_K(y, seed)
            c = vb_exact_cloud(y, VB_N; seed = seed)
            c.particles[:, VB_K + 1] .+= K
            return c
        end
        # A full smc run here does not terminate (job 343006: the identity bridge with
        # log_prob_old_data = log p(Y) + K spun in get_least_upper_bound for > 10 min, printing
        # "new weights sum: 0.0 / norm weights sum: NaN" ~4 million times), so check the pieces.
        oc = cloud_K(VB_Y_RT, 31)
        ll = SMC.get_loglh(oc)                       # identity: new loglh = old loglh
        w1 = SMC.bridge_incremental_weights(ll, ll, 1.0, 0.0, 0.0, lmdd_rt + K)
        println("  identity, Δϕ = 1: max incremental weight $(maximum(w1)) (should be exp(log p(Ỹ)) > 0 up to scale)")
        @test_broken any(>(0), w1)                   # all underflow to 0
        ess(ϕ) = SMC.compute_ESS(ll, ones(VB_N), ϕ, 0.0; old_loglh = ll, log_prob_old_data = lmdd_rt + K)
        println("  identity: ESS at ϕ = 0.1 → $(ess(0.1)), ϕ = 0.2 → $(ess(0.2)), ϕ = 1 → $(ess(1.0))")
        @test ess(0.1) ≈ VB_N
        @test_broken !isnan(ess(1.0))                # NaN: the adaptive solver cannot reach ϕ = 1
        # the same weights with the constant Δϕ·log p(Ỹ) left out are fine
        w0 = SMC.bridge_incremental_weights(ll, ll, 1.0, 0.0, 0.0, 0.0)
        @test all(==(1.0), w0)

        # Workaround: no log_prob_old_data (μ = 0 needs none); level = smc + log p(Ỹ)
        o = vb_run("big_rev_nolpod"; lik = lik_K, old_data = VB_Y_REV, old_cloud = cloud_K(VB_Y_REV, 32),
                   old_loglikelihood = old_K, smc_seed = 32)
        μ, Σ = vb_moments(o["cloud"])
        lvl = o["log_mdd_estimates"]["smc"] + lmdd_rev + K
        println("  workaround: stages $(o["cloud"].stage_index - 1), mean err/sd ", round.((μ .- μ_rt) ./ sd_rt, digits = 3),
                ", MDD err ", round(lvl - (lmdd_rt + K), digits = 4))
        @test all(abs.(μ .- μ_rt) .< 0.12 .* sd_rt)
        @test abs(lvl - (lmdd_rt + K)) < 0.1
    end
end
