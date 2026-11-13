# Tests of the marginal data density estimators (src/mdd_estimators.jl and modified_harmonic_mean).
# Everything lives in a module, so no global leaks into the other test files of the suite.
#
# Tolerances come from 40 replications of each conjugate case (d = 10-15, N = M = 4,000):
# replication sd of the bridge 0.010-0.017, of the real-line MHM (p = 0.5) 0.014-0.028, of the IS
# estimate 0.012-0.020; Frühwirth-Schnatter SEs within 10% of those sds. The slow d = 93 case runs
# only with ENV["SMC_SLOW_TESTS"] = "true".
module TestMDDEstimators

using Test, SMC, ModelConstructors, Distributions, LinearAlgebra, Random, Statistics

include("mdd_conjugate_cases.jl")

const R = 10            # replications per conjugate case
const N = 4_000         # posterior draws (= proposal draws)

# SMC main (95f0350) on mhm_compat_input(), printed by the branch work's compat_ref.jl
const COMPAT_LOG_MDD = -8.1608486924789094
const COMPAT_BY_P = [-8.870007986735521, -8.4038909648019473, -8.1752000880843969, -7.955956249926782,
                     -7.8036516377921537, -7.919973910857256, -8.1070446002441336, -8.1482703666698164,
                     -8.0636424271981717]

@testset "bounds_transform" begin
    lb = [0.0, -2.0, 0.0, -Inf, -Inf, 1e-5, 1.0]
    ub = [1.0, 5.0, Inf, 3.0, Inf, 1e5, 1.0 + 1e-6]
    bt = bounds_transform(lb, ub)
    @test length(bt) == 7
    @test bounds_transform([(a, b) for (a, b) in zip(lb, ub)]).lb == bt.lb
    @test_throws ArgumentError bounds_transform([0.0], [0.0])
    @test_throws ArgumentError bounds_transform([NaN], [1.0])
    @test_throws DimensionMismatch bounds_transform([0.0, 1.0], [1.0])
    rng = Xoshiro(1)
    for _ in 1:200
        u = 0.001 .+ 0.998 .* rand(rng, 7)
        θ = [u[1], -2 + 7u[2], 10u[3], 3 - 10u[4], 20u[5] - 10, 1e-5 + (1e5 - 1e-5) * u[6], 1 + 1e-6 * u[7]]
        y = bounds_to_real(bt, θ)
        @test bounds_to_model(bt, y) ≈ θ rtol = 1e-12
        @test bounds_logjac(bt, θ) ≈ bounds_logjac_real(bt, y) atol = 1e-8
        @test bounds_in_support(bt, θ)
        # |y| ≤ 15 (≤ 5 for the 1e-6-wide interval, where θ's own rounding near a bound sets the precision)
        y2 = vcat(30 .* rand(rng, 6) .- 15, 10 * rand(rng) - 5)
        @test bounds_to_real(bt, bounds_to_model(bt, y2)) ≈ y2 atol = 1e-6
    end
    # The log Jacobian against central finite differences of T⁻¹, coordinate by coordinate
    for y0 in (-4.0, -0.3, 0.0, 0.7, 6.0), j in 1:7
        h = 1e-4
        bj = bounds_transform([lb[j]], [ub[j]])
        dθ = (bounds_to_model(bj, [y0 + h])[1] - bounds_to_model(bj, [y0 - h])[1]) / (2h)
        lj = bounds_logjac_real(bj, [y0])
        @test log(abs(dθ)) ≈ lj atol = 1e-5 * max(1, abs(lj))
        @test bounds_logjac(bj, bounds_to_model(bj, [y0])) ≈ lj atol = 1e-6
    end
    # Matrices: one draw per row
    Θ = [0.5 1.0 2.0 0.0 1.0 1.0 1.0 + 5e-7; 0.1 -1.0 0.5 2.9 -3.0 50.0 1.0 + 1e-8]
    @test bounds_to_model(bt, bounds_to_real(bt, Θ)) ≈ Θ rtol = 1e-12
    @test bounds_logjac(bt, Θ) ≈ [bounds_logjac(bt, Θ[i, :]) for i in 1:2]
    @test bounds_in_support(bt, Θ) == trues(2)
    # Edge cases: at a bound, outside, far out on the real line
    b1 = bounds_transform([0.0, 0.0, -Inf], [1.0, Inf, 2.0])
    @test bounds_to_real(b1, [0.0, 0.0, 2.0]) == [-Inf, -Inf, Inf]
    @test bounds_to_real(b1, [1.0, 1.0, 1.0])[1] == Inf
    @test bounds_logjac(b1, [0.0, 1.0, 1.0]) == -Inf
    @test_throws DomainError bounds_to_real(b1, [1.5, 1.0, 1.0])
    @test_throws DomainError bounds_to_real(b1, [0.5, -1.0, 1.0])
    @test_throws DomainError bounds_to_real(b1, [NaN, 1.0, 1.0])
    far = bounds_to_model(b1, [800.0, -800.0, 800.0])
    @test far == [1.0, 0.0, 2.0]                     # rounded onto the bounds …
    @test !bounds_in_support(b1, far)                # … so no likelihood evaluation
    @test isfinite(bounds_logjac_real(b1, [800.0, -800.0, 800.0]))
    @test bounds_logjac_real(b1, [-40.0, 3.0, 0.0]) ≈ -40.0 + 3.0 rtol = 1e-12   # logit tail: log σ(y) → y
    near = bounds_to_model(b1, [-30.0, 0.0, 0.0])[1]
    @test near > 0 && isapprox(near, exp(-30.0); rtol = 1e-12)  # full relative precision near the lower bound
    @test 1 - bounds_to_model(b1, [30.0, 0.0, 0.0])[1] ≈ exp(-30.0) rtol = 1e-2
end

@testset "parameter_slots and bounds_transform(::ParameterVector)" begin
    m = GenericModel()
    m <= parameter(:a, 1.3, (1.0, 10.0), (1.0, 10.0), Untransformed(), Normal(1.25, 0.12), fixed = false)
    m <= parameter(:s, 0.5, (0.0, 1.0), (0.0, 1.0), Untransformed(), RootInverseGamma(2.0, 0.75), fixed = false)
    m <= parameter(:f, 0.3, (0.0, 1.0), (0.0, 1.0), Untransformed(), Beta(2.0, 5.0), fixed = true)
    m <= parameter(:g, 0.3, (0.0, 1.0), (0.0, 1.0), Untransformed(), Beta(2.0, 5.0), fixed = false)
    s = m[:s]
    ModelConstructors.set_regime_val!(s, 1, 0.5)
    ModelConstructors.set_regime_val!(s, 2, 0.4)
    ModelConstructors.set_regime_prior!(s, 1, RootInverseGamma(2.0, 0.75))
    ModelConstructors.set_regime_prior!(s, 2, RootInverseGamma(2.0, 0.3))
    ModelConstructors.set_regime_valuebounds!(s, 1, (0.0, 1.0))
    ModelConstructors.set_regime_valuebounds!(s, 2, (0.0, 0.5))
    g = m[:g]
    ModelConstructors.set_regime_val!(g, 1, 0.3)
    ModelConstructors.set_regime_val!(g, 2, 0.3)
    ModelConstructors.set_regime_fixed!(g, 1, false)
    ModelConstructors.set_regime_fixed!(g, 2, true)
    slots = parameter_slots(m.parameters; regime_switching = true)
    @test [(x.key, x.regime) for x in slots] == [(:a, 1), (:s, 1), (:f, 1), (:g, 1), (:s, 2), (:g, 2)]
    @test findall(x -> !x.fixed, slots) ==
          ModelConstructors.get_free_para_inds(m.parameters; regime_switching = true)
    @test slots[5].prior isa RootInverseGamma && slots[5].prior.τ == 0.3
    bt = bounds_transform(m.parameters; regime_switching = true)
    @test bt.lb == [1.0, 0.0, 0.0, 0.0] && bt.ub == [10.0, 1.0, 1.0, 0.5]
    bt1 = bounds_transform(m.parameters)
    @test bt1.lb == [1.0, 0.0, 0.0] && bt1.ub == [10.0, 1.0, 1.0]
    @test findall(x -> !x.fixed, parameter_slots(m.parameters)) ==
          ModelConstructors.get_free_para_inds(m.parameters)
end

@testset "effective_prior_lognorm against analytic and Monte Carlo masses" begin
    priors = [Normal(1.25, 0.12), Normal(0.0, 0.4), RootInverseGamma(2.0, 0.1), RootInverseGamma(2.0, 0.75),
              RootInverseGamma(4.0, 0.5), Gamma(2.0, 0.5), Beta(2.0, 5.0), Uniform(0.0, 1.0)]
    lb = [1.0, -0.999, 0.0, 0.0, 1e-8, 0.1, 0.05, -1.0]
    ub = [10.0, 0.999, 10.0, 1.0, 0.6, 3.0, 0.6, 2.0]
    out = effective_prior_lognorm(priors, lb, ub; p_valid = 1.0)
    @test all(==(:analytic), out.methods)
    @test out.lognorm ≈ -sum(out.log_mass)
    # RootInverseGamma(2, τ): θ² ~ InverseGamma(1, τ²), so P(θ < u) = exp(-τ²/u²)
    @test out.log_mass[4] ≈ -0.75^2 rtol = 1e-12
    @test out.log_mass[3] ≈ log(exp(-0.01 / 100)) rtol = 1e-10
    @test out.log_mass[8] == 0.0
    @test out.log_mass[1] ≈ log(1 - cdf(Normal(1.25, 0.12), 1.0)) rtol = 1e-12
    # Monte Carlo, with independent samplers (ModelConstructors' rand for RootInverseGamma)
    n = 400_000
    mc = SMC.preserving_default_rng() do
        Random.seed!(2)
        rng = Xoshiro(3)
        [count(_ -> lb[j] < (priors[j] isa RootInverseGamma ? rand(priors[j]) : rand(rng, priors[j])) < ub[j], 1:n) / n
         for j in eachindex(priors)]
    end
    for j in eachindex(priors)
        p = exp(out.log_mass[j])
        @test abs(p - mc[j]) <= 4 * sqrt(p * (1 - p) / n) + 1e-12
    end
    # P(valid): finite likelihood only for θ₁ < 1.4, given the bounds
    ll(θ) = θ[1] < 1.4 ? 0.0 : -Inf
    est = effective_prior_lognorm(priors[1:2], lb[1:2], ub[1:2]; loglikelihood = ll, n_draws = 20_000, rng = Xoshiro(4))
    F = Normal(1.25, 0.12)
    pv = (cdf(F, 1.4) - cdf(F, 1.0)) / ccdf(F, 1.0)
    @test est.n_draws == 20_000 && est.n_valid == round(Int, est.p_valid * 20_000)
    @test abs(est.p_valid - pv) <= 4 * sqrt(pv * (1 - pv) / 20_000)
    @test est.p_valid_se ≈ sqrt(est.p_valid * (1 - est.p_valid) / 20_000) rtol = 1e-10
    @test est.lognorm ≈ -est.sum_log_mass - log(est.p_valid)
    @test est.lognorm_se ≈ sqrt((1 - est.p_valid) / (est.p_valid * 20_000)) rtol = 1e-10
    tup = effective_prior_lognorm(priors, lb, ub; p_valid = (1505, 1600))
    @test tup.p_valid == 1505 / 1600 && tup.lognorm ≈ out.lognorm - log(1505 / 1600)
    # ParameterVector method (regime-switching slots, smc's loglikelihood signature)
    m = GenericModel()
    m <= parameter(:a, 1.3, (1.0, 10.0), (1.0, 10.0), Untransformed(), Normal(1.25, 0.12), fixed = false)
    m <= parameter(:s, 0.5, (0.0, 1.0), (0.0, 1.0), Untransformed(), RootInverseGamma(2.0, 0.75), fixed = false)
    ModelConstructors.set_regime_val!(m[:s], 1, 0.5)
    ModelConstructors.set_regime_val!(m[:s], 2, 0.4)
    ModelConstructors.set_regime_prior!(m[:s], 1, RootInverseGamma(2.0, 0.75))
    ModelConstructors.set_regime_prior!(m[:s], 2, RootInverseGamma(2.0, 0.3))
    ModelConstructors.set_regime_valuebounds!(m[:s], 1, (0.0, 1.0))
    ModelConstructors.set_regime_valuebounds!(m[:s], 2, (0.0, 0.5))
    pe = effective_prior_lognorm(m.parameters; regime_switching = true,
                                 loglikelihood = (p, data) -> p[1].value < 1.4 ? 0.0 : -Inf, n_draws = 4_000, seed = 5)
    @test pe.log_mass ≈ [log(ccdf(F, 1.0)), -0.75^2, -0.3^2 / 0.5^2] rtol = 1e-10
    @test abs(pe.p_valid - pv) <= 4 * sqrt(pv * (1 - pv) / 4_000)
    state = copy(Random.default_rng())
    effective_prior_lognorm(m.parameters; regime_switching = true, loglikelihood = (p, data) -> 0.0, n_draws = 10)
    @test Random.default_rng() == state           # the caller's RNG is left as it was
    @test (@test_logs (:warn,) effective_prior_lognorm(priors, lb, ub)).p_valid == 1.0
end

@testset "modified_harmonic_mean: default log_mdd unchanged (backward compatibility)" begin
    # The function on main (95f0350), verbatim
    function old_mhm(θ::AbstractMatrix, logpost::AbstractVector, weights::AbstractVector; p = 0.1:0.1:0.9)
        keep = findall(i -> isfinite(logpost[i]) && weights[i] > 0, eachindex(logpost))
        θ, logpost, W = θ[keep, :], logpost[keep], weights[keep] ./ sum(weights[keep])
        d = size(θ, 2)
        μ, Σ = SMC.weighted_mean_cov(θ, W)
        C = cholesky(Symmetric(SMC.floored_cov(Σ)))
        dist2 = vec(sum(abs2, (θ .- μ') / C.U, dims = 2))
        lognorm = -d / 2 * log(2π) - logdet(C) / 2
        by_p = map(p) do q
            inside = dist2 .< quantile(Chisq(d), q)
            any(inside) || return NaN
            terms = log.(W[inside]) .+ (lognorm - log(q)) .- dist2[inside] ./ 2 .- logpost[inside]
            -SMC.logsumexp(terms)
        end
        return (log_mdd = mean(filter(isfinite, by_p)), by_p = collect(by_p))
    end
    θ, lp, w = mhm_compat_input()
    new = modified_harmonic_mean(θ, lp, w)
    old = old_mhm(θ, lp, w)
    @test new.log_mdd === old.log_mdd
    @test new.by_p == old.by_p
    @test modified_harmonic_mean(θ, lp, w; p = [0.25, 0.5, 0.75], se = :none).by_p == old_mhm(θ, lp, w; p = [0.25, 0.5, 0.75]).by_p
    # Values printed by SMC main 95f0350 on this input (scripts/compat_ref.jl of the branch work)
    @test new.log_mdd ≈ COMPAT_LOG_MDD rtol = 1e-12
    @test new.by_p ≈ COMPAT_BY_P rtol = 1e-12
    # New fields
    @test new.p == collect(0.1:0.1:0.9) && length(new.se_by_p) == 9 && all(x -> x > 0, new.se_by_p)
    @test all(x -> 0 < x <= 1, new.max_share) && new.n_inside[end] >= new.n_inside[1]
    @test new.n_clusters == 597           # 600 rows, one -Inf, one zero weight, one duplicate pair
    @test modified_harmonic_mean(θ, lp, w; lognorm = 0.25, se = :none).by_p ≈ old.by_p .+ 0.25
    @test modified_harmonic_mean(θ, lp, w; se = :batch).se_method == :batch
    @test_throws ArgumentError modified_harmonic_mean(θ, lp, w; real_line = true)
    @test_throws ArgumentError modified_harmonic_mean(θ, lp, w; se = :jackknife)
end

# One replication of every estimator on a case
function one_rep(C, r; dup = false)
    rng = Xoshiro(1000r + 7)
    Θ = C.draw(rng, N)
    dup && (Θ = resampled(Xoshiro(3000r), Θ))
    lk = [C.logk(Θ[i, :]) for i in 1:N]; w = ones(N)
    b = bridge_sampling(C.logk, Θ, lk, w; bounds = (C.lb, C.ub), M = N, rng = Xoshiro(r))
    dup && return (b = b,)
    mr = modified_harmonic_mean(Θ, lk, w; real_line = true, bounds = (C.lb, C.ub), p = [0.5], se = :none)
    mm = modified_harmonic_mean(Θ, lk, w; p = [0.9], se = :none)
    ms = modified_harmonic_mean(Θ, lk, w; p = [0.9], se = :none, support_correction = true,
                                bounds = (C.lb, C.ub), rng = Xoshiro(r))
    is = is_lower_bound(C.logk, Θ, lk, w; bounds = (C.lb, C.ub), τ = 0.9, M = 2_000, rng = Xoshiro(50 + r))
    return (b = b, mhm_real = mr.by_p[1], mhm_model = mm.by_p[1], mhm_corr = ms.by_p[1], is = is)
end

for (name, C) in [("Gaussian", gauss_case()), ("skewed gamma near the bound 0", gamma_case()),
                  ("logit-bounded beta-binomial", beta_case())]
    @testset "conjugate $name (d = $(C.d), N = M = $N): analytic log MDD" begin
        reps = [one_rep(C, r) for r in 1:R]
        err(f) = [f(x) - C.truth for x in reps]
        eb = err(x -> x.b.log_mdd)
        @test all(abs.(eb) .< 0.08)                          # 40 replications: |error| ≤ 0.044, sd ≤ 0.017
        @test abs(mean(eb)) < 0.03
        nse = mean(x.b.nse for x in reps)
        @test 0.5 < nse / std(eb) < 2.0                       # Frühwirth-Schnatter SE calibrated
        @test all(x -> x.b.converged && x.b.iterations < 50, reps)
        @test all(x -> x.b.n_prop == N && x.b.n_post == N ÷ 2, reps)
        @test all(x -> isfinite(x.b.nse_batch) && length(x.b.batch_estimates) == 10, reps)
        eis = err(x -> x.b.is_log_mdd)
        @test all(abs.(eis) .< 0.1) && abs(mean(eis)) < 0.04  # IS from the same proposal draws
        @test all(x -> x.b.is_max_share < 0.1, reps)
        emr = err(x -> x.mhm_real)
        @test all(abs.(emr) .< 0.2) && abs(mean(emr)) < 0.08  # real-line MHM (bias ≈ -0.02)
        # IS lower bound: below the truth on average (and in every replication here), its
        # division by the posterior mass of the ellipsoid a point estimate
        @test mean(err(x -> x.is.log_bound)) < 0
        @test all(err(x -> x.is.log_bound) .< 0.05)
        @test abs(mean(err(x -> x.is.log_point))) < 0.05
        @test all(x -> 0.8 < x.is.post_mass < 1.0 && x.is.n_draws == 2_000, reps)
        if isfinite(C.lb[1])
            # model space: f leaks out of the support, the MHM is biased up (+0.4 to +0.7 at p = 0.9);
            # adding log F_τ removes the bias
            @test mean(err(x -> x.mhm_model)) > 0.3
            ec = err(x -> x.mhm_corr)
            @test abs(mean(ec)) < 0.1 && all(abs.(ec) .< 0.25)
        end
    end
    @testset "conjugate $name: SMC-like cloud with duplicates, cluster-robust SE" begin
        reps = [one_rep(C, r; dup = true) for r in 1:R]
        eb = [x.b.log_mdd - C.truth for x in reps]
        @test all(abs.(eb) .< 0.1) && abs(mean(eb)) < 0.04
        @test all(x -> x.b.n_clusters_post < x.b.n_post, reps)    # duplicates were grouped
        @test all(x -> x.b.nse_cluster > x.b.nse_ess, reps)       # duplicates add posterior-side variance
        @test all(x -> x.b.nse == x.b.nse_cluster && x.b.se_method == :cluster && x.b.rho_f1 > 1, reps)
        @test 0.5 < mean(x.b.nse_cluster for x in reps) / std(eb) < 2.0
    end
end

@testset "bridge sampling: two steps, stored draws, options" begin
    C = gamma_case()
    rng = Xoshiro(21)
    Θ = C.draw(rng, 2_000); lk = [C.logk(Θ[i, :]) for i in 1:2_000]; w = 0.5 .+ rand(rng, 2_000)
    one = bridge_sampling(C.logk, Θ, lk, w; bounds = (C.lb, C.ub), M = 2_000, rng = Xoshiro(5), lognorm = 0.1)
    bd = bridge_draws(Θ, lk, w; bounds = (C.lb, C.ub), M = 2_000, rng = Xoshiro(5))
    lkp = evaluate_proposals(C.logk, bd)
    two = bridge_combine(bd, lkp; lognorm = 0.1)
    @test two.log_mdd == one.log_mdd && two.nse == one.nse && two.is_log_mdd == one.is_log_mdd
    @test lkp == [C.logk(proposal_parameters(bd)[j, :]) for j in 1:2_000]
    # Precomputed kernels in any order of evaluation (e.g. an SGE array): same result
    @test bridge_combine(bd, [C.logk(bd.θ_prop[j, :]) for j in 1:2_000]; lognorm = 0.1).log_mdd == one.log_mdd
    # Stored proposal draws reproduce the estimate
    bd2 = bridge_draws(Θ, lk, w; bounds = (C.lb, C.ub), rng = Xoshiro(5), y_prop = bd.y_prop)
    @test bd2.logg_prop == bd.logg_prop && bridge_combine(bd2, lkp; lognorm = 0.1).log_mdd == one.log_mdd
    @test abs(one.log_mdd - 0.1 - C.truth) < 0.15
    # Weighted cloud: ESS < N/2 on the posterior side
    @test one.ess_post < one.n_post
    # Full parameter vectors with fixed slots
    Θf = hcat(fill(7.0, 2_000), Θ)
    lkf(v) = (@assert v[1] == 7.0; C.logk(v[2:end]))
    fb = bridge_sampling(lkf, Θf, lk, w; free_para_inds = 2:C.d + 1, bounds = (C.lb, C.ub), M = 2_000,
                         rng = Xoshiro(5), lognorm = 0.1)
    @test fb.log_mdd == one.log_mdd
    # Model-space proposal: draws outside the bounds get kernel 0 without an evaluation
    ncalls = Ref(0)
    counted(θ) = (ncalls[] += 1; C.logk(θ))
    bm = bridge_draws(Θ, lk, w; bounds = (C.lb, C.ub), real_line = false, M = 2_000, rng = Xoshiro(6))
    @test count(bm.in_support) < 2_000
    lkm = evaluate_proposals(counted, bm)
    @test ncalls[] == count(bm.in_support) && all(isinf, lkm[.!bm.in_support])
    @test isfinite(bridge_combine(bm, lkm).log_mdd)
    # Options
    @test bridge_combine(bd, lkp; se_method = :ess).nse == two.nse_ess
    @test bridge_combine(bd, lkp; se_method = :iid).nse == two.nse_iid
    @test_throws ArgumentError bridge_combine(bd, lkp; se_method = :hac)
    @test_throws DimensionMismatch bridge_combine(bd, lkp[1:10])
    @test_throws ArgumentError bridge_draws(Θ, lk, w; split = :thirds)
    sall = bridge_draws(Θ, lk, w; bounds = (C.lb, C.ub), split = :all, M = 10)
    @test sall.fit == sall.est == collect(1:2_000)
    eve = repeat(1:100, inner = 20)
    se = bridge_draws(Θ, lk, w; bounds = (C.lb, C.ub), clusters = eve, M = 10)
    @test isempty(intersect(Set(eve[se.fit]), Set(eve[se.est])))     # split by cluster
    # Cloud method
    cl = SMC.Cloud(C.d, 2_000)
    cl.particles .= 0.0
    cl.particles[:, 1:C.d] = Θ
    cl.particles[:, SMC.ind_loglh(C.d + 5)] = lk
    cl.particles[:, SMC.ind_logprior(C.d + 5)] .= 0.0
    cl.particles[:, end] = w
    cb = bridge_sampling(C.logk, cl, collect(1:C.d); bounds = (C.lb, C.ub), M = 2_000, rng = Xoshiro(5), lognorm = 0.1)
    @test cb.log_mdd == one.log_mdd
    # A posterior draw on a bound is dropped with a warning
    Θb = copy(Θ); Θb[1, 1] = 0.0
    @test_logs (:warn, r"on a bound") match_mode = :any bridge_draws(Θb, lk, w; bounds = (C.lb, C.ub), M = 10)
end

@testset "IS lower bound pieces" begin
    @test is_lower_bound([0.0, 0.0, -Inf, -Inf]).log_bound ≈ log(0.5)
    r = is_lower_bound([log(2.0), log(4.0)]; post_mass = 0.5, lognorm = 1.0)
    @test r.log_bound ≈ log(3.0) + 1.0 && r.log_point ≈ log(3.0) + 1.0 - log(0.5) && r.max_share ≈ 2 / 3
    @test r.n_zero == 0 && r.n_draws == 2
    C = beta_case()
    Θ = C.draw(Xoshiro(8), 2_000)
    bd = is_bound_draws(Θ, ones(2_000); bounds = (C.lb, C.ub), τ = 0.5, M = 500, rng = Xoshiro(9))
    @test size(bd.y_prop) == (500, C.d) && all(bd.in_support)
    @test abs(bd.post_mass - 0.5) < 0.1
    # the draws are inside the τ-ellipsoid: log f = const - q/2 with 0 ≤ q ≤ χ²_d(τ)
    @test maximum(bd.logf) - minimum(bd.logf) <= quantile(Chisq(C.d), 0.5) / 2 + 1e-8
end

if get(ENV, "SMC_SLOW_TESTS", "false") == "true"
    @testset "slow: d = 93 (63 well identified + 30 skewed near 0), N = M = 15,000" begin
        C = gamma_case(d = 93, n_good = 63, seed = 13)
        for r in 1:3
            Θ = C.draw(Xoshiro(1000r + 7), 15_000); lk = [C.logk(Θ[i, :]) for i in 1:15_000]
            b = bridge_sampling(C.logk, Θ, lk, ones(15_000); bounds = (C.lb, C.ub), M = 15_000, rng = Xoshiro(r))
            @test abs(b.log_mdd - C.truth) < 0.1 && 0.01 < b.nse < 0.04
            @test abs(b.is_log_mdd - C.truth) < 0.15
            m = modified_harmonic_mean(Θ, lk, ones(15_000); real_line = true, bounds = (C.lb, C.ub), p = [0.5], se = :none)
            @test -0.5 < m.by_p[1] - C.truth < 0.1          # bias ≈ -d²/(2N) = -0.29
        end
    end
end

end # module TestMDDEstimators
