using ModelConstructors, HDF5, Random, JLD2, FileIO, SMC, Test, Distributions, LinearAlgebra
include("modelsetup.jl")

# Reliability diagnostics (SMC reliability plan, step 3 part C, items 1-6)

@testset "Log MDD helpers" begin
    # Two stages: the second multiplies W = (1, 3) by w = (2, 4): mean of w·W = (2 + 12)/4 per 2 particles
    w = [0. 2.; 0. 4.]
    W = [1. 0.5; 3. 1.5]
    @test SMC.smc_log_mdd(w, W) ≈ log((2 * 1 + 4 * 3) / 4)
    @test SMC.smc_log_mdd_level(-3.0) == -3.0
    @test isnan(SMC.smc_log_mdd_level(-3.0; tempered_update = true))
    @test SMC.smc_log_mdd_level(-3.0; tempered_update = true, log_prob_old_data = -10.0) == -3.0
    @test SMC.smc_log_mdd_level(-3.0; tempered_update = true, tempered_update_prior_weight = 0.5,
                                log_prob_old_data = -10.0) == -3.0
    @test SMC.logmeanexp([log(1.0), log(3.0)]) ≈ log(2.0)
    est = Dict{String, Any}("smc_level" => -10.0, "smc_lineage_nse" => 0.1,
                            "bridge_sampling" => -12.0, "bridge_sampling_nse" => 0.1)
    @test occursin("log MDD gap", SMC.mdd_gap_flag(est))
    @test SMC.mdd_gap_flag(merge(est, Dict("bridge_sampling" => -10.1))) === nothing
    # A μ = 0 bridge has no level (NaN per run, absent once pooled): no flag, no error
    @test SMC.mdd_gap_flag(merge(est, Dict("smc_level" => NaN))) === nothing
    @test SMC.mdd_gap_flag(delete!(copy(est), "smc_level")) === nothing
end

@testset "Stage health pieces" begin
    @test SMC.conditional_ess(ones(5), ones(5)) ≈ 1.0
    @test SMC.conditional_ess([1.0, 0.0], [0.5, 0.5]) ≈ 0.5
    @test SMC.distinct_share([1. 2.; 1. 2.; 3. 4.; 5. 6.]) == 0.75

    rng = Xoshiro(1)
    before = randn(rng, 10_000, 2)
    after  = hcat(before[:, 1], randn(rng, 10_000))
    corr, esjd = SMC.particle_movement(before, after, ones(10_000))
    @test corr[1] ≈ 1.0
    @test abs(corr[2]) < 0.05
    @test esjd[1] == 0.0
    @test esjd[2] ≈ 2.0 atol = 0.1    # E(x - y)² = 2 for independent standard normals

    h = SMC.new_stage_health(2)
    for s in 2:21
        late = s > 17
        SMC.record_stage_health!(h, (stage = s, phi = s / 21, dphi = s == 10 ? 0.5 : 0.01,
                                     ess = s == 5 ? 0.05 : 0.9, cess = 0.95, max_weight = 0.01,
                                     resampled = false, distinct_selected = 1.0,
                                     distinct_mutated = s == 7 ? 0.3 : 0.9,
                                     accept = s == 8 ? 0.6 : 0.25, c = 0.5, inf_cloud = 0,
                                     inf_proposals = late ? 50 : 0, n_proposals = 1000, time = 1.0),
                                 [0.5, late ? 0.95 : 0.5], [1.0, 1.0])
    end
    flags = SMC.stage_health_flags(h; para_names = [:a, :b])
    @test length(flags) == 6
    @test any(f -> occursin("ESS/N < 0.1 at stages 5", f), flags)
    @test any(f -> occursin("distinct particles after mutation < 50% of N at stages 7", f), flags)
    @test any(f -> occursin("acceptance outside [0.1, 0.5] at stages 8", f), flags)
    @test any(f -> occursin("correlation > 0.9", f) && occursin("(e.g. b)", f), flags)
    @test any(f -> occursin("Δφ jumps by more than 10.0× at stages 10", f), flags)
    @test any(f -> occursin("-Inf proposals rise", f), flags)
    @test isempty(SMC.stage_health_flags(SMC.new_stage_health(2)))

    path = tempname() * ".csv"
    SMC.write_stage_health(path, h; para_names = [:a, :b])
    lines = readlines(path)
    @test length(lines) == 21
    @test endswith(lines[end], ",0.95,b,1.0,a")
end

@testset "Lineage NSE" begin
    # Without resampling it is the importance-sampling variance of the mean weight,
    # Var(w) / (N - 1) / mean(w)², and of the weighted mean
    rng = Xoshiro(2)
    N = 1_000
    w = exp.(randn(rng, N))
    θ = randn(rng, N, 1)
    out = SMC.lineage_nse(1:N, 0, w, θ)
    m, v = mean(w), mean((w .- mean(w)) .^ 2)
    @test out.log_mdd_nse ≈ sqrt(v / (m^2 * (N - 1)))
    W = w ./ sum(w)
    @test out.mean_nse[1] ≈ sqrt(N / (N - 1) * sum((W .* (θ[:, 1] .- sum(W .* θ[:, 1]))) .^ 2))
    @test out.n_eves == N
    # One surviving Eve: no information left
    @test SMC.lineage_nse(ones(Int, N), 5, w, θ).n_eves == 1
end

@testset "MHM and bridge sampling recover a known normalizing constant" begin
    rng = Xoshiro(3)
    d, N = 4, 20_000
    μ, A = randn(rng, d), randn(rng, d, d)
    Σ = A * A' + I
    target = MvNormal(μ, Σ)
    log_Z = -12.3
    θ = rand(rng, target, N)'
    logpost = log_Z .+ logpdf(target, θ')
    mhm = SMC.modified_harmonic_mean(θ, logpost, ones(N))
    @test mhm.log_mdd ≈ log_Z atol = 0.02
    @test length(mhm.by_p) == 9

    g = MvNormal(μ .+ 0.2, 1.5 * Σ)
    θg = rand(rng, g, N)'
    bs = SMC.bridge_sampling_estimate(logpost, logpdf(g, θ'), ones(N),
                                      log_Z .+ logpdf(target, θg'), logpdf(g, θg'))
    @test bs.log_mdd ≈ log_Z atol = 0.02
    @test 0 < bs.nse < 0.02
    # Weighted posterior draws: draws from g reweighted to the target
    wts = exp.(logpdf(target, θg') .- logpdf(g, θg'))
    @test SMC.modified_harmonic_mean(θg, log_Z .+ logpdf(target, θg'), wts).log_mdd ≈ log_Z atol = 0.05
end

@testset "Gaussian mixture BIC" begin
    rng = Xoshiro(4)
    one_mode = randn(rng, 2_000, 3)
    @test SMC.gaussian_mixture_bic(one_mode, ones(2_000); rng = Xoshiro(1)).best_k == 1
    two_modes = vcat(randn(rng, 1_000, 3), randn(rng, 1_000, 3) .+ [6.0 0.0 0.0])
    fit = SMC.gaussian_mixture_bic(two_modes, ones(2_000); rng = Xoshiro(1))
    @test fit.best_k == 2
    @test sort(round.(first.(fit.centers))) == [0.0, 6.0]
    @test all(w -> isapprox(w, 0.5, atol = 0.05), fit.mixture_weights)
end

@testset "NSE validation and run comparison" begin
    rng = Xoshiro(5)
    R, K, J = 400, 3, 10
    truth = [1.0, 2.0, 3.0]
    nse = [0.1, 0.2, 0.3]
    # Each run: the mean of J group estimates with sd √J·nse, and the NSE from their spread
    groups = [truth[k] + sqrt(J) * nse[k] * randn(rng) for r in 1:R, k in 1:K, j in 1:J]
    estimates = dropdims(mean(groups, dims = 3), dims = 3)
    nses = dropdims(std(groups, dims = 3), dims = 3) ./ sqrt(J)
    v = SMC.nse_validation(estimates, nses, J)
    @test all(r -> isapprox(r, 1.0, atol = 0.15), v.sd_ratio)
    @test all(c -> isapprox(c, 0.95, atol = 0.04), v.coverage)
    # Overconfident NSEs show up in both
    v_bad = SMC.nse_validation(estimates, nses ./ 2, J)
    @test all(v_bad.sd_ratio .> 1.6)
    @test all(v_bad.coverage .< 0.8)

    a, b = randn(rng, 500), randn(rng, 500)
    same = SMC.compare_estimates(a, ones(500), b, ones(500))
    @test same.share_above_2 < 0.1
    @test same.ks_pvalue > 0.01
    shifted = SMC.compare_estimates(a .+ 1, ones(500), b, ones(500))
    @test shifted.ks_pvalue < 1e-6

    # NSEs from J = 6 groups: raw t's exceed 2 too often; the small-J z's don't
    J, n = 6, 20_000
    ga, gb = randn(rng, n, J), randn(rng, n, J)
    ma, mb = vec(mean(ga, dims = 2)), vec(mean(gb, dims = 2))
    sa, sb = vec(std(ga, dims = 2)) ./ sqrt(J), vec(std(gb, dims = 2)) ./ sqrt(J)
    raw   = SMC.compare_estimates(ma, sa, mb, sb)
    small = SMC.compare_estimates(ma, sa, mb, sb; n_groups_a = J, n_groups_b = J)
    @test raw.share_above_2 > 0.065
    @test isapprox(small.share_above_2, 0.0455, atol = 0.006)
    @test small.ks_pvalue > 0.001
    @test SMC.comparison_z(1.0, 0.5, 0.0) ≈ 2.0
    @test SMC.comparison_z(1.0, 0.5, 0.0, 6, 6) < 2.0
    @test SMC.comparison_z(-1.0, 0.5, 0.0, 6, 6) ≈ -SMC.comparison_z(1.0, 0.5, 0.0, 6, 6)
    @test isfinite(SMC.comparison_z(100.0, 0.5, 0.5, 6, 6))
end

# Exact log MDD of the linear test model: per equation, integrate α and β analytically given σ,
# then σ ~ U(0, 1000) numerically (the likelihood is negligible outside the grid)
function linear_model_log_mdd(data; τ = 1e3, σ_max = 1e3)
    total = 0.0
    T = size(data, 2)
    for i in 1:3
        Z = hcat(ones(T), X[i, 1:T])
        logp(σ) = logpdf(MvNormal(zeros(T), Symmetric(σ^2 * I + τ^2 * Z * Z')), data[i, :])
        grid = range(0.3, 4.0, length = 8_001)
        total += SMC.logsumexp(logp.(grid)) + log(step(grid)) - log(σ_max)
    end
    return total
end

diag_data = h5read("reference/test_data.h5", "data")
diag_dir  = mktempdir()
diag_truth = linear_model_log_mdd(diag_data)

function diag_smc(name; kwargs...)
    SMC.smc(loglik_fn, setup_linear_model().parameters, diag_data; verbose = :none, n_parts = 1_000,
            n_mh_steps = 2, use_fixed_schedule = false, tempering_target = 0.95, α = 0.9,
            savepath = joinpath(diag_dir, name * ".jld2"),
            particle_store_path = joinpath(diag_dir, name * ".h5"), kwargs...)
end

@testset "One run: stage health and three log MDD estimates" begin
    diag_smc("one"; smc_seed = 11, bridge_sampling = true)
    out = load(joinpath(diag_dir, "one.jld2"))
    cloud, h, est = out["cloud"], out["stage_health"], out["log_mdd_estimates"]
    @test SMC.n_stages(h) == cloud.stage_index - 1
    @test h["phi"][end] == 1.0
    @test h["ess"] ≈ cloud.ESS[2:end] ./ 1_000
    @test size(h["corr"]) == (9, SMC.n_stages(h))
    @test all(0 .<= h["inf_proposals"] .<= h["n_proposals"])
    @test isfile(joinpath(diag_dir, "one_stage_health.csv"))
    @test out["health_flags"] isa Vector{String}

    @test est["smc"] == SMC.smc_log_mdd(out["w"], out["W"])
    @test est["smc_level"] == est["smc"]
    @test est["bridge_sampling"] ≈ diag_truth atol = 0.3
    @test est["mhm"] ≈ diag_truth atol = 0.5
    @test est["n_eves"] >= 1
    # With this model's very diffuse prior the particles stop moving in the middle stages and
    # the incremental-weight estimate falls far below the exact value; the checks catch it
    @test est["smc"] < diag_truth - 10
    @test any(f -> occursin("log MDD gap", f), out["health_flags"])
    @test any(f -> occursin("particles are not moving", f), out["health_flags"])
end

@testset "Particle groups" begin
    cloud = diag_smc("groups"; smc_seed = 12, n_smc_groups = 4, bridge_sampling = true)
    out = load(joinpath(diag_dir, "groups.jld2"))
    s = out["smc_groups"]
    @test length(cloud) == 1_000
    @test out["n_smc_groups"] == 4
    @test !haskey(out, "w")  # a pooled run has no single weight history
    @test sum(SMC.get_weights(cloud)) ≈ 1_000
    for g in 1:4
        group = load(joinpath(diag_dir, "groups_group=$g.jld2"))
        @test length(group["cloud"]) == 250
        @test group["smc_seed"] == s["group_seeds"][g]
        @test group["run_info"]["smc_group"] == "$g of 4"
    end
    @test length(unique(s["group_seeds"])) == 4
    @test s["mean"] ≈ vec(mean(reduce(hcat, [SMC.weighted_mean(load(joinpath(diag_dir, "groups_group=$g.jld2"), "cloud"))
                                             for g in 1:4]), dims = 2))
    @test all(s["mean_nse"] .> 0)
    @test all(isfinite, s["rne"])
    est = out["log_mdd_estimates"]
    @test est["bridge_sampling"] ≈ SMC.logmeanexp(est["bridge_sampling_groups"])
    @test isfile(joinpath(diag_dir, "groups_group_summary.csv"))

    # Resuming a finished grouped run reuses the groups instead of rerunning them
    before = mtime(joinpath(diag_dir, "groups_group=1.jld2"))
    again = diag_smc("groups"; smc_seed = 12, n_smc_groups = 4, bridge_sampling = true,
                     continue_intermediate = true)
    @test mtime(joinpath(diag_dir, "groups_group=1.jld2")) == before
    @test isequal(again.particles, cloud.particles)
    @test_throws ArgumentError diag_smc("bad"; smc_seed = 1, n_smc_groups = 3)

    # A single old cloud is split into disjoint parts, one per group
    parts = SMC.group_old_clouds(cloud, "", 4, true; run_seed = UInt64(5), verbose = :none)
    @test all(p -> size(p.particles) == (250, size(cloud.particles, 2)), parts)
    # Same particles apart from the weight column, which each part rescales to its size
    keep = 1:size(cloud.particles, 2) - 1
    @test sortslices(reduce(vcat, [p.particles[:, keep] for p in parts]), dims = 1) ==
          sortslices(cloud.particles[:, keep], dims = 1)
    @test parts[1].particles == SMC.group_old_clouds(cloud, "", 4, true; run_seed = UInt64(5),
                                                     verbose = :none)[1].particles
    @test cloud.particles == again.particles   # the old cloud itself is untouched
    # A grouped old run's group files are used one per group
    files = SMC.group_old_clouds(SMC.Cloud(0, 0), joinpath(diag_dir, "groups.jld2"), 4, true)
    @test files[3].particles == load(joinpath(diag_dir, "groups_group=3.jld2"), "cloud").particles

    # Each part's last ESS comes from its own weights, not the full cloud's (SMC-CHECK-5)
    for p in parts
        w = SMC.get_weights(p)
        @test p.ESS[end] ≈ sum(w)^2 / sum(abs2, w)
        @test p.ESS[end] <= 250 + 1e-8
        @test sum(w) ≈ 250   # weights rescaled to the part's size (SMC-CHECK-6)
        # ... in proportion to the old cloud's weights for the same particles
        rows = [findfirst(r -> cloud.particles[r, keep] == p.particles[k, keep], 1:length(cloud))
                for k in 1:2]
        @test w[1] / w[2] ≈ SMC.get_weights(cloud)[rows[1]] / SMC.get_weights(cloud)[rows[2]]
    end
end

@testset "Adaptive grouped bridge from a single old cloud" begin
    # Old run on the first half of the data, then a grouped bridge to the full data with
    # adaptive tempering. Before the split fix every part kept the old cloud's ESS (for
    # 1,000 particles), so a group of 250 could not reach tempering_target × ESS and the
    # adaptive root solver threw (SMC-CHECK-5: 4 of 5 groups failed).
    half = diag_data[:, 1:end÷2]
    SMC.smc(loglik_fn, setup_linear_model().parameters, half; verbose = :none, n_parts = 1_000,
            n_mh_steps = 2, use_fixed_schedule = false, tempering_target = 0.95, α = 0.9,
            smc_seed = 21, savepath = joinpath(diag_dir, "old_half.jld2"),
            particle_store_path = joinpath(diag_dir, "old_half.h5"))
    old = load(joinpath(diag_dir, "old_half.jld2"), "cloud")
    @test old.ESS[end] > 250        # so a part that kept it would have an unreachable target

    bridged = diag_smc("grouped_bridge"; smc_seed = 22, n_smc_groups = 4,
                       old_data = half, old_cloud = old)
    @test length(bridged) == 1_000
    for g in 1:4
        group = load(joinpath(diag_dir, "grouped_bridge_group=$g.jld2"))
        @test length(group["cloud"]) == 250
        @test group["cloud"].tempering_schedule[end] == 1.0
        @test all(isfinite, group["cloud"].ESS)
    end
end

@testset "Provenance in run_info" begin
    # A full commit hash, "-dirty" only if src/ or Project.toml differ from it
    commit = SMC.package_commit(SMC)
    @test occursin(r"^[0-9a-f]{40}(-dirty)?$", commit)
    # The caller can add keys but not overwrite SMC's own provenance
    @test_logs (:warn, r"set by SMC") match_mode = :any diag_smc("prov"; smc_seed = 31,
        run_info = Dict("smc_commit" => "fake", "caller_key" => "kept"))
    info = load(joinpath(diag_dir, "prov.jld2"), "run_info")
    @test info["smc_commit"] == commit
    @test info["caller_key"] == "kept"
    @test info["julia_version"] == string(VERSION)
end

rm(diag_dir; recursive = true, force = true)
