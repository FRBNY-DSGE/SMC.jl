using Test, SMC, ModelConstructors, Distributions, JLD2, Random, LinearAlgebra

# Starting scales of the block groups with blockwise_c (initial_c_blocks), e.g. the previous
# estimation's final c_blocks when a bridge starts from its cloud

@testset "resolve_initial_c_blocks" begin
    labels = ["r", "block2", "rest"]
    @test SMC.resolve_initial_c_blocks([1.0, 2, 0.5], labels, true) == [1.0, 2.0, 0.5]
    # A Dict by label (String or Symbol keys, any order) gives the scales in group order
    @test SMC.resolve_initial_c_blocks(Dict("rest" => 0.5, :r => 1.0, "block2" => 2.0), labels, true) ==
        [1.0, 2.0, 0.5]
    @test SMC.resolve_initial_c_blocks(Dict(:all => 0.3), ["all"], true) == [0.3]
    @test_throws ArgumentError SMC.resolve_initial_c_blocks([1.0, 2.0, 0.5], labels, false)   # needs blockwise_c
    @test_throws ArgumentError SMC.resolve_initial_c_blocks([1.0, 2.0], labels, true)         # one per group
    @test_throws ArgumentError SMC.resolve_initial_c_blocks([1.0, 2.0, 0.5, 1.0], labels, true)
    @test_throws ArgumentError SMC.resolve_initial_c_blocks([1.0, 0.0, 0.5], labels, true)    # > 0
    @test_throws ArgumentError SMC.resolve_initial_c_blocks([1.0, -2.0, 0.5], labels, true)
    @test_throws ArgumentError SMC.resolve_initial_c_blocks([1.0, NaN, 0.5], labels, true)    # finite
    @test_throws ArgumentError SMC.resolve_initial_c_blocks([1.0, Inf, 0.5], labels, true)
    @test_throws ArgumentError SMC.resolve_initial_c_blocks(Dict("r" => 1.0, "rest" => 0.5), labels, true)  # missing group
    @test_throws ArgumentError SMC.resolve_initial_c_blocks(Dict("r" => 1.0, "block2" => 2.0, "rest" => 0.5,
                                                                 "x1" => 1.0), labels, true)  # unknown label
    @test_throws ArgumentError SMC.resolve_initial_c_blocks(Dict(:r => 1.0, "r" => 1.0, "block2" => 2.0,
                                                                 "rest" => 0.5), labels, true) # twice
    @test_throws ArgumentError SMC.resolve_initial_c_blocks(Dict(1 => 1.0, "block2" => 2.0, "rest" => 0.5),
                                                            labels, true)                     # not a label
    @test_throws ArgumentError SMC.resolve_initial_c_blocks(Dict("r" => -1.0, "block2" => 2.0, "rest" => 0.5),
                                                            labels, true)
end

# Toy posterior as in param_blocks.jl: 6 normal-ish parameters and r ∈ (0, 1) concentrated near 1
function icb_test_parameters()
    m = GenericModel()
    for k in 1:6
        m <= parameter(Symbol("x$k"), 0., (-20., 20.), (-20., 20.), Untransformed(), Normal(0, 3), fixed = false)
    end
    m <= parameter(:r, 0.5, (0., 1.), (0., 1.), Untransformed(), Uniform(0, 1), fixed = false)
    return m.parameters
end
function icb_test_likelihood(p, data)
    r = p[7].value
    0 < r < 1 || return -Inf
    return -0.5 * sum(abs2, [p[k].value - 0.1 * k for k in 1:6]) + 60 * log(r) + log(1 - r)
end
icb_test_old_likelihood(p, data; kw...) = icb_test_likelihood(p, data) - 0.5 * sum(abs2, [p[k].value for k in 1:6])

@testset "smc with initial_c_blocks" begin
    dir = mktempdir()
    function run(name; resume = false, loadpath = "", kwargs...)
        path = joinpath(dir, name * ".jld2")
        kw = merge((param_blocks = [[:r]], blockwise_c = true), values(kwargs))
        SMC.smc(icb_test_likelihood, icb_test_parameters(), zeros(1, 1);
                n_parts = 200, n_Φ = 12, n_mh_steps = 2, verbose = :none, smc_seed = 7,
                α = 0.9, c = 0.5, savepath = path, particle_store_path = replace(path, ".jld2" => ".h5"),
                continue_intermediate = resume, loadpath = loadpath,
                save_intermediate = true, intermediate_stage_increment = 5, kw...)
        return load(path)
    end
    striptime(h) = Dict(k => v for (k, v) in h if k != "time")
    same(a, b) = isequal(a["cloud"].particles, b["cloud"].particles) && isequal(a["w"], b["w"]) &&
        isequal(a["W"], b["W"]) && isequal(striptime(a["stage_health"]), striptime(b["stage_health"])) &&
        isequal(a["c_blocks"], b["c_blocks"]) && a["cloud"].c === b["cloud"].c

    base = run("base")

    @testset "Default nothing is the shared starting c" begin
        # nothing, and every group given the initial c, are the run without the keyword, bitwise
        @test same(run("expl"; initial_c_blocks = nothing), base)
        @test same(run("vec05"; initial_c_blocks = [0.5, 0.5]), base)
        @test same(run("dict05"; initial_c_blocks = Dict("r" => 0.5, "rest" => 0.5)), base)
        @test all(base["stage_health"]["c_blocks"][:, 1] .== SMC.update_c(0.5, 0.25, 0.25))
    end

    c0 = [1.3, 0.26]
    icb = run("icb"; initial_c_blocks = Dict(:rest => c0[2], "r" => c0[1]))
    @testset "The groups start from the given scales, then follow update_c" begin
        h = icb["stage_health"]
        @test h["block_labels"] == ["r", "rest"]
        @test h["c_blocks"][:, 1] == SMC.update_c.(c0, 0.25, 0.25)      # stage 1
        for g in 1:2
            cg = c0[g]
            for s in 1:length(h["stage"])
                cg = SMC.update_c(cg, s == 1 ? 0.25 : h["accept_blocks"][g, s-1], 0.25)
                @test h["c_blocks"][g, s] == cg
            end
        end
        @test h["c"] == h["c_blocks"][2, :]
        @test !isequal(icb["cloud"].particles, base["cloud"].particles)
        # The vector form (group order) is the Dict form, bitwise
        @test same(run("icbvec"; initial_c_blocks = c0), icb)
    end

    @testset "Resuming ignores initial_c_blocks (the checkpoint's scales win), bitwise" begin
        ckpt = joinpath(dir, "icb_stage=5.jld2")
        @test load(ckpt, "c_blocks") == icb["stage_health"]["c_blocks"][:, 4]
        @test same(run("res_nothing"; resume = true, loadpath = ckpt), icb)
        @test same(run("res_other"; resume = true, loadpath = ckpt, initial_c_blocks = [9.0, 9.0]), icb)
        # Not even validated on a resume
        @test same(run("res_bad"; resume = true, loadpath = ckpt, initial_c_blocks = [-1.0]), icb)
        @test same(run("base_res"; resume = true, loadpath = joinpath(dir, "base_stage=5.jld2"),
                       initial_c_blocks = c0), base)
    end

    @testset "Bad inputs throw" begin
        @test_throws ArgumentError run("bad1"; initial_c_blocks = [0.5])
        @test_throws ArgumentError run("bad2"; initial_c_blocks = [0.5, -1.0])
        @test_throws ArgumentError run("bad3"; initial_c_blocks = [0.5, NaN])
        @test_throws ArgumentError run("bad4"; initial_c_blocks = [0.5, 0.5], blockwise_c = false)
        @test_throws ArgumentError run("bad5"; initial_c_blocks = [0.5], param_blocks = Vector{Vector{Symbol}}(),
                                       blockwise_c = false)
        @test_throws ArgumentError run("bad6"; initial_c_blocks = Dict("r" => 0.5))
        @test_throws ArgumentError run("bad7"; initial_c_blocks = Dict("r" => 0.5, "x1" => 0.5))
        # Without param_blocks there is one group, "all"
        @test run("all"; param_blocks = Vector{Vector{Symbol}}(),
                  initial_c_blocks = Dict("all" => 0.8))["stage_health"]["c_blocks"][1, 1] ==
            SMC.update_c(0.8, 0.25, 0.25)
    end

    @testset "Bridge from an old cloud starting from its final scales" begin
        old = base                                    # the old estimation, its final c_blocks unequal
        bridge(name; kw...) = run(name; old_data = zeros(1, 1), old_cloud = deepcopy(old["cloud"]),
                                  old_loglikelihood = icb_test_old_likelihood,
                                  tempered_update_prior_weight = 0.0, use_fixed_schedule = false,
                                  tempering_target = 0.95, save_intermediate = false, kw...)
        b0 = bridge("b0")
        b1 = bridge("b1"; initial_c_blocks = old["c_blocks"])
        @test b1["cloud"].tempering_schedule[end] == 1.0
        # Same starting acceptance in both runs, so the stage-1 update factor is the same
        @test b1["stage_health"]["c_blocks"][:, 1] ./ old["c_blocks"] ≈ b0["stage_health"]["c_blocks"][:, 1] ./ 0.5
        @test b1["stage_health"]["c_blocks"][:, 1] != b0["stage_health"]["c_blocks"][:, 1]
    end
    rm(dir; recursive = true, force = true)
end
