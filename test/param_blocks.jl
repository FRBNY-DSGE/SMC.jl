using Test, SMC, ModelConstructors, Distributions, JLD2, Random, LinearAlgebra

# Fixed mutation blocks (param_blocks) and a scale per block group (blockwise_c)

@testset "resolve_param_blocks" begin
    syms = [:a, :b, :r, :z, :b_reg2]
    free = [1, 2, 3, 5]                           # z is fixed
    fixed, rest = SMC.resolve_param_blocks([[:r]], syms, free)
    @test fixed == [[3]] && rest == [1, 2, 4]
    # Names (Symbol or String), full-vector indices and extra regimes; blocks are sorted
    fixed, rest = SMC.resolve_param_blocks([["b_reg2", 1], [3]], syms, free)
    @test fixed == [[1, 4], [3]] && rest == [2]
    fixed, rest = SMC.resolve_param_blocks([[:a, :b], [:r, :b_reg2]], syms, free)
    @test fixed == [[1, 2], [3, 4]] && isempty(rest)
    @test SMC.resolve_param_blocks(Vector{Vector{Int}}(), syms, free) == (Vector{Vector{Int}}(), [1, 2, 3, 4])
    @test_throws ArgumentError SMC.resolve_param_blocks([[:nope]], syms, free)
    @test_throws ArgumentError SMC.resolve_param_blocks([[:z]], syms, free)       # fixed
    @test_throws ArgumentError SMC.resolve_param_blocks([[:a], [:a]], syms, free) # twice
    @test_throws ArgumentError SMC.resolve_param_blocks([Symbol[]], syms, free)   # empty
    @test_throws ArgumentError SMC.resolve_param_blocks([[6]], syms, free)        # out of range
end

@testset "generate_stage_blocks, group_acceptance, update_c" begin
    # Without fixed blocks: exactly generate_free_blocks from the same RNG, one group
    for nb in (1, 2, 3)
        b1, g1 = SMC.generate_stage_blocks(Vector{Vector{Int}}(), collect(1:7), 7, nb; rng = Xoshiro(5))
        @test b1 == SMC.generate_free_blocks(7, nb; rng = Xoshiro(5))
        @test g1 == ones(Int, length(b1))
    end
    # Fixed blocks first, in the given order; the rest split at random into n_blocks blocks
    blocks, groups = SMC.generate_stage_blocks([[4], [1, 6]], [2, 3, 5, 7], 7, 2; rng = Xoshiro(5))
    @test blocks[1:2] == [[4], [1, 6]]
    @test groups == [1, 2, 3, 3]
    @test sort(vcat(blocks...)) == collect(1:7)
    blocks, groups = SMC.generate_stage_blocks([[4]], [1, 2, 3], 4, 1; rng = Xoshiro(5))
    @test blocks == [[4], [1, 2, 3]] && groups == [1, 2]

    # Acceptance per group: accepted proposals weighted by block size over n_trials proposals
    acc = SMC.group_acceptance([5, 10, 0], [[1], [2, 3], [4]], [1, 2, 2], 2, 20)
    @test acc ≈ [0.25, (2 * 10 + 1 * 0) / (20 * 3)]

    # update_c is the adaptive rule smc always used
    c, a, t = 0.37, 0.31, 0.25
    @test SMC.update_c(c, a, t) === c * (0.95 + 0.10 * exp(16.0 * (a - t)) / (1.0 + exp(16.0 * (a - t))))
    @test SMC.update_c(c, t, t) == c
end

# Toy posterior: 6 standard-normal-ish parameters and r ∈ (0, 1) concentrated near 1
function blocks_test_parameters()
    m = GenericModel()
    for k in 1:6
        m <= parameter(Symbol("x$k"), 0., (-20., 20.), (-20., 20.), Untransformed(), Normal(0, 3), fixed = false)
    end
    m <= parameter(:r, 0.5, (0., 1.), (0., 1.), Untransformed(), Uniform(0, 1), fixed = false)
    return m.parameters
end
function blocks_test_likelihood(p, data)
    r = p[7].value
    0 < r < 1 || return -Inf
    return -0.5 * sum(abs2, [p[k].value - 0.1 * k for k in 1:6]) + 60 * log(r) + log(1 - r)
end

@testset "mutation with a scale per block" begin
    params = blocks_test_parameters()
    θ0 = vcat(0.1 .* (1:6), 0.95)
    ModelConstructors.update!(params, θ0)
    p0 = vcat(θ0, blocks_test_likelihood(params, zeros(1, 1)), ModelConstructors.prior(params), 0.0, 0.0, 1.0)
    μ, Σ = θ0, Matrix(Diagonal(vcat(fill(1.0, 6), 0.02^2)))
    blocks = [[7], collect(1:6)]
    mut(c, n_acc) = (Random.seed!(11);
                     SMC.mutation(blocks_test_likelihood, params, zeros(1, 1), copy(p0), μ, Σ, 7,
                                  blocks, blocks, 1.0; c = c, α = 0.9, n_mh_steps = 3, n_accepted = n_acc))
    # A vector of equal scales is the scalar c, draw for draw
    @test mut(0.4, nothing) == mut(fill(0.4, 2), nothing)
    n_acc = zeros(Int, 2)
    p = mut([1.5, 0.3], n_acc)
    @test p != mut(0.4, nothing)
    @test p[end-1] ≈ (1 * n_acc[1] + 6 * n_acc[2]) / (7 * 3)
    @test all(0 .<= n_acc .<= 3)
    @test_throws ArgumentError mut([0.4], nothing)
    @test_throws ArgumentError mut(0.4, zeros(Int, 3))
end

mktempdir() do dir
    function run(name; resume = false, loadpath = "", kwargs...)
        path = joinpath(dir, name * ".jld2")
        SMC.smc(blocks_test_likelihood, blocks_test_parameters(), zeros(1, 1);
                n_parts = 200, n_Φ = 12, n_mh_steps = 2, verbose = :none, smc_seed = 7,
                α = 0.9, savepath = path, particle_store_path = replace(path, ".jld2" => ".h5"),
                continue_intermediate = resume, loadpath = loadpath,
                intermediate_stage_increment = 5, kwargs...)
        return load(path)
    end

    base     = run("base")
    explicit = run("explicit"; param_blocks = Vector{Vector{Symbol}}(), blockwise_c = false)
    shared   = run("shared"; param_blocks = [[:r]])
    blockc   = run("blockc"; param_blocks = [[:r]], blockwise_c = true, save_intermediate = true)
    resumed  = run("resumed"; param_blocks = [[:r]], blockwise_c = true, resume = true,
                   loadpath = joinpath(dir, "blockc_stage=5.jld2"))
    rand3    = run("rand3"; param_blocks = [[:r]], blockwise_c = true, n_blocks = 3)
    byindex  = run("byindex"; param_blocks = [[7], [1, 2]], blockwise_c = true)
    # A checkpoint written without blocks resumes with every group at the saved c
    plain    = run("plain"; save_intermediate = true)
    oldckpt  = @test_logs (:warn, r"no c_blocks") match_mode = :any run("oldckpt"; param_blocks = [[:r]],
               blockwise_c = true, resume = true, loadpath = joinpath(dir, "plain_stage=5.jld2"))

    @testset "Defaults are unchanged" begin
        @test explicit["cloud"].particles == base["cloud"].particles
        @test explicit["stage_health"]["c"] == base["stage_health"]["c"]
        @test !haskey(base["stage_health"], "accept_blocks") && !haskey(base, "c_blocks")
        csv = readlines(joinpath(dir, "base_stage_health.csv"))
        @test !occursin("accept_", csv[1])
    end

    @testset "Fixed blocks with a shared c" begin
        h = shared["stage_health"]
        @test h["block_labels"] == ["r", "rest"]
        @test size(h["accept_blocks"]) == (2, length(h["stage"]))
        @test all(h["c_blocks"][g, :] == h["c"] for g in 1:2)
        @test shared["param_blocks"] == [[:r], [:x1, :x2, :x3, :x4, :x5, :x6]]
        @test all(0 .<= h["accept_blocks"] .<= 1)
        # The overall rate is the size-weighted mean of the group rates
        @test h["accept"] ≈ vec((1 * h["accept_blocks"][1, :] + 6 * h["accept_blocks"][2, :]) / 7)
    end

    @testset "Blockwise c" begin
        h = blockc["stage_health"]
        # Each group's scale follows update_c from its own acceptance, starting from c = 0.5
        for g in 1:2
            cg = 0.5
            for s in 1:length(h["stage"])
                cg = SMC.update_c(cg, s == 1 ? 0.25 : h["accept_blocks"][g, s-1], 0.25)
                @test h["c_blocks"][g, s] == cg
            end
        end
        @test h["c"] == h["c_blocks"][2, :]           # cloud.c is the rest group's scale
        @test blockc["cloud"].c == h["c_blocks"][2, end]
        @test h["c_blocks"][1, end] != h["c_blocks"][2, end]
        @test blockc["c_blocks"] == h["c_blocks"][:, end]
        csv = readlines(joinpath(dir, "blockc_stage_health.csv"))
        header = split(csv[1], ",")
        @test header[end-3:end] == ["accept_r", "accept_rest", "c_r", "c_rest"]
        @test header[1:length(SMC.STAGE_HEALTH_COLUMNS)] == collect(SMC.STAGE_HEALTH_COLUMNS)
        @test parse(Float64, split(csv[end], ",")[end]) == h["c_blocks"][2, end]
        ckpt = joinpath(dir, "blockc_stage=5.jld2")
        @test load(ckpt, "c_blocks") == h["c_blocks"][:, 4]   # stage 5 is the 4th recorded stage
    end

    @testset "Blockwise c resumes bitwise" begin
        @test resumed["cloud"].particles == blockc["cloud"].particles
        @test resumed["w"] == blockc["w"] && resumed["W"] == blockc["W"]
        @test resumed["stage_health"]["c_blocks"] == blockc["stage_health"]["c_blocks"]
        @test resumed["stage_health"]["accept_blocks"] == blockc["stage_health"]["accept_blocks"]
        # Resuming with different blocks is refused
        @test_throws ArgumentError run("wrong"; param_blocks = [[:x1]], blockwise_c = true, resume = true,
                                       loadpath = joinpath(dir, "blockc_stage=5.jld2"))
    end

    @testset "Old checkpoints, random rest blocks, blocks by index" begin
        h = oldckpt["stage_health"]
        @test all(isnan, h["accept_blocks"][:, 1:4]) && all(isfinite, h["accept_blocks"][:, 5:end])
        @test h["c_blocks"][:, 5] == fill(SMC.update_c(plain["stage_health"]["c"][4],
                                                     plain["stage_health"]["accept"][4], 0.25), 2)
        @test oldckpt["cloud"].particles[1:4, 1:7] != plain["cloud"].particles[1:4, 1:7]

        @test rand3["stage_health"]["block_labels"] == ["r", "rest"]
        @test size(rand3["stage_health"]["c_blocks"], 1) == 2
        @test rand3["stage_health"]["n_proposals"][1] == 200 * 2 * 4   # {r} + 3 random rest blocks

        @test byindex["param_blocks"] == [[:r], [:x1, :x2], [:x3, :x4, :x5, :x6]]
        @test byindex["stage_health"]["block_labels"] == ["r", "block2", "rest"]
    end
end
