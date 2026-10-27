using Test, Random, SMC, Distributed, ModelConstructors, Distributions

@testset "Indexed SMC random streams" begin
    seed = UInt64(42)
    keys = [(op, stage, particle) for op in (:initial_particle, :mutation_particle, :stage_resample)
            for stage in 0:2 for particle in 0:12]
    derived = [SMC.stream_seed(seed, op, stage, particle) for (op, stage, particle) in keys]
    @test length(unique(derived)) == length(derived)
    @test derived == [SMC.stream_seed(seed, op, stage, particle) for (op, stage, particle) in keys]
    @test derived != [SMC.stream_seed(seed + 1, op, stage, particle) for (op, stage, particle) in keys]
    @test_throws ArgumentError SMC.checked_smc_seed(-1)

    weights = [0.1, 0.2, 0.3, 0.4]
    for method in (:systematic, :multinomial, :polyalgo)
        key = SMC.stream_seed(seed, :stage_resample, 2)
        first = SMC.resample(weights; n_parts=200, method=method, rng=Xoshiro(key))
        rand(1000)
        second = SMC.resample(weights; n_parts=200, method=method, rng=Xoshiro(key))
        @test first == second
        @test all(x -> 1 <= x <= 4, first)
    end

    # A bridge may request fewer output particles than there are source weights.
    unequal = SMC.resample([0.0, 0.0, 0.0, 1.0]; n_parts=2,
                           method=:systematic, rng=Xoshiro(seed))
    @test unequal == [4, 4]

    key = SMC.stream_seed(seed, :stage_blocks, 3)
    first = SMC.generate_free_blocks(12, 3; rng=Xoshiro(key))
    rand(1000)
    @test first == SMC.generate_free_blocks(12, 3; rng=Xoshiro(key))
    @test sort(vcat(first...)) == collect(1:12)

    # Appending new parameters to a bridge cloud also uses row-indexed streams.
    m = GenericModel()
    m <= parameter(:a, 0., (-10., 10.), (-10., 10.), Untransformed(), Normal(0, 1), fixed=false)
    m <= parameter(:b, 0., (-10., 10.), (-10., 10.), Untransformed(), Normal(0, 1), fixed=false)
    old = SMC.Cloud(1, 8)
    old.particles .= 0.0
    old.particles[:, end] .= 1.0
    old.ESS[1] = 8.0
    old_inds = BitVector([true, false])
    appended1 = SMC.add_parameters_to_cloud(old, deepcopy(m.parameters), old_inds; smc_seed=42)
    rand(1000)
    appended2 = SMC.add_parameters_to_cloud(old, deepcopy(m.parameters), old_inds; smc_seed=42)
    appended3 = SMC.add_parameters_to_cloud(old, deepcopy(m.parameters), old_inds; smc_seed=43)
    @test appended1.particles == appended2.particles
    @test appended1.particles != appended3.particles
    @test appended1.particles[:, 1] == old.particles[:, 1]
end
