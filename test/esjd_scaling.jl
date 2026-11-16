using Test, SMC, JLD2

@testset "ESJD scale adaptation" begin
    # A unit change in one parameter leaves its contribution to the objective unchanged.
    @test SMC.standardized_squared_jump([2.0, 1100.0], [1.0, 1000.0],
                                        [1.0, 10_000.0], [1, 2]) ≈ 2.0
    @test SMC.standardized_squared_jump([2.0, 1.1], [1.0, 1.0],
                                        [1.0, 0.01], [1, 2]) ≈ 2.0

    mktempdir() do dir
        multipliers = [0.5, 1.0, 2.0]
        function run(name; resume = false, loadpath = "", blockwise = true, kwargs...)
            path = joinpath(dir, name * ".jld2")
            options = merge((scale_adaptation = :esjd, esjd_multipliers = multipliers,
                             esjd_pilot_particles = 12), values(kwargs))
            SMC.smc(blocks_test_likelihood, blocks_test_parameters(), zeros(1, 1);
                    n_parts = 32, n_Φ = 4, n_mh_steps = 1, verbose = :none,
                    smc_seed = 41, param_blocks = [[:r]], blockwise_c = blockwise,
                    save_intermediate = true,
                    intermediate_stage_increment = 2,
                    savepath = path, particle_store_path = replace(path, ".jld2" => ".h5"),
                    continue_intermediate = resume, loadpath = loadpath, options...)
            return load(path)
        end

        full = run("full")
        h = full["stage_health"]
        @test full["scale_adaptation"] == :esjd
        @test length(h["esjd_pilot_scores"]) == length(h["c"])
        previous = fill(0.5, 2)
        for stage in eachindex(h["c"])
            scores = h["esjd_pilot_scores"][stage]
            @test size(scores) == (length(multipliers), 2)
            for group in 1:2
                best = 2  # The baseline (multiplier 1) wins ties.
                for candidate in eachindex(multipliers)
                    scores[candidate, group] > scores[best, group] && (best = candidate)
                end
                @test h["c_blocks"][group, stage] ≈ previous[group] * multipliers[best]
            end
            previous = h["c_blocks"][:, stage]
        end

        checkpoint = joinpath(dir, "full_stage=2.jld2")
        resumed = run("resumed"; resume = true, loadpath = checkpoint)
        @test resumed["cloud"].particles == full["cloud"].particles
        @test resumed["stage_health"]["c_blocks"] == h["c_blocks"]
        @test resumed["stage_health"]["esjd_pilot_scores"] == h["esjd_pilot_scores"]
        @test_throws ArgumentError run("wrong_multipliers"; resume = true,
                                       loadpath = checkpoint,
                                       esjd_multipliers = [0.75, 1.0, 1.5])

        shared = run("shared"; blockwise = false)["stage_health"]
        @test all(size(scores) == (length(multipliers), 1)
                  for scores in shared["esjd_pilot_scores"])
        @test all(shared["c_blocks"][g, :] == shared["c"] for g in 1:2)
        @test_throws ArgumentError run("bad_mode"; scale_adaptation = :unknown)
        @test_throws ArgumentError run("bad_grid"; esjd_multipliers = [0.5, 2.0])
    end
end
