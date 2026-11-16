using Test, SMC, JLD2

@testset "Full-cloud ESJD diagnostic" begin
    grid = SMC.esjd_scale_grid(0.5, 7, 4.0)
    @test length(grid) == 7
    @test 0.5 in grid
    @test issorted(grid)
    @test_throws ArgumentError SMC.esjd_scale_grid(0.5, 1, 4.0)

    mktempdir() do dir
        function run(name; diagnostic = false, resume = false, loadpath = "", kwargs...)
            path = joinpath(dir, name * ".jld2")
            options = merge((esjd_diagnostic = diagnostic, esjd_diagnostic_every = 2,
                             esjd_diagnostic_n_scales = 3, esjd_diagnostic_span = 2.0),
                            values(kwargs))
            SMC.smc(blocks_test_likelihood, blocks_test_parameters(), zeros(1, 1);
                    n_parts = 32, n_Φ = 5, n_mh_steps = 1, verbose = :none,
                    smc_seed = 73, save_intermediate = true,
                    intermediate_stage_increment = 2,
                    savepath = path, particle_store_path = replace(path, ".jld2" => ".h5"),
                    continue_intermediate = resume, loadpath = loadpath, options...)
            return load(path)
        end

        base = run("base")
        full = run("full"; diagnostic = true)
        @test full["cloud"].particles == base["cloud"].particles
        @test full["stage_health"]["c"] == base["stage_health"]["c"]
        reports = full["stage_health"]["esjd_diagnostics"]
        @test [r["stage"] for r in reports] == [2, 4]
        for report in reports
            @test length(report["scales"]) == 3
            @test report["current_c"] in report["scales"]
            @test report["best_c"] in report["scales"]
            @test all(0 .<= report["acceptance"] .<= 1)
            @test all(report["esjd"] .>= 0)
            if maximum(report["esjd"]) > 0
                @test report["efficiency"] ≈
                    report["esjd"][findfirst(==(report["current_c"]), report["scales"])] /
                    maximum(report["esjd"])
            else
                @test isnan(report["efficiency"])
            end
            snapshot = load(joinpath(dir, "full_esjd_stage=$(report["stage"]).jld2"))
            @test size(snapshot["particles"], 1) == 32
            @test length(snapshot["weights"]) == 32
            @test snapshot["phi"] == report["phi"]
            @test snapshot["current_c"] == report["current_c"]
            @test size(snapshot["proposal_covariance"]) == (7, 7)
        end
        csv = readlines(joinpath(dir, "full_esjd_diagnostic.csv"))
        @test length(csv) == 1 + 3 * length(reports)
        @test startswith(csv[1], "stage,phi,trial_c,acceptance,esjd")

        resumed = run("resumed"; diagnostic = true, resume = true,
                      loadpath = joinpath(dir, "full_stage=2.jld2"))
        @test resumed["cloud"].particles == full["cloud"].particles
        @test resumed["stage_health"]["esjd_diagnostics"] == reports
        @test_throws ArgumentError run("bad_frequency"; diagnostic = true,
                                       esjd_diagnostic_every = 0)
        @test_throws ArgumentError run("bad_scales"; diagnostic = true,
                                       esjd_diagnostic_scales = [0.1, -0.2])
    end
end
