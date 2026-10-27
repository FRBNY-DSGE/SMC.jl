using Test, Distributed, SMC, ModelConstructors, Distributions, JLD2, LinearAlgebra
@everywhere using SMC, ModelConstructors, Distributions
BLAS.set_num_threads(1)

@everywhere function issue9_test_likelihood(p, data)
    a, b = p[1].value, p[2].value
    return -0.5 * ((a - 0.25)^2 + (b + 0.4)^2)
end

function issue9_test_parameters()
    m = GenericModel()
    m <= parameter(:a, 0., (-10., 10.), (-10., 10.), Untransformed(), Normal(0, 1), fixed=false)
    m <= parameter(:b, 0., (-10., 10.), (-10., 10.), Untransformed(), Normal(0, 1), fixed=false)
    return m.parameters
end

mktempdir() do dir
    init_serial = SMC.Cloud(2, 8)
    init_parallel = SMC.Cloud(2, 8)
    init_serial.particles .= 0.0
    init_parallel.particles .= 0.0
    SMC.initial_draw!(issue9_test_likelihood, issue9_test_parameters(), zeros(1, 1),
                      init_serial; smc_seed=42)
    SMC.initial_draw!(issue9_test_likelihood, issue9_test_parameters(), zeros(1, 1),
                      init_parallel; smc_seed=42, parallel=nworkers()>1)
    @test init_serial.particles == init_parallel.particles

    function run(name; seed=nothing, parallel=nworkers()>1, resume=false, loadpath="", save_intermediate=false,
                 fixed=true, method=:multinomial, n_parts=32, old_data=zeros(1, 0),
                 old_cloud=SMC.Cloud(0, 0), weight=0.0, threshold=0.999,
                 log_prob_old_data=nothing)
        path = joinpath(dir, name * ".jld2")
        SMC.smc(issue9_test_likelihood, issue9_test_parameters(), zeros(1, 1);
                n_parts=n_parts, n_Φ=5, n_blocks=2, n_mh_steps=1,
                parallel=parallel, verbose=:none, smc_seed=seed,
                use_fixed_schedule=fixed, tempering_target=0.9,
                resampling_method=method, threshold_ratio=threshold,
                old_data=old_data, old_cloud=old_cloud,
                tempered_update_prior_weight=weight,
                log_prob_old_data=log_prob_old_data, savepath=path,
                particle_store_path=replace(path, ".jld2"=>".h5"),
                continue_intermediate=resume, loadpath=loadpath,
                save_intermediate=save_intermediate, intermediate_stage_increment=2)
        return load(path)
    end
    first = run("first"; seed=42, save_intermediate=true)
    second = run("second"; seed=42)
    checkpoint = joinpath(dir, "first_stage=2.jld2")
    resumed = run("resumed"; resume=true, loadpath=checkpoint)
    changed = run("changed"; seed=43)
    adaptive1 = run("adaptive1"; seed=42, fixed=false, threshold=0.9)
    adaptive2 = run("adaptive2"; seed=42, fixed=false, threshold=0.9)
    bridge_data = zeros(1, 1)
    first_w, first_W = first["w"], first["W"]
    first_mdd = sum(log.(vec(sum(first_w[:, 2:end] .* first_W[:, 1:end-1] ./
                                  length(first["cloud"]), dims = 1))))
    bridge1 = run("bridge1"; seed=42, method=:systematic, n_parts=40,
                  old_data=bridge_data, old_cloud=deepcopy(first["cloud"]),
                  weight=0.25, threshold=0.9, log_prob_old_data=first_mdd)
    bridge2 = run("bridge2"; seed=42, method=:systematic, n_parts=40,
                  old_data=bridge_data, old_cloud=deepcopy(first["cloud"]),
                  weight=0.25, threshold=0.9, log_prob_old_data=first_mdd)
    @testset "Seeded end-to-end SMC" begin
        @test first["smc_seed"] == UInt64(42)
        @test first["smc_seed_version"] == SMC.SMC_SEED_VERSION
        @test load(checkpoint, "smc_seed") == UInt64(42)
        @test load(checkpoint, "smc_seed_version") == SMC.SMC_SEED_VERSION
        for other in (second, resumed)
            @test first["cloud"].particles == other["cloud"].particles
            @test first["cloud"].ESS == other["cloud"].ESS
            @test first["cloud"].tempering_schedule == other["cloud"].tempering_schedule
            @test first["cloud"].resamples == other["cloud"].resamples
            @test first["cloud"].accept == other["cloud"].accept
            @test first["w"] == other["w"]
            @test first["W"] == other["W"]
        end
        for (a, b) in ((adaptive1, adaptive2), (bridge1, bridge2))
            @test a["cloud"].particles == b["cloud"].particles
            @test a["cloud"].ESS == b["cloud"].ESS
            @test a["cloud"].tempering_schedule == b["cloud"].tempering_schedule
            @test a["w"] == b["w"]
            @test a["W"] == b["W"]
        end
        @test length(bridge1["cloud"]) == 40
        @test first["cloud"].particles != changed["cloud"].particles
        @test_throws ArgumentError run("bad_seed"; seed=43, resume=true, loadpath=checkpoint)
        if nworkers() > 1
            serial = run("serial"; seed=42, parallel=false)
            @test first["cloud"].particles == serial["cloud"].particles
            @test first["w"] == serial["w"]
            @test first["W"] == serial["W"]
        end
    end
end
