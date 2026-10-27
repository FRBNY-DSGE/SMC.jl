using ModelConstructors, HDF5, Random, JLD2, FileIO, SMC, Test, Distributed
include("modelsetup.jl")

# A run with a given smc_seed must be bitwise identical serially and with workers, must resume
# from a stage checkpoint bitwise, and must not depend on (or disturb) the caller's RNG beyond
# drawing the default seed from it.

repro_data = h5read("reference/test_data.h5", "data")
repro_dir  = mktempdir()

# Adaptive tempering with 2 blocks, so the run uses every per-stage stream (resampling, blocks,
# mutation) and the adaptive ϕ root-solve that a resume has to reproduce
function repro_smc(name::String; parallel::Bool = false, kwargs...)
    savepath = joinpath(repro_dir, name * ".jld2")
    SMC.smc(loglik_fn, setup_linear_model().parameters, repro_data; verbose = :none,
            parallel = parallel, n_parts = 400, n_blocks = 2, n_mh_steps = 1, n_Φ = 100,
            use_fixed_schedule = false, tempering_target = 0.95, resampling_method = :systematic,
            α = .9, target = 0.25, threshold_ratio = .5, smc_iteration = 0, savepath = savepath,
            particle_store_path = joinpath(repro_dir, name * ".h5"), kwargs...)
    return load(savepath)
end

function same_run(a, b)
    ca, cb = a["cloud"], b["cloud"]
    return isequal(ca.particles, cb.particles) && isequal(a["w"], b["w"]) && isequal(a["W"], b["W"]) &&
        isequal(ca.tempering_schedule, cb.tempering_schedule) && isequal(ca.ESS, cb.ESS) &&
        ca.stage_index == cb.stage_index && ca.resamples == cb.resamples &&
        ca.c == cb.c && ca.accept == cb.accept && a["smc_seed"] == b["smc_seed"]
end

serial = repro_smc("serial"; smc_seed = 7, save_intermediate = true, intermediate_stage_increment = 2)

@testset "Same smc_seed: serial and 2 workers are bitwise identical" begin
    # A batch runner may already have reserved workers. SMC uses all available
    # workers, so initialize each of them and remove only workers added here.
    added_workers = length(workers()) < 2 ?
        Distributed.addprocs(2 - length(workers()); exeflags = "--project=$(Base.active_project())") : Int[]
    run_workers = workers()
    try
        @everywhere run_workers begin
            using SMC, ModelConstructors
            include(joinpath($(@__DIR__), "modelsetup.jl"))
        end
        @test same_run(serial, repro_smc("parallel"; parallel = true, smc_seed = 7))
    finally
        isempty(added_workers) || Distributed.rmprocs(added_workers)
    end
    @test serial["smc_seed"] == 7
    @test serial["smc_seed_version"] == SMC.SMC_SEED_VERSION
    @test serial["run_info"]["julia_version"] == string(VERSION)
    @test haskey(serial["run_info"], "smc_commit")
end

@testset "Default seed: drawn from (and leaves alone) the caller's RNG" begin
    Random.seed!(1793)
    a = repro_smc("default_a")
    after_a = rand()
    Random.seed!(1793)
    expected_seed = rand(UInt64)
    expected_after = rand()
    Random.seed!(1793)
    b = repro_smc("default_b")
    @test same_run(a, b)
    @test a["smc_seed"] == expected_seed
    # smc took exactly one draw (the seed) from the caller's RNG
    @test after_a == expected_after
    @test !same_run(a, serial)
end

@testset "Resume from a stage checkpoint is bitwise" begin
    # Stages 2 and 6: resuming there with ϕ_prop = sched[j] (instead of the saved live value)
    # changes the adaptive ϕ root-solve, so the run drifts (by ~1e-10 here)
    @test serial["cloud"].stage_index > 6
    for stage in (2, 6)
        checkpoint = joinpath(repro_dir, "serial_stage=$stage.jld2")
        @test same_run(serial, repro_smc("resumed$stage"; continue_intermediate = true, loadpath = checkpoint))
    end
    checkpoint = joinpath(repro_dir, "serial_stage=2.jld2")
    @test same_run(serial, repro_smc("resumed_seed"; continue_intermediate = true, loadpath = checkpoint,
                                     smc_seed = 7))
    @test_throws ArgumentError repro_smc("resumed_wrong"; continue_intermediate = true,
                                         loadpath = checkpoint, smc_seed = 8)

    # A checkpoint written before the seeding (no smc_seed, ϕ_prop, ...) still resumes, with
    # a warning since it can't replay the original run
    old_checkpoint = joinpath(repro_dir, "old_stage=2.jld2")
    jldopen(old_checkpoint, true, true, true, IOStream) do file
        for key in ("cloud", "w", "W", "j")
            write(file, key, load(checkpoint, key))
        end
    end
    old_resume = @test_logs (:warn, r"no smc_seed") (:warn, r"no ϕ_prop") match_mode = :any begin
        repro_smc("resumed_old"; continue_intermediate = true, loadpath = old_checkpoint)
    end
    @test old_resume["cloud"].tempering_schedule[end] == 1.0
end

rm(repro_dir; recursive = true, force = true)
