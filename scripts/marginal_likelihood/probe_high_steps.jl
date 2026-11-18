using Pkg
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(ROOT)
using SMC, ModelConstructors, HDF5, JLD2, Distributions, LinearAlgebra, Statistics
include(joinpath(ROOT, "test", "modelsetup.jl"))
include(joinpath(@__DIR__, "reference.jl"))

data = h5read(joinpath(ROOT, "test", "reference", "test_data.h5"), "data")
truth = exact_log_mdd(data)
outdir = joinpath(@__DIR__, "results", "high_steps")
mkpath(outdir)
for cfg in [(name = "blocked_adaptive_2_to_30", steps = 2, adaptive = true),
            (name = "blocked_fixed_10", steps = 10, adaptive = false)]
    name = "$(cfg.name)_seed=11"
    elapsed = @elapsed SMC.smc(loglik_fn, setup_linear_model().parameters, data;
        verbose = :none, n_parts = 1_000, n_mh_steps = cfg.steps,
        adaptive_mh_steps = cfg.adaptive, max_mh_steps = 30,
        param_blocks = [[:α1, :β1, :σ1], [:α2, :β2, :σ2], [:α3, :β3, :σ3]],
        blockwise_c = true, use_fixed_schedule = false, tempering_target = 0.95,
        α = 0.9, smc_seed = 11,
        savepath = joinpath(outdir, name * ".jld2"),
        particle_store_path = joinpath(outdir, name * ".h5"))
    out = JLD2.load(joinpath(outdir, name * ".jld2"))
    health = out["stage_health"]
    estimate = out["log_mdd_estimates"]["smc_level"]
    line = join((cfg.name, truth, estimate, estimate - truth,
                 round(Int, sum(health["n_proposals"])), elapsed), ",")
    open(joinpath(outdir, "probe.csv"), "a") do io
        println(io, line)
    end
    println(line)
    flush(stdout)
end
