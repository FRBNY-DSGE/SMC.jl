using Pkg
const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
Pkg.activate(ROOT)

using SMC, ModelConstructors, HDF5, JLD2, Distributions, LinearAlgebra, Statistics
include(joinpath(ROOT, "test", "modelsetup.jl"))
include(joinpath(@__DIR__, "reference.jl"))

data = h5read(joinpath(ROOT, "test", "reference", "test_data.h5"), "data")
truth = exact_log_mdd(data)
seeds = isempty(ARGS) ? collect(11:16) : parse.(Int, split(ARGS[1], ","))
variant = get(ENV, "SMC_BENCH_VARIANT", "standard")
configs = if variant == "blocks"
    [(name = "blocked_2", steps = 2, adaptive = false, blocks = true),
     (name = "blocked_adaptive_2_to_10", steps = 2, adaptive = true, blocks = true)]
else
    [(name = "fixed_2", steps = 2, adaptive = false, blocks = false),
     (name = "adaptive_2_to_10", steps = 2, adaptive = true, blocks = false),
     (name = "fixed_8", steps = 8, adaptive = false, blocks = false)]
end
outdir = variant == "blocks" ? joinpath(@__DIR__, "results", "blocks") :
    joinpath(@__DIR__, "results")
mkpath(outdir)
rows = NamedTuple[]
csv = joinpath(outdir, "runs.csv")
open(csv, "w") do io
    println(io, "method,seed,truth,log_mdd,error,abs_error,proposals,stages,mean_distinct,seconds")
end
for seed in seeds, cfg in configs
    name = "$(cfg.name)_seed=$(seed)"
    path = joinpath(outdir, name * ".jld2")
    elapsed = @elapsed SMC.smc(loglik_fn, setup_linear_model().parameters, data;
        verbose = :none, n_parts = 1_000, n_mh_steps = cfg.steps,
        adaptive_mh_steps = cfg.adaptive, max_mh_steps = 10,
        param_blocks = cfg.blocks ? [[:α1, :β1, :σ1], [:α2, :β2, :σ2],
                                     [:α3, :β3, :σ3]] : [],
        blockwise_c = cfg.blocks,
        use_fixed_schedule = false, tempering_target = 0.95, α = 0.9,
        smc_seed = seed, savepath = path,
        particle_store_path = joinpath(outdir, name * ".h5"))
    out = load(path)
    health = out["stage_health"]
    log_mdd = out["log_mdd_estimates"]["smc_level"]
    error = log_mdd - truth
    row = (method = cfg.name, seed = seed, truth = truth, log_mdd = log_mdd,
           error = error, abs_error = abs(error),
           proposals = round(Int, sum(health["n_proposals"])),
           stages = length(health["stage"]),
           mean_distinct = mean(health["distinct_mutated"]), seconds = elapsed)
    push!(rows, row)
    open(csv, "a") do io
        println(io, join(Tuple(row), ","))
    end
    println("Completed $name: error=$(round(error, digits = 3)), proposals=$(row.proposals)")
    flush(stdout)
end

summary = joinpath(outdir, "summary.csv")
open(summary, "w") do io
    println(io, "method,n,median_abs_error,rmse_log_mdd,mean_proposals,mean_seconds")
    for cfg in configs
        r = filter(row -> row.method == cfg.name, rows)
        isempty(r) && continue
        println(io, join((cfg.name, length(r), median(getproperty.(r, :abs_error)),
                          sqrt(mean(getproperty.(r, :error) .^ 2)),
                          mean(getproperty.(r, :proposals)),
                          mean(getproperty.(r, :seconds))), ","))
    end
end
println(read(summary, String))
