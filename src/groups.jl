# Independent particle groups (Durham and Geweke 2014; reliability plan item 1)
#
# smc(...; n_smc_groups = J) splits the N particles into J groups of N/J. Each group is a
# complete SMC run with its own seed (stream :group), schedule and proposal adaptation, saved
# next to `savepath` with a `_group=g` suffix. The pooled answer and its numerical standard
# errors (NSE = sd across groups / √J) are saved at `savepath`.

function group_path(path::AbstractString, g::Integer)
    base, ext = splitext(path)
    return "$(base)_group=$(g)$(ext)"
end

# The old cloud(s) a bridge starts each group from: a vector gives one per group, and groups
# of a grouped old run are found next to `loadpath`. A single old cloud (passed, or found at
# `loadpath`) is split into J disjoint random parts, so the groups don't share old particles
# and each bridges from N_old/J particles, like a group of a grouped old run. The parts still
# share the old run's Monte Carlo error, which the group NSEs leave out: a grouped old run
# avoids that.
function group_old_clouds(old_cloud, loadpath::AbstractString, n_groups::Int, tempered_update::Bool;
                          run_seed::UInt64 = UInt64(0), verbose::Symbol = :low)
    tempered_update || return fill(Cloud(0, 0), n_groups)
    if old_cloud isa AbstractVector
        length(old_cloud) == n_groups ||
            throw(ArgumentError("old_cloud has $(length(old_cloud)) clouds for $n_groups groups"))
        return collect(Cloud, old_cloud)
    elseif cloud_isempty(old_cloud) && all(g -> isfile(group_path(loadpath, g)), 1:n_groups)
        return [load(group_path(loadpath, g), "cloud") for g in 1:n_groups]
    end
    shared = cloud_isempty(old_cloud) ? load(loadpath, "cloud") : old_cloud
    verbose == :none ||
        @warn "Bridging $n_groups particle groups from one old cloud: each group gets a disjoint " *
              "part of it, but the group NSEs leave out the old run's Monte Carlo error"
    return split_cloud(shared, n_groups, Xoshiro(stream_seed(run_seed, :group, 0, 0)))
end

# J disjoint random parts of a cloud, each with ⌊N/J⌋ particles and the cloud's weights
function split_cloud(cloud::Cloud, n_groups::Int, rng::AbstractRNG)
    n_parts = size(cloud.particles, 1)
    n_each  = n_parts ÷ n_groups
    n_each > 0 || throw(ArgumentError("the old cloud has $n_parts particles for $n_groups groups"))
    order = randperm(rng, n_parts)
    return map(1:n_groups) do g
        part = deepcopy(cloud)
        part.particles = cloud.particles[order[(g-1)*n_each+1:g*n_each], :]
        part
    end
end

# Where to pick group g up on a resume: its finished output, its latest stage checkpoint, or
# a fresh start. Files from another seed are ignored.
function group_resume_point(path::AbstractString, seed::UInt64)
    same_seed(file) = "smc_seed" in JLD2.jldopen(keys, file, "r") && load(file, "smc_seed") == seed
    if isfile(path) && same_seed(path) && "log_mdd_estimates" in JLD2.jldopen(keys, path, "r")
        return (:done, path)
    end
    dir    = dirname(path) == "" ? "." : dirname(path)
    prefix = basename(splitext(path)[1]) * "_stage="
    stages = Int[]
    for file in readdir(dir)
        if startswith(file, prefix) && endswith(file, ".jld2")
            stage = tryparse(Int, file[length(prefix)+1:end-5])
            stage === nothing || push!(stages, stage)
        end
    end
    for stage in sort(stages; rev = true)
        file = joinpath(dir, prefix * "$(stage).jld2")
        same_seed(file) && return (:stage, file)
    end
    return (:fresh, "")
end

function smc_groups(loglikelihood::Function, parameters::ParameterVector, data::Matrix;
                    n_smc_groups::Int, smc_seed::Union{Nothing, Integer} = nothing,
                    n_parts::Int = 5_000, savepath::String = "smc_cloud.jld2",
                    particle_store_path::String = "smcsave.h5", loadpath::String = "",
                    old_cloud = Cloud(0, 0), old_data::Matrix = Matrix{Float64}(undef, size(data, 1), 0),
                    continue_intermediate::Bool = false, testing::Bool = false,
                    verbose::Symbol = :low,
                    run_info::AbstractDict{<:AbstractString, <:AbstractString} = Dict{String, String}(),
                    kwargs...)
    J = n_smc_groups
    n_parts % J == 0 || throw(ArgumentError("n_parts = $n_parts is not divisible by n_smc_groups = $J"))
    smc_seed === nothing &&
        throw(ArgumentError("resuming a run with n_smc_groups > 1 needs its smc_seed"))
    run_seed = checked_smc_seed(smc_seed)
    old_clouds = group_old_clouds(old_cloud, loadpath, J, !isempty(old_data);
                                  run_seed = run_seed, verbose = verbose)

    results = map(1:J) do g
        seed = stream_seed(run_seed, :group, 0, g)
        path = group_path(savepath, g)
        state, file = continue_intermediate ? group_resume_point(path, seed) : (:fresh, "")
        println(verbose, :low, "\n SMC particle group $g of $J ($state)\n")
        state == :done && return load(path)
        info = merge(Dict{String, String}(run_info),
                     Dict("smc_group" => "$g of $J", "smc_group_run_seed" => string(run_seed)))
        _smc(loglikelihood, parameters, data; kwargs..., n_parts = n_parts ÷ J, smc_seed = seed,
             savepath = path, particle_store_path = group_path(particle_store_path, g),
             old_cloud = old_clouds[g], old_data = old_data, loadpath = state == :stage ? file : loadpath,
             continue_intermediate = state == :stage, testing = testing, verbose = verbose,
             run_info = info)
    end

    n_para = ind_para_end(size(results[1]["cloud"].particles, 2))
    free_para_inds = ModelConstructors.get_free_para_inds(parameters;
                         regime_switching = get(kwargs, :regime_switching, false),
                         toggle = get(kwargs, :toggle, true))
    para_names = [θ.key for θ in parameters]
    para_names = length(para_names) == n_para ? para_names[free_para_inds] : Symbol.("θ" .* string.(free_para_inds))
    cloud, summary = pool_groups(results, free_para_inds; para_names = para_names)
    summary["group_seeds"] = [stream_seed(run_seed, :group, 0, g) for g in 1:J]

    if verbose != :none
        for flag in summary["flags"]
            @warn "SMC health: " * flag
        end
    end
    if !testing
        h5open(particle_store_path, "w") do file
            write(file, "smcparams", cloud.particles[:, 1:n_para])
        end
        jldopen(savepath, true, true, true, IOStream) do file
            write(file, "cloud", cloud)
            write(file, "smc_groups", summary)
            write(file, "log_mdd_estimates", summary["log_mdd_estimates"])
            write(file, "health_flags", summary["flags"])
            write(file, "n_smc_groups", J)
            write(file, "smc_seed", run_seed)
            write(file, "smc_seed_version", SMC_SEED_VERSION)
            write(file, "run_info", results[1]["run_info"])
        end
        write_group_summary(replace(savepath, r"\.jld2$" => "") * "_group_summary.csv", summary)
    end
    return cloud
end

"""
```
pool_groups(results, free_para_inds; para_names, quantiles = (0.05, 0.5, 0.95))
```
Pool J independent SMC runs (the result dicts `smc` saves) as in Durham and Geweke (2014):
- the pooled cloud stacks the groups, each group's weights normalized to its size, so the
  groups count equally;
- posterior means are the average of the group means, with NSE = sd across groups / √J and
  RNE = posterior variance / (N · NSE²);
- quantiles come from the pooled cloud, their NSEs from the group quantiles;
- each log MDD estimate is pooled on the level scale, log((1/J) Σ_j p̂_j), with
  NSE ≈ sd_j(p̂_j / p̂) / √J (delta method).
Returns `(cloud, summary::Dict)`. `summary["flags"]` collects the groups' health flags and
the MDD gap flag on the pooled estimates.
"""
function pool_groups(results::AbstractVector, free_para_inds::AbstractVector{<:Integer};
                     para_names = nothing, quantiles = (0.05, 0.5, 0.95))
    J = length(results)
    clouds = [r["cloud"] for r in results]
    N = sum(length, clouds)

    particles = reduce(vcat, [c.particles for c in clouds])
    col = ind_weight(size(particles, 2))
    offset = 0
    for c in clouds
        particles[offset .+ (1:length(c)), col] .= get_weights(c) .* (length(c) / sum(get_weights(c)))
        offset += length(c)
    end
    W = particles[:, col]
    cloud = Cloud(particles, clouds[1].tempering_schedule, [N^2 / sum(W .^ 2)],
                  maximum(c.stage_index for c in clouds), clouds[1].n_Φ,
                  sum(c.resamples for c in clouds), mean(c.c for c in clouds),
                  mean(c.accept for c in clouds), sum(c.total_sampling_time for c in clouds))

    θ = particles[:, free_para_inds]
    group_means = reduce(vcat, [weighted_mean_cov(c.particles[:, free_para_inds], get_weights(c))[1]'
                                for c in clouds])
    post_var = diag(weighted_mean_cov(θ, W)[2])
    summary = Dict{String, Any}("n_smc_groups" => J, "n_parts" => N,
                                "parameters" => para_names === nothing ? collect(free_para_inds) : para_names,
                                "mean" => vec(mean(group_means, dims = 1)),
                                "mean_nse" => vec(std(group_means, dims = 1)) ./ sqrt(J))
    summary["rne"] = post_var ./ (N .* summary["mean_nse"] .^ 2)
    for q in quantiles
        key = "q" * lpad(round(Int, 100q), 2, "0")
        summary[key] = [quantile(θ[:, k], Weights(W), q) for k in axes(θ, 2)]
        group_q = [quantile(c.particles[:, free_para_inds[k]], Weights(get_weights(c)), q)
                   for c in clouds, k in axes(θ, 2)]
        summary[key * "_nse"] = vec(std(group_q, dims = 1)) ./ sqrt(J)
    end

    estimates = Dict{String, Any}()
    for key in ("smc", "smc_level", "mhm", "bridge_sampling")
        all(r -> haskey(r["log_mdd_estimates"], key), results) || continue
        x = [r["log_mdd_estimates"][key] for r in results]
        all(isfinite, x) || continue
        pooled = logmeanexp(x)
        estimates[key] = pooled
        estimates[key * "_nse"] = std(exp.(x .- pooled)) / sqrt(J)
        estimates[key * "_groups"] = x
    end
    summary["log_mdd_estimates"] = estimates

    flags = String[]
    for (g, r) in enumerate(results), flag in r["health_flags"]
        push!(flags, "group $g: " * flag)
    end
    gap = mdd_gap_flag(estimates)
    gap === nothing || push!(flags, gap)
    summary["flags"] = flags
    return cloud, summary
end

function write_group_summary(path::AbstractString, summary::AbstractDict)
    q_keys = sort([k for k in keys(summary) if occursin(r"^q\d+$", k)])
    open(path, "w") do io
        println(io, join(["parameter", "mean", "mean_nse", "rne",
                          vcat([[k, k * "_nse"] for k in q_keys]...)...], ","))
        for (k, name) in enumerate(summary["parameters"])
            row = [string(name), string(summary["mean"][k]), string(summary["mean_nse"][k]),
                   string(summary["rne"][k])]
            for q in q_keys
                push!(row, string(summary[q][k]), string(summary[q * "_nse"][k]))
            end
            println(io, join(row, ","))
        end
        est = summary["log_mdd_estimates"]
        for key in ("smc", "smc_level", "mhm", "bridge_sampling")
            haskey(est, key) &&
                println(io, join(["log_mdd_" * key, est[key], est[key * "_nse"], "",
                                  fill("", 2 * length(q_keys))...], ","))
        end
    end
    return path
end
