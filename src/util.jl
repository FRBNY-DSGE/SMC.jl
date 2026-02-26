"""
```
function scalar_reduce(args...)
```
Each individual iteration returns n scalars. The output is reduced to n vectors,
where the i-th vector contains all of the i-th scalars from each iteration.

The return type of reduce functions must be the same type as the tuple of
arguments passed in. If args is a tuple of Vector{Float64}, then the return
argument will be a Vector{Float64}.

e.g.
a, b = @parallel (scalar_reduce) for i in 1:10000
           [[1], [2]]
       end
a = [1, 1, 1, ...]
b = [2, 2, 2, ...]

Input/Output type: Vector{Vector{Float64}}
"""
function scalar_reduce(args...)
    return_arg = args[1]
    for (i, arg) in enumerate(args[2:end])
        for (j, el) in enumerate(arg)
            append!(return_arg[j], el)
        end
    end
    return return_arg
end

"""
```
function vector_reduce(args...)
```
Each individual iteration returns n Vector types; we vector-reduce to n matrices, where
the i-th column of that matrix corresponds to the i-th vector from an individual iteration.

Input/Output type: Vector{Matrix{Float64}}
"""
function vector_reduce(args...)
    nargs1 = length(args)    # The number of times the loop is run
    nargs2 = length(args[1]) # The number of variables output by a single run

    return_arg = args[1]
    for i in 1:nargs2
        for j in 2:nargs1
            return_arg[i] = hcat(return_arg[i], args[j][i])
        end
    end
    return return_arg
end

"""
```
function scalar_reshape(args...)
```
Function ensures type conformity of the return arguments.
"""
function scalar_reshape(args...)
    n_args = length(args)
    return_arg = Vector{Vector{Float64}}(undef, n_args)
    for i in 1:n_args
        arg = typeof(args[i]) <: Vector ? args[i] : [args[i]]
        return_arg[i] = arg
    end
    return return_arg
end

"""
```
function vector_reshape(args...)
```
Function ensures type conformity of the return arguments.
"""
function vector_reshape(args...)
    n_args = length(args)
    return_arg = Vector{Matrix{Float64}}(undef, n_args)
    for i in 1:n_args
        arg = typeof(args[i]) <: Vector ? args[i] : [args[i]]
        return_arg[i] = reshape(arg, length(arg), 1)
    end
    return return_arg
end

"""
```
sendto(p::Int; args...)
```
Function to send data from master process to particular worker, p.
Code from ChrisRackauckas, avavailable at:
 https://github.com/ChrisRackauckas/ParallelDataTransfer.jl/blob/master/src/ParallelDataTransfer.jl.
"""
function sendto(p::Int; args...)
    for (nm, val) in args
        @spawnat(p, Core.eval(Main, Expr(:(=), nm, val)))
    end
end

"""
```
sendto(ps::AbstractVector{Int}; args...)
```
Function to send data from master process to list of workers.
Code from ChrisRackauckas, available at:
https://github.com/ChrisRackauckas/ParallelDataTransfer.jl/blob/master/src/ParallelDataTransfer.jl.
"""
function sendto(ps::AbstractVector{Int}; args...)
    for p in ps
        sendto(p; args...)
    end
end

function get_cloud(filepath::String)
    return load(filepath, "cloud")
end

function init_stage_print(cloud::Cloud, para_symbols::Vector{Symbol};
                          verbose::Symbol=:low, use_fixed_schedule::Bool = true)
    if VERBOSITY[verbose] >= VERBOSITY[:low]
        if use_fixed_schedule
            println("--------------------------")
            println("Iteration = $(cloud.stage_index) / $(cloud.n_Φ)")
        else
            println("--------------------------")
            println("Iteration = $(cloud.stage_index)")
        end
	    println("--------------------------")
        println("phi = $(cloud.tempering_schedule[cloud.stage_index])")
	    println("--------------------------")
        println("c = $(cloud.c)")
        println("ESS = $(cloud.ESS[cloud.stage_index])   ($(cloud.resamples) total resamples.)")
	    println("--------------------------")
    end
    if VERBOSITY[verbose] >= VERBOSITY[:high]
        μ = weighted_mean(cloud)
        σ = weighted_std(cloud)
        println("Mean and standard deviation of parameter estimates")
        for n = 1:length(para_symbols)
            println("$(para_symbols[n]) = $(round(μ[n], digits = 5)), $(round(σ[n], digits = 5))")
	    end
    end
end

function end_stage_print(cloud::Cloud, para_symbols::Vector{Symbol};
                         verbose::Symbol=:low, use_fixed_schedule::Bool = true)
    if VERBOSITY[verbose] >= VERBOSITY[:low]
        total_sampling_time_minutes = cloud.total_sampling_time/60
        if use_fixed_schedule
            expected_time_remaining_sec = (cloud.total_sampling_time/cloud.stage_index) *
                (cloud.n_Φ - cloud.stage_index)
            expected_time_remaining_minutes = expected_time_remaining_sec / 60
        end

        println("--------------------------")
        if use_fixed_schedule
            println("Iteration = $(cloud.stage_index) / $(cloud.n_Φ)")
            println("time elapsed: $(round(total_sampling_time_minutes, digits = 4)) minutes")
            println("estimated time remaining: " *
                    "$(round(expected_time_remaining_minutes, digits = 4)) minutes")
        else
            println("Iteration = $(cloud.stage_index)")
            println("time elapsed: $(round(total_sampling_time_minutes, digits = 4)) minutes")
        end
        println("--------------------------")
        println("phi = $(cloud.tempering_schedule[cloud.stage_index])")
        println("--------------------------")
        println("c = $(cloud.c)")
        println("accept = $(cloud.accept)")
        println("ESS = $(cloud.ESS[cloud.stage_index])   ($(cloud.resamples) total resamples.)")
        println("--------------------------")
    end
    if VERBOSITY[verbose] >= VERBOSITY[:high]
        μ = weighted_mean(cloud)
        σ = weighted_std(cloud)
        println("Mean and standard deviation of parameter estimates")
        for n=1:length(para_symbols)
            println("$(para_symbols[n]) = $(round(μ[n], digits = 5)), $(round(σ[n], digits = 5))")
        end
    end
end

"""
    <(a::Complex, b::Complex)

Compare real values of complex numbers.
"""
function <(a::Complex, b::Complex)
    return a.re < b.re
end

"""
    <(a::Real, b::Complex)

Compare real values of complex numbers.
"""
function <(a::Real, b::Complex)
    return a < b.re
end

"""
    <(a::Complex, b::Real)

Compare real values of complex numbers.
"""
function <(a::Complex, b::Real)
    return a.re < b
end

function min(a::Complex, b::Real)
    return min(a.re, b)
end

function min(a::Complex, b::Complex)
    return min(a.re, b.re)
end

function min(a::Real, b::Complex)
    return min(a, b.re)
end

function max(a::Complex, b::Real)
    return max(a.re, b)
end

function max(a::Complex, b::Complex)
    return max(a.re, b.re)
end

function max(a::Real, b::Complex)
    return max(a, b.re)
end

function isempty(c::Cloud)
    isempty(c.particles)
end

function sample_RootInverseGamma(d; n = 1000000, return_mean = false, return_std = false)
    # Try to obtain draws from RootInverseGamma
    draws = []
    for i = 1:n
        push!(draws, sqrt(d.ν * d.τ^2 / sum(randn(round(Int, d.ν)).^2)))
    end
    return draws
end

#=
function get_prior_covariance(parameters; regime_switching = false, max_regimes = 5) #arbitrarily large
    if regime_switching
        free_priors = []
        regime_priors = Dict(i => [] for i in 1:max_regimes)

        for p in parameters
            # Case where parameters have only one regime and unfixed
            if !haskey(p.regimes, :prior) && !p.fixed
                if isa(p.prior.value, RootInverseGamma)

                push!(regime_priors[1], std(p.prior.value)^2)

            # Case where parameters have more than one regime, and at least one of which is estimated
            elseif haskey(p.regimes, :prior)
                n_regimes = length(p.regimes[:value])
                for reg_ind in 1:n_regimes
                    if p.regimes[:fixed][reg_ind] == false
                        push!(regime_priors[reg_ind], std(p.regimes[:prior][reg_ind].value)^2)
                    end
                end
            end
        end

        # Now want to concatenate in order of regime (all regime 1 free param priors, then regime 2...)
        for i in 1:max_regimes
            append!(free_priors, regime_priors[i])
        end

        prior_cov = diagm(Float64.(free_priors))

    else
        free_para_inds = ModelConstructors.get_free_para_inds(parameters, regime_switching = false, toggle = true)
        free_params = parameters[free_para_inds]
        free_priors = [std(p.prior.value)^2 for p in free_params]
        prior_cov = diagm(Float64.(free_priors))
    end
    return prior_cov
end
=#

# Idea here is that we want to draw from the bounded prior distribution (whose covariance structure is different from the unbounded prior

function get_prior_covariance(parameters; regime_switching = false, n_draws = 1e6)

    # Obtain priors for free variables only
    free_para_inds = ModelConstructors.get_free_para_inds(parameters; regime_switching = regime_switching)

    # Obtain draws
    draws = []

    if regime_switching
        for n = 1:n_draws
            draw = ModelConstructors.rand_regime_switching(parameters)
            push!(draws, draw)
        end
    else
        for n = 1:n_draws
            draw = zeros(length(parameters))
            for (i, para) in enumerate(parameters)
                draw[i] = if para.fixed
                    para.value
                else
                    # Resample until all prior draws are within the value bounds
                    prio = rand(para.prior.value)
                    while !(para.valuebounds[1] < prio < para.valuebounds[2])
                        prio = rand(para.prior.value)
                    end
                    prio
                end
            end
            push!(draws, draw)
        end
    end

    draw_matrix = reduce(vcat, transpose.(draws))
    free_matrix = draw_matrix[free_para_inds, free_para_inds]

    return cov(free_matrix)
end
