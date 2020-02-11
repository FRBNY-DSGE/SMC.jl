"""
```
function smc(loglikelihood::Function, parameters::ParameterVector{U}, data::Matrix{S};
             kwargs...) where {S<:AbstractFloat, U<:Number}
```

### Arguments:

- `loglikelihood::Function`: Log-likelihood function of model being estimated. Takes `parameters`
    and `data` as arguments.
- `parameters::ParameterVector{U}`: Model parameter vector, which stores parameter values,
    prior dists, and bounds.
- `data`: A matrix or dataframe containing the time series of the observables used in
    the calculation of the posterior/loglikelihood
- `old_data`: A matrix containing the time series of observables of previous data
    (with `data` being the new data) for the purposes of a time tempered estimation
    (that is, using the posterior draws from a previous estimation as the initial set
    of draws for an estimation with new data)

### Keyword Arguments:
- `verbose::Symbol`: Desired frequency of function progress messages printed to standard out.
	- `:none`: No status updates will be reported.
	- `:low`: Status updates for SMC initialization and recursion will be included.
	- `:high`: Status updates for every iteration of SMC is output, which includes
    the mean and standard deviation of each parameter draw after each iteration,
    as well as calculated acceptance rate, ESS, and number of times resampled.

- `parallel::Bool`: Flag for running algorithm in parallel.
- `n_parts::Int`: Number of particles.
- `n_blocks::Int`: Number of parameter blocks in mutation step.
- `n_mh_steps::Int`: Number of Metropolis Hastings steps to attempt during the mutation step.
- `λ::S`: The 'bending coefficient' λ in Φ(n) = (n/N(Φ))^λ
- `n_Φ::Int`: Number of stages in the tempering schedule.
- `resampling_method::Symbol`: Which resampling method to use.
    - `:systematic`: Will use sytematic resampling.
    - `:multinomial`: Will use multinomial resampling.
    - `:polyalgo`: Samples using a polyalgorithm.
- `threshold_ratio::S`: Threshold s.t. particles will be resampled when the population
    drops below threshold * N.
- `c::S`: Scaling factor for covariance of the particles. Controls size of steps in mutation step.
- `α::S`: The mixture proportion for the mutation step's proposal distribution.
- `target::S`: The initial target acceptance rate for new particles during mutation.

- `use_chand_recursion::Bool`: Flag for using Chandrasekhar Recursions in Kalman filter.
- `use_fixed_schedule::Bool`: Flag for whether or not to use a fixed tempering (ϕ) schedule.
- `tempering_target::S`: Coefficient of the sample size metric to be targeted when solving
    for an endogenous ϕ.

- `old_data::Matrix{S}`: data from vintage of last SMC estimation.
- `old_cloud::Cloud`: associated cloud borne of old data in previous SMC estimation.
- `old_vintage::String`: String for vintage date of old data
- `smc_iteration::Int`: The iteration index for the number of times SMC has been run on the
     same data vintage. Primarily for numerical accuracy/testing purposes.

- `run_test::Bool`: Flag for when testing accuracy of program
- `filestring_addl::Vector{String}`: Additional file string extension for loading old cloud.
- `continue_intermediate::Bool`: Flag to indicate whether one is continuing SMC from an
    intermediate stage.
- `intermediate_stage_start::Int`: Intermediate stage at which one wishes to begin the estimation.
- `save_intermediate::Bool`: Flag for whether one wants to save intermediate Cloud objects
- `intermediate_stage_increment::Int`: Save Clouds at every increment
    (1 = each stage, 10 = every 10th stage, etc.)

### Outputs

- `cloud`: The Cloud object containing all of the information about the
    parameter values from the sample, their respective log-likelihoods, the ESS
    schedule, tempering schedule etc., which is saved in the saveroot.

### Overview

Sequential Monte Carlo can be used in lieu of Random Walk Metropolis Hastings to
    generate parameter samples from high-dimensional parameter spaces using
    sequentially constructed proposal densities to be used in iterative importance
    sampling.

This implementation is based on Edward Herbst and Frank Schorfheide's 2014 paper
    'Sequential Monte Carlo Sampling for DSGE Models' and the code accompanying their
    book 'Bayesian Estimation of DSGE Models'.

SMC is broken up into three main steps:

- `Correction`: Reweight the particles from stage n-1 by defining incremental weights,
    which gradually "temper in" the loglikelihood function p(Y|θ)^(ϕ_n - ϕ_n-1) into the
    normalized particle weights.
- `Selection`: Resample the particles if the distribution of particles begins to
    degenerate, according to a tolerance level for the ESS.
- `Mutation`: Propagate particles {θ(i), W(n)} via N(MH) steps of a Metropolis
    Hastings algorithm.
"""
function smc(loglikelihood::Function, parameters::ParameterVector{U}, data::Matrix{S};
             verbose::Symbol = :low,
             testing::Bool   = false,
             data_vintage::String = Dates.format(today(), "yymmdd"),

             parallel::Bool  = false,
             n_parts::Int    = 5_000,
             n_blocks::Int   = 1,
             n_mh_steps::Int = 1,

             λ::S = 2.1,
             n_Φ::Int64 = 300,

             resampling_method::Symbol = :systematic,
             threshold_ratio::S = 0.5,

             # Mutation Settings
             c::S = 0.5,       # step size
             α::S = 1.0,       # mixture proportion
             target::S = 0.25, # target accept rate

             use_fixed_schedule::Bool = true,
             tempering_target::S = 0.97,

             old_data::Matrix{S} = Matrix{S}(undef, size(data, 1), 0),
             old_cloud::Cloud = Cloud(0, 0),
             old_vintage::String = "",
             smc_iteration::Int = 1,

             run_test::Bool = false,
             filestring_addl::Vector{String} = Vector{String}(),
             loadpath::String = "",
             savepath::String = "smc_cloud.jld2",
             particle_store_path::String = "smcsave.h5",
             continue_intermediate::Bool = false,
             intermediate_stage_start::Int = 0,
             save_intermediate::Bool = false,
             intermediate_stage_increment::Int = 10) where {S<:AbstractFloat, U<:Number}

    ########################################################################################
    ### Settings
    ########################################################################################

    # Construct closure of mutation function so as to avoid issues with serialization
    # across workers with different Julia system images
    sendto(workers(), parameters = parameters)
    sendto(workers(), data = data)

    function mutation_closure(p::Vector{S}, d_μ::Vector{S}, d_Σ::Matrix{S},
                 blocks_free::Vector{Vector{Int64}}, blocks_all::Vector{Vector{Int64}},
                 ϕ_n::S, ϕ_n1::S; c::S = 1.0, α::S = 1.0, n_mh_steps::Int = 1,
                 old_data::T = Matrix{S}(undef, size(data, 1), 0)) where {S<:Float64, T<:Matrix}
        return mutation(loglikelihood, parameters, data, p, d_μ, d_Σ, blocks_free, blocks_all,
                        ϕ_n, ϕ_n1; c = c, α = α, n_mh_steps = n_mh_steps, old_data = old_data)
    end
    @everywhere function mutation_closure(p::Vector{S}, d_μ::Vector{S}, d_Σ::Matrix{S},
                 blocks_free::Vector{Vector{Int64}}, blocks_all::Vector{Vector{Int64}},
                 ϕ_n::S, ϕ_n1::S; c::S = 1.0, α::S = 1.0, n_mh_steps::Int = 1,
                 old_data::T = Matrix{S}(undef, size(data, 1), 0)) where {S<:Float64, T<:Matrix}
        return mutation(loglikelihood, parameters, data, p, d_μ, d_Σ, blocks_free, blocks_all,
                        ϕ_n, ϕ_n1; c = c, α = α, n_mh_steps = n_mh_steps, old_data = old_data)
    end

    # Check that if there's a tempered update, old and current vintages are different
    tempered_update = !isempty(old_data) # Time tempering
    @assert !(tempered_update&(old_vintage==data_vintage)) "Old & current vintages the same!"

    # General
    i   = 1             # Index tracking the stage of the algorithm
    j   = 2             # Index tracking the fixed_schedule entry ϕ_prop
    ϕ_n = ϕ_prop = 0.   # Instantiate ϕ_n and ϕ_prop variables

    resampled_last_period = false # Ensures proper resetting of ESS_bar after resample
    #use_fixed_schedule = (tempering_target == 0.0)
    threshold          = threshold_ratio * n_parts

    fixed_para_inds = findall([ θ.fixed for θ in parameters])
    free_para_inds  = findall([!θ.fixed for θ in parameters])
    para_symbols    = [θ.key for θ in parameters]

    n_para          = length(parameters)
    n_free_para     = length(free_para_inds)
    @assert n_free_para > 0 "All model parameters are fixed!"

    # Initialization of Particle Array Cloud
    cloud = Cloud(n_para, n_parts)

    #################################################################################
    ### Initialize Algorithm: Draws from prior
    #################################################################################
    println(verbose, :low, "\n\n SMC " * (testing ? "testing " : "") * "starts ....\n\n")

    if tempered_update
        # If user does not input Cloud object themselves, looks for cloud in loadpath.
        cloud = cloud_isempty(old_cloud) ? load(loadpath, "cloud") : old_cloud

        initialize_cloud_settings!(cloud; tempered_update = tempered_update,
                                   n_parts = n_parts, n_Φ = n_Φ, c = c, accept = target)
        initialize_likelihoods!(loglikelihood, parameters, data, cloud; parallel = parallel)

    elseif continue_intermediate
        cloud = load(loadpath, "cloud")
    else
        # Instantiating Cloud object, update draws, loglh, & logpost
        initial_draw!(loglikelihood, parameters, data, cloud; parallel = parallel)
        initialize_cloud_settings!(cloud; tempered_update = tempered_update,
                                   n_parts = n_parts, n_Φ = n_Φ, c = c, accept = target)
    end

    # Fixed schedule for construction of ϕ_prop
    if use_fixed_schedule
        cloud.tempering_schedule = ((collect(1:n_Φ) .- 1) / (n_Φ-1)) .^ λ
    else
        proposed_fixed_schedule  = ((collect(1:n_Φ) .- 1) / (n_Φ-1)) .^ λ
    end

    # Instantiate incremental and normalized weight matrices for logMDD calculation
    if continue_intermediate
        w_matrix = load(loadpath, "w")
        W_matrix = load(loadpath, "W")
        j        = load(loadpath, "j")
        i        = cloud.stage_index
        c        = cloud.c
        ϕ_prop   = proposed_fixed_schedule[j]
    else
        w_matrix = zeros(n_parts, 1)
        W_matrix = tempered_update ? (sum(get_weights(cloud)) <= 1.0 ?
                                      get_weights(cloud)*n_parts : get_weights(cloud)) :
                                      fill(1,(n_parts, 1))
    end

    # Printing
    init_stage_print(cloud, para_symbols; verbose = verbose,
                     use_fixed_schedule = use_fixed_schedule)
    println(verbose, :low, "\n\n SMC recursion starts... \n\n")

    #################################################################################
    ### Recursion
    #################################################################################
    while ϕ_n < 1.

        start_time = time_ns()
        cloud.stage_index = i += 1

        #############################################################################
        ### Setting ϕ_n (either adaptively or by the fixed schedule)
        #############################################################################
        ϕ_n1 = cloud.tempering_schedule[i-1]

        if use_fixed_schedule
            ϕ_n = cloud.tempering_schedule[i]
        else
            ϕ_n, resampled_last_period, j, ϕ_prop = solve_adaptive_ϕ(cloud,
                                                       proposed_fixed_schedule,
                                                       i, j, ϕ_prop, ϕ_n1,
                                                       tempering_target,
                                                       resampled_last_period)
        end

        #############################################################################
        ### Step 1: Correction
        #############################################################################

        # Calculate incremental weights (if no old data, get_old_loglh(cloud) = 0)
        incremental_weights = exp.((ϕ_n1 - ϕ_n) * get_old_loglh(cloud) +
                                   (ϕ_n - ϕ_n1) * get_loglh(cloud))

        # Update weights
        update_weights!(cloud, incremental_weights)
        mult_weights = get_weights(cloud)

        # Normalize weights
        normalized_weights = normalize_weights!(cloud) #get_weights(cloud)

        w_matrix = hcat(w_matrix, incremental_weights)
        W_matrix = hcat(W_matrix, normalized_weights)

        ##############################################################################
        ### Step 2: Selection
        ##############################################################################

        # Calculate the degeneracy/effective sample size metric
        push!(cloud.ESS, n_parts ^ 2 / sum(normalized_weights .^ 2))

        # If this assertion does not hold then there are likely too few particles
        @assert !isnan(cloud.ESS[i]) "No particles have non-zero weight."

        # Resample if degeneracy/ESS metric falls below the accepted threshold
        if (cloud.ESS[i] < threshold)

            # Resample according to particle weights, uniformly reset weights to 1/n_parts
            new_inds = resample(normalized_weights/n_parts; method = resampling_method)
            cloud.particles = [deepcopy(cloud.particles[k,j]) for k in new_inds,
                               j=1:size(cloud.particles, 2)]
            reset_weights!(cloud)
            cloud.resamples += 1
            resampled_last_period = true
            W_matrix[:, i] .= 1
        end

        ##############################################################################
        ### Step 3: Mutation
        ##############################################################################

        # Calculate adaptive c-step for use as scaling coefficient in mutation MH step
        c = c * (0.95 + 0.10 * exp(16.0 * (cloud.accept - target)) /
                 (1.0 + exp(16.0 * (cloud.accept - target))))
        cloud.c = c

        θ_bar = weighted_mean(cloud)
        R     = weighted_cov(cloud)

        # Ensures marix is positive semi-definite symmetric
        # (not off due to numerical error) and values haven't changed
        R_fr = (R[free_para_inds, free_para_inds] + R[free_para_inds, free_para_inds]') / 2.

        # MvNormal centered at ̄θ with var-cov ̄Σ, subsetting out the fixed parameters
        θ_bar_fr = θ_bar[free_para_inds]

        # Generate random parameter blocks
        blocks_free = generate_free_blocks(n_free_para, n_blocks)
        blocks_all  = generate_all_blocks(blocks_free, free_para_inds)

        new_particles = if parallel
            @distributed (hcat) for k in 1:n_parts
                mutation_closure(cloud.particles[k, :], θ_bar_fr, R_fr, blocks_free,
                                 blocks_all, ϕ_n, ϕ_n1; c = c, α = α,
                                 n_mh_steps = n_mh_steps, old_data = old_data)
            end
        else
            hcat([mutation_closure(cloud.particles[k, :], θ_bar_fr, R_fr,
                                   blocks_free, blocks_all, ϕ_n, ϕ_n1; c = c,
                                   α = α, n_mh_steps = n_mh_steps,
                                   old_data = old_data) for k=1:n_parts]...)
        end
        update_cloud!(cloud, new_particles)
        update_acceptance_rate!(cloud)

        ##############################################################################
        ### Timekeeping and Output Generation
        ##############################################################################
        cloud.total_sampling_time += Float64((time_ns() - start_time) * 1e-9)

        end_stage_print(cloud, para_symbols; verbose = verbose,
                        use_fixed_schedule = use_fixed_schedule)

        if run_test && (i == 3)
            break
        end

        if mod(cloud.stage_index, intermediate_stage_increment) == 0 && save_intermediate
            jldopen(replace(savepath, ".jld2" => "_stage=$(cloud.stage_index).jld2"),
                    true, true, true, IOStream) do file
                write(file, "cloud", cloud)
                write(file, "w", w_matrix)
                write(file, "W", W_matrix)
                write(file, "j", j)
            end
        end
    end

    ##################################################################################
    ### Saving data
    ##################################################################################
    if !testing
        simfile = h5open(particle_store_path, "w")
        particle_store = d_create(simfile, "smcparams", datatype(Float64),
                                  dataspace(n_parts, n_para))
        for k in 1:n_parts; particle_store[k,:] = cloud.particles[k, 1:n_para] end
        close(simfile)
        jldopen(savepath, true, true, true, IOStream) do file
            write(file, "cloud", cloud)
            write(file, "w", w_matrix)
            write(file, "W", W_matrix)
        end
    end
end
