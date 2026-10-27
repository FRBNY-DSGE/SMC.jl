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

### Keyword Arguments:
- `verbose::Symbol`: Desired frequency of function progress messages printed to standard out.
	- `:none`: No status updates will be reported.
	- `:low`: Status updates for SMC initialization and recursion will be included (default).
	- `:high`: Status updates for every iteration of SMC is output, which includes
    the mean and standard deviation of each parameter draw after each iteration,
    as well as calculated acceptance rate, ESS, and number of times resampled.
- `parallel::Bool = false`: Flag for running algorithm in parallel.
- `smc_seed::Union{Nothing,Integer} = nothing`: Nonnegative run seed for reproducible
    particle, resampling, and block streams (see `src/rng.jl`). A run with a given seed is
    bitwise identical serially and with any number of workers. If omitted, the seed is drawn
    from the caller's task-local RNG, so `Random.seed!(x)` before `smc` still makes the run
    reproducible. Either way it is logged. smc() leaves the caller's task-local RNG as it found
    it (apart from that one draw). The seed and derivation version are stored under `smc_seed`
    and `smc_seed_version` in saved JLD2 files. A stage checkpoint supplies its own seed on
    resume; an explicit seed must match it. Bitwise replay assumes the same Julia and
    dependency versions, which are saved under `run_info`.
- `run_info::AbstractDict = Dict{String,String}()`: Extra provenance (e.g. the caller's package
    commit) saved under `run_info` in the JLD2 files, next to the Julia version and SMC commit.
- `n_smc_groups::Int = 1`: Number of independent particle groups (Durham and Geweke 2014). With
    J > 1, the `n_parts` particles are split into J complete SMC runs of `n_parts / J` particles,
    each with its own seed (derived from `smc_seed`), saved with a `_group=g` suffix. The pooled
    cloud is saved at `savepath` with `smc_groups` (pooled means, quantiles, their numerical
    standard errors and RNEs; also written to `<savepath>_group_summary.csv`) and the pooled
    `log_mdd_estimates`; it has no `w`/`W`. `continue_intermediate = true` (with the same
    `smc_seed`) reuses finished groups and resumes the others from their latest checkpoint.
- `bridge_sampling::Bool = false`: Also estimate the log MDD by bridge sampling on the final
    cloud, which costs `n_bridge_draws` (default `n_parts`) extra likelihood evaluations.
- `health_thresholds`: Flag thresholds for the per-stage health record
    (see `SMC.DEFAULT_HEALTH_THRESHOLDS`).
- `n_parts::Int = 5_000`: Number of particles.
- `n_blocks::Int = 1`: Number of parameter blocks in mutation step.
- `n_mh_steps::Int = 1`: Number of Metropolis Hastings steps to attempt during the mutation step.
- `λ::S = 2.1`: The 'bending coefficient' λ in Φ(n) = (n/N(Φ))^λ
- `n_Φ::Int = 300`: Number of stages in the tempering schedule.
- `resampling_method::Symbol`: Which resampling method to use.
    - `:systematic`: Will use systematic resampling (default).
    - `:multinomial`: Will use multinomial resampling.
    - `:polyalgo`: Samples using a polyalgorithm.
- `threshold_ratio::S = 0.5`: Threshold s.t. particles will be resampled when the population
    drops below threshold * N.
- `c::S = 0.5`: Initial scaling factor for covariance of the particles. Controls size of steps in mutation step.
    This value will be adaptively set afterward to reach an accept rate of `target` (see kwarg below).
- `α::S = 1.0`: The mixture proportion for the mutation step's proposal distribution. See `?mvnormal_mixture_draw` for details.
    Note that a value of 0.9 has commonly been used in applications to DSGE models (see citations below).
- `target::S = 0.25`: The target acceptance rate for new particles during mutation.
- `use_fixed_schedule::Bool = true`: Flag for whether or not to use a fixed tempering (ϕ) schedule.
- `tempering_target::S = 0.97`: Coefficient of the sample size metric to be targeted when solving
    for an endogenous ϕ.
- `old_data::Matrix{S} = []`: A matrix containing the time series of observables of previous data
    (with `data` being the new data) for the purposes of a time tempered estimation
    (that is, using the posterior draws from a previous estimation as the initial set
    of draws for an estimation with new data). This matrix is used to compute the
    log-likelihood according to the `old_loglikelihood` function when particles change during tempering.
- `old_cloud::Cloud = Cloud(0, 0)`: associated cloud borne of old data in previous SMC estimation.
    Running a bridge estimation requires `old_data` and `old_cloud`. If no `old_cloud`
    is provided, then we will attempt to load one using `loadpath`.
- `old_loglikelihood::Function = loglikelihood`: the old log-likelihood function when running
    a time tempered estimation. This function should be able to evaluate the log-likelihood
    of `old_data` given the current `parameters`, which may be nontrivial if the current
    `parameters` include additional parameters relative to the old estimation.
    By default, we assume the log-likelihood function has not changed
    and therefore coincides with the current one, so that the log-likelihood of `old_data`
    can be computed as `loglikelihood(parameters, old_data)`
- `old_vintage::String = ""`: String for vintage date of old data
- `smc_iteration::Int = 1`: The iteration index for the number of times SMC has been run on the
     same data vintage. Primarily for numerical accuracy/testing purposes.
- `run_test::Bool = false`: Flag for when testing accuracy of program
- `filestring_addl::Vector{String} = []`: Additional file string extension for loading old cloud.
- `save_intermediate::Bool = false`: Flag for whether one wants to save intermediate Cloud objects
- `intermediate_stage_increment::Int`: Save Clouds at every increment
   (1 = each stage, 10 = every 10th stage, etc.). Useful if you are using a cluster with time
    limits because if you hit the time limit, then you can just
    start from an intermediate stage rather than start over.
- `continue_intermediate::Bool = false`: Flag to indicate whether one is continuing SMC from an
    intermediate stage.
- `intermediate_stage_start::Int = 10`: Intermediate stage at which one wishes to begin the estimation.
- `tempered_update_prior_weight::Float64 = 0.0`: Weight placed on the current priors of parameters
    to construct a convex combination of draws from current priors and the previous estimation's
    cloud. The convex combination serves as the bridge distribution for a time tempered estimation.
- `run_csminwel::Bool = true`: Flag to run the csminwel algorithm to identify the true posterior mode
    (which may not exist) after completing an estimation. The mode identified by SMC is just
    the particle with the highest posterior value, but we do not check it is actually a mode (i.e.
    the Hessian is negative definite).
- `regime_switching::Bool = false`: Flag if there are regime-switching parameters. Otherwise, not all the values of the
    regimes will be used or saved.
- `toggle::Bool = true`: Flag for resetting the fields of parameter values to regime 1 anytime
    the loglikelihood is computed. The regime-switching version of SMC assumes at various points
    that this resetting occurs. If speed is important, then ensure that the fields of parameters
    take their regime 1 values at the end of the loglikelihood computation and set `toggle = false`.
- `debug_assertion::Bool = false`: if true, then when an assertion error is thrown during the estimation,
    output is created in a JLD2 file to help the user debug the problem.
- `log_prob_old_data::Union{Nothing,Float64} = nothing`: Log MDD of the old data.
    Supplying it puts a bridge's saved incremental weights on the full-data MDD scale.
    At zero prior share, omitting it retains the old incremental-MDD convention; do not
    also add the old MDD when it is supplied. A prior share strictly between zero and
    one requires this value to normalize the mixture.

### Outputs

- `cloud`: The Cloud object containing all of the information about the
    parameter values from the sample, their respective log-likelihoods, the ESS
    schedule, tempering schedule etc., which is saved in the saveroot.

Besides the cloud and weights, the JLD2 file at `savepath` holds the run's reliability
diagnostics (SMC reliability plan, step 3 part C):
- `stage_health`: per stage, φ and Δφ, ESS and conditional ESS, the largest weight, the share of
    distinct particles after selection and after mutation, the acceptance rate per MH step, the
    scale c, -Inf counts, and per free parameter the pre/post-mutation correlation and relative
    jump distance (also written to `<savepath>_stage_health.csv`);
- `health_flags`: the flags these raise (also logged as warnings unless `verbose = :none`);
- `log_mdd_estimates`: the incremental-weight log MDD (`smc`; `smc_level` on the level scale,
    see `smc_log_mdd_level`), the modified harmonic mean (`mhm`), bridge sampling if requested,
    and the ancestral-lineage NSEs (`smc_lineage_nse`, `mean_lineage_nse`, `n_eves`).

### Overview

Sequential Monte Carlo can be used in lieu of Random Walk Metropolis Hastings to
    generate parameter samples from high-dimensional parameter spaces using
    sequentially constructed proposal densities to be used in iterative importance
    sampling.

This implementation is based on Edward Herbst and Frank Schorfheide's 2014 paper
    'Sequential Monte Carlo Sampling for DSGE Models' and the code accompanying their
    book 'Bayesian Estimation of DSGE Models'. Cai et al. (2021)
    'Online Estimation of DSGE Models' extend the algorithm in Herbst and Schorfheide (2014)
    to include an adaptive schedule and generalized tempering, which are both
    features of this package.


SMC is broken up into three main steps:

- `Correction`: Reweight the particles from stage n-1 by defining incremental weights,
    which gradually "temper in" the loglikelihood function ``p(Y \vert \\theta)^(\\phi_n - \\phi_n-1)`` into the
    normalized particle weights.
- `Selection`: Resample the particles if the distribution of particles begins to
    degenerate, according to a tolerance level for the ESS.
- `Mutation`: Propagate particles ``{\\theta(i), W(n)}`` via a Metropolis
    Hastings algorithm (the number of steps are specified by `n_mh_steps`).
"""
function smc(loglikelihood::Function, parameters::ParameterVector{U}, data::Matrix{S};
             smc_seed::Union{Nothing, Integer} = nothing, continue_intermediate::Bool = false,
             n_smc_groups::Int = 1, kwargs...) where {S<:AbstractFloat, U<:Number}
    # With no seed, draw it from the caller's RNG so Random.seed! still controls the run (on a
    # resume the checkpoint supplies it)
    if smc_seed === nothing && !continue_intermediate
        smc_seed = rand(Random.default_rng(), UInt64)
    end
    n_smc_groups >= 1 || throw(ArgumentError("n_smc_groups must be at least 1"))
    # The per-particle streams reseed the task-local RNG: give the caller's back afterwards
    return preserving_default_rng() do
        if n_smc_groups == 1
            _smc(loglikelihood, parameters, data; smc_seed = smc_seed,
                 continue_intermediate = continue_intermediate, kwargs...)["cloud"]
        else
            smc_groups(loglikelihood, parameters, data; n_smc_groups = n_smc_groups,
                       smc_seed = smc_seed, continue_intermediate = continue_intermediate, kwargs...)
        end
    end
end

function _smc(loglikelihood::Function, parameters::ParameterVector{U}, data::Matrix{S};
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
             old_loglikelihood::Function = loglikelihood,
             old_vintage::String = "",
             smc_iteration::Int = 1,

             run_test::Bool = false,
             filestring_addl::Vector{String} = Vector{String}(),
             loadpath::String = "",
             savepath::String = "smc_cloud.jld2",
             particle_store_path::String = "smcsave.h5",
             save_intermediate::Bool = false,
             intermediate_stage_increment::Int = 10,
             continue_intermediate::Bool = false,
             intermediate_stage_start::Int = 0,
             tempered_update_prior_weight::S = 0.0,
             regime_switching::Bool = false,
             toggle::Bool = true,
             debug_assertion::Bool = false,
             log_prob_old_data::Union{Nothing, Float64} = nothing,
             add_zlb_duration::Tuple{Bool, Int} = (false, 1),
             cholesky_fix = :none,
             smc_seed::Union{Nothing, Integer} = nothing,
             run_info::AbstractDict{<:AbstractString, <:AbstractString} = Dict{String, String}(),
             health_thresholds = DEFAULT_HEALTH_THRESHOLDS,
             bridge_sampling::Bool = false, n_bridge_draws::Int = 0) where {S<:AbstractFloat, U<:Number}

    ########################################################################################
    ### Settings
    ########################################################################################

    0.0 <= tempered_update_prior_weight <= 1.0 ||
        throw(ArgumentError("tempered_update_prior_weight must be in [0, 1]"))
    if 0.0 < tempered_update_prior_weight < 1.0 && log_prob_old_data === nothing
        throw(ArgumentError("log_prob_old_data is required when tempered_update_prior_weight is between 0 and 1"))
    end
    log_prob_old_data = something(log_prob_old_data, 0.0)

    # Checkpoint seed takes precedence on resume; a caller-supplied seed must match it.
    # Checkpoints written before the seeding (no smc_seed) still resume, with a new seed.
    checkpoint_keys = continue_intermediate ? JLD2.jldopen(keys, loadpath, "r") : String[]
    run_seed = if continue_intermediate && "smc_seed" in checkpoint_keys
        saved_seed = load(loadpath, "smc_seed")
        saved_version = load(loadpath, "smc_seed_version")
        saved_version == SMC_SEED_VERSION || error("Unsupported SMC seed derivation version: $saved_version")
        smc_seed === nothing || checked_smc_seed(smc_seed) == saved_seed ||
            throw(ArgumentError("smc_seed differs from the checkpoint seed"))
        saved_seed
    else
        new_seed = smc_seed === nothing ? rand(Random.default_rng(), UInt64) : checked_smc_seed(smc_seed)
        continue_intermediate &&
            @warn "Checkpoint $loadpath has no smc_seed (written before SMC seeding): resuming with " *
                  "a new seed, so the resumed stages cannot replay the original run" smc_seed=new_seed
        new_seed
    end
    @info "SMC run seed" smc_seed=run_seed smc_seed_version=SMC_SEED_VERSION

    # Provenance saved next to the seed: bitwise replay needs the same Julia and package versions
    run_info = merge!(Dict{String, String}("julia_version" => string(VERSION),
                                           "smc_version"   => string(pkgversion(SMC)),
                                           "smc_commit"    => package_commit(SMC)), run_info)

    # Construct closure of mutation function so as to avoid issues with serialization
    # across workers with different Julia system images
    # Ship everything the worker-local closure reads as a Main global. The @distributed
    # body below lexically binds the *local* `mutation_closure` (which captures these by
    # value and is serialized to workers), so these sends are only exercised if the global
    # `@everywhere` definition is ever dispatched — but keep them complete so that path is
    # correct too (previously only parameters/data were sent, so loglikelihood /
    # old_loglikelihood / regime_switching / toggle would be UndefVar on workers).
    sendto(workers(), parameters = parameters)
    sendto(workers(), data = data)
    sendto(workers(), loglikelihood = loglikelihood)
    sendto(workers(), old_loglikelihood = old_loglikelihood)
    sendto(workers(), regime_switching = regime_switching)
    sendto(workers(), toggle = toggle)
    sendto(workers(), tempered_update_prior_weight = tempered_update_prior_weight)
    sendto(workers(), log_prob_old_data = log_prob_old_data)

    function mutation_closure(p::Vector{S}, d_μ::Vector{S}, d_Σ::Matrix{S},
                              n_free_para::Int,
                              blocks_free::Vector{Vector{Int64}}, blocks_all::Vector{Vector{Int64}},
                              ϕ_n::S; c::S = 1.0, α::S = 1.0, n_mh_steps::Int = 1,
                              old_data::Matrix{S} = Matrix{S}(undef, size(data, 1), 0)) where {S<:Float64}
        # The mutated particle plus its number of -Inf proposals (for the stage health record)
        n_failed = Ref(0)
        p = mutation(loglikelihood, parameters, data, p, d_μ, d_Σ, n_free_para, blocks_free, blocks_all,
                     ϕ_n; c = c, α = α, n_mh_steps = n_mh_steps, old_data = old_data,
                     old_loglikelihood = old_loglikelihood, regime_switching = regime_switching,
                     toggle = toggle, tempered_update_prior_weight = tempered_update_prior_weight,
                     log_prob_old_data = log_prob_old_data, n_failed = n_failed)
        return vcat(p, n_failed[])
    end
    # NOTE: positional args MUST match the local closure above and mutation()'s signature
    # (n_free_para, blocks_free, blocks_all). A prior version had blocks_free/blocks_all/
    # n_free_para swapped here, which would MethodError if this global were ever dispatched.
    @everywhere function mutation_closure(p::Vector{S}, d_μ::Vector{S}, d_Σ::Matrix{S},
                                          n_free_para::Int,
                                          blocks_free::Vector{Vector{Int64}}, blocks_all::Vector{Vector{Int64}},
                                          ϕ_n::S; c::S = 1.0, α::S = 1.0, n_mh_steps::Int = 1,
                                          old_data::Matrix{S} = Matrix{S}(undef, size(data, 1), 0)) where {S<:Float64}

        n_failed = Ref(0)
        p = mutation(loglikelihood, parameters, data, p, d_μ, d_Σ, n_free_para, blocks_free, blocks_all,
                     ϕ_n; c = c, α = α, n_mh_steps = n_mh_steps, old_data = old_data,
                     old_loglikelihood = old_loglikelihood, regime_switching = regime_switching,
                     toggle = toggle, tempered_update_prior_weight = tempered_update_prior_weight,
                     log_prob_old_data = log_prob_old_data, n_failed = n_failed)
        return vcat(p, n_failed[])
    end


    (tempered_update_prior_weight == 0.0 || tempered_update_prior_weight == 1.0 || !isempty(old_data)) ||
        throw(ArgumentError("old_data is required when tempered_update_prior_weight is between 0 and 1"))

    # A full prior share is an ordinary prior initialization and does not need
    # any old-data cloud, even if old_data was supplied by the caller.
    tempered_update = !isempty(old_data) && tempered_update_prior_weight < 1.0
    if (verbose != :none) && tempered_update && (old_vintage == data_vintage)
        @info "Old & current vintages the same!"
    end

    # General
    i   = 1             # Index tracking the stage of the algorithm
    j   = 2             # Index tracking the fixed_schedule entry ϕ_prop
    ϕ_n = ϕ_prop = 0.   # Instantiate ϕ_n and ϕ_prop variables

    resampled_last_period = false # Ensures proper resetting of ESS_bar after resample
    threshold             = threshold_ratio * n_parts
    #use_fixed_schedule = (tempering_target == 0.0)

    # Now count number of regime switching parameter values (excluding regime 1 values), i.e. if one parameter has 3 regimes, then add 2 to n_para

    n_para = length(parameters)
    for para in parameters
        if !isempty(para.regimes)
            for (ind, val) in para.regimes[:value]
                if ind != 1
                    n_para += 1
                end
            end
        end
    end

fixed_para_inds = ModelConstructors.get_fixed_para_inds(parameters; regime_switching = regime_switching, toggle = toggle)
free_para_inds  = ModelConstructors.get_free_para_inds(parameters; regime_switching = regime_switching, toggle = toggle)

old_cloud_cholesky_fixes = [
    :fixed_weight_mixture_1_prev_cloud,
    :fixed_weight_mixture_2_prev_cloud,
    :fixed_weight_mixture_scaling_prev_cloud
]
#[ID] Get prior covariance matrix (for free paras)
if tempered_update || cholesky_fix ∈ old_cloud_cholesky_fixes
    # If we do data tempering, our "prior covariance" is just the covariance of the the old cloud
    prev_cloud = cloud_isempty(old_cloud) ? load(loadpath, "cloud") : old_cloud
    R_prior = weighted_cov(prev_cloud)[free_para_inds, free_para_inds]
else
    # If we don't do data tempering, our "prior covariance" is the covariance of draws from marginal
    Random.seed!(stream_seed(run_seed, :prior_covariance))
    R_prior = SMC.get_prior_covariance(parameters, regime_switching = regime_switching, n_draws = 1e5)
end

para_symbols    = [θ.key for θ in parameters]
if regime_switching
    # Concatenate regime symbols for each extra regimes
    reg_switch_symbols = Vector{Symbol}(undef, n_para - length(parameters))
    ind = 0
    for θ in parameters
        if !isempty(θ.regimes)
            for i in 2:length(θ.regimes[:value])
                ind += 1
                reg_switch_symbols[ind] = Symbol(θ.key, "_reg$(i)")
            end
        end
    end
    push!(para_symbols, reg_switch_symbols...)
end

n_free_para = length(free_para_inds)
n_fixed_para = length(fixed_para_inds)
@assert n_free_para > 0 "All model parameters are fixed!"

    #################################################################################
    ### Initialize Algorithm: Draws from prior
    #################################################################################
println(verbose, :low, "\n\n SMC " * (testing ? "testing " : "") * "starts ....\n\n")
    # RESUME FIRST. This branch used to sit below `if tempered_update`, which made
    # it unreachable for a bridge run (tempered_update is true whenever old_data is
    # passed) -- so resuming a bridge step silently fell into the bridge
    # initialization below, which calls initialize_cloud_settings! and
    # initialize_likelihoods! and thereby resets stage_index/c and recomputes every
    # likelihood, i.e. it restarted the step instead of continuing it.
    #
    # A stage file already holds the fully initialized bridged cloud: draws, loglh,
    # OLD loglh, weights, ESS history, stage_index and c. So on resume none of that
    # initialization should run again, for a bridge step or otherwise -- we just
    # load the cloud and let the w/W/j restore below pick up the rest.
    if continue_intermediate
        cloud = load(loadpath, "cloud")
    elseif tempered_update
        # If user does not input Cloud object themselves, looks for cloud in loadpath.
        cloud = cloud_isempty(old_cloud) ? load(loadpath, "cloud") : old_cloud
        old_n_parts = length(cloud)

        if (tempered_update_prior_weight == 0.0) && (old_n_parts == n_parts)
            # Initialize settings first, with the ESS properly set since starting from an old estimation
            #initialize_cloud_settings!(cloud; tempered_update = tempered_update,
            #n_parts = n_parts, n_Φ = n_Φ, c = c, accept = target)
            #println("Above initalize cloud")
            # Initialize remaining cloud settings
            #println("Initializing cloud")
            initialize_cloud_settings!(cloud; tempered_update = tempered_update,
                                       n_parts = n_parts, n_Φ = n_Φ, c = c, accept = target)

            # Update the old_loglh column in cloud.particles with the values
            # from the current loglh column in cloud.particles.
            # Then compute log-likelihood of *old* estimation's particles on *new* data and parameters.


            # 8/9/24 - stuck here. Trying to update a size 130 vector of parameters with a length 144 set of parameter draws.
            #For starters, the parameter vector should be 184. Secondarily, we need a mechanism of including the draws of new parameters

            #println("Initialize likelihood")
            initialize_likelihoods!(loglikelihood, parameters, data, cloud; parallel = parallel,
                                    toggle = toggle)
            #println("After initialize likelihood")


        elseif (1. >= tempered_update_prior_weight > 0.) || (old_n_parts != n_parts)
            # Resample from bridge distribution
            n_to_resample = Int(round((1-tempered_update_prior_weight) * n_parts))

            n_from_prior  = n_parts - n_to_resample

            if n_to_resample > 0
                new_inds = resample(get_weights(cloud); n_parts = n_to_resample,
                                    method = resampling_method, parallel = parallel,
                                    rng = Xoshiro(stream_seed(run_seed, :bridge_resample_1)))

            else
                new_inds = Int64[] # empty
            end

            # Add in the samples from the old bridge.
            # Note the update_old_loglh! is commented out b/c
            # it is redundant, as we call initialize_likelihoods! later
            bridge_cloud = Cloud(n_para, n_to_resample)
            update_cloud!(bridge_cloud, cloud.particles[new_inds, :])
            update_loglh!(bridge_cloud, get_loglh(cloud)[new_inds])
            update_logprior!(bridge_cloud, get_logprior(cloud)[new_inds])
            # The old cloud was selected according to its weights. Clear the
            # copied weights so the selected particles are not weighted again.
            reset_weights!(bridge_cloud)
            #update_old_loglh!(bridge_cloud, get_old_loglh(cloud)[new_inds])

            # Add to the cloud draws from the prior. Note that we are drawing from the current prior,
            # but evaluating the old log-likelihood function on the old data. Thus,
            # you should also make sure that old_loglikelihood can be evaluated on the
            # current set of parameters (which may be nontrivial if new parameters have been added
            # relative to the previous estimation)
            if n_from_prior > 0
                # Evaluate old_loglikelihood on old data and new draws from prior
                prior_cloud = Cloud(n_para, n_from_prior)
                initial_draw!(old_loglikelihood, parameters, old_data, prior_cloud, parallel = parallel,
                              regime_switching = regime_switching, toggle = toggle,
                              smc_seed = run_seed, seed_operation = :bridge_prior_particle,
                              seed_particle_offset = n_to_resample)

                # Instantiating Cloud object, update draws, loglh, & logprior.
                # Note the update_old_loglh! is commented out b/c
                # it is redundant, as we call initialize_likelihoods! later
                cloud = Cloud(n_para, n_to_resample + n_from_prior)
                update_cloud!(cloud, vcat(bridge_cloud.particles, prior_cloud.particles))
                update_loglh!(cloud, vcat(get_loglh(bridge_cloud), get_loglh(prior_cloud)))
                update_logprior!(cloud, vcat(get_logprior(bridge_cloud), get_logprior(prior_cloud)))
                update_old_loglh!(cloud, vcat(get_old_loglh(bridge_cloud), get_old_loglh(prior_cloud)))
            else
                cloud = bridge_cloud
            end
            # Update the old_loglh column in cloud.particles with the values
            # from the current loglh column in cloud.particles.
            # Then compute current log-likelihood of *old* estimation's particles on *new* data and parameters.

            initialize_likelihoods!(loglikelihood, parameters, data, cloud; parallel = parallel,
                                    toggle = toggle)

            # Preserve particles with -Inf new-data likelihood through the
            # first correction step. Their discarded mass contributes to the
            # bridge evidence and would be lost by pre-resampling here.
            push!(cloud.ESS, n_parts)

            # Initialize remaining cloud settings
            initialize_cloud_settings!(cloud; tempered_update = tempered_update,
                                       n_parts = n_parts, n_Φ = n_Φ, c = c, accept = target)
else
            throw(DomainError("The keyword tempered_update_prior_weight must be within the interval [0, 1] but " *
                              "is currently set to $(tempered_update_prior_weight)"))
end
else
# Initialization of Particle Array Cloud
#println("Above new cloud?")
cloud = Cloud(n_para, n_parts)
# Instantiating Cloud object, update draws, loglh, & logprior
#println("Making new cloud object")
        initial_draw!(loglikelihood, parameters, data, cloud; parallel = parallel, regime_switching = regime_switching,
                      toggle = toggle, smc_seed = run_seed)
#println("initializing cloud")
        initialize_cloud_settings!(cloud; tempered_update = tempered_update,
                                   n_parts = n_parts, n_Φ = n_Φ, c = c, accept = target)
end
    # Fixed schedule for construction of ϕ_prop
    if use_fixed_schedule
        cloud.tempering_schedule = ((collect(1:n_Φ) .- 1) / (n_Φ-1)) .^ λ
    else
        proposed_fixed_schedule  = ((collect(1:n_Φ) .- 1) / (n_Φ-1)) .^ λ
    end

    # Diagnostics (reliability plan items 1-3): the per-stage health record, and each
    # particle's Eve (stage-0 ancestor) with the number of resampling steps for lineage_nse
    health            = new_stage_health(n_free_para)
    eve               = collect(1:n_parts)
    n_resample_events = 0
    lineage_valid     = true

    # Instantiate incremental and normalized weight matrices for logMDD calculation
    if continue_intermediate
        w_matrix = load(loadpath, "w")
        W_matrix = load(loadpath, "W")
        j        = load(loadpath, "j")
        println(j)
        i        = cloud.stage_index
        c        = cloud.c
        # Older checkpoints lack these: fall back to what resuming used to assume. The saved
        # ϕ_prop is the live one; sched[j] is generally not (it is sched[j-1] after
        # solve_adaptive_ϕ, or off the grid), which made an adaptive resume diverge.
        if "resampled_last_period" in checkpoint_keys && "ϕ_prop" in checkpoint_keys
            resampled_last_period = load(loadpath, "resampled_last_period")
            ϕ_prop                = load(loadpath, "ϕ_prop")
        else
            use_fixed_schedule ||
                @warn "Checkpoint $loadpath has no ϕ_prop/resampled_last_period: an adaptive " *
                      "schedule will not match the original run"
            ϕ_prop = (((collect(1:n_Φ) .- 1) / (n_Φ-1)) .^ λ)[j]
            if "resampled_last_period" in checkpoint_keys
                resampled_last_period = load(loadpath, "resampled_last_period")
            end
        end
        # Per-stage health and the Eve indices for the lineage NSEs. A checkpoint without them
        # resumes with the record restarted, and without lineage NSEs.
        if "stage_health" in checkpoint_keys && "eve" in checkpoint_keys
            health            = load(loadpath, "stage_health")
            eve               = load(loadpath, "eve")
            n_resample_events = load(loadpath, "n_resample_events")
        else
            lineage_valid = false
        end
        println("I have all vars")
    else
        w_matrix = zeros(n_parts, 1)
        W_matrix = tempered_update ? (sum(get_weights(cloud)) <= 1.0 ?
                                      get_weights(cloud) * n_parts : get_weights(cloud)) :
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
            #println("Proposed phi = $(Φ_prop)")
            ϕ_n, resampled_last_period, j, ϕ_prop = solve_adaptive_ϕ(cloud,
                                                       proposed_fixed_schedule,
                                                       i, j, ϕ_prop, ϕ_n1,
                                                       tempering_target,
                                                       resampled_last_period,
                                                       tempered_update_prior_weight = tempered_update_prior_weight,
                                                       log_prob_old_data = log_prob_old_data)
        end

        #############################################################################
        ### Step 1: Correction
        #############################################################################
    # Calculate incremental weights (if no old data, get_old_loglh(cloud) = 0)
        incremental_weights = bridge_incremental_weights(get_loglh(cloud), get_old_loglh(cloud),
                                                         ϕ_n, ϕ_n1, tempered_update_prior_weight,
                                                         log_prob_old_data)

        previous_weights = get_weights(cloud)
        inf_cloud        = count(==(-Inf), get_loglh(cloud))

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

        # Check whether ESS is a NaN and throws an assertion error if it is.
        # In many cases, the problem is that there too few particles.
        check_nan_ess(cloud, i, incremental_weights,
                      normalized_weights, savepath, debug_assertion)
        # Resample if degeneracy/ESS metric falls below the accepted threshold
    if (cloud.ESS[i] < threshold)

            # Resample according to particle weights, uniformly reset weights to 1/n_parts
            new_inds = resample(normalized_weights/n_parts; method = resampling_method,
                                parallel = parallel,
                                rng = Xoshiro(stream_seed(run_seed, :stage_resample, i)))
            cloud.particles = [deepcopy(cloud.particles[k,j]) for k in new_inds,
                               j=1:size(cloud.particles, 2)]
            reset_weights!(cloud)
            cloud.resamples += 1
            resampled_last_period = true
            eve = eve[new_inds]
            n_resample_events += 1
        W_matrix[:, i] .= 1

        println("Resampled at stage: $(i)")
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

    # [ID] Try out different solutions to covariance issue
    R_fr_mix = zeros(size(R_fr, 1), size(R_fr, 1))
    if cholesky_fix == :fixed_weight_mixture_1 || cholesky_fix == :fixed_weight_mixture_1_prev_cloud
        # [ID] New implementation: weighted mixture of current covariance (3/4) and prior covariance (1/4)
        println("Using fixed_weight_mixture (3/4, 1/4)")
        R_fr_mix = 0.75 * R_fr + 0.25 * R_prior

    elseif cholesky_fix == :fixed_weight_mixture_2 || cholesky_fix == :fixed_weight_mixture_2_prev_cloud
        println("Using fixed_weight_mixture (9/10, 1/10)")
        R_fr_mix = 0.90 * R_fr + 0.10 * R_prior

    elseif cholesky_fix == :fixed_weight_mixture_scaling || cholesky_fix == :fixed_weight_mixture_scaling_prev_cloud
        #[ID] Keep previous mixture but scale according to trace of covariance and prior covariance
        println("Using fixed_weight_mixture_scaling (3/4, 1/4, s=tr(Vn/Prior)")
        s_factor = tr(R_fr)/tr(R_prior)
        R_fr_mix = 0.75 * R_fr + 0.25 * s_factor *  R_prior

    elseif cholesky_fix == :regularization_1
        # We regularize with an identity matrix scaled by the trace of proposal cov.
        λ = 0.1
        p = size(R_fr, 1)
        I_mat = Matrix{Float64}(I, p, p)
        R_fr_mix = (1 - λ) * R_fr + λ * (tr(R_fr)/p) * I_mat

    elseif cholesky_fix == :scaled_prior
        # We use only the prior distribution, but scaled to the same magnitude as the proposal cov
        println("Using scaled prior")
        s = tr(R_fr)/tr(R_prior)
        R_fr_mix = s * R_prior

    else
        println("Using no covariance fixes")
        R_fr_mix = R_fr
    end

R_fr_mix = (R_fr_mix + R_fr_mix') / 2.

        # Julia 1.12's stricter LAPACK rejects covariances that are only positive *semi*-definite
        # (a zero / tiny-negative eigenvalue from roundoff or a near-degenerate weighted cloud),
        # which crashes every downstream proposal Cholesky (the mutation MvNormal, the c²·Σ mixture
        # draw, the proposal densities). Project R_fr_mix onto the PD cone by flooring its eigenvalues
        # at a small fraction of the largest, so it — and c²·any principal submatrix — factorizes.
        if !isposdef(R_fr_mix)
            F      = eigen(Symmetric(R_fr_mix))
            λfloor = max(maximum(F.values) * 1e-10, eps(eltype(R_fr_mix)))
            R_fr_mix   = F.vectors * Diagonal(max.(F.values, λfloor)) * F.vectors'
            R_fr_mix   = (R_fr_mix + R_fr_mix') / 2.
        end

        # MvNormal centered at ̄θ with var-cov ̄Σ, subsetting out the fixed parameters
        θ_bar_fr = θ_bar[free_para_inds]

# Generate random parameter blocks
@show n_blocks
blocks_free = generate_free_blocks(n_free_para, n_blocks;
                                   rng = Xoshiro(stream_seed(run_seed, :stage_blocks, i)))
blocks_all  = generate_all_blocks(blocks_free, free_para_inds)

θ_selected      = cloud.particles[:, free_para_inds]
weights_mutated = get_weights(cloud)

# Each column is a particle plus, in the last row, its number of -Inf (failed) proposals
new_particles = if parallel
    @distributed (hcat) for k in 1:n_parts
        Random.seed!(stream_seed(run_seed, :mutation_particle, i, k))
        mutation_closure(cloud.particles[k, :], θ_bar_fr, R_fr_mix, n_free_para,
                         blocks_free, blocks_all, ϕ_n; c = c, α = α,
                         n_mh_steps = n_mh_steps, old_data = old_data)
    end
else
    hcat([begin
        Random.seed!(stream_seed(run_seed, :mutation_particle, i, k))
        mutation_closure(cloud.particles[k, :], θ_bar_fr, R_fr_mix, n_free_para,
                           blocks_free, blocks_all, ϕ_n; c = c,
                           α = α, n_mh_steps = n_mh_steps,
                           old_data = old_data)
    end for k=1:n_parts]...)
end

inf_proposals = sum(new_particles[end, :])
update_cloud!(cloud, new_particles[1:end-1, :])
update_acceptance_rate!(cloud)


        ##############################################################################
        ### Timekeeping and Output Generation
        ##############################################################################
        stage_sampling_time = Float64((time_ns() - start_time) * 1e-9)
        cloud.total_sampling_time += stage_sampling_time

        θ_mutated  = cloud.particles[:, free_para_inds]
        corr, esjd = particle_movement(θ_selected, θ_mutated, weights_mutated)
        record_stage_health!(health,
                             (stage = i, phi = ϕ_n, dphi = ϕ_n - ϕ_n1, ess = cloud.ESS[i] / n_parts,
                              cess = conditional_ess(incremental_weights, previous_weights),
                              max_weight = maximum(normalized_weights) / n_parts,
                              resampled = cloud.ESS[i] < threshold,
                              distinct_selected = distinct_share(θ_selected),
                              distinct_mutated = distinct_share(θ_mutated),
                              # cloud.accept sums acceptances over the MH steps; record the rate
                              accept = cloud.accept / max(n_mh_steps, 1),
                              c = c, inf_cloud = inf_cloud, inf_proposals = inf_proposals,
                              n_proposals = n_parts * n_mh_steps * length(blocks_free),
                              time = stage_sampling_time), corr, esjd)

        end_stage_print(cloud, para_symbols; verbose = verbose,
                        use_fixed_schedule = use_fixed_schedule)

        if run_test && (i == 3)
            break
        end

        if mod(cloud.stage_index, intermediate_stage_increment) == 0 && save_intermediate
            #jldopen(replace(savepath, ".jld2" => "_stage=$(cloud.stage_index).jld2"),
            #true, true, true, IOStream) do file
            JLD2.jldopen(replace(savepath, ".jld2" => "_stage=$(cloud.stage_index).jld2"), true, true, true, IOStream) do file
                write(file, "cloud", cloud)
                write(file, "w", w_matrix)
                write(file, "W", W_matrix)
                write(file, "j", j)
                write(file, "resampled_last_period", resampled_last_period)
                write(file, "ϕ_prop", ϕ_prop)
                write(file, "smc_seed", run_seed)
                write(file, "smc_seed_version", SMC_SEED_VERSION)
                write(file, "run_info", run_info)
                write(file, "stage_health", health)
                write(file, "eve", eve)
                write(file, "n_resample_events", n_resample_events)
            end

        end
    end

    ##################################################################################
    ### Diagnostics: log MDD estimates and flags (reliability plan items 1-3)
    ##################################################################################
    θ_free  = cloud.particles[:, free_para_inds]
    weights = get_weights(cloud)
    log_mdd = smc_log_mdd(w_matrix, W_matrix)
    estimates = Dict{String, Any}(
        "smc"       => log_mdd,
        "smc_level" => smc_log_mdd_level(log_mdd; tempered_update = tempered_update,
                                         tempered_update_prior_weight = tempered_update_prior_weight,
                                         log_prob_old_data = log_prob_old_data),
        "mhm"       => modified_harmonic_mean(θ_free, get_logpost(cloud), weights).log_mdd)
    if lineage_valid
        lineage = lineage_nse(eve, n_resample_events, weights, θ_free)
        estimates["smc_lineage_nse"] = lineage.log_mdd_nse
        estimates["mean_lineage_nse"] = lineage.mean_nse
        estimates["n_eves"] = lineage.n_eves
    end
    if bridge_sampling
        bs = smc_bridge_sampling(loglikelihood, parameters, data, cloud, free_para_inds;
                                 n_draws = n_bridge_draws > 0 ? n_bridge_draws : n_parts,
                                 rng = Xoshiro(stream_seed(run_seed, :bridge_sampling)),
                                 parallel = parallel, toggle = toggle)
        estimates["bridge_sampling"]     = bs.log_mdd
        estimates["bridge_sampling_nse"] = bs.nse
    end

    flags = stage_health_flags(health; thresholds = health_thresholds,
                               para_names = para_symbols[free_para_inds])
    gap = mdd_gap_flag(estimates)
    gap === nothing || push!(flags, gap)
    if verbose != :none
        for flag in flags
            @warn "SMC health: " * flag
        end
    end

    result = Dict{String, Any}("cloud" => cloud, "w" => w_matrix, "W" => W_matrix,
                               "smc_seed" => run_seed, "smc_seed_version" => SMC_SEED_VERSION,
                               "run_info" => run_info, "stage_health" => health,
                               "health_flags" => flags, "log_mdd_estimates" => estimates)

    ##################################################################################
    ### Saving data
    ##################################################################################
    if !testing
        simfile = h5open(particle_store_path, "w")
        particle_store = isdefined(HDF5, :create_dataset) ?
            HDF5.create_dataset(simfile, "smcparams", datatype(Float64), dataspace(n_parts, n_para)) :
            HDF5.d_create(simfile, "smcparams", datatype(Float64), dataspace(n_parts, n_para))

        for k in 1:n_parts; particle_store[k,:] = cloud.particles[k, 1:n_para] end
        close(simfile)
        jldopen(savepath, true, true, true, IOStream) do file
            for (key, value) in result
                write(file, key, value)
            end
        end
        write_stage_health(replace(savepath, r"\.jld2$" => "") * "_stage_health.csv", health;
                           para_names = para_symbols[free_para_inds])
    end
    return result
end
