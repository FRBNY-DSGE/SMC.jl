"""
```
one_draw(loglikelihood::Function, parameters::ParameterVector{U},
         data::Matrix{Float64}; regime_switching::Bool = false,
         toggle::Bool = true) where {U<:Number}
```

Finds and returns one valid draw from parameter distribution, along with its
log likelihood and log prior.

Set `regime_switching` to true if
there are regime-switching parameters. Otherwise, not all the values of the
regimes will be used or saved.

Set `toggle` to false if, after calculating the loglikelihood,
the values in the fields of every parameter in `parameters`
are set to their regime 1 values. The regime-switching version of `rand`
requires that the fields of all parameters take their regime 1 values,
or else sampling may be wrong. The default is `true` as a safety, but
if speed is a paramount concern, setting `toggle = true` will avoid
unnecessary computations.
"""
function one_draw(loglikelihood::Function, parameters::ParameterVector{U},
                  data::Matrix{Float64}; regime_switching::Bool = false,
                  toggle::Bool = true) where {U<:Number}
    success    = false
    draw       = vec(rand(parameters, 1, regime_switching = regime_switching, toggle = toggle))

    draw_loglh = draw_logprior = 0.0

    while !success
        try
            update!(parameters, draw)

            draw_loglh = loglikelihood(parameters, data)

            if toggle
                toggle_regime!(parameters, 1)
            end

            draw_logprior = prior(parameters)

            if (draw_loglh == -Inf) || (draw_loglh === NaN)
                draw_loglh = draw_logprior = -Inf
            end
        catch err
            if isa(err, ParamBoundsError) || isa(err, SingularException) ||
               isa(err, LinearAlgebra.LAPACKException) || isa(err, PosDefException) ||
               isa(err, DomainError)
                draw_loglh = draw_logprior = -Inf
            else
                throw(err)
            end
        end

        if any(isinf.(draw_loglh))
            draw = vec(rand(parameters, 1, regime_switching = regime_switching, toggle = false))
        else
            success = true
        end
    end
    return vector_reshape(draw, draw_loglh, draw_logprior)
end

"""
```
function initial_draw!(loglikelihood::Function, parameters::ParameterVector{U},
                       data::Matrix{Float64}, c::Cloud; parallel::Bool = false,
                       regime_switching::Bool = false, toggle::Bool = true) where {U<:Number}
```

Draw from a general starting distribution (set by default to be from the prior) to
initialize the SMC algorithm. Returns a tuple (loglh, logprior) and modifies the
particle objects in the particle cloud in place.

Set `regime_switching` to true if
there are regime-switching parameters. Otherwise, not all the values of the
regimes will be used or saved.

Set `toggle` to false if, after calculating the loglikelihood,
the values in the fields of every parameter in `parameters`
are set to their regime 1 values. The regime-switching version of `rand`
requires that the fields of all parameters take their regime 1 values,
or else sampling may be wrong. The default is `true` as a safety, but
if speed is a paramount concern, setting `toggle = true` will avoid
unnecessary computations.
"""
function initial_draw!(loglikelihood::Function, parameters::ParameterVector{U},
                       data::Matrix{Float64}, c::Cloud; parallel::Bool = false,
                       regime_switching::Bool = false, toggle::Bool = true) where {U<:Number}
    n_parts = length(c)

    # ================== Define closure on one_draw function ==================
    sendto(workers(), loglikelihood = loglikelihood)
    sendto(workers(), parameters = parameters)
    sendto(workers(), data       = data)

    one_draw_closure() = one_draw(loglikelihood, parameters, data, regime_switching = regime_switching, toggle = toggle)
    @everywhere one_draw_closure() = one_draw(loglikelihood, parameters, data, regime_switching = regime_switching, toggle = toggle)
    # =========================================================================

    # For each particle, finds valid parameter draw and returns loglikelihood & prior
    draws, loglh, logprior = if parallel
        @sync @distributed (vector_reduce) for i in 1:n_parts
            one_draw_closure()
        end
    else
        vector_reduce([one_draw_closure() for i in 1:n_parts]...)
    end

    update_draws!(c, draws)
    update_loglh!(c, vec(loglh))
    update_logprior!(c, vec(logprior))
    update_old_loglh!(c, zeros(n_parts))

    # Need to call `set_weights` as opposed to `update_weights`
    # since update_weights will multiply and 0*anything = 0
    set_weights!(c, ones(n_parts))
end

"""
```
draw_likelihood(loglikelihood::Function, parameters::ParameterVector{U},
                data::Matrix{Float64}, draw::Vector{Float64};
                toggle::Bool = true) where {U<:Number}
```
Computes likelihood of a particular parameter draw; returns loglh and logprior.
"""
function draw_likelihood(loglikelihood::Function, parameters::ParameterVector{U},
                         data::Matrix{Float64}, draw::Vector{Float64};
                         toggle::Bool = true) where {U<:Number}
    update!(parameters, draw)
    loglh   = loglikelihood(parameters, data)
    if toggle
        ModelConstructors.toggle_regime!(parameters, 1)
    end
    logprior = prior(parameters)
    return scalar_reshape(loglh, logprior)
end

"""
```
initialize_likelihoods!(loglikelihood::Function, parameters::ParameterVector{U},
                        data::Matrix{Float64}, c::Cloud;
                        parallel::Bool = false,
                        toggle::Bool = true) where {U<:Number}
```
This function is made for transfering the log-likelihood values saved in the
Cloud from a previous estimation to each particle's respective old_loglh
field, and for evaluating/saving the likelihood and prior at the new data, which
here is just the argument, data.
"""
function initialize_likelihoods!(loglikelihood::Function, parameters::ParameterVector{U},
                                 data::Matrix{Float64}, c::Cloud;
                                 parallel::Bool = false,
                                 toggle::Bool = true) where {U<:Number}
    n_parts = length(c)
    draws   = get_vals(c; transpose = false)

    # Retire log-likelihood values from the old estimation to the field old_loglh
    update_old_loglh!(c, get_loglh(c))

    # ============== Define closure on draw_likelihood function ===============
    sendto(workers(), parameters = parameters)
    sendto(workers(), loglikelihood = loglikelihood) # TODO: Check if this is necessary
    sendto(workers(), data = data)

    draw_likelihood_closure(draw::Vector{Float64}) = draw_likelihood(loglikelihood, parameters,
                                                                     data, draw, toggle = toggle)
    @everywhere draw_likelihood_closure(draw::Vector{Float64}) = draw_likelihood(loglikelihood,
                                                                                 parameters,
                                                                                 data, draw, toggle = toggle)
    # =========================================================================

    # TODO: handle when the likelihood with new data cannot be evaluated (returns -Inf),
    # even if the likelihood was not -Inf prior to incorporating new data
    loglh, logprior = if parallel
        @sync @distributed (scalar_reduce) for i in 1:n_parts
            draw_likelihood_closure(draws[i, :])
        end
    else
        scalar_reduce([draw_likelihood_closure(draws[i, :]) for i in 1:n_parts]...)
    end
    update_loglh!(c, loglh)
    update_logprior!(c, logprior)
end

"""
```
function initialize_cloud_settings!(cloud::Cloud; tempered_update::Bool = false,
                                    n_parts::Int = 5_000, n_Φ::Int = 300, c::S = 0.5,
                                    accept::S = 0.25) where {S<:AbstractFloat}
```
Initializes stage index, number of Φ stages, c, resamples, acceptance, and sampling time.
"""
function initialize_cloud_settings!(cloud::Cloud; tempered_update::Bool = false,
                                    n_parts::Int = 5_000, n_Φ::Int = 300, c::S = 0.5,
                                    accept::S = 0.25) where {S<:AbstractFloat}
    if tempered_update
        cloud.ESS = [cloud.ESS[end]]
    else
        cloud.ESS[1] = n_parts
    end
    cloud.stage_index = 1
    cloud.n_Φ         = n_Φ
    cloud.resamples   = 0
    cloud.c           = c
    cloud.accept      = accept
    cloud.total_sampling_time = 0.
    cloud.tempering_schedule  = zeros(1)
end
