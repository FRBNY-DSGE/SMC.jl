"""
```
function one_draw(loglikelihood::Function, parameters::ParameterVector{U},
                  data::Matrix{Float64}) where {U<:Number}
```
Finds and returns one valid draw from parameter distribution, along with its
log likelihood and log posterior.
"""
function one_draw(loglikelihood::Function, parameters::ParameterVector{U},
                  data::Matrix{Float64}) where {U<:Number}

    success    = false
    draw       = vec(rand(parameters, 1))
    draw_loglh = draw_logpost = 0.0

    while !success
        try
            update!(parameters, draw)
            draw_loglh   = loglikelihood(parameters, data)
            draw_logpost = prior(parameters)

            if (draw_loglh == -Inf) || (draw_loglh === NaN)
                draw_loglh = draw_logpost = -Inf
            end
        catch err
            if isa(err, ParamBoundsError) || isa(err, SingularException) ||
               isa(err, LinearAlgebra.LAPACKException) || isa(err, PosDefException)
                draw_loglh = draw_logpost = -Inf
            else
                throw(err)
            end
        end

        if any(isinf.(draw_loglh))
            draw = vec(rand(parameters, 1))
        else
            success = true
        end
    end
    return vector_reshape(draw, draw_loglh, draw_logpost)
end

"""
```
function initial_draw!(loglikelihood::Function, parameters::ParameterVector{U},
                       data::Matrix{Float64}, c::Cloud; parallel::Bool = false) where {U<:Number}
```
Draw from a general starting distribution (set by default to be from the prior) to
initialize the SMC algorithm. Returns a tuple (logpost, loglh) and modifies the
particle objects in the particle cloud in place.
"""
function initial_draw!(loglikelihood::Function, parameters::ParameterVector{U},
                       data::Matrix{Float64}, c::Cloud; parallel::Bool = false) where {U<:Number}
    n_parts = length(c)

    # ================== Define closure on one_draw function ==================
    sendto(workers(), loglikelihood = loglikelihood)
    sendto(workers(), parameters = parameters)
    sendto(workers(), data       = data)

    one_draw_closure() = one_draw(loglikelihood, parameters, data)
    @everywhere one_draw_closure() = one_draw(loglikelihood, parameters, data)
    # =========================================================================

    # For each particle, finds valid parameter draw and returns loglikelihood & posterior
    draws, loglh, logpost = if parallel
        @sync @distributed (vector_reduce) for i in 1:n_parts
            one_draw_closure()
        end
    else
        vector_reduce([one_draw_closure() for i in 1:n_parts]...)
    end

    update_draws!(c, draws)
    update_loglh!(c, vec(loglh))
    update_logpost!(c, vec(logpost))
    update_old_loglh!(c, zeros(n_parts))

    # Need to call `set_weights` as opposed to `update_weights`
    # since update_weights will multiply and 0*anything = 0
    set_weights!(c, ones(n_parts))
end

"""
```
function draw_likelihood(loglikelihood::Function, parameters::ParameterVector{U},
                         data::Matrix{Float64}, draw::Vector{Float64}) where {U<:Number}
```
Computes likelihood of a particular parameter draw; returns loglh and logpost.
"""
function draw_likelihood(loglikelihood::Function, parameters::ParameterVector{U},
                         data::Matrix{Float64}, draw::Vector{Float64}) where {U<:Number}
    update!(parameters, draw)
    loglh   = loglikelihood(parameters, data)
    logpost = prior(parameters)
    return scalar_reshape(loglh, logpost)
end

"""
```
function initialize_likelihoods!(loglikelihood::Function, parameters::ParameterVector{U},
                                 data::Matrix{Float64}, c::Cloud;
                                 parallel::Bool = false) where {U<:Number}
```
This function is made for transfering the log-likelihood values saved in the
Cloud from a previous estimation to each particle's respective old_loglh
field, and for evaluating/saving the likelihood and posterior at the new data, which
here is just the argument, data.
"""
function initialize_likelihoods!(loglikelihood::Function, parameters::ParameterVector{U},
                                 data::Matrix{Float64}, c::Cloud;
                                 parallel::Bool = false) where {U<:Number}
    n_parts = length(c)
    draws   = get_vals(c; transpose = false)

    # Retire log-likelihood values from the old estimation to the field old_loglh
    update_old_loglh!(c, get_loglh(c))

    # ============== Define closure on draw_likelihood function ===============
    sendto(workers(), parameters = parameters)
    sendto(workers(), loglikelihood = loglikelihood) # TODO: Check if this is necessary
    sendto(workers(), data = data)

    draw_likelihood_closure(draw::Vector{Float64}) = draw_likelihood(loglikelihood, parameters,
                                                                     data, draw)
    @everywhere draw_likelihood_closure(draw::Vector{Float64}) = draw_likelihood(loglikelihood,
                                                                                 parameters,
                                                                                 data, draw)
    # =========================================================================

    # TODO: handle when the likelihood with new data cannot be evaluated (returns -Inf),
    # even if the likelihood was not -Inf prior to incorporating new data
    loglh, logpost = if parallel
        @sync @distributed (scalar_reduce) for i in 1:n_parts
            draw_likelihood_closure(draws[i, :])
        end
    else
        scalar_reduce([draw_likelihood_closure(draws[i, :]) for i in 1:n_parts]...)
    end
    update_loglh!(c, loglh)
    update_logpost!(c, logpost)
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
