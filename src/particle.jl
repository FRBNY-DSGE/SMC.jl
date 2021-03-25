"""
```
Cloud
```

The `Cloud` type contains all of the relevant information for a given cloud of
particles in the SMC algorithm. Information for a single iteration is stored at
any given time (and thus the final output will be the final cloud of particles,
 of which only the particle values will be saved).

### Fields
- `particles::Matrix{Float64}`: The vector of particles. The first `1:(size(particles, 2) - 5)`
    columns hold the parameter values. The last five columns
    contain "metadata" on the particles, namely (in order from left to right)
    (1) the log-likelihood, (2) log-prior, (3) the old log-likelihood from a previous estimation
    (used for tempered updates, set to zero if not tempering), (4) the accept rate
    for the particle's mutation steps, and (5) the particle's normalized weight.
- `tempering_schedule::Vector{Float64}`: The vector of ϕ_ns (tempering factors)
- `ESS::Vector{Float64}`: The vector of effective sample sizes (resample if ESS
    falls under the threshold)
- `stage_index::Int`: The current iteration index of the algorithm
- `n_Φ::Int`: The total number of stages of in the fixed tempering schedule
    (if the algorithm is run with an adaptive ϕ schedule then this is used to
    calibrate the ϕ_prop)
- `resamples::Int`: The number of times the particle population was resampled
- `c::Float64`: The mutation step size
- `accept::Float64`: The average acceptance rate of mutation steps
- `total_sampling_time::Float64`: Total amount of time that SMC algorithm took
     to execute
"""
mutable struct Cloud
    particles::Matrix{Float64}
    tempering_schedule::Vector{Float64}
    ESS::Vector{Float64}
    stage_index::Int
    n_Φ::Int
    resamples::Int
    c::Float64
    accept::Float64
    total_sampling_time::Float64
end

"""
```
function Cloud(n_params::Int, n_parts::Int)
```
Easier constructor for Cloud, which initializes the weights to be
equal, and everything else in the particle object to be empty.
"""
function Cloud(n_params::Int, n_parts::Int)
    return Cloud(Matrix{Float64}(undef, n_parts, n_params + 5),
                 zeros(1), zeros(1), 1, 0, 0, 0., 0.25, 0.)
end

"""
Find correct indices for accessing columns of cloud array.
"""
ind_para_end(N::Int)  = N-5
ind_loglh(N::Int)     = N-4
ind_logprior(N::Int)  = N-3
ind_old_loglh(N::Int) = N-2
ind_accept(N::Int)    = N-1
ind_weight(N::Int)    = N

"""
```
function get_weights(c::Matrix{Float64})
```
Returns Vector{Float64}(n_parts) of weights of particles in cloud.
"""
function get_weights(c::Matrix{Float64})
    return c[:, ind_weight(size(c,2))]
end
"""
```
function get_weights(c::Cloud)
```
Returns Vector{Float64}(n_parts) of weights of particles in cloud.
"""
function get_weights(c::Cloud)
    return c.particles[:, ind_weight(size(c.particles,2))]
end

"""
```
function get_vals(c::Cloud; transposed::Bool = true)
```
Returns Matrix{Float64}(n_params, n_parts) of parameter values in particle cloud.
"""
function get_vals(c::Matrix{Float64}; transpose::Bool = true)
    return transpose ? Matrix{Float64}(c[:, 1:ind_para_end(size(c, 2))]') :
                                       c[:, 1:ind_para_end(size(c, 2))]
end
"""
```
function get_vals(c::Matrix{Float64})
```
Returns Matrix{Float64}(n_params, n_parts) of parameter values in particle cloud.
"""
function get_vals(c::Cloud; transpose::Bool = true)
    return transpose ? Matrix{Float64}(c.particles[:, 1:ind_para_end(size(c.particles,2))]') :
                                       c.particles[:, 1:ind_para_end(size(c.particles,2))]
end

"""
```
function get_loglh(c::Matrix{Float64})
function get_loglh(c::Cloud)
```
Returns Vector{Float64}(n_parts) of log-likelihood of particles in cloud.
"""
function get_loglh(c::Matrix{Float64})
    return c[:, ind_loglh(size(c,2))]
end
function get_loglh(c::Cloud)
    return c.particles[:, ind_loglh(size(c.particles,2))]
end

"""
```
function cloud_isempty(c::Matrix{Float64})
function cloud_isempty(c::Cloud)
```
Check if cloud has no particles.
"""
function cloud_isempty(c::Matrix{Float64})
    return isempty(c)
end
function cloud_isempty(c::Cloud)
    # TODO: 'ISEMPTY' IS INCOMPATIBLE WITH INITIALIZATION
    return isempty(c.particles)
end


"""
```
function get_old_loglh(c::Matrix{Float64})
function get_old_loglh(c::Cloud)
```
Returns Vector{Float64}(n_parts) of old log-likelihood of particles in cloud.
"""
function get_old_loglh(c::Matrix{Float64})
    return c[:, ind_old_loglh(size(c,2))]
end
function get_old_loglh(c::Cloud)
    return c.particles[:, ind_old_loglh(size(c.particles,2))]
end

"""
```
function get_logpost(c::Matrix{Float64})
function get_logpost(c::Cloud)
```
Returns Vector{Float64}(n_parts) of log-posterior of particles in cloud.
"""
function get_logpost(c::Matrix{Float64})
    return c[:, ind_loglh(size(c,2))] .+ c[:, ind_logprior(size(c,2))]
end
function get_logpost(c::Cloud)
    return c.particles[:, ind_loglh(size(c.particles,2))] .+
        c.particles[:, ind_logprior(size(c.particles,2))]
end

"""
```
function get_logprior(c::Matrix{Float64})
function get_logprior(c::Cloud)
```
Returns Vector{Float64}(n_parts) of log-prior of particles in cloud.
"""
function get_logprior(c::Matrix{Float64})
    return c[:, ind_logprior(size(c,2))]
end
function get_logprior(c::Cloud)
    return c.particles[:, ind_logprior(size(c.particles,2))]
end

"""
```
function get_accept(c::Matrix{Float64})
function get_accept(c::Cloud)
```
Returns Vector{Float64}(n_parts) of old log-likelihood of particles in cloud.
"""
function get_accept(c::Matrix{Float64})
    return c[:, ind_accept(size(c,2))]
end
function get_accept(c::Cloud)
    return c.particles[:, ind_accept(size(c.particles,2))]
end

"""
```
function get_likeliest_particle_value(c::Matrix{Float64})
function get_likeliest_particle_value(c::Cloud)
```
Return parameter vector of particle with highest log-likelihood.
"""
function get_likeliest_particle_value(c::Matrix{Float64})
    return c[argmax(get_loglh(c)), 1:ind_para_end(size(c,2))]
end
function get_likeliest_particle_value(c::Cloud)
    return c.particles[argmax(get_loglh(c)), 1:ind_para_end(size(c.particles,2))]
end

"""
```
function get_highest_posterior_particle_value(c::Matrix{Float64})
function get_highest_posterior_particle_value(c::Cloud)
```
Return parameter vector of particle with highest log-posterior.
"""
function get_highest_posterior_particle_value(c::Matrix{Float64})
    return c[argmax(get_logpost(c)), 1:ind_para_end(size(c,2))]
end
function get_highest_posterior_particle_value(c::Cloud)
    return c.particles[argmax(get_logpost(c)), 1:ind_para_end(size(c.particles,2))]
end

"""
```
function update_draws!(c::Cloud, draws::Matrix{Float64})
```
Update parameter draws in cloud.
"""
function update_draws!(c::Cloud, draws::Matrix{Float64})
    I, J     = size(draws)
    n_parts  = length(c)
    n_params = ind_para_end(size(c.particles, 2))
    if (I, J) == (n_parts, n_params)
        for i = 1:I, j=1:J
            c.particles[i, j] = draws[i, j]
        end
    elseif (I, J) == (n_params, n_parts)
        for i = 1:I, j=1:J
            c.particles[j, i] = draws[i, j]
        end
    else
        throw(error("update_draws!(c::Cloud, draws::Matrix): Draws are incorrectly sized!"))
    end
end

"""
```
function update_weights!(c::Matrix{Float64}, incweight::Vector{Float64})
function update_weights!(c::Cloud, weights::Vector{Float64})
```
Update weights in cloud.
"""
function update_weights!(c::Matrix{Float64}, incweight::Vector{Float64})
    @assert size(c, 1) == length(incweight) "Dimensional mismatch in inc. weights"
    N = ind_weight(size(c,2))
    for i=1:length(incweight)
        c[i, N] *= incweight[i]
    end
end
function update_weights!(c::Cloud, weights::Vector{Float64})
    update_weights!(c.particles, weights)
end

"""
```
function set_weights!(c::Cloud, weights::Vector{Float64})
```
Set weights to specific values. Contrast to update_weights, which multiplies
existing weights by provided incremental weights.
"""
function set_weights!(c::Cloud, weights::Vector{Float64})
    @assert length(c) == length(weights) "Dimensional mismatch in set_weights"
    N = ind_weight(size(c.particles,2))
    for i=1:length(c)
        c.particles[i, N] = weights[i]
    end
end


"""
```
function update_loglh!(c::Matrix{Float64}, incweight::Vector{Float64})
function update_loglh!(c::Cloud, loglh::Vector{Float64})
```
Update log-likelihood in cloud.
"""
function update_loglh!(c::Matrix{Float64}, loglh::Vector{Float64})
    @assert size(c,1) == length(loglh) "Dimensional mismatch"
    N = ind_loglh(size(c,2))
    for i=1:length(loglh)
        c[i, N] = loglh[i]
    end
end
function update_loglh!(c::Cloud, loglh::Vector{Float64})
    update_loglh!(c.particles, loglh)
end

"""
```
function update_logprior!(c::Matrix{Float64}, incweight::Vector{Float64})
function update_logprior!(c::Cloud, logprior::Vector{Float64})
```
Update log-prior in cloud.
"""
function update_logprior!(c::Matrix{Float64}, logprior::Vector{Float64})
    @assert size(c, 1) == length(logprior) "Dimensional mismatch"
    N = ind_logprior(size(c,2))
    for i=1:length(logprior)
        c[i, N] = logprior[i]
    end
end
function update_logprior!(c::Cloud, logprior::Vector{Float64})
    update_logprior!(c.particles, logprior)
end


"""
```
function update_old_loglh!(c::Matrix{Float64}, incweight::Vector{Float64})
function update_old_loglh!(c::Cloud, old_loglh::Vector{Float64})
```
Update log-likelihood in cloud.
"""
function update_old_loglh!(c::Matrix{Float64}, old_loglh::Vector{Float64})
    @assert size(c, 1) == length(old_loglh) "Dimensional mismatch"
    N = ind_old_loglh(size(c,2))
    for i=1:length(old_loglh)
        c[i, N] = old_loglh[i]
    end
end
function update_old_loglh!(c::Cloud, old_loglh::Vector{Float64})
    update_old_loglh!(c.particles, old_loglh)
end

"""
```
function normalize_weights!(c::Matrix{Float64})
function normalize_weights!(c::Cloud)
```
Normalize weights in cloud to N, the number of particles.
"""
function normalize_weights!(c::Matrix{Float64})
    sum_weights = sum(get_weights(c))
    c[:, ind_weight(size(c,2))] *= size(c,1)
    c[:, ind_weight(size(c,2))] /= sum_weights
end
function normalize_weights!(c::Cloud)
    normalize_weights!(c.particles)
end

"""
```
function reset_weights!(c::Matrix{Float64})
function reset_weights!(c::Cloud)
```
Uniformly reset weights of all particles to 1/n_parts.
"""
function reset_weights!(c::Matrix{Float64})
    n_parts = size(c, 1)
    c[:, ind_weight(size(c,2))] .= 1.0
end
function reset_weights!(c::Cloud)
    reset_weights!(c.particles)
end

"""
```
function update_mutation!(p::Vector{Float64}, para::Vector{Float64},
                          like::Float64, prior::Float64, old_like::Float64,
                          accept::Float64)
```
Update a particle's parameter vector, log-likelihood, log-prior,
old log-likelihood, and acceptance rate at the end of mutation.
"""
function update_mutation!(p::Vector{Float64}, para::Vector{Float64}, like::Float64,
                          prior::Float64, old_like::Float64, accept::Float64)
    N = length(p)
    p[1:ind_para_end(N)] = para
    p[ind_loglh(N)]      = like
    p[ind_logprior(N)]   = prior
    p[ind_old_loglh(N)]  = old_like
    p[ind_accept(N)]     = accept
end

"""
```
function update_cloud!(cloud::Cloud, new_particles::Matrix{Float64})
```
Updates cloud values with those of new particles in particle array.
"""
function update_cloud!(cloud::Cloud, new_particles::Matrix{Float64})
    I, J = size(new_particles)
    if I == length(cloud)
        cloud.particles = new_particles
    elseif J == length(cloud)
        for k = 1:length(cloud)
            cloud.particles[k,:] = new_particles[:, k]
        end
    else
        throw(error("update_cloud!(c::Cloud, draws): draws are incorrect size!"))
    end
end

"""
```
function update_val!(p::Vector{Float64}, val::Vector{Float64})
```
Update parameter vector of particle.
"""
function update_val!(p::Vector{Float64}, val::Vector{Float64})
    @assert ind_para_end(length(p)) == length(val) "Parameter vector length is wrong!"
    p[1:length(val)] = val
end

"""
```
function update_weight!(p::Vector{Float64}, weight::Vector{Float64})
```
Update weight of particle.
"""
function update_weight!(p::Vector{Float64}, weight::Float64)
    p[ind_weight(length(p))] = weight
end

"""
```
function update_acceptance_rate!(c::Cloud)
```
Update cloud's acceptance rate with the mean of its particle acceptance rates.
"""
function update_acceptance_rate!(c::Cloud)
    c.accept = mean(get_accept(c))
end

function Base.length(c::Cloud)
        return size(c.particles, 1)
end

"""
```
function weighted_mean(c::Cloud)
function weighted_mean(c::Matrix{Float64})
```
Compute weighted mean of particle cloud.
"""
function weighted_mean(c::Matrix{Float64})
    return get_vals(c) * get_weights(c) / sum(get_weights(c))
end
function weighted_mean(c::Cloud)
    return weighted_mean(c.particles)
end

"""
```
function weighted_quantile(c::Matrix{Float64}, i::Int64)
function weighted_quantile(c::Cloud, i::Int64)
```
Compute weighted quantiles of particle cloud for input parameter, indexed by i.
"""
function weighted_quantile(c::Matrix{Float64}, i::Int64)
    @assert i <= ind_end_para(size(c,2)) "Parameter index invalid."
    lb = quantile(c[:, i], Weights(get_weights(c)), .05)
    ub = quantile(c[:, i], Weights(get_weights(c)), .95)
    return lb, ub
end
function weighted_quantile(c::Cloud, i::Int64)
    return weighted_quantile(c.particles, i)
end

"""
```
function weighted_std(c::Cloud)
function weighted_std(c::Matrix{Float64})
```
Compute weighted standard deviation of particle cloud.
"""
function weighted_std(c::Matrix{Float64})
    return sqrt.(diag(weighted_cov(c)))
end
function weighted_std(c::Cloud)
    return weighted_std(c.particles)
end

"""
```
function weighted_cov(c::Cloud)
function weighted_cov(c::Matrix{Float64})
```
Compute weighted covariance of particle cloud.
"""
function weighted_cov(c::Matrix{Float64})
    return cov(get_vals(c; transpose = false),
               Weights(get_weights(c) / sum(get_weights(c))), corrected = false)
end
function weighted_cov(c::Cloud)
    return weighted_cov(c.particles)
end

"""
```
split_cloud(filename::String, n_pieces::Int)
```
splits the cloud saved in `filename` to multiple files (`n_pieces` of them).
For large clouds, the memory usage by one file may be too large. For example,
the file may exceed GitHub's 100MB memory limit.
"""
function split_cloud(filename::String, n_pieces::Int)
    cloud = load(filename, "cloud")
    w    = load(filename, "w")
    W    = load(filename, "W")
    n_part = size(cloud.particles, 1)
    n_para = size(cloud.particles, 2)

    @assert mod(n_part, n_pieces) == 0
    npart_small = Int(n_part / n_pieces)
    clouds = Vector{Cloud}(undef, n_pieces)
    ws     = Vector{Matrix{Float64}}(undef, n_pieces)
    Ws     = Vector{Matrix{Float64}}(undef, n_pieces)

    for i = 1:n_pieces
        inds = ((i-1)*npart_small+1):(i*npart_small)
        ws[i] = w[inds, :]
        Ws[i] = W[inds, :]
        clouds[i] = Cloud(n_para, npart_small)

        # Since loglh, logprior, old_loglh are all stored in cloud.particles, this updates them all too!
        SMC.update_cloud!(clouds[i], cloud.particles[inds, :])

        clouds[i].ESS = cloud.ESS
        clouds[i].c = cloud.c
        clouds[i].stage_index = cloud.stage_index
        clouds[i].total_sampling_time = cloud.total_sampling_time
        clouds[i].accept = cloud.accept
        clouds[i].n_Φ = cloud.n_Φ
        clouds[i].resamples = cloud.resamples
        clouds[i].tempering_schedule = cloud.tempering_schedule

        new_filename = replace(filename, ".jld2" => "_part$(i).jld2")
        jldopen(new_filename, true, true, true, IOStream) do file
            write(file, "cloud", clouds[i])
            write(file, "w", ws[i])
            write(file, "W", Ws[i])
        end
    end
end

"""
```
join_cloud(filename::String, n_pieces::Int; save_cloud::Bool = true)
```
joins a cloud saved in `filename` that had previously been
split into multiple files (`n_pieces` of them) and saves the newly
joined cloud when the kwarg `save_cloud` is `true`.
For large clouds, the memory usage by one file may be too large. For example,
the file may exceed GitHub's 100MB memory limit. For this reason,
it is useful to split the cloud file into multiple files and then
rejoin later.
"""
function join_cloud(filename::String, n_pieces::Int; save_cloud::Bool = true)
    clouds     = Vector{Cloud}(undef, n_pieces)
    ws         = Vector{Matrix{Float64}}(undef, n_pieces)
    Ws         = Vector{Matrix{Float64}}(undef, n_pieces)
    for i = 1:n_pieces
        small_filename = replace(filename, ".jld2" => "_part$(i).jld2")

        clouds[i]     = load(small_filename, "cloud")
        ws[i]         = load(small_filename, "w")
        Ws[i]         = load(small_filename, "W")
    end
    n_part  = sum([size(y.particles, 1) for y in clouds])
    n_para  = size(clouds[1].particles, 2)
    n_stage = size(ws[1], 2)

    cloud = Cloud(n_para, n_part)

    particles  = Matrix{Float64}(undef, 0, n_para)
    loglhs     = Vector{Float64}(undef, 0)
    old_loglhs = Vector{Float64}(undef, 0)
    logpriors  = Vector{Float64}(undef, 0)
    w          = Matrix{Float64}(undef, 0, n_stage)
    W          = Matrix{Float64}(undef, 0, n_stage)

    for i = 1:n_pieces
        particles     = vcat(particles, clouds[i].particles)
        loglhs        = vcat(loglhs, SMC.get_loglh(clouds[i]))
        old_loglhs    = vcat(old_loglhs, SMC.get_old_loglh(clouds[i]))
        logpriors     = vcat(logpriors, SMC.get_logprior(clouds[i]))
        w             = vcat(w, ws[i])
        W             = vcat(W, Ws[i])
    end

    # Since loglh, logprior, old_loglh are all stored in cloud.particles, this updates them all too!
    SMC.update_cloud!(cloud, particles)

    cloud.ESS                 = clouds[1].ESS
    cloud.c                   = clouds[1].c
    cloud.stage_index         = clouds[1].stage_index
    cloud.total_sampling_time = clouds[1].total_sampling_time
    cloud.accept              = clouds[1].accept
    cloud.n_Φ                 = clouds[1].n_Φ
    cloud.resamples           = clouds[1].resamples
    cloud.tempering_schedule  = clouds[1].tempering_schedule

    if save_cloud
        jldopen(filename, true, true, true, IOStream) do file
            write(file, "cloud", cloud)
            write(file, "w", w)
            write(file, "W", W)
        end
    end

    return cloud
end

"""
```
add_parameters_to_cloud(old_cloud_file::String, para::ParameterVector,
                        old_para_inds::BitVector; regime_switching::Bool = false)

add_parameters_to_cloud(old_cloud::Cloud, para::ParameterVector, old_para_inds::BitVector;
                        regime_switching::Bool = false) where {T <: Real}
```
extends a `Cloud` from a previous estimation to include new parameters.
This function helps construct a bridge distribution when
you want to estimate a model that extends a previous model by
adding additional parameters.

To be concrete, suppose we have two models ``\\mathcal{M}_1``
and ``\\mathcal{M}_2`` such that the parameters of the first model
are a subset of the parameters of the second model. For example,
suppose ``\\theta_1`` are the parameters for the first model,
and ``\\theta_2 = [\\theta_1, \\tilde{\\theta}]^T``,
where ``\\tilde{\\theta}`` are the new parameters for ``\\mathcal{M}_2``.
Assume that

(1) the likelihood function for ``\\mathcal{M}_1` does not depend on ``\tilde{\\theta}``,
(2) the priors for ``\theta_1`` and ``\tilde{\theta}`` are independent.

Then the posterior for ``\\theta_2`` given ``\\mathcal{M}_1`` is just
``math
\\begin{aligned}
  \\pi(\\theta_2 \\vert Y, \\mathcal{M}_1) = \\pi(\\theta_1 \\vert Y, \\mathcal{M}_1) p(\\tilde{\\theta}).
\\end{aligned}
``
Therefore, we can construct the posterior ``\\pi(\\theta_2 \\vert Y, \\mathcal{M}_1)``
by concatenating draws from the prior for ``\\tilde{\\theta}`` to the previous estimation of ``\\mathcal{M}_1``.
This function performs this concatenation and allows for regime-switching.

### Inputs
- `old_cloud` or `old_cloud_file`: this input should specify the `Cloud` from a previous estimation
    of the old model. If a different package was used to estimate the model, then the user
    can construct a new `Cloud` object. The only fields which must be set correctly
    are the `particles` and `ESS` field. The others can be set to default values (see `?Cloud` and
    the source code for the construction of the `Cloud` object).

- `para`: the parameter vector of the new model (i.e. ``\\theta_2``).
    The parameters in `para` that belonged to the old model should have the same
    settings as the ones used in previous estimation, e.g. the same prior.

- `old_para_inds`: indicates which parameters were used in the old estimation.
    If `regime_switching = true`, then this vector should specify which values
    correspond to old parameter values based on the matrix returned by
    `SMC.get_values(para)`. For example, if `old_para` is the `ParameterVector`
    used by the old estimation, then `SMC.get_values(old_para) == SMC.get_values(para)[old_para_inds]`

### Keyword Arguments
- `regime_switching`: this kwarg is needed to be able to draw from the prior correctly
    for the new parameters.
"""
function add_parameters_to_cloud(old_cloud_file::String, para::ParameterVector,
                                 old_para_inds::BitVector; regime_switching::Bool = false)
    old_cloud = load(old_cloud_file, "cloud")
    return add_parameters_to_cloud(old_cloud, para, old_para_inds;
                                   regime_switching = regime_switching)
end

function add_parameters_to_cloud(old_cloud::Cloud, para::ParameterVector{T}, old_para_inds::BitVector;
                                 regime_switching::Bool = false) where {T <: Real}

    # Sample from prior
    n_parts   = length(old_cloud)
    para_vals = regime_switching ? Matrix{T}(undef, n_parts, n_parameters_regime_switching(para)) :
        Matrix{T}(undef, n_parts, length(para))
    for i in 1:n_parts
        para_vals[i, :] = rand(para; regime_switching = regime_switching)
    end

    # Use same particles as the old cloud. Re-sampling is unnecessary
    # because old_cloud.particles has the needed weights information
    part_dim2 = size(old_cloud.particles, 2)
    old_para  = old_cloud.particles[:, 1:ind_para_end(part_dim2)]

    # Combine old parameters (drawn from their posterior from the old estimation)
    # and the new parameters (drawn from a prior)
    # It is assumed that the order of old_para matches the order of old_para_inds
    # and that the order of parameters haven't been switched around.
    # If that is the case, then the user need to write a function
    # that maps the old parameters to their correct indices.
    para_vals[:, old_para_inds] = old_para

    # Create logprior columns
    meta_info = Matrix{T}(undef, n_parts, 5) # additional 5 columns of "meta" information about particles
    for i in 1:n_parts
        # Update ParameterVector para
        update!(para, view(para_vals, i, :))

        # Compute logprior
        meta_info[i, 2] = prior(para)
    end

    # Copy loglh from old model
    meta_info[:, 1] = view(old_cloud.particles, :, ind_loglh(part_dim2))

    # Compute old loglh (just zeros since we will be creating a "new" Cloud)
    meta_info[:, 3] .= 0.

    # Add acceptance rate based on the bold cloud
    meta_info[:, 4] = view(old_cloud.particles, :, ind_accept(part_dim2))

    # Add weights for each particle based on the old cloud
    meta_info[:, 5] = view(old_cloud.particles, :, ind_weight(part_dim2))

    # Form a new Cloud
    return Cloud(hcat(para_vals, meta_info), zeros(1), old_cloud.ESS, 1, 0, 0, 0., .25, 0.)
end
