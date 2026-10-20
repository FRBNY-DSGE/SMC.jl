function mutation_block_proposal(μ::Vector{S}, Σ::Matrix{S}, block::Vector{Int},
                                 c::S, α::S) where {S<:AbstractFloat}
    @assert 0 <= α <= 1
    # The covariance and its factorizations remain fixed across MH steps.
    subset = MvNormal(μ[block], Σ[block, block])
    scaled_draw = c^2 * subset.Σ
    bar = MvNormal(subset.μ, scaled_draw)
    diagonal = MvNormal(subset.μ, Diagonal(diag(scaled_draw)))

    # Match compute_proposal_densities, which passes a covariance matrix
    # to DegenerateMvNormal with stdev = false.
    density_cov = c^2 * get_cov(subset)
    singular_values = svd(density_cov).S
    positive = findall(x -> x > 0.0, singular_values)
    normalizer = length(positive) * Distributions.log2π +
                 sum(log.(singular_values[positive]))
    inverse = inv(density_cov[positive, positive])

    return (full_cov = bar.Σ, diagonal_cov = diagonal.Σ, bar = bar,
            weights = [α, (1 - α) / 2, (1 - α) / 2],
            diagonal_stdev = sqrt.(diag(get_cov(subset))),
            density_indices = positive, density_normalizer = normalizer,
            density_inverse = inverse)
end

function draw_mutation_proposal(θ_old::Vector{S}, proposal) where {S<:AbstractFloat}
    full_old = MvNormal(θ_old, proposal.full_cov)
    diagonal_old = MvNormal(θ_old, proposal.diagonal_cov)
    mixture = MixtureModel(MvNormal[full_old, diagonal_old, proposal.bar], proposal.weights)
    return rand(mixture)
end

function mutation_normal_logpdf(x::Vector{S}, μ::Vector{S}, proposal) where {S<:AbstractFloat}
    positive = proposal.density_indices
    difference = x[positive] .- μ[positive]
    return -(proposal.density_normalizer +
             (difference' * proposal.density_inverse * difference)) / 2.0
end

function mutation_proposal_densities(θ_new::Vector{S}, θ_old::Vector{S},
                                     proposal, α::S) where {S<:AbstractFloat}
    q0 = α * exp(mutation_normal_logpdf(θ_old, θ_new, proposal))
    q1 = α * exp(mutation_normal_logpdf(θ_new, θ_old, proposal))
    diagonal = one(S)
    for i in eachindex(θ_old)
        σ = proposal.diagonal_stdev[i]
        z = (θ_old[i] - θ_new[i]) / σ
        diagonal = diagonal / (σ * sqrt(2.0 * π)) * exp(-0.5 * z^2)
    end
    q0 += (1.0 - α) / 2.0 * diagonal
    q1 += (1.0 - α) / 2.0 * diagonal
    q0 += (1.0 - α) / 2.0 * exp(mutation_normal_logpdf(θ_old, proposal.bar.μ, proposal))
    q1 += (1.0 - α) / 2.0 * exp(mutation_normal_logpdf(θ_new, proposal.bar.μ, proposal))
    q0, q1 = log(q0), log(q1)
    if q0 == Inf && q1 == Inf
        q0 = zero(S)
    end
    return q0, q1
end

"""
```
mutation(loglikelihood::Function, parameters::ParameterVector{U},
         data::Matrix{S}, p::Vector{S}, d_μ::Vector{S}, d_Σ::Matrix{S},
         blocks_free::Vector{Vector{Int}}, blocks_all::Vector{Vector{Int}},
         ϕ_n::S, ϕ_n1::S; c::S = 1., α::S = 1., n_mh_steps::Int = 1,
         old_data::T = T(undef, size(data, 1), 0),
         old_loglikelihood::Function = loglikelihood,
         regime_switching::Bool = false) where {S<:AbstractFloat,T<:AbstractMatrix, U<:Number})
```

Execute one proposed move of the Metropolis-Hastings algorithm for a given parameter

### Arguments:
- `loglikelihood::Function`: Likelihood function of model being estimated.
- `parameters::ParameterVector{U}`: Model parameter vector, which stores parameter
    values, prior dists, and bounds
- `data::Matrix{Float64}`: Matrix of data
- `p::Vector{Float64}`: Initial particle value
- `d::Distribution`: A distribution with μ = the weighted mean, and Σ = the weighted
    variance/covariance matrix
- `blocks_free::Vector{Vector{Int64}}`: A vector of index blocks, where the indices
    in each block corresponds to the ordering of free parameters only (e.g. all the
    indices will be ∈ 1:n_free_parameters)
- `blocks_all::Vector{Vector{Int64}}`: A vector of index blocks, where the indices in
    each block corresponds to the ordering of all parameters (e.g. all the indices will
    be in ∈ 1:n_para, free and fixed)
- `ϕ_n::Float64`: The current tempering factor
- `ϕ_n1::Float64`: The previous tempering factor

### Keyword Arguments:
- `c::Float64`: The scaling parameter for the proposed covariance matrix
- `α::Float64`: The mixing proportion
- `n_mh_steps::Int`: Number of Metropolis Hastings steps to attempt
- `old_data::Matrix{Float64}`: The matrix of old data to be used in calculating the
    old_loglh, old_logprior in time tempering
- `old_loglikelihood::Function = loglikelihood`: the old log-likelihood function when running
    a time tempered estimation. This function should be able to evaluate the log-likelihood
    of `old_data` given the current `parameters`, which may be nontrivial if the current
    `parameters` include additional parameters relative to the old estimation.
    By default, we assume the log-likelihood function has not changed
    and therefore coincides with the current one, so that the log-likelihood of `old_data`
    can be computed as `loglikelihood(parameters, old_data)`
- `regime_switching::Bool`: Set to true if there are regime-switching parameters.
    Otherwise, not all the values of the regimes will be used or saved.
- `toggle::Bool = true`: Flag for resetting the fields of parameter values to regime 1 anytime
    the loglikelihood is computed. The regime-switching version of SMC assumes at various points
    that this resetting occurs. If speed is important, then ensure that the fields of parameters
    take their regime 1 values at the end of the loglikelihood computation and set `toggle = false`.

### Outputs:
- `p::Vector{Float64}`: An updated particle containing updated parameter values,
    log-likelihood, prior, and acceptance indicator.

"""
function mutation(loglikelihood::Function, parameters::ParameterVector{U},
                  data::Matrix{S}, p::Vector{S}, d_μ::Vector{S}, d_Σ::Matrix{S},
                  n_free_para::Int,
                  blocks_free::Vector{Vector{Int}}, blocks_all::Vector{Vector{Int}},
                  ϕ_n::S, ϕ_n1::S; c::S = 1., α::S = 1., n_mh_steps::Int = 1,
                  old_data::T = T(undef, size(data, 1), 0),
                  old_loglikelihood::Function = loglikelihood,
                  regime_switching::Bool = false,
                  toggle::Bool = true) where {S<:AbstractFloat,T<:AbstractMatrix, U<:Number}

    step_prob   = rand() # Draw initial step probability

    N         = length(p)
    para      = p[1:ind_para_end(N)]
    like      = p[ind_loglh(N)]
    logprior  = p[ind_logprior(N)]
    like_prev = p[ind_old_loglh(N)] # Likelihood evaluated at the old data (for time tempering)
    accept    = 0.0

    if n_mh_steps <= 0
        update_mutation!(p, para, like, logprior, like_prev, accept / n_free_para)
        return p
    end
    block_proposals = [mutation_block_proposal(d_μ, d_Σ, block, c, α)
                       for (block, _) in zip(blocks_free, blocks_all)]

    for step in 1:n_mh_steps
        for (block_a, proposal) in zip(blocks_all, block_proposals)

            # Only the proposal center changes between MH steps.
            para_subset = para[block_a]
            para_draw = draw_mutation_proposal(para_subset, proposal)
            q0, q1 = mutation_proposal_densities(para_draw, para_subset,
                                                 proposal, α)

            para_new          = deepcopy(para)
            para_new[block_a] = para_draw
            like_init, prior_init = like, logprior
            prior_new = like_new = like_old_data = -Inf
            try
                @assert length(para_new) == sum(ModelConstructors.n_param_regs(parameters)) ## Delete for speed after testing
                update!(parameters, para_new)
                prior_new = prior(parameters)
                like_new  = loglikelihood(parameters, data)
                if toggle
                    toggle_regime!(parameters, 1)
                end
                if like_new == -Inf
                    prior_new = like_old_data = -Inf
                end


                if isempty(old_data)
                    like_old_data = 0.
                elseif old_loglikelihood === loglikelihood
                    like_old_data = old_loglikelihood(parameters, old_data)
                else
                    like_old_data = old_loglikelihood(parameters, old_data; new_model_params = true)
                end
                if toggle && isempty(old_data)
                    toggle_regime!(parameters, 1)
                end

            catch err
                if isa(err, ParamBoundsError) || isa(err, LinearAlgebra.LAPACKException) ||
                   isa(err, PosDefException)  || isa(err, SingularException)             ||
                   isa(err, DomainError)
                    prior_new = like_new = like_old_data = -Inf
                else
                    throw(err)
                end
            end

            η = exp(ϕ_n * (like_new - like_init) + (1 - ϕ_n) * (like_old_data - like_prev) +
                    (prior_new - prior_init) + (q0 - q1))



            if step_prob < η
                para      = para_new
                like      = like_new
                logprior  = prior_new
                like_prev = like_old_data
                accept   += length(block_a)

            end
            step_prob = rand() # Draw again for next step
        end
    end

    update_mutation!(p, para, like, logprior, like_prev, accept / n_free_para)
    return p
end
