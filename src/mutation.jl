"""
```
mutation(loglikelihood::Function, parameters::ParameterVector{U},
         data::Matrix{S}, p::Vector{S}, d_μ::Vector{S}, d_Σ::Matrix{S},
         blocks_free::Vector{Vector{Int}}, blocks_all::Vector{Vector{Int}},
         ϕ_n::S, ϕ_n1::S; c::S = 1., α::S = 1., n_mh_steps::Int = 1,
         old_data::T = T(undef, size(data, 1), 0)) where {S<:AbstractFloat,
                                                          T<:AbstractMatrix, U<:Number}
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
    old_loglh, old_logpost in time tempering

### Outputs:
- `p::Vector{Float64}`: An updated particle containing updated parameter values,
    log-likelihood, prior, and acceptance indicator.

"""
function mutation(mutation_method::Symbol, loglikelihood::Function, parameters::ParameterVector{U},
                  data::Matrix{S}, p::Vector{S}, d_μ::Vector{S}, d_Σ::Matrix{S},
                  blocks_free::Vector{Vector{Int}}, blocks_all::Vector{Vector{Int}},
                  ϕ_n::S, ϕ_n1::S; c::S = 1., α::S = 1., n_mh_steps::Int = 1,
                  old_data::T = T(undef, size(data, 1), 0)) where {S<:AbstractFloat,
                                                                   T<:AbstractMatrix, U<:Number}
    if mutation_method == :MH
        mutation_mh(loglikelihood, parameters, data, p, d_μ, d_Σ, blocks_free, blocks_all,
                    ϕ_n, ϕ_n1; c = c, α = α, n_mh_steps = n_mh_steps, old_data = old_data)
    #else if mutation_method == :HMC
    else
        throw(error("Method for mutation not recognized. Options are " *
                    "Metropolis-Hastings (:MH) and Hamiltonian Monte Carlo (:HMC)."))
    end

end

function mutation_mh(loglikelihood::Function, parameters::ParameterVector{U},
                     data::Matrix{S}, p::Vector{S}, d_μ::Vector{S}, d_Σ::Matrix{S},
                     blocks_free::Vector{Vector{Int}}, blocks_all::Vector{Vector{Int}},
                     ϕ_n::S, ϕ_n1::S; c::S = 1., α::S = 1., n_mh_steps::Int = 1,
                     old_data::T = T(undef, size(data, 1), 0)) where {S<:AbstractFloat,
                                                                      T<:AbstractMatrix, U<:Number}

    n_free_para = length([!θ.fixed for θ in parameters])
    step_prob   = rand() # Draw initial step probability

    N         = length(p)
    para      = p[1:ind_para_end(N)]
    like      = p[ind_loglh(N)]
    logprior  = p[ind_logprior(N)]
    like_prev = p[ind_old_loglh(N)] # Likelihood evaluated at the old data (for time tempering)
    accept    = 0.0

    d = MvNormal(d_μ, d_Σ)

    for step in 1:n_mh_steps
        for (block_f, block_a) in zip(blocks_free, blocks_all)

            # Index out parameters corresponding to given random block, create distribution
            # centered at weighted mean, with Σ corresponding to the same random block
            para_subset = para[block_a]
            d_subset    = MvNormal(d.μ[block_f], d.Σ.mat[block_f, block_f])
            para_draw   = mvnormal_mixture_draw(para_subset, d_subset; c = c, α = α)

            q0, q1 = compute_proposal_densities(para_draw, para_subset,
                                                d_subset, c = c, α = α)

            para_new          = deepcopy(para)
            para_new[block_a] = para_draw

            like_init, prior_init = like, logprior
            prior_new = like_new = like_old_data = -Inf
            try
                update!(parameters, para_new)
                para_new  = [θ.value for θ in parameters]
                prior_new = prior(parameters)
                like_new  = loglikelihood(parameters, data)

                if like_new == -Inf
                    prior_new = like_old_data = -Inf
                end

                like_old_data = isempty(old_data) ? 0. : loglikelihood(parameters, old_data)

            catch err
                if isa(err, ParamBoundsError) || isa(err, LinearAlgebra.LAPACKException) ||
                   isa(err, PosDefException)  || isa(err, SingularException)
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
