## Helper Functions

```@docs
Cloud(n_params::Int, n_parts::Int)
SMC.get_weights(c::Cloud)
SMC.get_vals(c::Cloud; transpose::Bool = true)
prior(parameters::ParameterVector{T}) where {T<:Number}
SMC.solve_adaptive_ϕ(cloud::Cloud, proposed_fixed_schedule::Vector{Float64},
                               i::Int64, j::Int64, ϕ_prop::Float64, ϕ_n1::Float64,
                               tempering_target::Float64, resampled_last_period::Bool)
mvnormal_mixture_draw(θ_old::Vector{T}, d_prop::Distribution;
                                        c::T = 1.0, α::T = 1.0) where T<:AbstractFloat
SMC.compute_ESS(loglh::Vector{T}, current_weights::Vector{T}, ϕ_n::T, ϕ_n1::T;
                              old_loglh::Vector{T} = zeros(length(loglh))) where {T<:AbstractFloat}
SMC.generate_free_blocks(n_free_para::Int64, n_blocks::Int64)
SMC.generate_all_blocks(blocks_free::Vector{Vector{Int64}}, free_para_inds::Vector{Int64})
mutation(loglikelihood::Function, parameters::ParameterVector{U},
                  data::Matrix{S}, p::Vector{S}, d_μ::Vector{S}, d_Σ::Matrix{S},
                  blocks_free::Vector{Vector{Int}}, blocks_all::Vector{Vector{Int}},
                  ϕ_n::S, ϕ_n1::S; c::S = 1., α::S = 1., n_mh_steps::Int = 1,
                  old_data::T = T(undef, size(data, 1), 0)) where {S<:AbstractFloat,
                                                                   T<:AbstractMatrix, U<:Number}
one_draw(loglikelihood::Function, parameters::ParameterVector{U},
                  data::Matrix{Float64}) where {U<:Number}
initial_draw!(loglikelihood::Function, parameters::ParameterVector{U},
                       data::Matrix{Float64}, c::Cloud; parallel::Bool = false) where {U<:Number}
draw_likelihood(loglikelihood::Function, parameters::ParameterVector{U},
                         data::Matrix{Float64}, draw::Vector{Float64}) where {U<:Number}
initialize_likelihoods!(loglikelihood::Function, parameters::ParameterVector{U},
                                 data::Matrix{Float64}, c::Cloud;
                                 parallel::Bool = false) where {U<:Number}
initialize_cloud_settings!(cloud::Cloud; tempered_update::Bool = false,
                                    n_parts::Int = 5_000, n_Φ::Int = 300, c::S = 0.5,
                                    accept::S = 0.25) where {S<:AbstractFloat}
```
