var documenterSearchIndex = {"docs":
[{"location":"license/#License-1","page":"License","title":"License","text":"","category":"section"},{"location":"license/#","page":"License","title":"License","text":"Copyright (c) 2015, Federal Reserve Bank of New York All rights reserved.","category":"page"},{"location":"license/#","page":"License","title":"License","text":"Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:","category":"page"},{"location":"license/#","page":"License","title":"License","text":"Redistributions of source code must retain the above copyright notice, this list of","category":"page"},{"location":"license/#","page":"License","title":"License","text":"conditions and the following disclaimer.","category":"page"},{"location":"license/#","page":"License","title":"License","text":"Redistributions in binary form must reproduce the above copyright notice, this list of","category":"page"},{"location":"license/#","page":"License","title":"License","text":"conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.","category":"page"},{"location":"license/#","page":"License","title":"License","text":"Neither the name of the copyright holder nor the names of its contributors may be used to","category":"page"},{"location":"license/#","page":"License","title":"License","text":"endorse or promote products derived from this software without specific prior written permission.","category":"page"},{"location":"license/#","page":"License","title":"License","text":"THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.","category":"page"},{"location":"smc/#SMC-Main-Function-1","page":"Using SMC","title":"SMC Main Function","text":"","category":"section"},{"location":"smc/#","page":"Using SMC","title":"Using SMC","text":"You can use SMC to estimate any Bayesian model. This requires (1) parameters and their associated prior distributions (2) data (3) a log-likelihood function. These three ingredients are the only inputs into the smc driver.","category":"page"},{"location":"smc/#","page":"Using SMC","title":"Using SMC","text":"smc(loglikelihood::Function, parameters::ParameterVector, data::Matrix) function.","category":"page"},{"location":"smc/#","page":"Using SMC","title":"Using SMC","text":"Let's look at an example. First, we'll make a model. We need the ModelConstructors package to do that.","category":"page"},{"location":"smc/#","page":"Using SMC","title":"Using SMC","text":"using ModelConstructors\nols_model = GenericModel()","category":"page"},{"location":"smc/#","page":"Using SMC","title":"Using SMC","text":"Next, assign our intercept and coefficient parameters to the model.","category":"page"},{"location":"smc/#","page":"Using SMC","title":"Using SMC","text":"reg <= parameter(:α1, 0., (-1e5, 1e5), (-1e5, 1e5), Untransformed(), Normal(0, 10), fixed = false)\nreg <= parameter(:β1, 0., (-1e5, 1e5), (-1e5, 1e5), Untransformed(), Normal(0, 10), fixed = false)","category":"page"},{"location":"smc/#","page":"Using SMC","title":"Using SMC","text":"The first argument is the name of the parameter, the second is its initial value, the third and fourther are bounds on the parameter (they should always be the same). If the sampler draws a value outside of these bounds, it'll try again. The fifth argument is whether you would like to perform a transformation on the parameter when using it (ignore this for now–it's for more advanced users). The sixth argument is the prior, N(0 10), and last argument says the parameters aren't fixed parameters (i.e. we're trying to estimate them!)","category":"page"},{"location":"smc/#","page":"Using SMC","title":"Using SMC","text":"We'll make some artifical data to test our model","category":"page"},{"location":"smc/#","page":"Using SMC","title":"Using SMC","text":"X = rand(100) #hide\nβ = 1.\nα = 1.\ny = β*X + α","category":"page"},{"location":"smc/#","page":"Using SMC","title":"Using SMC","text":"We need a log-likelihood function! Note that it's important you define a log-likelihood function rather than a likelihood function.","category":"page"},{"location":"smc/#","page":"Using SMC","title":"Using SMC","text":"function likelihood_fnct(p, d)\n    α = p[1]\n    β = p[2]\n    Σ = 1\n    det_Σ = det(Σ)\n    inv_Σ = inv(Σ)\n    term1 = -N / 2 * log(2 * π) - 1 /2 * log(det_Σ)\n    logprob = 0.\n    errors = d[:, 1] .- α .- β .* d[:, 2]\n    for t in 1:size(d,1)\n        logprob += term1 - 1/2 * dot(errors, inv_Σ * errors)\n    end\n    return logprob\nend","category":"page"},{"location":"smc/#","page":"Using SMC","title":"Using SMC","text":"And that's it! Now let's run SMC.","category":"page"},{"location":"smc/#","page":"Using SMC","title":"Using SMC","text":"smc(likelihood_fnct, reg.parameters, data, n_parts = 10, use_fixed_schedule = false, tempering_target = 0.97)","category":"page"},{"location":"smc/#","page":"Using SMC","title":"Using SMC","text":"smc(loglikelihood::Function, parameters::ParameterVector{U}, data::Matrix{S};\n         kwargs...) where {S<:AbstractFloat, U<:Number}","category":"page"},{"location":"smc/#SMC.smc-Union{Tuple{U}, Tuple{S}, Tuple{Function,Array{AbstractParameter{U},1},Array{S,2}}} where U<:Number where S<:AbstractFloat","page":"Using SMC","title":"SMC.smc","text":"function smc(loglikelihood::Function, parameters::ParameterVector{U}, data::Matrix{S};\n             kwargs...) where {S<:AbstractFloat, U<:Number}\n\nArguments:\n\nloglikelihood::Function: Log-likelihood function of model being estimated. Takes parameters   and data as arguments.\nparameters::ParameterVector{U}: Model parameter vector, which stores parameter values,   prior dists, and bounds.\ndata: A matrix or dataframe containing the time series of the observables used in   the calculation of the posterior/loglikelihood\nold_data: A matrix containing the time series of observables of previous data   (with data being the new data) for the purposes of a time tempered estimation   (that is, using the posterior draws from a previous estimation as the initial set   of draws for an estimation with new data)\n\nKeyword Arguments:\n\nverbose::Symbol: Desired frequency of function progress messages printed to standard out.\n\n- `:none`: No status updates will be reported.\n- `:low`: Status updates for SMC initialization and recursion will be included.\n- `:high`: Status updates for every iteration of SMC is output, which includes\nthe mean and standard deviation of each parameter draw after each iteration,\nas well as calculated acceptance rate, ESS, and number of times resampled.\n\nparallel::Bool: Flag for running algorithm in parallel.\nn_parts::Int: Number of particles.\nn_blocks::Int: Number of parameter blocks in mutation step.\nn_mh_steps::Int: Number of Metropolis Hastings steps to attempt during the mutation step.\nλ::S: The 'bending coefficient' λ in Φ(n) = (n/N(Φ))^λ\nn_Φ::Int: Number of stages in the tempering schedule.\nresampling_method::Symbol: Which resampling method to use.\n:systematic: Will use sytematic resampling.\n:multinomial: Will use multinomial resampling.\n:polyalgo: Samples using a polyalgorithm.\nthreshold_ratio::S: Threshold s.t. particles will be resampled when the population   drops below threshold * N.\nc::S: Scaling factor for covariance of the particles. Controls size of steps in mutation step.\nα::S: The mixture proportion for the mutation step's proposal distribution.\ntarget::S: The initial target acceptance rate for new particles during mutation.\nuse_chand_recursion::Bool: Flag for using Chandrasekhar Recursions in Kalman filter.\nuse_fixed_schedule::Bool: Flag for whether or not to use a fixed tempering (ϕ) schedule.\ntempering_target::S: Coefficient of the sample size metric to be targeted when solving   for an endogenous ϕ.\nold_data::Matrix{S}: data from vintage of last SMC estimation. Running a bridge   estimation requires old_data and old_cloud.\nold_cloud::Cloud: associated cloud borne of old data in previous SMC estimation.   Running a bridge estimation requires old_data and old_cloud. If no old_cloud   is provided, then we will attempt to load one using loadpath.\nold_vintage::String: String for vintage date of old data\nsmc_iteration::Int: The iteration index for the number of times SMC has been run on the    same data vintage. Primarily for numerical accuracy/testing purposes.\nrun_test::Bool: Flag for when testing accuracy of program\nfilestring_addl::Vector{String}: Additional file string extension for loading old cloud.\nsave_intermediate::Bool: Flag for whether one wants to save intermediate Cloud objects\nintermediate_stage_increment::Int: Save Clouds at every increment  (1 = each stage, 10 = every 10th stage, etc.). Useful if you are using a cluster with time   limits because if you hit the time limit, then you can just   start from an intermediate stage rather than start over.\ncontinue_intermediate::Bool: Flag to indicate whether one is continuing SMC from an   intermediate stage.\nintermediate_stage_start::Int: Intermediate stage at which one wishes to begin the estimation.\ntempered_update_prior_weight::Float64 = 0.0: Weight placed on the current priors of parameters   to construct a convex combination of draws from current priors and the previous estimation's   cloud. The convex combination serves as the bridge distribution for a time tempered estimation.\nrun_csminwel::Bool = true: Flag to run the csminwel algorithm to identify the true posterior mode   (which may not exist) after completing an estimation. The mode identified by SMC is just   the particle with the highest posterior value, but we do not check it is actually a mode (i.e.   the Hessian is negative definite).\nregime_switching::Bool = false: Flag if there are regime-switching parameters. Otherwise, not all the values of the   regimes will be used or saved.\ntoggle::Bool = true: Flag for resetting the fields of parameter values to regime 1 anytime   the loglikelihood is computed. The regime-switching version of SMC assumes at various points   that this resetting occurs. If speed is important, then ensure that the fields of parameters   take their regime 1 values at the end of the loglikelihood computation and set toggle = false.\n\nOutputs\n\ncloud: The Cloud object containing all of the information about the   parameter values from the sample, their respective log-likelihoods, the ESS   schedule, tempering schedule etc., which is saved in the saveroot.\n\nOverview\n\nSequential Monte Carlo can be used in lieu of Random Walk Metropolis Hastings to     generate parameter samples from high-dimensional parameter spaces using     sequentially constructed proposal densities to be used in iterative importance     sampling.\n\nThis implementation is based on Edward Herbst and Frank Schorfheide's 2014 paper     'Sequential Monte Carlo Sampling for DSGE Models' and the code accompanying their     book 'Bayesian Estimation of DSGE Models'.\n\nSMC is broken up into three main steps:\n\nCorrection: Reweight the particles from stage n-1 by defining incremental weights,   which gradually \"temper in\" the loglikelihood function p(Y|θ)^(ϕn - ϕn-1) into the   normalized particle weights.\nSelection: Resample the particles if the distribution of particles begins to   degenerate, according to a tolerance level for the ESS.\nMutation: Propagate particles {θ(i), W(n)} via N(MH) steps of a Metropolis   Hastings algorithm.\n\n\n\n\n\n","category":"method"},{"location":"#SMC.jl-Documentation-1","page":"Home","title":"SMC.jl Documentation","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"Pages = [\n      \"smc.md\",\n      \"helpers.md\",\n      \"license.md\"\n]","category":"page"},{"location":"helpers/#Helper-Functions-1","page":"Helper Functions","title":"Helper Functions","text":"","category":"section"},{"location":"helpers/#","page":"Helper Functions","title":"Helper Functions","text":"Cloud(n_params::Int, n_parts::Int)\nSMC.get_weights(c::Cloud)\nSMC.get_vals(c::Cloud; transpose::Bool = true)\nprior(parameters::ParameterVector{T}) where {T<:Number}\nSMC.solve_adaptive_ϕ(cloud::Cloud, proposed_fixed_schedule::Vector{Float64},\n                               i::Int64, j::Int64, ϕ_prop::Float64, ϕ_n1::Float64,\n                               tempering_target::Float64, resampled_last_period::Bool)\nmvnormal_mixture_draw(θ_old::Vector{T}, d_prop::Distribution;\n                                        c::T = 1.0, α::T = 1.0) where T<:AbstractFloat\nSMC.compute_ESS(loglh::Vector{T}, current_weights::Vector{T}, ϕ_n::T, ϕ_n1::T;\n                              old_loglh::Vector{T} = zeros(length(loglh))) where {T<:AbstractFloat}\nSMC.generate_free_blocks(n_free_para::Int64, n_blocks::Int64)\nSMC.generate_all_blocks(blocks_free::Vector{Vector{Int64}}, free_para_inds::Vector{Int64})\nmutation(loglikelihood::Function, parameters::ParameterVector{U},\n                  data::Matrix{S}, p::Vector{S}, d_μ::Vector{S}, d_Σ::Matrix{S},\n                  blocks_free::Vector{Vector{Int}}, blocks_all::Vector{Vector{Int}},\n                  ϕ_n::S, ϕ_n1::S; c::S = 1., α::S = 1., n_mh_steps::Int = 1,\n                  old_data::T = T(undef, size(data, 1), 0)) where {S<:AbstractFloat,\n                                                                   T<:AbstractMatrix, U<:Number}\none_draw(loglikelihood::Function, parameters::ParameterVector{U},\n                  data::Matrix{Float64}) where {U<:Number}\ninitial_draw!(loglikelihood::Function, parameters::ParameterVector{U},\n                       data::Matrix{Float64}, c::Cloud; parallel::Bool = false) where {U<:Number}\ndraw_likelihood(loglikelihood::Function, parameters::ParameterVector{U},\n                         data::Matrix{Float64}, draw::Vector{Float64}) where {U<:Number}\ninitialize_likelihoods!(loglikelihood::Function, parameters::ParameterVector{U},\n                                 data::Matrix{Float64}, c::Cloud;\n                                 parallel::Bool = false) where {U<:Number}\ninitialize_cloud_settings!(cloud::Cloud; tempered_update::Bool = false,\n                                    n_parts::Int = 5_000, n_Φ::Int = 300, c::S = 0.5,\n                                    accept::S = 0.25) where {S<:AbstractFloat}","category":"page"},{"location":"helpers/#SMC.Cloud-Tuple{Int64,Int64}","page":"Helper Functions","title":"SMC.Cloud","text":"function Cloud(n_params::Int, n_parts::Int)\n\nEasier constructor for Cloud, which initializes the weights to be equal, and everything else in the particle object to be empty.\n\n\n\n\n\n","category":"method"},{"location":"helpers/#SMC.get_weights-Tuple{Cloud}","page":"Helper Functions","title":"SMC.get_weights","text":"function get_weights(c::Cloud)\n\nReturns Vector{Float64}(n_parts) of weights of particles in cloud.\n\n\n\n\n\n","category":"method"},{"location":"helpers/#SMC.get_vals-Tuple{Cloud}","page":"Helper Functions","title":"SMC.get_vals","text":"function get_vals(c::Matrix{Float64})\n\nReturns Matrix{Float64}(nparams, nparts) of parameter values in particle cloud.\n\n\n\n\n\n","category":"method"},{"location":"helpers/#SMC.solve_adaptive_ϕ-Tuple{Cloud,Array{Float64,1},Int64,Int64,Float64,Float64,Float64,Bool}","page":"Helper Functions","title":"SMC.solve_adaptive_ϕ","text":"`function solve_adaptive_ϕ(cloud::Cloud, proposed_fixed_schedule::Vector{Float64},\n                           i::Int64, j::Int64, ϕ_prop::Float64, ϕ_n1::Float64,\n                           tempering_target::Float64, resampled_last_period::Bool)`\n\nSolves for next Φ. Returns ϕn, resampledlastperiod, j, ϕprop.\n\n\n\n\n\n","category":"method"},{"location":"helpers/#SMC.mvnormal_mixture_draw-Union{Tuple{T}, Tuple{Array{T,1},Distribution}} where T<:AbstractFloat","page":"Helper Functions","title":"SMC.mvnormal_mixture_draw","text":"`mvnormal_mixture_draw(θ_old::Vector{T}, d_prop::Distribution;\n                       c::T = 1.0, α::T = 1.0) where T<:AbstractFloat`\n\nCreate a DegenerateMvNormal distribution object, d, from a parameter vector, p, and a standard deviation matrix (obtained from SVD), σ.\n\nGenerate a draw from the mixture distribution of:\n\nA DegenerateMvNormal centered at θ_old with the standard deviation matrix σ, scaled by cc^2 and with mixture proportion α.\nA DegenerateMvNormal centered at the same mean, but with a standard deviation matrix of the diagonal entries of σ scaled by cc^2 with mixture proportion (1 - α)/2.\nA DegenerateMvNormal  with the same standard deviation matrix σ but centered at the new proposed mean, θ_prop, scaled by cc^2, and with mixture proportion (1 - α)/2.\n\nIf no θ_prop is given, but an α is specified, then the mixture will consist of α of the standard distribution and (1 - α) of the diagonalized distribution.\n\nArguments\n\nθ_old::Vector{T}: The mean of the desired distribution\nσ::Matrix{T}: The standard deviation matrix of the desired distribution\n\nKeyword Arguments\n\ncc::T: The standard deviation matrix scaling factor\nα::T: The mixing proportion\nθ_prop::Vector{T}: The proposed parameter vector to be used as part of the mixture distribution, set by default to be the weighted mean of the particles, prior to mutation.\n\nOutputs\n\nθ_new::Vector{T}: The draw from the mixture distribution to be used as the MH proposed step\n\n\n\n\n\n","category":"method"},{"location":"helpers/#SMC.compute_ESS-Union{Tuple{T}, Tuple{Array{T,1},Array{T,1},T,T}} where T<:AbstractFloat","page":"Helper Functions","title":"SMC.compute_ESS","text":"function `compute_ESS(loglh::Vector{T}, current_weights::Vector{T}, ϕ_n::T, ϕ_n1::T;\n                     old_loglh::Vector{T} = zeros(length(loglh))) where {T<:AbstractFloat}`\n\nCompute ESS given log likelihood, current weights, ϕn, ϕ{n-1}, and old log likelihood.\n\n\n\n\n\n","category":"method"},{"location":"helpers/#SMC.generate_free_blocks-Tuple{Int64,Int64}","page":"Helper Functions","title":"SMC.generate_free_blocks","text":"`generate_free_blocks(n_free_para::Int64, n_blocks::Int64)`\n\nReturn a Vector{Vector{Int64}} where each internal Vector{Int64} contains a subset of the range 1:nfreepara of randomly permuted indices. This is used to index out random blocks of free parameters from the covariance matrix for the mutation step.\n\n\n\n\n\n","category":"method"},{"location":"helpers/#SMC.generate_all_blocks-Tuple{Array{Array{Int64,1},1},Array{Int64,1}}","page":"Helper Functions","title":"SMC.generate_all_blocks","text":"`generate_all_blocks(blocks_free::Vector{Vector{Int64}}, free_para_inds::Vector{Int64})`\n\nReturn a Vector{Vector{Int64}} where each internal Vector{Int64} contains indices corresponding to those in blocks_free but mapping to 1:n_para (as opposed to 1:n_free_para). These blocks are used to reconstruct the particle vector by inserting the mutated free parameters into the size n_para, particle vector, which also contains fixed parameters.\n\n\n\n\n\n","category":"method"},{"location":"helpers/#SMC.initial_draw!-Union{Tuple{U}, Tuple{Function,Array{AbstractParameter{U},1},Array{Float64,2},Cloud}} where U<:Number","page":"Helper Functions","title":"SMC.initial_draw!","text":"function initial_draw!(loglikelihood::Function, parameters::ParameterVector{U},\n                       data::Matrix{Float64}, c::Cloud; parallel::Bool = false,\n                       regime_switching::Bool = false, toggle::Bool = true) where {U<:Number}\n\nDraw from a general starting distribution (set by default to be from the prior) to initialize the SMC algorithm. Returns a tuple (logpost, loglh) and modifies the particle objects in the particle cloud in place.\n\nSet regime_switching to true if there are regime-switching parameters. Otherwise, not all the values of the regimes will be used or saved.\n\nSet toggle to false if, after calculating the loglikelihood, the values in the fields of every parameter in parameters are set to their regime 1 values. The regime-switching version of rand requires that the fields of all parameters take their regime 1 values, or else sampling may be wrong. The default is true as a safety, but if speed is a paramount concern, setting toggle = true will avoid unnecessary computations.\n\n\n\n\n\n","category":"method"}]
}
