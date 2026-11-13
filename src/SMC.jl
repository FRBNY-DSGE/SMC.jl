isdefined(Base, :__precompile__) && __precompile__(false)

module SMC
    using Dates, Distributed, Distributions
    using FileIO, HDF5, JLD2, LinearAlgebra, Random
    using ModelConstructors

    using Roots: fzero, ConvergenceFailed
    using StatsBase: sample, Weights

    import Base.<, Base.isempty, Base.min, Base.max
    import Calculus, ModelConstructors

    export
        mutation, resample, smc,
        mvnormal_mixture_draw, initial_draw!, Cloud, get_cloud,
        isempty, join_cloud, split_cloud, add_parameters_to_cloud,
        smc_log_mdd, stage_health_flags, write_stage_health, lineage_nse,
        modified_harmonic_mean, bridge_sampling_estimate, gaussian_mixture_bic,
        nse_validation, compare_estimates, comparison_z, pool_groups, mutation_shift,
        # marginal data density estimators (mdd_estimators.jl)
        BoundsTransform, bounds_transform, bounds_to_real, bounds_to_model, bounds_logjac,
        bounds_logjac_real, bounds_in_support, parameter_slots, duplicate_clusters,
        bridge_sampling, bridge_draws, bridge_combine, bridge_estimate, proposal_parameters,
        evaluate_proposals, is_lower_bound, is_bound_draws, is_bound_combine,
        effective_prior_lognorm, prior_inbounds_logmass

    const VERBOSITY   = Dict(:none => 0, :low => 1, :high => 2)
    const DATE_FORMAT = "yymmdd"

    include("rng.jl")
    include("particle.jl")
    include("initialization.jl")
    include("helpers.jl")
    include("util.jl")
    include("mutation.jl")
    include("resample.jl")
    include("diagnostics.jl")
    include("drift.jl")
    include("mdd_estimators.jl")
    include("groups.jl")
    include("smc_main.jl")
end
