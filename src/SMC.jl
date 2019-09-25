isdefined(Base, :__precompile__) && __precompile__(false)

module SMC
    using BenchmarkTools, DataFrames, Dates, Distributed, Distributions
    using FileIO, HDF5, JLD2, LinearAlgebra, Random, Test
    using ModelConstructors
    using LogDensityProblems, TransformVariables, DynamicHMC

    using Roots: fzero, ConvergenceFailed
    using StatsBase: sample, Weights

    import Base.<, Base.isempty, Base.min, Base.max
    import Calculus, ModelConstructors

    export
        compute_parameter_covariance, get_estimation_output_files,
        compute_moments, find_density_bands, mutation, resample, smc,
        mvnormal_mixture_draw, nearest_spd, marginal_data_density,
        initial_draw!, Cloud, get_cloud

    const VERBOSITY   = Dict(:none => 0, :low => 1, :high => 2)
    const DATE_FORMAT = "yymmdd"

    include("particle.jl")
    include("initialization.jl")
    include("helpers.jl")
    include("util.jl")
    include("mutation.jl")
    include("resample.jl")
    include("smc.jl")
end
