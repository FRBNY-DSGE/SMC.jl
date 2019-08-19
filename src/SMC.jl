isdefined(Base, :__precompile__) && __precompile__(false)

module SMC
    using BenchmarkTools, DataFrames, Distributed, Distributions
    using FileIO, HDF5, JLD2, LinearAlgebra, Random, Test
    using ModelConstructors

    using Roots: fzero, ConvergenceFailed
    using StatsBase: sample, Weights

    import Base: <, isempty, min, max
    import Calculus, ModelConstructors
    import ModelConstructors: update!,
                              @test_matrix_approx_eq,
                              @test_matrix_approx_eq_eps

    export
        compute_parameter_covariance, prior, get_estimation_output_files,
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
