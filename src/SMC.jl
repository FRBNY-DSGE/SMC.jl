isdefined(Base, :__precompile__) && __precompile__()

module SMC
    using CSV, DataFrames, DataStructures, OrderedCollections
    using Distributed, Distributions, Dates, Test, BenchmarkTools
    using FileIO, FredData, HDF5, JLD2, LinearAlgebra, Missings
    using Nullables, Printf, Random, RecipesBase, SparseArrays
    using SpecialFunctions, ModelConstructors

    using Roots: fzero, ConvergenceFailed
    using StatsBase: sample, Weights

    import Base: <, isempty, min, max
    import Calculus
    import ModelConstructors
    import ModelConstructors: update!

    export
        compute_parameter_covariance, prior, get_estimation_output_files,
        compute_moments, find_density_bands, mutation, resample, smc,
        mvnormal_mixture_draw, nearest_spd, marginal_data_density,
        initial_draw!, Cloud, get_cloud,

        # util
        @test_matrix_approx_eq, @test_matrix_approx_eq_eps

    const VERBOSITY = Dict(:none => 0, :low => 1, :high => 2)

    include("particle.jl")
    include("initialization.jl")
    include("helpers.jl")
    include("util.jl")
    include("mutation.jl")
    include("resample.jl")
    include("smc.jl")
end
