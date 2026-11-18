using Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..", "..")))
using Test, ModelConstructors, SMC, LinearAlgebra, Distributions, Random, HDF5, FileIO, JLD2
cd(joinpath(@__DIR__, "..", "..", "test")) do
    include(joinpath(pwd(), "diagnostics.jl"))
end
