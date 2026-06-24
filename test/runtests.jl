using Test
using ModelConstructors, SMC
using LinearAlgebra, PDMats, Distributions
using Printf, Distributed, Random, HDF5, FileIO, JLD2
using BenchmarkTools

my_tests = [
            "helpers",
            "initialization",
            "resample",
            "util",
            "mutation",
            "particle",
            "smc",
            "regime_switching_smc"
            ]

for test in my_tests
    test_file = string("$test.jl")
    @printf " * %s\n" test_file
    result = @timed include(test_file)
    @printf "   time: %.3f s  |  memory: %.2f MiB\n" result.time result.bytes/1024^2
end
