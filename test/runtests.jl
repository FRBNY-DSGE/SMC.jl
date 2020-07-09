using Test
using ModelConstructors, SMC
using LinearAlgebra, PDMats, Distributions, SparseArrays
using Printf, Distributed, Random, HDF5, FileIO, JLD2

import SparseArrays.SparseMatrixCSC
import ModelConstructors.Setting

my_tests = [
            "smc",
            "helpers",
            "initialization",
            "resample",
            "util",
            "mutation",
            "particle"
            ]

for test in my_tests
    test_file = string("$test.jl")
    @printf " * %s\n" test_file
    include(test_file)
end

