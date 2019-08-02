using Test, Distributed, Dates, DataFrames, OrderedCollections, FileIO, DataStructures, LinearAlgebra, StatsBase, Random
@everywhere using DSGE, JLD2, Printf, LinearAlgebra

my_tests = [
            "smc",
            "helpers",
            "initialization",
            "resample",
            "util",
            "mutation"
            ]

for test in my_tests
    test_file = string("$test.jl")
    @printf " * %s\n" test_file
    include(test_file)
end
