using ModelConstructors, SMC, DSGE, Test, Distributed, Dates, DataFrames, OrderedCollections, FileIO, DataStructures, LinearAlgebra, StatsBase, Random
@everywhere using SMC, DSGE, ModelConstructors, JLD2, Printf, LinearAlgebra

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
