using Test
using ModelConstructors, SMC
using LinearAlgebra, PDMats, Distributions
using Printf, Distributed, Random, HDF5, FileIO, JLD2
using BenchmarkTools

# The test files reference data/output with paths relative to this directory (e.g.
# "reference/test_data.h5"). cd here so they resolve no matter where Julia was launched
# from (Pkg.test, include from the package root, the REPL, …).
cd(@__DIR__)

# Benchmarks are for manual runs only — the files share this scope, so without a definition
# here the first file to default run_benchmarks = true would leak it on to every later file
# (running expensive/under-set-up @benchmark/@btime blocks in CI). Pin it off for the suite;
# standalone `include("<file>.jl")` runs still honor each file's own default.
run_benchmarks = false

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

# Run every test file to completion, even if some throw, then report at the end.
# (A failing file still makes the suite exit non-zero — but only after all have run.)
failures = Tuple{String, Any}[]
for test in my_tests
    @printf " * %s.jl\n" test
    try
        result = @timed include("$test.jl")
        @printf "   time: %.3f s  |  memory: %.2f MiB\n" result.time result.bytes/1024^2
    catch err
        push!(failures, (test, err))
        @error "Test file failed" file = "$test.jl" exception = (err, catch_backtrace())
    end
end

println("\n", "="^70)
@printf "TEST SUMMARY: %d of %d test files passed\n" (length(my_tests) - length(failures)) length(my_tests)
if !isempty(failures)
    println("Failed test files:")
    for (test, _) in failures
        println("  ✗ ", test, ".jl")
    end
    error("$(length(failures)) test file(s) failed")
else
    println("All test files passed.")
end
