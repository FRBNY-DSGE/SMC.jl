using ModelConstructors, SMC, DSGE, Test, Printf, Distributed

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
