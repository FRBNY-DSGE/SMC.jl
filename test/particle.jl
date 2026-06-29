if !@isdefined(run_benchmarks); run_benchmarks = false; end

# split/join is a deterministic serialization round-trip — RNG-independent — so it reads
# the existing "150" cloud regardless of the Julia version that generated it.
ver = VERSION < v"1.5" ? "111" : "150"

file = string("reference/smc_cloud_fix=true_version=", ver, ".jld2")
cloud = load(file, "cloud")
split_cloud(file, 2)
rejoined_cloud = join_cloud(file, 2)

if run_benchmarks
    # Benchmarked here (before the rm cleanup below) so the _part* files split_cloud
    # writes are still present for join_cloud to read.
    b_split = @benchmark split_cloud($file, 2)
    b_join  = @benchmark join_cloud($file, 2)
    println("\n===== particle.jl benchmark results =====")
    for (label, b) in (("split_cloud", b_split), ("join_cloud", b_join))
        println(rpad(label, 14), " time: ",
                rpad(BenchmarkTools.prettytime(median(b).time), 12),
                "memory: ", BenchmarkTools.prettymemory(median(b).memory))
    end
end

@testset "Test split and join clouds" begin
    @test cloud.particles           == rejoined_cloud.particles
    @test SMC.get_vals(cloud)       == SMC.get_vals(rejoined_cloud)
    @test SMC.get_loglh(cloud)      == SMC.get_loglh(rejoined_cloud)
    @test SMC.get_old_loglh(cloud)   == SMC.get_old_loglh(rejoined_cloud)
    @test SMC.get_logpost(cloud)    == SMC.get_logpost(rejoined_cloud)
    @test cloud.ESS                 == rejoined_cloud.ESS
    @test cloud.c                   == rejoined_cloud.c
    @test cloud.stage_index         == rejoined_cloud.stage_index
    @test cloud.total_sampling_time == rejoined_cloud.total_sampling_time
    @test cloud.accept              == rejoined_cloud.accept
    @test cloud.n_Φ                 == rejoined_cloud.n_Φ
    @test cloud.resamples           == rejoined_cloud.resamples
    @test cloud.tempering_schedule  == rejoined_cloud.tempering_schedule
end

rm("reference/smc_cloud_fix=true_version=$(ver)_part1.jld2")
rm("reference/smc_cloud_fix=true_version=$(ver)_part2.jld2")
