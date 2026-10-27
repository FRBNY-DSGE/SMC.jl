using Test, Distributed, SMC, ModelConstructors, Distributions, JLD2, LinearAlgebra
@everywhere using SMC, ModelConstructors, Distributions
BLAS.set_num_threads(1)

@everywhere function issue9_regime_likelihood(p, data)
    return -0.5 * ((regime_val(p[1], 1) - 0.2)^2 + (regime_val(p[1], 2) + 0.3)^2)
end

function issue9_regime_parameters()
    m = GenericModel()
    m <= parameter(:a, 0., (-10., 10.), (-10., 10.), Untransformed(), Normal(0, 1), fixed=false)
    set_regime_val!(m[:a], 1, 0.)
    set_regime_val!(m[:a], 2, 0.)
    set_regime_prior!(m[:a], 1, Normal(0, 1))
    set_regime_prior!(m[:a], 2, Normal(0, 1))
    return m.parameters
end

mktempdir() do dir
    function run(name; parallel=false)
        path = joinpath(dir, name * ".jld2")
        SMC.smc(issue9_regime_likelihood, issue9_regime_parameters(), zeros(1, 1);
                n_parts=32, n_Φ=5, n_blocks=2, n_mh_steps=1,
                parallel=parallel, verbose=:none, smc_seed=42,
                regime_switching=true, use_fixed_schedule=true,
                resampling_method=:systematic, threshold_ratio=0.9,
                savepath=path, particle_store_path=replace(path, ".jld2"=>".h5"))
        return load(path)
    end
    serial = run("serial")
    repeated = run("repeated"; parallel=nworkers()>1)
    @testset "Regime-switching SMC streams" begin
        @test serial["cloud"].particles == repeated["cloud"].particles
        @test serial["cloud"].ESS == repeated["cloud"].ESS
        @test serial["w"] == repeated["w"]
        @test serial["W"] == repeated["W"]
    end
end
