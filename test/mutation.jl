write_test_output = false
if !@isdefined(run_benchmarks); run_benchmarks = false; end
include("modelsetup.jl")

# Julia 1.7+ switched the default RNG to a per-Task Xoshiro256++, so the seeded MH
# mutation draws differ from the "150" data — regenerate with write_test_output on.
if VERSION < v"1.5"
    ver = "111"
elseif VERSION < v"1.7"
    ver = "150"
else
    ver = "1126"
end

path = dirname(@__FILE__)

m = setup_linear_model()

save = normpath(joinpath(dirname(@__FILE__),"save"))
m <= Setting(:saveroot, saveroot)

data = h5read("reference/test_data.h5", "data")

n_parts = get_setting(m, :n_particles)
n_params = length(m.parameters)

file = JLD2.jldopen("reference/mutation_inputs.jld2", "r")
old_cloud = read(file, "particles")

d = as_mvnormal(read(file, "d"))
blocks_free = read(file, "blocks_free")
blocks_all = read(file, "blocks_all")
ϕ_n = read(file, "ϕ_n")
ϕ_n1 = read(file, "ϕ_n1")
c = read(file, "c")
α = read(file, "α")
old_data = read(file, "old_data")
close(file)


Random.seed!(42)

new_cloud = Cloud(n_params, n_parts)
for i in 1:n_parts
    new_cloud.particles[i,:] = SMC.mutation(loglik_fn, m.parameters, data,
                          	       old_cloud.particles[i, :], d.μ, Matrix(d.Σ),
                              	       n_params, 
                              	       blocks_free, blocks_all, ϕ_n, ϕ_n1;
                              	       c = c, α = α, old_data = old_data)
end

if write_test_output
    JLD2.jldopen(string("reference/mutation_outputs_version=", ver, ".jld2"), "w") do file
        write(file, "particles", new_cloud)
    end
end

saved_cloud = load(string("reference/mutation_outputs_version=", ver, ".jld2"), "particles")

@testset "Test mutation outputs, particle by particle" begin
    for i = 1:n_parts
        @test isapprox(saved_cloud.particles[i, :], new_cloud.particles[i, :], nans = true)
    end
end

# Benchmark last so it doesn't consume the RNG before the seeded mutation loop above.
if run_benchmarks
    @btime SMC.mutation($loglik_fn, $m.parameters, $data,
                        $(old_cloud.particles[1, :]), $(d.μ), $(Matrix(d.Σ)),
                        $n_params, $blocks_free, $blocks_all, $ϕ_n, $ϕ_n1;
                        c = $c, α = $α, old_data = $old_data)
end

# N3: the acceptance column is the rate per MH step (it used to sum over the steps, so the
# adaptation of c targeted target / n_mh_steps). With a tiny step every proposal is accepted.
@testset "Acceptance is per MH step" begin
    # A particle whose stored likelihood and prior match its parameters, with no old data
    p0 = copy(old_cloud.particles[1, :])
    N  = length(p0)
    ModelConstructors.update!(m.parameters, p0[1:SMC.ind_para_end(N)])
    p0[SMC.ind_loglh(N)]     = loglik_fn(m.parameters, data)
    p0[SMC.ind_logprior(N)]  = ModelConstructors.prior(m.parameters)
    p0[SMC.ind_old_loglh(N)] = 0.0
    for n_mh_steps in (1, 3)
        Random.seed!(7)
        p = SMC.mutation(loglik_fn, m.parameters, data, copy(p0), d.μ, Matrix(d.Σ), n_params,
                         blocks_free, blocks_all, ϕ_n; c = 1e-8, α = 1.0, n_mh_steps = n_mh_steps)
        @test SMC.get_accept(reshape(p, 1, :))[1] ≈ 1.0
    end
end
