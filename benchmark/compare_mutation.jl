using Pkg
project_dir = dirname(@__DIR__)
Pkg.activate(project_dir)
using SMC, ModelConstructors, Distributions, LinearAlgebra, HDF5, JLD2, BenchmarkTools, Random
cd(joinpath(project_dir, "test"))
include(joinpath(project_dir, "test", "modelsetup.jl"))

# Load the main-branch implementation under a separate name in the same process.
baseline_source = read(`git -C $project_dir show cc1211d:src/mutation.jl`, String)
baseline_source = replace(baseline_source, "function mutation(" => "function mutation_baseline("; count=1)
Base.include_string(SMC, baseline_source)

model_old = setup_linear_model()
model_new = setup_linear_model()
data = h5read("reference/test_data.h5", "data")
file = JLD2.jldopen("reference/mutation_inputs.jld2", "r")
old_cloud = read(file, "particles")
d = read(file, "d")
ϕ_n, ϕ_n1 = read(file, "ϕ_n"), read(file, "ϕ_n1")
c, α = read(file, "c"), read(file, "α")
old_data = read(file, "old_data")
close(file)

p = Vector{Float64}(old_cloud.particles[1, :])
μ = Vector{Float64}(d.μ)
Σ = Matrix{Float64}(d.Σ.mat)
blocks = [collect(1:3), collect(4:6), collect(7:9)]
function benchmark_old_loglikelihood(parameters, data; new_model_params=false)
    loglik_fn(parameters, data)
end
Random.seed!(20261020)
expected = SMC.mutation_baseline(loglik_fn, model_old.parameters, data, copy(p), μ, Σ,
                                 9, blocks, blocks, ϕ_n, ϕ_n1;
                                 c=c, α=α, old_data=old_data,
                                 old_loglikelihood=benchmark_old_loglikelihood, n_mh_steps=3)
Random.seed!(20261020)
actual = SMC.mutation(loglik_fn, model_new.parameters, data, copy(p), μ, Σ,
                      9, blocks, blocks, ϕ_n, ϕ_n1;
                      c=c, α=α, old_data=old_data,
                      old_loglikelihood=benchmark_old_loglikelihood, n_mh_steps=3)
@assert actual == expected
println("MUTATION_OUTPUT_EXACT=true")

old_bench = @benchmark SMC.mutation_baseline($loglik_fn, $(model_old.parameters), $data,
                                             copy($p), $μ, $Σ, 9, $blocks, $blocks,
                                             $ϕ_n, $ϕ_n1; c=$c, α=$α,
                                             old_data=$old_data,
                                             old_loglikelihood=$benchmark_old_loglikelihood,
                                             n_mh_steps=3) samples=80 evals=1
new_bench = @benchmark SMC.mutation($loglik_fn, $(model_new.parameters), $data,
                                    copy($p), $μ, $Σ, 9, $blocks, $blocks,
                                    $ϕ_n, $ϕ_n1; c=$c, α=$α,
                                    old_data=$old_data,
                                    old_loglikelihood=$benchmark_old_loglikelihood,
                                    n_mh_steps=3) samples=80 evals=1
println("BASELINE_MEDIAN_NS=", median(old_bench).time)
println("OPTIMIZED_MEDIAN_NS=", median(new_bench).time)
println("BASELINE_MEDIAN_BYTES=", median(old_bench).memory)
println("OPTIMIZED_MEDIAN_BYTES=", median(new_bench).memory)
