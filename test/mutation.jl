write_test_output = false
include("modelsetup.jl")

path = dirname(@__FILE__)

m = setup_linear_model()

save = normpath(joinpath(dirname(@__FILE__),"save"))
m <= Setting(:saveroot, saveroot)

data = h5read("reference/test_data.h5", "data")

n_parts = get_setting(m, :n_particles)
n_params = length(m.parameters)

file = JLD2.jldopen("reference/mutation_inputs.jld2", "r")
old_cloud = read(file, "particles")

d = read(file, "d")
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
    JLD2.jldopen("reference/mutation_outputs.jld2", "w") do file
        write(file, "particles", new_cloud)
    end
end

saved_cloud = load("reference/mutation_outputs.jld2", "particles")

@testset "Test mutation outputs, particle by particle" begin
    for i = 1:n_parts
        @test isapprox(saved_cloud.particles[i, :], new_cloud.particles[i, :], nans = true)
    end
end
