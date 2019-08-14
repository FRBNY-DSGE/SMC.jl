using DSGE, ModelConstructors, SMC
import ModelConstructors: ParameterVector
import SMC: smc

## Initialize model object
m = AnSchorfheide()
#data = df_to_matrix(m, load_data(m))

## Specify settings
verbose = :low
use_chand_recursion = true
filestring_addl     = Vector{String}()

loadpath = rawpath(m, "estimate", "smc_cloud.jld2", filestring_addl)
savepath = rawpath(m, "estimate", "", filestring_addl)
# load_path = rawpath(m, "estimate", "smc_cloud_stage=$(intermediate_stage_start).jld2",
#                     filestring_addl)
#replace(rawpath(m, "estimate", "smc_cloud.jld2", filestring_addl),
#           r"vint=[0-9]{6}", "vint=" * old_vintage)

particle_store_path = rawpath(m, "estimate", "smcsave.h5", filestring_addl)

## Define a likelihood function with correct input/output format
function my_likelihood(parameters::ParameterVector, data::Matrix{Float64})::Float64
    DSGE.update!(m, parameters)
    DSGE.likelihood(m, data; sampler = false, catch_errors = true,
                    use_chand_recursion = use_chand_recursion, verbose = verbose)
end

#smc(my_likelihood, m.parameters, data; data_vintage = data_vintage(m),
#    n_parts = 400, n_Φ = 100, λ = 2.0, resampling_method = :polyalgo, parallel = true,
#    particle_store_path = particle_store_path, loadpath = loadpath, savepath = savepath)

println("Trying out: Parallel") # 93.492 s (410.42 M alloc: 72.527 GiB)
@time smc(my_likelihood, m.parameters, data; data_vintage = data_vintage(m),
          n_parts = 400, n_Φ = 100, λ = 2.0, resampling_method = :polyalgo, parallel = false,
          particle_store_path = particle_store_path, loadpath = loadpath, savepath = savepath)
