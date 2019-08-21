## Add number of workers of one's choosing:
#addprocs(20)

using DSGE, DSGEModels, ModelConstructors, SMC
@everywhere using DSGE, DSGEModels, ModelConstructors, SMC
@everywhere import ModelConstructors: ParameterVector
@everywhere import SMC: smc

## Initialize model object
m = AnSchorfheide()
m <= Setting(:λ, 2.0)
m <= Setting(:n_particles, 400)
m <= Setting(:n_Φ, 100)
m <= Setting(:n_smc_blocks, 1)
m <= Setting(:n_mh_steps_smc, 1)
m <= Setting(:use_parallel_workers, true)
m <= Setting(:resampler_smc, :systematic)
m <= Setting(:adaptive_tempering_target_smc, false)
m <= Setting(:use_fixed_schedule, true)

data = df_to_matrix(m, load_data(m))

## Specify settings
verbose = :low
use_chand_recursion = true
filestring_addl     = Vector{String}()

## Specify output paths
loadpath            = rawpath(m, "estimate", "smc_cloud.jld2", filestring_addl)
savepath            = rawpath(m, "estimate", "smc_cloud.jld2", filestring_addl)
particle_store_path = rawpath(m, "estimate", "smcsave.h5",     filestring_addl)

## Define likelihood function in correct form, on all workers
DSGE.sendto(workers(), m = m)
function my_loglikelihood(parameters::ParameterVector, data::Matrix{Float64})::Float64
    DSGE.update!(m, parameters)
    DSGE.likelihood(m, data; sampler = false, catch_errors = true,
                    use_chand_recursion = true, verbose = :low)
end
@everywhere function my_loglikelihood(parameters::ParameterVector, data::Matrix{Float64})::Float64
    DSGE.update!(m, parameters)
    DSGE.likelihood(m, data; sampler = false, catch_errors = true,
                    use_chand_recursion = true, verbose = :low)
end

## Run SMC
smc(my_loglikelihood, m.parameters, data; verbose = verbose, data_vintage = data_vintage(m),
    n_parts = 400, n_Φ = 100, λ = 2.0, parallel = true, save_intermediate = false,
    particle_store_path = particle_store_path, loadpath = loadpath,
    savepath = savepath)
