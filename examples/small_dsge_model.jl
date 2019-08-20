## Add number of workers of one's choosing:
#addprocs_frbny(20)
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
savepath            = rawpath(m, "estimate", "")
particle_store_path = rawpath(m, "estimate", "smcsave.h5",     filestring_addl)

DSGE.sendto(workers(), m = m)
@everywhere function my_likelihood(parameters::ParameterVector, data::Matrix{Float64})::Float64
    DSGE.update!(m, parameters)
    DSGE.likelihood(m, data; sampler = false, catch_errors = true,
                    use_chand_recursion = true, verbose = :low)
end

# Old SMC: ~9.6 s (12.70 M alloc: 827 MiB)
@time DSGE.smc2(m, data)
@time DSGE.smc(m, data)

# New SMC: ~8.1 s (10.85 M alloc: 673 MiB)
@time smc(my_likelihood, m.parameters, data; data_vintage = data_vintage(m),
          n_parts = 400, n_Φ = 100, λ = 2.0, parallel = true, save_intermediate = false,
          particle_store_path = particle_store_path, loadpath = loadpath,
          savepath = savepath)
