## Add number of workers of one's choosing:
using DSGE, DSGEModels, ModelConstructors, SMC
#addprocs_frbny(40)
@everywhere using DSGE, DSGEModels, ModelConstructors, SMC, HDF5
@everywhere import ModelConstructors: ParameterVector
@everywhere import SMC: smc

## Initialize model object
fix_gamma = false
subspec = fix_gamma ? "ss1" : "ss0"
m = SmetsWoutersOrig(subspec)

saveroot = ""

m <= Setting(:saveroot, saveroot, "Output data directory path")
m <= Setting(:data_vintage,"041231")
m <= Setting(:date_presample_start, quartertodate("1965-Q4"))
m <= Setting(:date_mainsample_start, quartertodate("1966-Q4"))
m <= Setting(:date_forecast_start, quartertodate("2004-Q4"))
m <= Setting(:date_conditional_end, quartertodate("2004-Q4"))

# Load data
data = h5read("data/sw_orig_smc.h5", "data")

m <= Setting(:n_particles, 400)
m <= Setting(:n_Φ, 500)
m <= Setting(:λ, 2.1)
m <= Setting(:n_smc_blocks, 3)
m <= Setting(:n_mh_steps_smc, 1)
m <= Setting(:step_size_smc, 0.4)
m <= Setting(:use_parallel_workers, true)
m <= Setting(:resampler_smc, :multinomial)
m <= Setting(:target_accept, 0.25)

m <= Setting(:mixture_proportion, .9)
m <= Setting(:resampling_threshold, 0.5)
m <= Setting(:adaptive_tempering_target_smc, false)

# If want adaptive, just set the tempering target you want
#m <= Setting(:adaptive_tempering_target_smc, 0.95)

m <= Setting(:use_chand_recursion, true)

## Specify settings
verbose = :low
use_chand_recursion = true
filestring_addl     = Vector{String}()

## Specify output paths
loadpath            = rawpath(m, "estimate", "smc_cloud.jld2", filestring_addl)
savepath            = rawpath(m, "estimate", "",               filestring_addl)
particle_store_path = rawpath(m, "estimate", "smcsave.h5",     filestring_addl)

DSGE.sendto(workers(), m = m)
@everywhere function my_likelihood(parameters::ParameterVector, data::Matrix{Float64})::Float64
    DSGE.update!(m, parameters)
    DSGE.likelihood(m, data; sampler = false, catch_errors = true,
                    use_chand_recursion = true, verbose = :low)
end

# Old SMC: ~9.6 s (12.70 M alloc: 827 MiB)
#@time DSGE.smc(m, data)

# New SMC: ~8.1 s (10.85 M alloc: 673 MiB)
@time smc(my_likelihood, m.parameters, data; data_vintage = data_vintage(m),
          n_parts = 1_000, n_Φ = 100, λ = 2.0, parallel = true, save_intermediate = false,
          particle_store_path = particle_store_path, loadpath = loadpath,
          savepath = savepath)
