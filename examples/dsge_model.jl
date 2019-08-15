## Add number of workers of one's choosing:
#addprocs_frbny(20)

@everywhere using DSGE, ModelConstructors, SMC
@everywhere import ModelConstructors: ParameterVector
@everywhere import SMC: smc

## Initialize model object
m = AnSchorfheide()
data = df_to_matrix(m, load_data(m))

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

smc(my_likelihood, m.parameters, data; data_vintage = data_vintage(m),
    n_parts = 400, n_Φ = 100, λ = 2.0, parallel = true, save_intermediate = true,
    particle_store_path = particle_store_path, loadpath = loadpath,
    savepath = savepath)
