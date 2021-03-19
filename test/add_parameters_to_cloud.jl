using ModelConstructors, HDF5, Random, JLD2, FileIO, SMC, Test
include("modelsetup.jl")

path = dirname(@__FILE__)

if VERSION < v"1.5"
    ver = "111"
else
    ver = "150"
end

m    = setup_linear_model(; regime_switching = false)
m_rs = setup_linear_model(; regime_switching = true)
save = normpath(joinpath(dirname(@__FILE__),"save"))
m    <= Setting(:saveroot, save)
m_rs <= Setting(:saveroot, save)
savepath = rawpath(m, "estimate", "smc_cloud.jld2")
particle_store_path = rawpath(m, "estimate", "smcsave.h5")

data    = h5read("reference/test_data.h5", "data")
data_rs = h5read("reference/test_data.h5", "rsdata")

Random.seed!(1793)

# mean_para = mean(SMC.get_vals(test_cloud), dims = 2)
true_para = [1., 1., 1., # α1, β1, σ1 (regime 1)
             2., 2., 1., # α2, β2, σ2 (regime 1)
             3., 3., 1., # α3, β3, σ3 (regime 1)
             1., 1.,     # α1 regimes = 2-3
             2., 3.,     # β1 regimes = 2-3
             2., 2.,     # α2 regimes = 2-3
             3., 4.,     # β2 regimes = 2-3
             3., 3.,     # α3 regimes = 2-3
             4., 5.]     # β3 regimes = 2-3

saved_file  = JLD2.jldopen(string("reference/smc_cloud_fix=true_version=", ver, ".jld2"), "r")
saved_cloud = saved_file["cloud"]
saved_w     = saved_file["w"]
saved_W     = saved_file["W"]

saved_file_rs  = JLD2.jldopen(string("reference/smc_cloud_fix=true_rs=true_version=", ver, ".jld2"), "r")
saved_cloud_rs = saved_file_rs["cloud"]
saved_w_rs     = saved_file_rs["w"]
saved_W_rs     = saved_file_rs["W"]

n_para    = length(m.parameters)
n_para_rs = n_parameters_regime_switching(m_rs)
old_para_inds = vcat(trues(n_para), falses(n_para_rs - n_para))

m_rs.parameters[7].fixed = false # need to unfix this or the log-likelihoods won't match
m_rs.parameters[7].valuebounds = m_rs.parameters[4].valuebounds
set_regime_fixed!(m_rs.parameters[7], 1, false) # update! doesn't seem to be working properly???
set_regime_valuebounds!(m_rs.parameters[7], 1, m_rs.parameters[4].valuebounds)
new_cloud = SMC.add_parameters_to_cloud(saved_cloud, m_rs.parameters,
                                        old_para_inds; regime_switching = true)
new_cloud2 = SMC.add_parameters_to_cloud(joinpath("reference", "smc_cloud_fix=true_version=" *
                                                  ver * ".jld2"), m_rs.parameters,
                                         old_para_inds; regime_switching = true)

@test saved_cloud.particles[:, 1:9] ≈ new_cloud.particles[:, 1:9]
@test saved_cloud.particles[:, end - 2:end] ≈ new_cloud.particles[:, end - 2:end] # old_loglh, accept, and weight should be the same
@test saved_cloud.particles[:, end - 4] ≈ new_cloud.particles[:, end - 4] # loglh should be the same
@test !(saved_cloud.particles[:, end - 3] ≈ new_cloud.particles[:, end - 3]) # prior should not be the same b/c new parameters & resampling from prior
@test saved_cloud.particles[:, 1:9] ≈ new_cloud2.particles[:, 1:9]
@test saved_cloud.particles[:, end - 2:end] ≈ new_cloud2.particles[:, end - 2:end] # old_loglh, accept, and weight should be the same
@test saved_cloud.particles[:, end - 4] ≈ new_cloud2.particles[:, end - 4] # loglh should be the same
@test !(saved_cloud.particles[:, end - 3] ≈ new_cloud2.particles[:, end - 3]) # prior should not be the same b/c new parameters & resampling from prior
