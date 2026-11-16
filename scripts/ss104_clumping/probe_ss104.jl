using Pkg
Pkg.activate(joinpath(@__DIR__, "env"))
using DSGE, ModelConstructors, SMC, JLD2, CSV, DataFrames, Dates
println("DSGE source=", pkgdir(DSGE), " SMC source=", pkgdir(SMC))

source = "/data/dsge_data_dir/dsgejl/output_data/m1002/ss104/estimate/raw/" *
         "smc_cloud_fcastdate=2026Q3_mhsteps=3_nparts=15000_phi=0.95_scratch_vint=261013_stage=300.jld2"
cloud = JLD2.load(source, "cloud")
m = Model1002("ss104", custom_settings = [
    Setting(:date_forecast_start, Date(2026, 9, 30)),
    Setting(:date_conditional_end, Date(2026, 9, 30)),
    Setting(:add_ant_markup_shocks_ind, 2),
    Setting(:fix_ant_markup_shocks_0, false),
    Setting(:add_avgshortinfl, true),
])
m <= Setting(:data_vintage, "261013")
m <= Setting(:cond_vintage, "261013")
m <= Setting(:dataroot, "/data/dsge_data_dir/dsgejl/input_data")
m <= Setting(:use_population_forecast, true)
m <= Setting(:cond_full_names, [:obs_gdp, :obs_corepce, :obs_spread,
    :obs_longrate, :obs_nominalrate, :obs_avgshortinflation,
    :obs_exp_nominalrate1, :obs_exp_nominalrate2, :obs_exp_nominalrate3,
    :obs_exp_nominalrate4, :obs_exp_nominalrate5, :obs_exp_nominalrate6])
m <= Setting(:cond_semi_names, [:obs_spread, :obs_longrate])
for key in (:σ_biidc_prop, :σ_φ_prop, :σ_ziid_prop)
    if key in [p.key for p in m.parameters]
        ModelConstructors.set_regime_fixed!(m[key], 1, true)
        ModelConstructors.set_regime_fixed!(m[key], 2, true)
    end
end
m <= Setting(:exp_ffr_data_start_date, Date(2022, 3, 31))
m <= Setting(:use_chand_recursion, false)

data_path = "/data/dsge_data_dir/dsgejl/input_data/data/data_cdid=02_cdvt=261013_dsid=03_vint=261013.csv"
df = CSV.read(data_path, DataFrame)
df = DSGE.post_covid_data_mods(m, df, :full, Int[])
data = DSGE.df_to_matrix(m, df; cond_type = :none)
println("model total params=", ModelConstructors.n_parameters_regime_switching(m.parameters),
        " cloud params=", size(cloud.particles, 2)-5,
        " data=", size(data), " model obs=", DSGE.n_observables(m))
θ = cloud.particles[1, 1:end-5]
params = deepcopy(m.parameters)
ModelConstructors.toggle_regime!(params, 1)
ModelConstructors.update!(params, θ)
ModelConstructors.toggle_regime!(m.parameters, 1)
DSGE.update!(m, params; regime_switching = true)
m <= Setting(:preprocessed_transitions, Dict())
ll = DSGE.likelihood(m, data; sampler = false, catch_errors = true,
                     use_chand_recursion = false, verbose = :none)
println("source loglh=", cloud.particles[1, end-4], " recomputed=", ll,
        " difference=", ll-cloud.particles[1, end-4])
println("free params=", length(ModelConstructors.get_free_para_inds(m.parameters; regime_switching = true, toggle = true)))
