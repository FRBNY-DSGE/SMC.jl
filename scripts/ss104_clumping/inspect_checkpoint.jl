using JLD2, SMC, Statistics, LinearAlgebra

const SOURCE_DIR = "/data/dsge_data_dir/dsgejl/output_data/m1002/ss104/estimate/raw"
const PREFIX = "smc_cloud_fcastdate=2026Q3_mhsteps=3_nparts=15000_phi=0.95_scratch_vint=261013_stage="

for stage in (100, 200, 300)
    path = joinpath(SOURCE_DIR, PREFIX * string(stage) * ".jld2")
    println("SOURCE $path")
    names = JLD2.jldopen(keys, path, "r")
    println("KEYS ", join(sort!(collect(names)), ","))
    cloud = JLD2.load(path, "cloud")
    θ = SMC.get_vals(cloud; transpose = false)
    distinct = length(Set(Tuple(row) for row in eachrow(θ)))
    weights = SMC.get_weights(cloud)
    println("CLOUD stage=$(cloud.stage_index) phi=$(cloud.tempering_schedule[end]) n=$(size(θ,1)) " *
            "d=$(size(θ,2)) distinct=$distinct c=$(cloud.c) accept=$(cloud.accept) " *
            "ess=$(cloud.ESS[end]) weights=($(minimum(weights)),$(maximum(weights))) " *
            "resamples=$(cloud.resamples)")
    if "stage_health" in names
        health = JLD2.load(path, "stage_health")
        println("HEALTH keys=", join(sort!(collect(keys(health))), ","))
        println("HEALTH last_resampled=", health["resampled"][end],
                " last_phi=", health["phi"][end])
    end
    println("SEED ", "smc_seed" in names ? string(JLD2.load(path, "smc_seed")) : "none")
    if stage == 300
        println("RUN_INFO ", JLD2.load(path, "run_info"))
        println("LOGLH ", extrema(SMC.get_loglh(cloud)))
        println("OLD_LOGLH ", extrema(SMC.get_old_loglh(cloud)))
    end
end
