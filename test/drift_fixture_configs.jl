# The runs behind test/reference/drift_off_main.jld2: written by SMC main (95f0350) with
# `drift_off_runs(dir)`, and rerun by drift.jl with every drift option off. The fixture
# is compared within floating-point roundoff across Julia releases; default and explicit
# off settings must be bitwise equal when run in the same process.
# Needs drift_workers.jl.
using ModelConstructors, Distributions, SMC, JLD2

function drift_parameters(d::Int, sdp::Float64)
    m = GenericModel()
    for k in 1:d
        m <= parameter(Symbol("x$k"), 0., (-1e4, 1e4), (-1e4, 1e4), Untransformed(), Normal(0, sdp),
                       fixed = false)
    end
    return m.parameters
end

const DRIFT_OFF_CONFIGS = [
    # adaptive tempering, fixed blocks with their own scales, random blocks of the rest, α < 1
    ("ridge_blocks", :ridge, (n_parts = 300, n_mh_steps = 2, use_fixed_schedule = false,
                              tempering_target = 0.95, param_blocks = [[:x1], [:x2, :x3, :x4]],
                              blockwise_c = true, n_blocks = 2, α = 0.9, c = 0.5)),
    # fixed schedule, one block, multinomial resampling
    ("gauss3_plain", :gauss3, (n_parts = 200, n_mh_steps = 1, n_Φ = 25, λ = 2.0,
                               use_fixed_schedule = true, resampling_method = :multinomial)),
]

function drift_off_run(dir::AbstractString, name::AbstractString, target::Symbol; kwargs...)
    lik, params = target === :ridge ? (drift_ridge_loglh, drift_parameters(DRIFT_D, DRIFT_SDP)) :
                                      (drift_gauss3_loglh, drift_parameters(3, 2.0))
    path = joinpath(dir, name * ".jld2")
    SMC.smc(lik, params, zeros(1, 1); verbose = :none, smc_seed = 11, savepath = path,
            particle_store_path = replace(path, ".jld2" => ".h5"), kwargs...)
    out = load(path)
    return Dict{String, Any}("particles" => out["cloud"].particles, "w" => out["w"], "W" => out["W"],
                             "schedule" => out["cloud"].tempering_schedule,
                             "log_mdd" => out["log_mdd_estimates"]["smc"],
                             "accept" => out["stage_health"]["accept"], "c" => out["stage_health"]["c"])
end

drift_off_runs(dir::AbstractString) =
    Dict(name => drift_off_run(dir, name, target; kw...) for (name, target, kw) in DRIFT_OFF_CONFIGS)
