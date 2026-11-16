using DSGE, ModelConstructors, SMC, HDF5, JLD2

const DEFAULT_DATA_PATH = joinpath(pkgdir(DSGE), "test", "estimate", "smc", "reference", "smc.h5")

env_int(name, default) = parse(Int, get(ENV, name, string(default)))

function run_anschorfheide_scaling(adaptation::Symbol)
    adaptation in (:acceptance, :esjd) || error("Unknown scale adaptation: $adaptation")
    isdefined(SMC, :mutation_esjd_grid) ||
        error("The loaded SMC package does not include the ESJD diagnostic: $(pathof(SMC))")

    seed = env_int("SMC_COMPARE_SEED", 1234)
    n_parts = env_int("SMC_COMPARE_N", 300)
    max_obs = env_int("SMC_COMPARE_MAX_OBS", 0)
    diagnostic_every = env_int("SMC_COMPARE_DIAGNOSTIC_EVERY", 10)
    diagnostic_scales = env_int("SMC_COMPARE_DIAGNOSTIC_SCALES", 5)
    pilot_particles = env_int("SMC_COMPARE_PILOT_PARTICLES", 64)
    n_parts > 0 || error("SMC_COMPARE_N must be positive")
    max_obs >= 0 || error("SMC_COMPARE_MAX_OBS must be nonnegative")
    diagnostic_every > 0 || error("SMC_COMPARE_DIAGNOSTIC_EVERY must be positive")
    diagnostic_scales >= 2 || error("SMC_COMPARE_DIAGNOSTIC_SCALES must be at least 2")
    pilot_particles > 0 || error("SMC_COMPARE_PILOT_PARTICLES must be positive")

    data_path = get(ENV, "SMC_COMPARE_DATA_PATH", DEFAULT_DATA_PATH)
    data = h5read(data_path, "data")
    if max_obs > 0
        data = data[:, 1:min(max_obs, size(data, 2))]
    end

    label = adaptation == :acceptance ? "bounded_sigmoid" : "esjd"
    outdir = joinpath(get(ENV, "SMC_COMPARE_OUTDIR", joinpath(@__DIR__, "results")), label)
    mkpath(outdir)
    stem = "anschorfheide_seed=$(seed)_N=$(n_parts)_T=$(size(data, 2))"
    savepath = joinpath(outdir, stem * ".jld2")
    particle_store_path = joinpath(outdir, stem * "_particles.h5")

    model = AnSchorfheide()
    model <= Setting(:use_chand_recursion, true)
    loglikelihood = function (parameters, observations)
        DSGE.update!(model, parameters)
        likelihood(model, observations; sampler = false, catch_errors = true)
    end

    println("Scaling method: $label")
    println("Seed: $seed; particles: $n_parts; observations: $(size(data, 2))")
    println("DSGE: $(pathof(DSGE)); SMC: $(pathof(SMC))")
    println("Output: $savepath")
    SMC.smc(loglikelihood, model.parameters, data;
            smc_seed = seed, n_parts = n_parts, n_smc_groups = 1,
            n_blocks = 1, n_mh_steps = 3, α = 0.975, c = 0.5, target = 0.25,
            use_fixed_schedule = false, tempering_target = 0.95, n_Φ = 300, λ = 2.1,
            threshold_ratio = 0.5, resampling_method = :multinomial,
            scale_adaptation = adaptation,
            esjd_multipliers = [0.5, 1.0, 2.0],
            esjd_pilot_particles = min(pilot_particles, n_parts),
            esjd_diagnostic = true,
            esjd_diagnostic_every = diagnostic_every,
            esjd_diagnostic_n_scales = diagnostic_scales,
            esjd_diagnostic_span = 4.0,
            savepath = savepath, particle_store_path = particle_store_path,
            run_info = Dict("scaling_method" => label,
                            "dsge_commit" => SMC.package_commit(DSGE)),
            verbose = :low)

    isfile(savepath) || error("SMC did not create $savepath")
    health = JLD2.load(savepath, "stage_health")
    reports = health["esjd_diagnostics"]
    n_stages = length(health["phi"])
    diagnostic_csv = replace(savepath, r"\.jld2$" => "_esjd_diagnostic.csv")
    println("Finished $label: $n_stages stages, $(length(reports)) ESJD checks")
    println("Diagnostic CSV: $diagnostic_csv")
    return savepath
end
