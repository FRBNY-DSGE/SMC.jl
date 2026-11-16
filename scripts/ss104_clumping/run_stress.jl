using Pkg
Pkg.activate(joinpath(@__DIR__, "env"))
using DSGE, ModelConstructors, SMC, JLD2, CSV, DataFrames, Dates
using Random, Statistics, LinearAlgebra, StatsBase

const METHOD = occursin("acceptance", basename(PROGRAM_FILE)) ? :acceptance : :esjd
METHOD in (:acceptance, :esjd) || error("method must be acceptance or esjd")
const ROOT = @__DIR__
const OUT = joinpath(ROOT, "results")
mkpath(OUT)
const SOURCE = "/data/dsge_data_dir/dsgejl/output_data/m1002/ss104/estimate/raw/" *
    "smc_cloud_fcastdate=2026Q3_mhsteps=3_nparts=15000_phi=0.95_scratch_vint=261013_stage=300.jld2"
const DATA = "/data/dsge_data_dir/dsgejl/input_data/data/data_cdid=02_cdvt=261013_dsid=03_vint=261013.csv"
const SEED = UInt64(20261113)
const FRACTION = 0.35
const N_SWEEPS = 3
const N_MH_STEPS = 3
const ALPHA = 0.975
const TARGET = 0.25
const MULTIPLIERS = [0.5, 1.0, 2.0]
const N_PILOT = 32
const N_DIAGNOSTIC = 32

function make_model()
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
    return m
end

function distinct_count(particles)
    npara = size(particles, 2) - 5
    return length(Set(Tuple(row) for row in eachrow(view(particles, :, 1:npara))))
end

function covariance_summary(particles, free)
    c = SMC.Cloud(size(particles, 2)-5, size(particles, 1))
    c.particles .= particles
    cov = SMC.weighted_cov(c)[free, free]
    eig = eigvals(Symmetric((cov + cov') / 2))
    return cov, minimum(eig), maximum(eig), tr(cov)
end

source = JLD2.load(SOURCE, "cloud")
phi = source.tempering_schedule[end]
m = make_model()
df = DSGE.post_covid_data_mods(m, CSV.read(DATA, DataFrame), :full, Int[])
data = DSGE.df_to_matrix(m, df; cond_type = :none)
parameters = deepcopy(m.parameters)
ModelConstructors.toggle_regime!(parameters, 1)
free = ModelConstructors.get_free_para_inds(parameters; regime_switching = true, toggle = true)
nfree = length(free)
nparts = max(480, ceil(Int, (nfree + 10) / FRACTION))
nunique = round(Int, nparts * FRACTION)
@assert nunique > nfree
@assert size(source.particles, 2)-5 == ModelConstructors.n_parameters_regime_switching(parameters)

# Select distinct source draws, then duplicate those draws. This represents a
# post-resampling cloud with equal weights and exactly the requested clumping.
rng = Xoshiro(SEED)
source_w = SMC.get_weights(source)
source_indices = sample(rng, 1:size(source.particles, 1), Weights(source_w),
                        min(size(source.particles, 1), 3nunique); replace = false)
selected = Int[]
seen = Set{Any}()
for k in source_indices
    key = Tuple(view(source.particles, k, 1:size(source.particles, 2)-5))
    if !(key in seen)
        push!(selected, k)
        push!(seen, key)
        length(selected) == nunique && break
    end
end
length(selected) == nunique || error("source cloud has too few distinct draws")
indices = vcat(selected, rand(rng, selected, nparts - nunique))
shuffle!(rng, indices)
initial = copy(source.particles[indices, :])
initial[:, end] .= 1.0
initial[:, end-1] .= 0.0
@assert distinct_count(initial) == nunique
source_cov, _, _, source_trace = covariance_summary(source.particles, free)
initial_cov, initial_min, initial_max, initial_trace = covariance_summary(initial, free)
println("method=$METHOD source_stage=$(source.stage_index) phi=$phi n=$nparts free=$nfree " *
        "initial_unique=$nunique initial_eig=($initial_min,$initial_max) " *
        "initial_trace_ratio=$(initial_trace/source_trace)")
flush(stdout)

function loglikelihood(params, data)
    ModelConstructors.toggle_regime!(m.parameters, 1)
    DSGE.update!(m, params; regime_switching = true)
    m <= Setting(:preprocessed_transitions, Dict())
    return DSGE.likelihood(m, data; sampler = false, catch_errors = true,
                           use_chand_recursion = false, verbose = :none)
end

# A stored particle must have the same likelihood under this model and dataset.
probe = selected[1]
ModelConstructors.update!(parameters, source.particles[probe, 1:end-5])
check_ll = loglikelihood(parameters, data)
stored_ll = source.particles[probe, end-4]
println("likelihood_check stored=$stored_ll recomputed=$check_ll")
flush(stdout)
isfinite(check_ll) && isapprox(check_ll, stored_ll; atol = 1e-4, rtol = 1e-8) ||
    error("SS104 likelihood does not reproduce checkpoint; refusing stress test")
ModelConstructors.toggle_regime!(parameters, 1)

let particles = initial, c = source.c, previous_accept = source.accept
rows = DataFrame(method=String[], sweep=Int[], phi=Float64[], c=Float64[],
                 acceptance=Float64[], failed_proposals=Int[], distinct=Int[],
                 distinct_share=Float64[], covariance_min_eigenvalue=Float64[],
                 covariance_max_eigenvalue=Float64[], covariance_trace=Float64[],
                 spread_vs_source=Float64[], diagnostic_best_c=Float64[],
                 diagnostic_efficiency=Float64[])
grids = DataFrame(method=String[], sweep=Int[], c=Float64[],
                  acceptance=Float64[], normalized_esjd=Float64[])

for sweep in 1:N_SWEEPS
    stage = source.stage_index + sweep
    cloud = SMC.Cloud(size(particles,2)-5, nparts)
    cloud.particles .= particles
    proposal_mean = SMC.weighted_mean(cloud)[free]
    empirical, eigmin, eigmax, trace_cov = covariance_summary(particles, free)
    covariance = (empirical + empirical') / 2
    if !isposdef(covariance)
        F = eigen(Symmetric(covariance))
        floor = max(maximum(F.values)*1e-10, eps(Float64))
        covariance = F.vectors * Diagonal(max.(F.values, floor)) * F.vectors'
        covariance = (covariance + covariance') / 2
    end
    blocks_free = [collect(1:nfree)]
    blocks_all = [free]
    if METHOD == :acceptance
        c = SMC.update_c(c, previous_accept, TARGET)
    else
        pilot_indices = sample(Xoshiro(SMC.stream_seed(SEED, :esjd_pilot_selection, stage)),
                               1:nparts, Weights(SMC.get_weights(cloud)), N_PILOT; replace = true)
        variances = max.(diag(covariance), eps(Float64))
        scores = zeros(length(MULTIPLIERS))
        for (j, multiplier) in enumerate(MULTIPLIERS)
            pilot_params = deepcopy(parameters)
            for (draw, k) in enumerate(pilot_indices)
                Random.seed!(SMC.stream_seed(SEED, :esjd_pilot_particle, stage, draw))
                score = zeros(1)
                SMC.mutation(loglikelihood, pilot_params, data, copy(particles[k,:]), proposal_mean,
                             covariance, nfree, blocks_free, blocks_all, phi;
                             c=c*multiplier, α=ALPHA, n_mh_steps=1,
                             regime_switching=true, toggle=true,
                             esjd_scores=score, esjd_variances=variances)
                scores[j] += score[1]
            end
        end
        c *= MULTIPLIERS[argmax(scores)]
        println("sweep=$sweep pilot_scores=$(scores ./ N_PILOT) chosen_c=$c")
        flush(stdout)
    end

    diagnostic_indices = sample(Xoshiro(SMC.stream_seed(SEED, :stress_diagnostic_selection, stage)),
                                1:nparts, N_DIAGNOSTIC; replace = false)
    diagnostic_particles = particles[diagnostic_indices, :]
    grid_scales = sort!(unique(vcat(SMC.esjd_scale_grid(c, 5, 4.0), c)))
    report = SMC.mutation_esjd_grid(loglikelihood, parameters, data,
                diagnostic_particles, ones(N_DIAGNOSTIC), proposal_mean, covariance,
                blocks_free, blocks_all, phi, c, grid_scales;
                α=ALPHA, regime_switching=true, toggle=true, run_seed=SEED, stage=stage)
    for j in eachindex(report["scales"])
        push!(grids, (String(METHOD), sweep, report["scales"][j],
                      report["acceptance"][j], report["esjd"][j]))
    end

    next_particles = similar(particles)
    nfailed = 0
    for k in 1:nparts
        Random.seed!(SMC.stream_seed(SEED, :mutation_particle, stage, k))
        failed = Ref(0)
        next_particles[k,:] = SMC.mutation(loglikelihood, parameters, data,
                    copy(particles[k,:]), proposal_mean, covariance, nfree,
                    blocks_free, blocks_all, phi; c=c, α=ALPHA,
                    n_mh_steps=N_MH_STEPS, regime_switching=true,
                    toggle=true, n_failed=failed)
        nfailed += failed[]
        if k % 50 == 0
            println("method=$METHOD sweep=$sweep mutated=$k/$nparts")
            flush(stdout)
        end
    end
    particles = next_particles
    previous_accept = mean(particles[:,end-1])
    _, after_min, after_max, after_trace = covariance_summary(particles, free)
    ndistinct = distinct_count(particles)
    push!(rows, (String(METHOD), sweep, phi, c, previous_accept, nfailed,
                 ndistinct, ndistinct/nparts, after_min, after_max, after_trace,
                 after_trace/source_trace, report["best_c"], report["efficiency"]))
    CSV.write(joinpath(OUT, "$(METHOD)_metrics.csv"), rows)
    CSV.write(joinpath(OUT, "$(METHOD)_esjd_grid.csv"), grids)
    JLD2.jldsave(joinpath(OUT, "$(METHOD)_stage$(stage).jld2");
                 particles=particles, phi=phi, c=c, method=METHOD, seed=SEED)
    efficiency = report["efficiency"]
    println("completed method=$METHOD sweep=$sweep distinct=$ndistinct/$nparts " *
            "accept=$previous_accept spread=$(after_trace/source_trace) " *
            "efficiency=$efficiency")
    flush(stdout)
end
end # let: keep evolving particles, c, and acceptance local to the sweeps
