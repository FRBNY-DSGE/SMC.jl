# Reliability diagnostics for a single SMC run (SMC reliability plan, step 3 part C):
#
#   item 1  numerical standard errors: independent particle groups (groups.jl) and, as a
#           cross-check, the ancestral-lineage estimators (lineage_nse)
#   item 2  per-stage health (record_stage_health!, stage_health_flags)
#   item 3  log MDD from the incremental weights, bridge sampling and the modified harmonic mean
#   item 4  multimodality: a Gaussian mixture on the cloud (gaussian_mixture_bic); the
#           optimizer/Hessian part needs a model and lives in DSGE (smc_mode_check)
#   items 5, 6  comparisons across runs (nse_validation, compare_estimates)

function logsumexp(x::AbstractVector{<:Real})
    m = maximum(x)
    isfinite(m) || return m
    return m + log(sum(exp.(x .- m)))
end
logmeanexp(x::AbstractVector{<:Real}) = logsumexp(x) - log(length(x))

"""
```
smc_log_mdd(w::AbstractMatrix, W::AbstractMatrix)
```
The SMC log MDD estimate Σ_n log((1/N) Σ_i w̃ⁱ_n Wⁱ_{n-1}) from the incremental (`w`) and
normalized (`W`) weight matrices that `smc` saves. Same formula as DSGE's
`marginal_data_density(...; calculation_method = :incremental_weights)`.
"""
function smc_log_mdd(w::AbstractMatrix, W::AbstractMatrix)
    size(w, 2) < 2 && return 0.0
    n_parts = sum(W[:, 1])
    return sum(log.(sum(w[:, 2:end] .* W[:, 1:end-1], dims = 1) ./ n_parts))
end

"""
```
smc_log_mdd_level(log_mdd; tempered_update, tempered_update_prior_weight, log_prob_old_data)
```
Put `smc_log_mdd` on the level scale log p(Y). A bridge with μ = 0 omits 1/p̃(Ỹ) from its
incremental weights, so its product is the increment log p(Y) - log p̃(Ỹ) (bug F4); add
`log_prob_old_data` back, or return NaN if it wasn't supplied. Update this together with F4.
"""
function smc_log_mdd_level(log_mdd::Real; tempered_update::Bool = false,
                           tempered_update_prior_weight::Real = 0.0,
                           log_prob_old_data::Real = 0.0)
    if tempered_update && tempered_update_prior_weight == 0.0
        return log_prob_old_data == 0.0 ? NaN : log_mdd + log_prob_old_data
    end
    return Float64(log_mdd)
end

########################################################################################
### Item 2: per-stage health
########################################################################################

const STAGE_HEALTH_COLUMNS = ("stage", "phi", "dphi", "ess", "cess", "max_weight", "resampled",
                              "distinct_selected", "distinct_mutated", "accept", "c",
                              "inf_cloud", "inf_proposals", "n_proposals", "time")

"""
```
new_stage_health(n_free_para::Int)
```
Empty per-stage health record: one Float64 vector per column of `STAGE_HEALTH_COLUMNS`, plus
`"corr"` and `"esjd"` (n_free_para × stages): per free parameter, the weighted correlation
of the particles before and after mutation, and the expected squared jump distance relative
to the weighted variance before mutation (Fearnhead and Taylor 2013).

`accept` is the mutation acceptance rate per MH step (`cloud.accept` divided by the number of
MH steps, since the particles' acceptance column sums over the steps), `ess` and `cess` are fractions of N (ESS before resampling; conditional ESS of Zhou, Johansen
and Aston 2016), `max_weight` the largest normalized weight's share, `distinct_*` the shares
of distinct particles after selection and after mutation, `inf_cloud` the number of -Inf
log-likelihoods in the cloud, `inf_proposals`/`n_proposals` the -Inf (or failed) MH
proposals out of all proposals.
"""
function new_stage_health(n_free_para::Int)
    h = Dict{String, Any}(col => Float64[] for col in STAGE_HEALTH_COLUMNS)
    h["corr"] = zeros(n_free_para, 0)
    h["esjd"] = zeros(n_free_para, 0)
    return h
end

function record_stage_health!(h::AbstractDict, row::NamedTuple, corr::Vector{Float64},
                              esjd::Vector{Float64})
    for col in STAGE_HEALTH_COLUMNS
        push!(h[col], Float64(row[Symbol(col)]))
    end
    h["corr"] = hcat(h["corr"], corr)
    h["esjd"] = hcat(h["esjd"], esjd)
    return h
end

n_stages(h::AbstractDict) = length(h["stage"])

"""
```
conditional_ess(incremental_weights, previous_weights)
```
Conditional ESS / N (Zhou, Johansen and Aston 2016): (Σ Wᵢ wᵢ)² / Σ Wᵢ wᵢ² with the previous
normalized weights W summing to one.
"""
function conditional_ess(incremental_weights::AbstractVector, previous_weights::AbstractVector)
    W = previous_weights ./ sum(previous_weights)
    w = incremental_weights ./ maximum(incremental_weights)
    return sum(W .* w)^2 / sum(W .* w .^ 2)
end

distinct_share(X::AbstractMatrix) = length(unique(eachrow(X))) / size(X, 1)

"""
```
particle_movement(before, after, weights)
```
Per column (parameter), the weighted correlation between the particles before and after
mutation, and the expected squared jump distance divided by the weighted variance before
mutation. Rows are particles.
"""
function particle_movement(before::AbstractMatrix, after::AbstractMatrix, weights::AbstractVector)
    w  = weights ./ sum(weights)
    db = before .- (w' * before)
    da = after  .- (w' * after)
    vb = vec(w' * (db .^ 2))
    va = vec(w' * (da .^ 2))
    corr = vec(w' * (db .* da)) ./ sqrt.(vb .* va)
    esjd = vec(w' * ((after .- before) .^ 2)) ./ vb
    return corr, esjd
end

"""
Initial flag thresholds for `stage_health_flags`, to be calibrated by item 5.
- `min_ess`: ESS/N before resampling
- `min_distinct`: distinct particles after mutation, as a share of N
- `accept_range`: mutation acceptance rate
- `max_late_corr`: pre/post-mutation correlation of any parameter in the last `late_share` of stages
- `max_dphi_jump`: Δφ_n / Δφ_{n-1}
- `inf_rise`, `min_inf_share`: share of -Inf proposals in the last `late_share` of stages vs the
  first `late_share`
"""
const DEFAULT_HEALTH_THRESHOLDS = (min_ess = 0.1, min_distinct = 0.5, accept_range = (0.1, 0.5),
                                   max_late_corr = 0.9, late_share = 0.2, max_dphi_jump = 10.0,
                                   inf_rise = 2.0, min_inf_share = 0.01)

function stage_list(stages)
    s = Int.(stages)
    return length(s) <= 8 ? join(s, ", ") : join(s[1:8], ", ") * ", … ($(length(s)) stages)"
end

"""
```
stage_health_flags(h; thresholds = DEFAULT_HEALTH_THRESHOLDS, para_names = nothing)
```
Flags (strings) for the per-stage record `h` of a run (see `new_stage_health`). An empty
vector means no flag fired.
"""
function stage_health_flags(h::AbstractDict; thresholds = DEFAULT_HEALTH_THRESHOLDS,
                            para_names = nothing)
    flags = String[]
    n = n_stages(h)
    n == 0 && return flags
    stage = h["stage"]

    low_ess = findall(h["ess"] .< thresholds.min_ess)
    isempty(low_ess) || push!(flags, "ESS/N < $(thresholds.min_ess) at stages $(stage_list(stage[low_ess]))")

    low_distinct = findall(h["distinct_mutated"] .< thresholds.min_distinct)
    isempty(low_distinct) ||
        push!(flags, "distinct particles after mutation < $(Int(100 * thresholds.min_distinct))% of N " *
                     "at stages $(stage_list(stage[low_distinct]))")

    lo, hi = thresholds.accept_range
    bad_accept = findall(a -> !(lo <= a <= hi), h["accept"])
    isempty(bad_accept) ||
        push!(flags, "acceptance outside [$lo, $hi] at stages $(stage_list(stage[bad_accept]))")

    late = max(1, n - ceil(Int, thresholds.late_share * n) + 1):n
    corr = h["corr"]
    if size(corr, 1) > 0
        late_corr = [maximum(x -> isnan(x) ? -Inf : x, corr[:, s]) for s in late]
        sticky = late[late_corr .> thresholds.max_late_corr]
        if !isempty(sticky)
            worst = findmax(x -> isnan(x) ? -Inf : x, corr[:, sticky[end]])[2]
            name = para_names === nothing ? "parameter $worst" : string(para_names[worst])
            push!(flags, "pre/post-mutation correlation > $(thresholds.max_late_corr) in the last " *
                         "$(Int(100 * thresholds.late_share))% of stages, at stages $(stage_list(stage[sticky])) " *
                         "(e.g. $name): particles are not moving")
        end
    end

    dphi = h["dphi"]
    jumps = [s for s in 2:n if dphi[s-1] > 0 && dphi[s] / dphi[s-1] > thresholds.max_dphi_jump]
    isempty(jumps) || push!(flags, "Δφ jumps by more than $(thresholds.max_dphi_jump)× at stages $(stage_list(stage[jumps]))")

    share = h["inf_proposals"] ./ max.(h["n_proposals"], 1)
    early = 1:min(n, ceil(Int, thresholds.late_share * n))
    late_share, early_share = mean(share[late]), mean(share[early])
    if late_share > thresholds.min_inf_share && late_share > thresholds.inf_rise * early_share
        push!(flags, "-Inf proposals rise from $(round(100 * early_share, digits = 2))% (first stages) " *
                     "to $(round(100 * late_share, digits = 2))% (last stages)")
    end
    return flags
end

"""
```
write_stage_health(path, h; para_names = nothing)
```
Write the per-stage record as CSV: the scalar columns plus, per stage, the largest pre/post
correlation and the smallest relative jump distance with the parameters they belong to.
"""
function write_stage_health(path::AbstractString, h::AbstractDict; para_names = nothing)
    name(k) = para_names === nothing ? string(k) : string(para_names[k])
    nanmax(v) = isempty(v) || all(isnan, v) ? (NaN, 0) : findmax(x -> isnan(x) ? -Inf : x, v)
    nanmin(v) = isempty(v) || all(isnan, v) ? (NaN, 0) : findmin(x -> isnan(x) ? Inf : x, v)
    open(path, "w") do io
        println(io, join((STAGE_HEALTH_COLUMNS..., "max_corr", "max_corr_para", "min_esjd", "min_esjd_para"), ","))
        for s in 1:n_stages(h)
            mc, kc = nanmax(h["corr"][:, s])
            me, ke = nanmin(h["esjd"][:, s])
            row = [string(h[col][s]) for col in STAGE_HEALTH_COLUMNS]
            push!(row, string(mc), kc == 0 ? "" : name(kc), string(me), ke == 0 ? "" : name(ke))
            println(io, join(row, ","))
        end
    end
    return path
end

########################################################################################
### Item 1 cross-check: ancestral-lineage variance estimators
########################################################################################

"""
```
lineage_nse(eve, n_resamples, weights, θ)
```
Single-run NSEs from the particles' Eve indices (the stage-0 ancestor of each particle), as
in Lee and Whiteley (2018) and, for adaptive SMC, Du and Guyader (2021):

    relative variance of Ẑ  ≈ 1 - c (1 - Σ_e S_e²)
    variance of the mean of g ≈ c Σ_e (Σ_{i∈e} Wᵢ (gᵢ - ḡ))²

with c = (N/(N-1))^(n_resamples+1), W the final weights normalized to one and S_e the total
weight of Eve e. Returns `(log_mdd_nse, mean_nse, n_eves)`. The estimates are for multinomial
resampling (others only approximately) and collapse to zero once few Eves survive, so
report them with `n_eves`; the group NSEs (item 1) are the primary ones.
"""
function lineage_nse(eve::AbstractVector{<:Integer}, n_resamples::Integer,
                     weights::AbstractVector, θ::AbstractMatrix)
    N = length(weights)
    W = weights ./ sum(weights)
    c = (N / (N - 1))^(n_resamples + 1)
    θ_bar = vec(W' * θ)
    ids = unique(eve)
    slot = Dict(e => k for (k, e) in enumerate(ids))
    S  = zeros(length(ids))
    Sθ = zeros(length(ids), size(θ, 2))
    for i in 1:N
        k = slot[eve[i]]
        S[k] += W[i]
        Sθ[k, :] .+= W[i] .* (θ[i, :] .- θ_bar)
    end
    rel_var = 1 - c * (1 - sum(S .^ 2))
    return (log_mdd_nse = sqrt(max(rel_var, 0.0)), mean_nse = sqrt.(c .* vec(sum(Sθ .^ 2, dims = 1))),
            n_eves = length(ids))
end

########################################################################################
### Item 3: log MDD by bridge sampling and the modified harmonic mean
########################################################################################

weighted_mean_cov(X::AbstractMatrix, W::AbstractVector) =
    (vec((W ./ sum(W))' * X), cov(X, Weights(W ./ sum(W)), corrected = false))

# Floor tiny/negative eigenvalues so the fitted covariance factorizes
function floored_cov(Σ::AbstractMatrix)
    S = Symmetric((Σ + Σ') / 2)
    isposdef(S) && return Matrix(S)
    F = eigen(S)
    λ = max.(F.values, max(maximum(F.values) * 1e-10, eps()))
    return Matrix(Symmetric(F.vectors * Diagonal(λ) * F.vectors'))
end

"""
```
modified_harmonic_mean(θ, logpost, weights; p = 0.1:0.1:0.9)
```
Geweke's (1999) modified harmonic mean estimate of log p(Y) from the final cloud: θ is the
N × d matrix of free parameters, `logpost` the log likelihood + log prior. The weighting
density is a normal fitted to the weighted cloud, truncated at the p-quantile of χ²_d.
Returns `(log_mdd, by_p)`, `log_mdd` the mean over p. Uses no new likelihood evaluations.
"""
function modified_harmonic_mean(θ::AbstractMatrix, logpost::AbstractVector, weights::AbstractVector;
                                p = 0.1:0.1:0.9)
    keep = findall(i -> isfinite(logpost[i]) && weights[i] > 0, eachindex(logpost))
    θ, logpost, W = θ[keep, :], logpost[keep], weights[keep] ./ sum(weights[keep])
    d = size(θ, 2)
    μ, Σ = weighted_mean_cov(θ, W)
    C = cholesky(Symmetric(floored_cov(Σ)))
    dist2 = vec(sum(abs2, (θ .- μ') / C.U, dims = 2))
    lognorm = -d / 2 * log(2π) - logdet(C) / 2
    by_p = map(p) do q
        inside = dist2 .< quantile(Chisq(d), q)
        any(inside) || return NaN
        # 1/p̂ = Σ Wᵢ f(θᵢ)/k(θᵢ), f the truncated normal
        terms = log.(W[inside]) .+ (lognorm - log(q)) .- dist2[inside] ./ 2 .- logpost[inside]
        -logsumexp(terms)
    end
    return (log_mdd = mean(filter(isfinite, by_p)), by_p = collect(by_p))
end

"""
```
bridge_sampling_estimate(logq_post, logg_post, w_post, logq_prop, logg_prop; tol, maxiter)
```
Meng and Wong's (1996) iterative optimal bridge sampling estimate of log p(Y). `logq_*` are
log posterior kernels (log likelihood + log prior) and `logg_*` log proposal densities, at the
posterior draws (with weights `w_post`) and at draws from the proposal. Returns
`(log_mdd, nse, iterations)`, where `nse` is Frühwirth-Schnatter's (2004) approximate relative
error treating both samples as independent (posterior sample size = its ESS).
"""
function bridge_sampling_estimate(logq_post::AbstractVector, logg_post::AbstractVector,
                                  w_post::AbstractVector, logq_prop::AbstractVector,
                                  logg_prop::AbstractVector; tol::Real = 1e-10, maxiter::Int = 1_000)
    W  = w_post ./ sum(w_post)
    n1 = 1 / sum(W .^ 2)
    n2 = length(logq_prop)
    s1, s2 = n1 / (n1 + n2), n2 / (n1 + n2)
    l1 = logq_post .- logg_post
    l2 = logq_prop .- logg_prop
    lstar = median(l1[isfinite.(l1)])
    e1, e2 = exp.(l1 .- lstar), exp.(l2 .- lstar)
    r, iterations = 1.0, 0
    for it in 1:maxiter
        iterations = it
        r_new = mean(e2 ./ (s1 .* e2 .+ s2 * r)) / sum(W ./ (s1 .* e1 .+ s2 * r))
        converged = abs(log(r_new) - log(r)) < tol
        r = r_new
        converged && break
    end
    # Frühwirth-Schnatter (2004): RE² ≈ Var_g(f2)/(n2 E_g(f2)²) + Var_p(f1)/(n1 E_p(f1)²)
    f1 = 1 ./ (s1 .* e1 ./ r .+ s2)
    f2 = (e2 ./ r) ./ (s1 .* e2 ./ r .+ s2)
    m1 = sum(W .* f1)
    v1 = sum(W .* (f1 .- m1) .^ 2)
    re2 = var(f2) / (n2 * mean(f2)^2) + v1 / (n1 * m1^2)
    return (log_mdd = log(r) + lstar, nse = sqrt(re2), iterations = iterations)
end

"""
```
evaluate_log_kernel(loglikelihood, parameters, data, draws; parallel = false, toggle = true)
```
Log likelihood and log prior at each row of `draws` (full parameter vectors), with -Inf for
draws outside the bounds or whose likelihood cannot be evaluated.
"""
function evaluate_log_kernel(loglikelihood::Function, parameters::ParameterVector, data::Matrix{Float64},
                             draws::Matrix{Float64}; parallel::Bool = false, toggle::Bool = true)
    function one_kernel(draw::Vector{Float64})
        try
            update!(parameters, draw)
            loglh = loglikelihood(parameters, data)
            toggle && toggle_regime!(parameters, 1)
            logprior = prior(parameters)
            return (loglh == -Inf || isnan(loglh)) ? [-Inf -Inf] : [loglh logprior]
        catch err
            if isa(err, ParamBoundsError) || isa(err, SingularException) ||
               isa(err, LinearAlgebra.LAPACKException) || isa(err, PosDefException) ||
               isa(err, DomainError)
                return [-Inf -Inf]
            end
            rethrow(err)
        end
    end
    out = if parallel
        @distributed (vcat) for m in 1:size(draws, 1)
            one_kernel(draws[m, :])
        end
    else
        reduce(vcat, [one_kernel(draws[m, :]) for m in 1:size(draws, 1)])
    end
    return out[:, 1], out[:, 2]
end

"""
```
smc_bridge_sampling(loglikelihood, parameters, data, cloud, free_para_inds; n_draws, rng, ...)
```
Item 3(b): bridge sampling estimate of log p(Y) from the final cloud. The particles are split
at random in two halves (Overstall and Forster 2010): a normal proposal g is fitted to the
first, weighted half, and the second half are the posterior draws. `n_draws` draws from g
(fixed parameters at their values) cost one likelihood evaluation each.
"""
function smc_bridge_sampling(loglikelihood::Function, parameters::ParameterVector, data::Matrix{Float64},
                             cloud::Cloud, free_para_inds::Vector{Int}; n_draws::Int = length(cloud),
                             rng::AbstractRNG = Xoshiro(0), parallel::Bool = false, toggle::Bool = true)
    N = length(cloud)
    order = randperm(rng, N)
    fit, post = order[1:N÷2], order[N÷2+1:end]
    θ = cloud.particles[:, free_para_inds]
    W = get_weights(cloud)
    μ, Σ = weighted_mean_cov(θ[fit, :], W[fit])
    g = MvNormal(μ, Symmetric(floored_cov(Σ)))
    draws = repeat(cloud.particles[1:1, 1:ind_para_end(size(cloud.particles, 2))], n_draws)
    draws[:, free_para_inds] = rand(rng, g, n_draws)'
    loglh, logprior = evaluate_log_kernel(loglikelihood, parameters, data, draws;
                                          parallel = parallel, toggle = toggle)
    return bridge_sampling_estimate(get_logpost(cloud)[post], logpdf(g, θ[post, :]'), W[post],
                                    loglh .+ logprior, logpdf(g, draws[:, free_para_inds]'))
end

"""
```
mdd_gap_flag(estimates; n_se = 2)
```
Item 3 flag: the SMC log MDD (level) and the bridge-sampling one differ by more than `n_se`
combined NSEs. Uses the group NSEs when the estimates are pooled, otherwise the single-run
lineage and Frühwirth-Schnatter approximations. Returns the flag or `nothing`.
"""
function mdd_gap_flag(estimates::AbstractDict; n_se::Real = 2)
    haskey(estimates, "bridge_sampling") || return nothing
    a, b = estimates["smc_level"], estimates["bridge_sampling"]
    nse_a = get(estimates, "smc_level_nse", get(estimates, "smc_lineage_nse", NaN))
    nse_b = estimates["bridge_sampling_nse"]
    (isfinite(a) && isfinite(b)) || return nothing
    se = sqrt(nse_a^2 + nse_b^2)
    abs(a - b) > n_se * se || return nothing
    return "log MDD gap: SMC $(round(a, digits = 3)) vs bridge sampling $(round(b, digits = 3)) " *
           "(more than $n_se combined NSEs, $(round(se, sigdigits = 3))): mass may be missed early on"
end

########################################################################################
### Item 4: multimodality of the cloud
########################################################################################

"""
```
gaussian_mixture_bic(X, weights; k_max = 4, n_components = 5, n_starts = 3, rng)
```
Fit weighted Gaussian mixtures with k = 1, …, k_max components to the cloud X (N × d), after
standardizing it and keeping its first `n_components` principal components (a full mixture in
all ~100 DSGE parameters would have far more parameters than particles). Returns
`(best_k, bic, mixture_weights, centers)`: BIC per k (lower is better), and for the best k
the component weights and centers mapped back to the original parameter space.
"""
function gaussian_mixture_bic(X::AbstractMatrix, weights::AbstractVector; k_max::Int = 4,
                              n_components::Int = 5, n_starts::Int = 3, maxiter::Int = 500,
                              tol::Real = 1e-8, rng::AbstractRNG = Xoshiro(0))
    W = weights ./ sum(weights)
    N = size(X, 1)
    μ, Σ = weighted_mean_cov(X, W)
    sd = sqrt.(max.(diag(Σ), eps()))
    Z = (X .- μ') ./ sd'
    F = eigen(Symmetric(cov(Z, Weights(W), corrected = false)))
    k_pc = min(n_components, size(X, 2))
    V = F.vectors[:, end:-1:end-k_pc+1]
    Y = Z * V
    fits = [best_mixture_fit(Y, W, k, n_starts, maxiter, tol, rng) for k in 1:k_max]
    d = size(Y, 2)
    n_free = [k * d + k * d * (d + 1) ÷ 2 + k - 1 for k in 1:k_max]
    bic = [-2 * N * fits[k].loglik + n_free[k] * log(N) for k in 1:k_max]
    best = argmin(bic)
    centers = [μ .+ sd .* (V * fits[best].means[:, j]) for j in 1:best]
    return (best_k = best, bic = bic, mixture_weights = fits[best].weights, centers = centers)
end

# Weighted EM; loglik is the weighted mean log density (weights sum to one)
function best_mixture_fit(Y, W, k, n_starts, maxiter, tol, rng)
    best = nothing
    for _ in 1:n_starts
        fit = mixture_em(Y, W, k, maxiter, tol, rng)
        (best === nothing || fit.loglik > best.loglik) && (best = fit)
    end
    return best
end

function mixture_em(Y, W, k, maxiter, tol, rng)
    N, d = size(Y)
    means = Y[sample(rng, 1:N, Weights(W), k; replace = false), :]'
    covs  = [Matrix(cov(Y, Weights(W), corrected = false)) + 1e-6I for _ in 1:k]
    props = fill(1 / k, k)
    loglik = -Inf
    logdens = zeros(N, k)
    for _ in 1:maxiter
        for j in 1:k
            logdens[:, j] = log(props[j]) .+ logpdf(MvNormal(means[:, j], Symmetric(covs[j])), Y')
        end
        lse = [logsumexp(logdens[i, :]) for i in 1:N]
        new_loglik = sum(W .* lse)
        R = exp.(logdens .- lse) .* W
        for j in 1:k
            r = R[:, j]
            nj = sum(r)
            nj < 1e-10 && continue
            props[j] = nj
            means[:, j] = Y' * r ./ nj
            D = Y .- means[:, j]'
            covs[j] = (D' * (D .* r)) ./ nj + 1e-6I
        end
        props ./= sum(props)
        converged = abs(new_loglik - loglik) < tol
        loglik = new_loglik
        converged && break
    end
    return (loglik = loglik, weights = props, means = Matrix(means))
end

########################################################################################
### Items 5 and 6: comparisons across runs
########################################################################################

"""
```
nse_validation(estimates, nses, n_groups; level = 0.95)
```
Item 5: R independent runs (rows) of K pooled estimates (columns), each with its in-run NSE
from `n_groups` groups. Returns, per estimate, the ratio of the cross-run sd to the RMS in-run
NSE (≈ 1 if the NSEs are right) and the coverage of the t_{J-1} `level` interval around each
run's estimate for the leave-one-out mean of the other runs (≈ `level`, slightly less
for small R since the comparison mean is itself noisy).
"""
function nse_validation(estimates::AbstractMatrix, nses::AbstractMatrix, n_groups::Integer;
                        level::Real = 0.95)
    R = size(estimates, 1)
    R > 1 || throw(ArgumentError("need at least two runs"))
    cross_sd = vec(std(estimates, dims = 1))
    rms_nse  = vec(sqrt.(mean(nses .^ 2, dims = 1)))
    t = quantile(TDist(n_groups - 1), (1 + level) / 2)
    total = sum(estimates, dims = 1)
    loo_mean = (total .- estimates) ./ (R - 1)
    coverage = vec(mean(abs.(estimates .- loo_mean) .<= t .* nses, dims = 1))
    return (sd_ratio = cross_sd ./ rms_nse, coverage = coverage, cross_sd = cross_sd,
            rms_nse = rms_nse, grand_mean = vec(total) ./ R)
end

"""
```
compare_estimates(a, nse_a, b, nse_b)
```
Item 6: z = (a - b)/√(nse_a² + nse_b²) per estimate, the share with |z| > 2 (≈ 5% if the two
runs agree), and a Kolmogorov–Smirnov test of the z's against N(0, 1) (asymptotic p-value).
"""
function compare_estimates(a::AbstractVector, nse_a::AbstractVector, b::AbstractVector,
                           nse_b::AbstractVector)
    z = (a .- b) ./ sqrt.(nse_a .^ 2 .+ nse_b .^ 2)
    zs = sort(filter(isfinite, z))
    n = length(zs)
    Φ = cdf.(Normal(), zs)
    ks = n == 0 ? NaN : maximum(max.((1:n) ./ n .- Φ, Φ .- (0:n-1) ./ n))
    return (z = z, share_above_2 = n == 0 ? NaN : count(x -> abs(x) > 2, zs) / n,
            ks_statistic = ks, ks_pvalue = n == 0 ? NaN : ccdf(Kolmogorov(), sqrt(n) * ks))
end
