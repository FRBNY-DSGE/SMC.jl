# Mutation drift diagnostic and the opt-in fixes it can drive (ridge move, adaptive MH steps).
#
# Idea. If the selected (pre-mutation) cloud is distributed as the stage target π_ϕ and the MH
# kernel K leaves π_ϕ invariant, the cloud after mutation is distributed as π_ϕ too, so the
# weighted mean shift of mutation, Δ = mean_post - mean_pre, has expectation zero. A cloud
# that lags a moving target (each stage's kernel only partly closes the gap) shows a
# systematic Δ ≠ 0 that points, stage after stage, in the same direction.
#
# Per stage n, on the free parameters, with the stage's normalized weights W (unchanged by
# mutation) and the pre-mutation weighted sd s_j:
#   Δ̃_i   = (θ⁺_i - θ⁻_i) ./ s                     each particle's own displacement, in sd units
#   δ_n   = Σ_i W_i Δ̃_i                            the mutation shift (= (mean_post - mean_pre)/s)
#   SE    : the particles are paired (same particle before and after), so the SE comes from the
#           spread of the displacements themselves. Copies of the same pre-mutation point (from
#           resampling) share their starting point, so they form one cluster:
#               SE(v)² = K/(K-1) Σ_k (Σ_{i∈k} W_i v'(Δ̃_i - δ_n))²
#           for any fixed direction v, K the number of clusters. Without copies and with equal
#           weights this is Var(v'Δ̃)/N, and per parameter Var(Δ̃_j) = (σ⁻² + σ⁺² - 2ρ σ⁻σ⁺)/σ⁻²
#           ≈ 2(1 - ρ_j), so SE_j ≈ √(2(1 - ρ_j)/ESS) with ρ_j the pre/post correlation.
#   u_n   : the leading direction, the normalized EWMA (span m) of δ_1..δ_{n-1-L} over the drift
#           parameters, i.e. estimated from PREVIOUS stages only, leaving out the last L
#           (`drift_direction_lag`, default 3). So u_n is independent of stage n's mutation noise.
#   z_n   = u_n'δ_n / SE(u_n)
#   Why the lag: a finite cloud's mean error e_n = mean_pre - E_π θ is partly the previous
#           stages' mutation noise, and a mixing kernel pulls it back: E[δ_n | past] ≈ -(1-ρ) e_n.
#           So δ_n is negatively correlated with δ_{n-1}, δ_{n-2}, … (for an AR(1)-like kernel
#           with per-stage autocorrelation ρ, corr(δ_n, δ_{n-k}) ∝ -ρ^(k-1)(1-ρ)), and a direction
#           built from the latest shifts would make z_n negative on average even under exact
#           invariance (about -0.6 in a well-mixed 10-dim toy). Leaving out the last L shifts
#           shrinks this to order ρ^L (1-ρ): negligible both when mixing is fast (ρ small) and
#           when it is slow (1-ρ small, the case the diagnostic is for). Under invariance z_n is
#           then ≈ N(0, 1); a lagging cloud gives z_n ≫ 0.
#   dloglh: the same paired shift for the log likelihood (direction-free lag signal).
#
# Alarm (used by the trigger options): `fire` when the last `drift_trigger_stages` values of
# z_n are all > `drift_trigger_z`; `calm` when they are all |z_n| < `drift_calm_z`.

const DRIFT_COLUMNS = ("drift_rms", "drift_rms_null", "drift_proj", "drift_proj_se", "drift_z",
                       "drift_cos", "drift_dloglh", "drift_dloglh_z", "drift_alarm", "n_mh",
                       "ridge_on", "ridge_tried", "ridge_accept", "ridge_c", "ridge_step")

"""
```
mutation_shift(θ_pre, θ_post, W; loglh_pre, loglh_post, u = nothing, sd = nothing)
```
The mutation shift of one stage (rows of `θ_pre`/`θ_post` are the same particles before and
after mutation, `W` their weights). Returns a NamedTuple: `delta` (per column, in units of the
pre-mutation weighted sd `sd`), `se` (its paired, cluster-robust SE per column), `proj`/`proj_se`
(u'δ and its SE for a unit vector `u` in the same standardized coordinates; NaN without `u`),
`dloglh`/`dloglh_se` (the log-likelihood shift) and `n_clusters` (distinct pre-mutation rows).
"""
function mutation_shift(θ_pre::AbstractMatrix, θ_post::AbstractMatrix, W::AbstractVector;
                        loglh_pre::AbstractVector = zeros(size(θ_pre, 1)),
                        loglh_post::AbstractVector = zeros(size(θ_pre, 1)),
                        u::Union{Nothing, AbstractVector} = nothing,
                        sd::Union{Nothing, AbstractVector} = nothing)
    N, d = size(θ_pre)
    w = W ./ sum(W)
    if sd === nothing
        μ  = vec(w' * θ_pre)
        sd = sqrt.(vec(w' * ((θ_pre .- μ') .^ 2)))
    end
    s = map(x -> x > 0 ? x : one(x), sd)          # a parameter that does not vary: shift in levels
    D = (θ_post .- θ_pre) ./ s'
    δ = vec(w' * D)
    L = loglh_post .- loglh_pre
    L[.!isfinite.(L)] .= 0.0
    dL = sum(w .* L)
    # Clusters: identical pre-mutation rows (copies of one resampled particle)
    cluster = Dict{Vector{Float64}, Int}()
    id = Vector{Int}(undef, N)
    for i in 1:N
        id[i] = get!(cluster, θ_pre[i, :], length(cluster) + 1)
    end
    K = length(cluster)
    Sk  = zeros(K, d)
    SkL = zeros(K)
    for i in 1:N
        k = id[i]
        @views Sk[k, :] .+= w[i] .* (D[i, :] .- δ)
        SkL[k] += w[i] * (L[i] - dL)
    end
    corr = K > 1 ? K / (K - 1) : NaN
    se = sqrt.(corr .* vec(sum(Sk .^ 2, dims = 1)))
    proj = proj_se = NaN
    if u !== nothing && any(!iszero, u)
        proj    = dot(u, δ)
        proj_se = sqrt(corr * sum(abs2, Sk * u))
    end
    return (delta = δ, se = se, proj = proj, proj_se = proj_se, dloglh = dL,
            dloglh_se = sqrt(corr * sum(abs2, SkL)), n_clusters = K)
end

"""
Settings of the drift diagnostic and the fixes (see `smc`'s keyword arguments).
"""
struct DriftSettings
    gate::Vector{Int}            # free-parameter indices the direction uses (drift_params)
    span::Int                    # EWMA span m of the leading direction
    lag::Int                     # the direction leaves out the last `lag` shifts (see below)
    trigger_z::Float64
    trigger_stages::Int
    calm_z::Float64
    ridge_mode::Symbol           # :off, :always, :trigger
    ridge_prob::Float64
    ridge_direction::Union{Symbol, Vector{Float64}}   # :drift, :pc, or a fixed direction
    ridge_c::Float64
    ridge_target::Float64
    adaptive_mh::Bool
    n_mh_base::Int
    n_mh_max::Int
end

function DriftSettings(gate::Vector{Int}, n_free::Int; span::Int = 10, lag::Int = 3, trigger_z::Real = 3.0,
                       trigger_stages::Int = 3, calm_z::Real = 2.0, ridge_mode::Symbol = :off,
                       ridge_prob::Real = 1.0, ridge_direction = :drift, ridge_c::Real = 1.0,
                       ridge_target::Real = 0.35, adaptive_mh::Bool = false, n_mh_base::Int = 1,
                       n_mh_max::Int = n_mh_base)
    span >= 1 || throw(ArgumentError("drift_ewma_span must be at least 1"))
    lag >= 0 || throw(ArgumentError("drift_direction_lag must be nonnegative"))
    trigger_stages >= 1 || throw(ArgumentError("drift_trigger_stages must be at least 1"))
    0 <= calm_z <= trigger_z || throw(ArgumentError("need 0 ≤ drift_calm_z ≤ drift_trigger_z"))
    ridge_mode in (:off, :always, :trigger) ||
        throw(ArgumentError("ridge_mode must be :off, :always or :trigger, got $(repr(ridge_mode))"))
    0 < ridge_prob <= 1 || throw(ArgumentError("ridge_prob must be in (0, 1]"))
    (ridge_c > 0 && isfinite(ridge_c)) || throw(ArgumentError("ridge_c must be finite and positive"))
    0 < ridge_target < 1 || throw(ArgumentError("ridge_target must be in (0, 1)"))
    n_mh_max >= n_mh_base || throw(ArgumentError("n_mh_max must be at least n_mh_steps"))
    dir = if ridge_direction isa Symbol
        ridge_direction in (:drift, :pc) ||
            throw(ArgumentError("ridge_direction must be :drift, :pc or a vector, got $(repr(ridge_direction))"))
        ridge_direction
    else
        v = Vector{Float64}(ridge_direction)
        length(v) == n_free ||
            throw(ArgumentError("ridge_direction has $(length(v)) entries for $n_free free parameters"))
        (all(isfinite, v) && any(!iszero, v)) || throw(ArgumentError("ridge_direction must be finite and nonzero"))
        v
    end
    return DriftSettings(gate, span, lag, Float64(trigger_z), trigger_stages, Float64(calm_z), ridge_mode,
                         Float64(ridge_prob), dir, Float64(ridge_c), Float64(ridge_target),
                         adaptive_mh, n_mh_base, n_mh_max)
end

"""
The state the diagnostic carries from stage to stage (saved in checkpoints as `drift_state`):
the EWMA `g` of the shifts (standardized, drift parameters only), the `pc` direction from the
previous stage's mutated cloud, the current number of MH steps, whether the ridge is on, its
scale, and the recent z values.
"""
function new_drift_state(n_free::Int, ds::DriftSettings)
    return Dict{String, Any}("g" => zeros(n_free), "pending" => Vector{Float64}[], "pc" => zeros(n_free), "n_mh" => ds.n_mh_base,
                             "ridge_on" => ds.ridge_mode == :always, "ridge_c" => ds.ridge_c,
                             "recent_z" => Float64[])
end

# Unit vector of the EWMA (zeros if none yet)
function leading_direction(g::AbstractVector)
    n = norm(g)
    return n > 0 ? g ./ n : zeros(length(g))
end

"""
```
ridge_move_for_stage(ds, state, sd, R_fr, free_para_inds, n_para)
```
The ridge move of this stage, or `nothing`: the direction in standardized coordinates (the
EWMA leading direction, the previous stage's top principal component, or the fixed vector),
mapped to parameter units e = s ⊙ u / ‖s ⊙ u‖ with the pre-mutation sd s, and the step
c_r √(e'Σe) with Σ the pre-mutation covariance. Both are fixed for the stage.
"""
function ridge_move_for_stage(ds::DriftSettings, state::AbstractDict, sd::AbstractVector,
                              R_fr::AbstractMatrix, free_para_inds::Vector{Int}, n_para::Int)
    ds.ridge_mode == :off && return nothing
    state["ridge_on"] || return nothing
    u = ds.ridge_direction === :drift ? leading_direction(state["g"]) :
        ds.ridge_direction === :pc    ? state["pc"] : ds.ridge_direction
    any(!iszero, u) || return nothing
    e = sd .* u
    ne = norm(e)
    (ne > 0 && isfinite(ne)) || return nothing
    e ./= ne
    σe = sqrt(max(dot(e, R_fr * e), 0.0))
    (σe > 0 && isfinite(σe)) || return nothing
    direction = zeros(n_para)
    direction[free_para_inds] = e
    return (direction = direction, step = state["ridge_c"] * σe, prob = ds.ridge_prob)
end

# Top eigenvector of the weighted correlation matrix of the drift parameters (zeros elsewhere)
function top_correlation_pc(θ::AbstractMatrix, W::AbstractVector, gate::Vector{Int})
    w = W ./ sum(W)
    X = θ[:, gate]
    μ = vec(w' * X)
    D = X .- μ'
    C = D' * (w .* D)
    s = sqrt.(max.(diag(C), 0.0))
    keep = findall(>(0), s)
    out = zeros(size(θ, 2))
    isempty(keep) && return out
    Rc = C[keep, keep] ./ (s[keep] * s[keep]')
    F = eigen(Symmetric((Rc + Rc') / 2))
    out[gate[keep]] = F.vectors[:, end]
    return out
end

"""
```
update_drift_state!(state, ds, z, delta, ridge_accept)
```
After stage n: update the EWMA with δ_n (drift parameters), the alarm, the ridge switch (with
`ridge_mode = :trigger`: on at `fire`, off at `calm`), the ridge scale (towards `ridge_target`,
c_r ⋅ exp(3 (acceptance - target)), the factor kept in [0.5, 2]) and, with `adaptive_mh`, the
number of MH steps for the next stage (+1 at `fire` up to `n_mh_max`, -1 at `calm` down to
`n_mh_steps`). Returns the alarm (1 fire, -1 calm, 0 neither).
"""
function update_drift_state!(state::AbstractDict, ds::DriftSettings, z::Real, delta::AbstractVector,
                             ridge_accept::Real)
    # δ_n waits `lag` stages before it enters the EWMA (see the note on mean reversion above)
    β = 2 / (ds.span + 1)
    g = state["g"]
    pending = state["pending"]
    push!(pending, Vector{Float64}(delta))
    while length(pending) > ds.lag
        d = popfirst!(pending)
        for j in ds.gate
            isfinite(d[j]) && (g[j] = (1 - β) * g[j] + β * d[j])
        end
    end
    recent = state["recent_z"]
    isfinite(z) && push!(recent, z)
    length(recent) > ds.trigger_stages && deleteat!(recent, 1:length(recent) - ds.trigger_stages)
    full  = length(recent) == ds.trigger_stages
    fire  = full && all(>(ds.trigger_z), recent)
    calm  = full && all(x -> abs(x) < ds.calm_z, recent)
    alarm = fire ? 1 : calm ? -1 : 0
    if ds.ridge_mode == :trigger
        fire && (state["ridge_on"] = true)
        calm && (state["ridge_on"] = false)
    end
    if isfinite(ridge_accept)
        state["ridge_c"] *= clamp(exp(3.0 * (ridge_accept - ds.ridge_target)), 0.5, 2.0)
    end
    if ds.adaptive_mh
        fire && (state["n_mh"] = min(state["n_mh"] + 1, ds.n_mh_max))
        calm && (state["n_mh"] = max(state["n_mh"] - 1, ds.n_mh_base))
    end
    return alarm
end

"""
```
add_drift_health!(h, n_free_para)
```
Add the drift columns (`DRIFT_COLUMNS`, plus the n_free × stages matrices `drift_delta`, the
shift δ_n, and `drift_dir`, the leading direction u_n used at stage n) to a stage health
record; stages recorded without them (a resumed older checkpoint) get NaN.
"""
function add_drift_health!(h::AbstractDict, n_free_para::Int)
    n = n_stages(h)
    for col in DRIFT_COLUMNS
        haskey(h, col) || (h[col] = fill(NaN, n))
    end
    haskey(h, "drift_delta") || (h["drift_delta"] = fill(NaN, n_free_para, n))
    haskey(h, "drift_dir")   || (h["drift_dir"]   = fill(NaN, n_free_para, n))
    return h
end

function record_drift_health!(h::AbstractDict, row::NamedTuple, delta::AbstractVector,
                              dir::AbstractVector)
    for col in DRIFT_COLUMNS
        push!(h[col], Float64(row[Symbol(col)]))
    end
    h["drift_delta"] = hcat(h["drift_delta"], delta)
    h["drift_dir"]   = hcat(h["drift_dir"], dir)
    return h
end

"""
```
write_drift_health(path, h; para_names = nothing, n_top = 5)
```
Write the drift diagnostic per stage as CSV (a sidecar of `write_stage_health`, whose layout is
unchanged): `stage`, `phi`, the `DRIFT_COLUMNS`, and the `n_top` parameters with the largest
weights in the leading direction u used at that stage (`dir_top`, as `name:+0.41;…`).
"""
function write_drift_health(path::AbstractString, h::AbstractDict; para_names = nothing, n_top::Int = 5)
    name(k) = para_names === nothing ? string(k) : string(para_names[k])
    open(path, "w") do io
        println(io, join(("stage", "phi", DRIFT_COLUMNS..., "dir_top"), ","))
        for s in 1:n_stages(h)
            u = h["drift_dir"][:, s]
            top = all(x -> isnan(x) || iszero(x), u) ? "" :
                join(["$(name(k)):$(round(u[k], digits = 3))" for k in sortperm(abs.(replace(u, NaN => 0.0)), rev = true)[1:min(n_top, length(u))]], ";")
            row = vcat(string(h["stage"][s]), string(h["phi"][s]), [string(h[col][s]) for col in DRIFT_COLUMNS], top)
            println(io, join(row, ","))
        end
    end
    return path
end
