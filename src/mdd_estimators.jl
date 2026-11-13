# Marginal data density (MDD) estimators for a weighted posterior cloud:
#
#   bounds transform     BoundsTransform / bounds_transform, bounds_to_real, bounds_to_model,
#                        bounds_logjac, bounds_logjac_real, bounds_in_support
#   bridge sampling      bridge_sampling (one shot), or bridge_draws → caller evaluates the
#                        proposals → bridge_combine (the two-step form for SGE arrays)
#   IS lower bound       is_lower_bound, or is_bound_draws → is_bound_combine
#   prior normalization  effective_prior_lognorm
#
# Conventions: parameter matrices have one draw per ROW (N × d), as `modified_harmonic_mean`
# and `cloud.particles`. A "log kernel" is log likelihood + log prior; every estimator returns
# log ∫ kernel + `lognorm`, where `lognorm` (default 0) is the log normalizing constant of the
# prior (see `effective_prior_lognorm` for SMC's effective prior).

########################################################################################
### Bounds transform
########################################################################################

"""
```
BoundsTransform(lb, ub)
bounds_transform(lb, ub)
bounds_transform(bounds::AbstractVector{<:Tuple})
bounds_transform(parameters::ParameterVector; regime_switching = false, toggle = true)
```
Per-coordinate map of θ, with `lb[j] < θ[j] < ub[j]`, to the real line:

| bounds   | y = T(θ)                 | θ = T⁻¹(y)        | log abs(dθ/dy)                     |
|:---------|:-------------------------|:------------------|:-----------------------------------|
| (a, b)   | log(θ - a) - log(b - θ)  | a + (b - a) σ(y)  | log(b - a) + log σ(y) + log σ(-y)  |
| (a, ∞)   | log(θ - a)               | a + exp(y)        | y                                  |
| (-∞, b)  | -log(b - θ)              | b - exp(-y)       | -y                                 |
| (-∞, ∞)  | θ                        | y                 | 0                                  |

(σ is the logistic function.) A density k(θ) on the bounded space is k(T⁻¹(y)) abs(dθ/dy) on the
real line, so adding the log Jacobian to a log kernel leaves its integral unchanged.

Numerical edge cases:
- θ exactly at a finite bound maps to ∓Inf and its log Jacobian is -Inf (zero density on
  the real line). θ outside [a, b] (or NaN) throws a `DomainError`.
- T⁻¹ is computed from the nearer bound (a + (b - a)σ(y) for y < 0, b - (b - a)σ(-y) for
  y ≥ 0), so θ keeps its relative precision near either bound. For large abs(y) θ still rounds
  onto the bound (abs(y) ≳ 37 + log(b - a) for a logit); `bounds_in_support` then returns false
  and the estimators below treat such a draw as having kernel 0 without evaluating it.
- `bounds_logjac_real(bt, y)` evaluates the log Jacobian from y and stays exact where θ has
  rounded onto a bound; `bounds_logjac(bt, θ)` evaluates it from θ (for posterior draws).
- Very wide finite bounds (such as ±1e5) are fine: the logit is then nearly linear.

The `ParameterVector` method takes the value bounds of the free slots, in the order of
`ModelConstructors.get_free_para_inds(parameters; regime_switching, toggle)`, with
regime-specific value bounds where set: the bounds that `rand(::ParameterVector)` and
`rand_regime_switching` draw within (see `parameter_slots`).
"""
struct BoundsTransform
    lb::Vector{Float64}
    ub::Vector{Float64}
    function BoundsTransform(lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real})
        length(lb) == length(ub) ||
            throw(DimensionMismatch("$(length(lb)) lower bounds but $(length(ub)) upper bounds"))
        for j in eachindex(lb, ub)
            (isnan(lb[j]) || isnan(ub[j]) || !(lb[j] < ub[j])) &&
                throw(ArgumentError("coordinate $j: the lower bound $(lb[j]) must be below the upper bound $(ub[j])"))
        end
        return new(Float64.(collect(lb)), Float64.(collect(ub)))
    end
end
Base.length(bt::BoundsTransform) = length(bt.lb)

bounds_transform(lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real}) = BoundsTransform(lb, ub)
bounds_transform(bounds::AbstractVector{<:Tuple}) = BoundsTransform(first.(bounds), last.(bounds))
bounds_transform(bt::BoundsTransform) = bt
"""Identity transform (no finite bounds) in dimension d."""
bounds_transform(d::Integer) = BoundsTransform(fill(-Inf, d), fill(Inf, d))
function bounds_transform(parameters::ParameterVector; regime_switching::Bool = false,
                          toggle::Bool = true)
    slots = Base.filter(s -> !s.fixed, parameter_slots(parameters; regime_switching = regime_switching,
                                                       toggle = toggle))
    return BoundsTransform([s.bounds[1] for s in slots], [s.bounds[2] for s in slots])
end

# log σ(y) and log σ(-y), without overflow or cancellation
@inline function _log_logistic_pair(y::Float64)
    if y >= 0
        l = log1p(exp(-y))
        return -l, -y - l
    else
        l = log1p(exp(y))
        return y - l, -l
    end
end
@inline _logistic(x::Float64) = x >= 0 ? 1 / (1 + exp(-x)) : (e = exp(x); e / (1 + e))

@inline function _to_real(θ::Real, a::Float64, b::Float64)
    (a <= θ <= b) || throw(DomainError(θ, "outside the bounds ($a, $b)"))
    fa, fb = isfinite(a), isfinite(b)
    return fa && fb ? log(θ - a) - log(b - θ) :
           fa       ? log(θ - a) :
           fb       ? -log(b - θ) : Float64(θ)
end
@inline function _to_model(y::Real, a::Float64, b::Float64)
    fa, fb = isfinite(a), isfinite(b)
    if fa && fb
        return y >= 0 ? b - (b - a) * _logistic(-Float64(y)) : a + (b - a) * _logistic(Float64(y))
    end
    return fa ? a + exp(y) : fb ? b - exp(-y) : Float64(y)
end
@inline function _logjac_model(θ::Real, a::Float64, b::Float64)
    fa, fb = isfinite(a), isfinite(b)
    return fa && fb ? log(θ - a) + log(b - θ) - log(b - a) :
           fa       ? log(θ - a) :
           fb       ? log(b - θ) : 0.0
end
@inline function _logjac_real(y::Real, a::Float64, b::Float64)
    fa, fb = isfinite(a), isfinite(b)
    if fa && fb
        ls, l1s = _log_logistic_pair(Float64(y))
        return log(b - a) + ls + l1s
    end
    return fa ? Float64(y) : fb ? -Float64(y) : 0.0
end

function _check_dim(bt::BoundsTransform, d::Integer)
    length(bt) == d || throw(DimensionMismatch("the transform has $(length(bt)) coordinates, the draws $d"))
end

"""
```
bounds_to_real(bt, θ)
```
y = T(θ) (see `BoundsTransform`). θ is a vector, or a matrix with one draw per row.
"""
function bounds_to_real(bt::BoundsTransform, θ::AbstractVector{<:Real})
    _check_dim(bt, length(θ))
    return [_to_real(θ[j], bt.lb[j], bt.ub[j]) for j in eachindex(θ)]
end
function bounds_to_real(bt::BoundsTransform, Θ::AbstractMatrix{<:Real})
    _check_dim(bt, size(Θ, 2))
    return [_to_real(Θ[i, j], bt.lb[j], bt.ub[j]) for i in axes(Θ, 1), j in axes(Θ, 2)]
end

"""
```
bounds_to_model(bt, y)
```
θ = T⁻¹(y) (see `BoundsTransform`). y is a vector, or a matrix with one draw per row.
"""
function bounds_to_model(bt::BoundsTransform, y::AbstractVector{<:Real})
    _check_dim(bt, length(y))
    return [_to_model(y[j], bt.lb[j], bt.ub[j]) for j in eachindex(y)]
end
function bounds_to_model(bt::BoundsTransform, Y::AbstractMatrix{<:Real})
    _check_dim(bt, size(Y, 2))
    return [_to_model(Y[i, j], bt.lb[j], bt.ub[j]) for i in axes(Y, 1), j in axes(Y, 2)]
end

"""
```
bounds_logjac(bt, θ)
```
log abs(dθ/dy), summed over the coordinates, at the model-space point θ (vector), or per row of a
matrix. -Inf at a bound.
"""
function bounds_logjac(bt::BoundsTransform, θ::AbstractVector{<:Real})
    _check_dim(bt, length(θ))
    return sum(j -> _logjac_model(θ[j], bt.lb[j], bt.ub[j]), eachindex(θ); init = 0.0)
end
bounds_logjac(bt::BoundsTransform, Θ::AbstractMatrix{<:Real}) =
    [bounds_logjac(bt, view(Θ, i, :)) for i in axes(Θ, 1)]

"""
```
bounds_logjac_real(bt, y)
```
log abs(dθ/dy), summed over the coordinates, evaluated from the real-line point y (vector), or per
row of a matrix. Exact even where T⁻¹(y) rounds onto a bound.
"""
function bounds_logjac_real(bt::BoundsTransform, y::AbstractVector{<:Real})
    _check_dim(bt, length(y))
    return sum(j -> _logjac_real(y[j], bt.lb[j], bt.ub[j]), eachindex(y); init = 0.0)
end
bounds_logjac_real(bt::BoundsTransform, Y::AbstractMatrix{<:Real}) =
    [bounds_logjac_real(bt, view(Y, i, :)) for i in axes(Y, 1)]

"""
```
bounds_in_support(bt, θ)
```
Whether every coordinate of θ is strictly inside its bounds (per row for a matrix).
"""
bounds_in_support(bt::BoundsTransform, θ::AbstractVector{<:Real}) =
    all(j -> bt.lb[j] < θ[j] < bt.ub[j], eachindex(θ))
bounds_in_support(bt::BoundsTransform, Θ::AbstractMatrix{<:Real}) =
    BitVector([bounds_in_support(bt, view(Θ, i, :)) for i in axes(Θ, 1)])

"""
```
parameter_slots(parameters; regime_switching = false, toggle = true)
```
One entry per parameter slot, in the layout of `rand(parameters; regime_switching)` (regime 1
of every parameter, then the extra regimes of the regime-switching parameters): a named tuple
`(key, regime, fixed, prior, bounds)`. `prior` is the slot's prior distribution (the
regime-specific one where set; `nothing` without a prior) and `bounds` its value bounds (the
regime-specific ones where set). `findall(s -> !s.fixed, slots)` equals
`get_free_para_inds(parameters; regime_switching)`.
"""
function parameter_slots(parameters::ParameterVector; regime_switching::Bool = false,
                         toggle::Bool = true)
    regime_switching && toggle && ModelConstructors.toggle_regime!(parameters, 1)
    prior_of(p) = ModelConstructors.hasprior(p) ? p.prior.value : nothing
    slots = NamedTuple{(:key, :regime, :fixed, :prior, :bounds),
                       Tuple{Symbol, Int, Bool, Any, Tuple{Float64, Float64}}}[]
    for p in parameters
        pr = (regime_switching && haskey(p.regimes, :prior) && haskey(p.regimes[:prior], 1)) ?
             ModelConstructors.regime_prior(p, 1).value : prior_of(p)
        push!(slots, (key = p.key, regime = 1, fixed = p.fixed, prior = pr,
                      bounds = (Float64(p.valuebounds[1]), Float64(p.valuebounds[2]))))
    end
    regime_switching || return slots
    for p in parameters
        (isempty(p.regimes) || !haskey(p.regimes, :value)) && continue
        for r in keys(p.regimes[:value])
            r == 1 && continue
            fixed = haskey(p.regimes, :fixed) ? ModelConstructors.regime_fixed(p, r) : p.fixed
            pr = (haskey(p.regimes, :prior) && haskey(p.regimes[:prior], r)) ?
                 ModelConstructors.regime_prior(p, r).value : prior_of(p)
            vb = haskey(p.regimes, :valuebounds) ? ModelConstructors.regime_valuebounds(p, r) : p.valuebounds
            push!(slots, (key = p.key, regime = r, fixed = fixed, prior = pr,
                          bounds = (Float64(vb[1]), Float64(vb[2]))))
        end
    end
    return slots
end

########################################################################################
### Shared pieces: clusters, the proposal, log-space helpers
########################################################################################

_logaddexp(a::Float64, b::Float64) = (m = max(a, b); isfinite(m) ? m + log(exp(a - m) + exp(b - m)) : m)
_logmeanexp_all(x::AbstractVector) = logsumexp(x) - log(length(x))   # -Inf entries count in n

"""
```
duplicate_clusters(θ)
```
Cluster id per row of θ: identical rows (the duplicates that resampling leaves in an SMC
cloud) share an id. Ids are 1, 2, … in order of first appearance.
"""
function duplicate_clusters(θ::AbstractMatrix)
    ids = Dict{Vector{Float64}, Int}()
    return [get!(ids, Vector{Float64}(θ[i, :]), length(ids) + 1) for i in axes(θ, 1)]
end

function _resolve_clusters(clusters, θ::AbstractMatrix)
    n = size(θ, 1)
    clusters === :duplicates && return duplicate_clusters(θ)
    clusters === :none && return collect(1:n)
    clusters isa AbstractVector{<:Integer} && length(clusters) == n && return collect(Int, clusters)
    throw(ArgumentError("clusters must be :duplicates, :none or a vector of $n integer ids " *
                        "(for example the particles' Eve indices)"))
end

# Relabel cluster ids 1, …, G in order of first appearance
function _relabel(c::AbstractVector{<:Integer})
    ids = Dict{Int, Int}()
    return [get!(ids, x, length(ids) + 1) for x in c]
end

"""
```
MvTProposal(μ, Σ, ν)
```
Multivariate t proposal with mean μ and covariance Σ (scale Σ(ν - 2)/ν, so ν > 2), or a
normal for ν = Inf. `logpdf_proposal(g, Y)` and `rand_proposal(rng, g, n)` work on draws in
rows.
"""
struct MvTProposal
    μ::Vector{Float64}
    L::LowerTriangular{Float64, Matrix{Float64}}
    ν::Float64
    logc::Float64
end
_loggamma(a::Real) = -1.0 - logpdf(Gamma(a, 1.0), 1.0)   # log Γ(a), no SpecialFunctions dependency
function MvTProposal(μ::AbstractVector, Σ::AbstractMatrix, ν::Real)
    ν > 2 || throw(ArgumentError("ν must exceed 2 (ν = Inf gives a normal proposal)"))
    d = length(μ)
    S = isfinite(ν) ? Σ .* ((ν - 2) / ν) : Matrix(Σ)
    L = cholesky(Symmetric(floored_cov(S))).L
    ld = 2 * sum(log, diag(L))
    logc = isfinite(ν) ? _loggamma((ν + d) / 2) - _loggamma(ν / 2) - d / 2 * log(ν * π) - ld / 2 :
                         -d / 2 * log(2π) - ld / 2
    return MvTProposal(Vector{Float64}(μ), L, Float64(ν), logc)
end
# Columns are draws internally (d × n)
function _logpdf_cols(g::MvTProposal, Y::AbstractMatrix)
    q = vec(sum(abs2, g.L \ (Y .- g.μ), dims = 1)); d = length(g.μ)
    return isfinite(g.ν) ? g.logc .- (g.ν + d) / 2 .* log1p.(q ./ g.ν) : g.logc .- q ./ 2
end
function _rand_cols(rng::AbstractRNG, g::MvTProposal, n::Integer)
    d = length(g.μ); Z = g.L * randn(rng, d, n)
    isfinite(g.ν) || return g.μ .+ Z
    s = sqrt.(g.ν ./ rand(rng, Chisq(g.ν), n))
    return g.μ .+ Z .* s'
end
logpdf_proposal(g::MvTProposal, Y::AbstractMatrix) = _logpdf_cols(g, Matrix(Y'))
rand_proposal(rng::AbstractRNG, g::MvTProposal, n::Integer) = Matrix(_rand_cols(rng, g, n)')

# The cloud's free-slot matrix, the template for the fixed slots and the free indices
function _free_part(θ::AbstractMatrix, free_para_inds)
    if free_para_inds === nothing
        return Matrix{Float64}(θ), Vector{Float64}(θ[1, :]), collect(1:size(θ, 2))
    end
    free = collect(Int, free_para_inds)
    return Matrix{Float64}(θ[:, free]), Vector{Float64}(θ[1, :]), free
end

_resolve_bounds(bounds, d::Integer) = bounds === nothing ? bounds_transform(d) :
                                      bounds isa Tuple ? bounds_transform(bounds...) :
                                      bounds_transform(bounds)

# Posterior rows usable on the chosen space: finite kernel, positive weight, finite y
function _usable_posterior(θ_free, logkernel, weights, bt, real_line)
    N = size(θ_free, 1)
    length(logkernel) == N && length(weights) == N ||
        throw(DimensionMismatch("θ has $N rows, logkernel $(length(logkernel)), weights $(length(weights))"))
    Y = real_line ? bounds_to_real(bt, θ_free) : copy(θ_free)
    lj = real_line ? bounds_logjac(bt, θ_free) : zeros(N)
    ok = [isfinite(logkernel[i]) && weights[i] > 0 && isfinite(lj[i]) && all(isfinite, view(Y, i, :)) for i in 1:N]
    nbad = count(i -> isfinite(logkernel[i]) && weights[i] > 0 && !ok[i], 1:N)
    nbad > 0 && @warn "$nbad posterior draw(s) sit on a bound (zero density on the real line) and are dropped"
    return Y, lj, ok
end

########################################################################################
### Bridge sampling
########################################################################################

"""
```
BridgeDraws
```
Step 1 of the two-step bridge sampling (see `bridge_draws`): the posterior side, the fitted
proposal and its draws. Fields: `transform`, `real_line`, `proposal::MvTProposal`,
`free_para_inds`, `template` (a full parameter vector; its free slots are replaced by each
proposal draw), the posterior cloud `θ` (free slots, model space), `logkernel`, `weights`,
`clusters`, `usable`, the `fit` and `est` row indices, and per proposal draw `y_prop` (proposal
space), `θ_prop` (model space), `logg_prop`, `logjac_prop` and `in_support`.
"""
struct BridgeDraws
    transform::BoundsTransform
    real_line::Bool
    proposal::MvTProposal
    free_para_inds::Vector{Int}
    template::Vector{Float64}
    θ::Matrix{Float64}
    logkernel::Vector{Float64}
    weights::Vector{Float64}
    clusters::Vector{Int}
    usable::BitVector
    fit::Vector{Int}
    est::Vector{Int}
    y_prop::Matrix{Float64}
    θ_prop::Matrix{Float64}
    logg_prop::Vector{Float64}
    logjac_prop::Vector{Float64}
    in_support::BitVector
end
n_proposals(bd::BridgeDraws) = size(bd.y_prop, 1)

"""
```
bridge_draws(θ, logkernel, weights; free_para_inds = nothing, bounds = nothing,
             real_line = true, ν = 7.0, M = size(θ, 1), split = :half, rng = Xoshiro(0),
             clusters = :duplicates, y_prop = nothing)
```
Step 1 of bridge sampling: fit the proposal to the cloud and draw from it. No likelihood
evaluations.

- `θ`: the posterior cloud, one draw per row: either full parameter vectors with
  `free_para_inds` the free slots (the fixed slots are taken from the first row), or only the
  free slots (`free_para_inds = nothing`). `logkernel`: log likelihood + log prior at each
  row. `weights`: the cloud's weights (any scale). Rows with a non-finite kernel or zero
  weight are not used.
- `bounds`: the free slots' bounds, as a `BoundsTransform`, `(lb, ub)` or a vector of
  tuples. `nothing` means unbounded.
- `real_line = true`: fit and draw on the real line y = T(θ) (see `BoundsTransform`), with the
  log Jacobian added to the kernel, so the proposal's support never leaves the posterior's.
  `false`: fit and draw in model space, where draws outside the bounds get kernel 0.
- `ν`: degrees of freedom of the multivariate t proposal (covariance = the weighted cloud
  covariance); `Inf` gives a normal. `M`: number of proposal draws.
- `split`: `:half` (default) fits the proposal to a random half of the clusters and uses the
  other half as the posterior side (Overstall and Forster 2010; splitting by cluster keeps
  duplicates of one particle on one side); `:all` fits and estimates on all draws (biased:
  about -0.15 at d = 93); or a tuple `(fit, est)` of row indices.
- `clusters`: groups of draws that are not independent, for the split and the standard
  errors: `:duplicates` (identical rows, the default), `:none`, or a vector of ids such as the
  particles' Eve indices.
- `y_prop`: proposal draws (M × d, on the proposal's space) to use instead of new ones, for
  example draws stored by an earlier run.

Then evaluate the log kernel at `proposal_parameters(bd)` (a likelihood evaluation each;
`evaluate_proposals` does it with any map, or use SGE arrays) and call `bridge_combine`.
"""
function bridge_draws(θ::AbstractMatrix, logkernel::AbstractVector, weights::AbstractVector;
                      free_para_inds = nothing, bounds = nothing, real_line::Bool = true,
                      ν::Real = 7.0, M::Integer = size(θ, 1), split = :half,
                      rng::AbstractRNG = Xoshiro(0), clusters = :duplicates,
                      y_prop::Union{Nothing, AbstractMatrix} = nothing)
    θf, template, free = _free_part(θ, free_para_inds)
    N, d = size(θf)
    bt = _resolve_bounds(bounds, d)
    _check_dim(bt, d)
    Y, _, usable = _usable_posterior(θf, logkernel, weights, bt, real_line)
    cl = _resolve_clusters(clusters, θf)
    fit, est = if split === :half
        ids = unique(cl[usable])
        perm = randperm(rng, length(ids))
        half = Set(ids[perm[1:length(ids) ÷ 2]])
        [i for i in 1:N if usable[i] && cl[i] in half], [i for i in 1:N if usable[i] && !(cl[i] in half)]
    elseif split === :all
        findall(usable), findall(usable)
    elseif split isa Tuple && length(split) == 2
        Base.filter(i -> usable[i], collect(Int, split[1])), Base.filter(i -> usable[i], collect(Int, split[2]))
    else
        throw(ArgumentError("split must be :half, :all or a tuple (fit, est) of row indices"))
    end
    (length(fit) > d && !isempty(est)) ||
        throw(ArgumentError("too few usable draws: $(length(fit)) to fit the proposal (d = $d), $(length(est)) on the posterior side"))
    μ, Σ = weighted_mean_cov(Y[fit, :], weights[fit])
    g = MvTProposal(μ, Σ, ν)
    yp = if y_prop === nothing
        _rand_cols(rng, g, M)
    else
        size(y_prop, 2) == d || throw(DimensionMismatch("y_prop has $(size(y_prop, 2)) columns, not $d"))
        Matrix{Float64}(y_prop')
    end
    logg = _logpdf_cols(g, yp)
    ypr = Matrix(yp')
    θp = real_line ? bounds_to_model(bt, ypr) : ypr
    ljp = real_line ? bounds_logjac_real(bt, ypr) : zeros(size(ypr, 1))
    insup = bounds_in_support(bt, θp)
    return BridgeDraws(bt, real_line, g, free, template, θf, Vector{Float64}(logkernel),
                       Vector{Float64}(weights), cl, BitVector(usable), fit, est, ypr, θp, logg, ljp, insup)
end

"""
```
proposal_parameters(bd)
```
The parameter vectors at which to evaluate the log kernel, one per row (M × length of the
template): the template with its free slots replaced by each proposal draw. Rows with
`bd.in_support[j] == false` need no evaluation (their kernel is 0).
"""
function proposal_parameters(bd::BridgeDraws)
    P = repeat(bd.template', n_proposals(bd))
    P[:, bd.free_para_inds] = bd.θ_prop
    return P
end

"""
```
evaluate_proposals(logkernel, bd; mapfun = map)
```
log kernel at every proposal draw: `logkernel(v)` for each in-support row v of
`proposal_parameters(bd)` (as a `Vector{Float64}`), -Inf for the others; NaN counts as -Inf.
`mapfun` can be `Distributed.pmap` (then `logkernel` must be defined on the workers).
"""
function evaluate_proposals(logkernel::Function, bd::BridgeDraws; mapfun = map)
    P = proposal_parameters(bd)
    rows = findall(bd.in_support)
    vals = mapfun(j -> Float64(logkernel(P[j, :])), rows)
    out = fill(-Inf, n_proposals(bd))
    for (k, j) in enumerate(rows)
        v = vals[k]
        out[j] = isnan(v) ? -Inf : v
    end
    return out
end

# Meng-Wong iterative optimal bridge in logs. l1 = log q - log g at the posterior draws (normalized
# weights W), l2 = log q - log g at the proposal draws (-Inf where q = 0; they count in n2).
function _bridge_iterate(l1::AbstractVector, W::AbstractVector, l2::AbstractVector;
                         tol::Real = 1e-10, maxiter::Integer = 10_000)
    logW = log.(W)
    n1 = 1 / sum(abs2, W); n2 = length(l2)
    ls1, ls2 = log(n1 / (n1 + n2)), log(n2 / (n1 + n2))
    fin2 = l2[isfinite.(l2)]
    logr = isempty(fin2) ? median(l1) : _logmeanexp_all(l2)
    iterations, converged = 0, false
    for k in 1:maxiter
        iterations = k
        num = logsumexp([x - _logaddexp(ls1 + x, ls2 + logr) for x in fin2]) - log(n2)
        den = logsumexp(logW .- [_logaddexp(ls1 + x, ls2 + logr) for x in l1])
        new = num - den
        converged = abs(new - logr) < tol
        logr = new
        converged && break
    end
    f1 = [exp(ls2 - _logaddexp(ls1 + x - logr, ls2)) for x in l1]
    f2 = [exp(x - logr - _logaddexp(ls1 + x - logr, ls2)) for x in l2]
    return (logr = logr, iterations = iterations, converged = converged, n1 = n1, n2 = n2, f1 = f1, f2 = f2)
end

# Variance of the weighted posterior-side mean m1 = Σ Wᵢ f1ᵢ, by cluster: the sandwich
# G/(G-1) Σ_c (Σ_{i∈c} Wᵢ (f1ᵢ - m1))²; singleton clusters give the independent-draws version
function _cluster_var(f::AbstractVector, W::AbstractVector, c::AbstractVector{<:Integer})
    m = sum(W .* f)
    S = zeros(maximum(c))
    for i in eachindex(f)
        S[c[i]] += W[i] * (f[i] - m)
    end
    G = length(unique(c))
    return G > 1 ? G / (G - 1) * sum(abs2, S) : NaN
end

"""
```
bridge_estimate(l1, w1, l2; clusters = nothing, se_method = :cluster, n_batches = 10,
                batch_rng = Xoshiro(99), lognorm = 0.0, tol = 1e-10, maxiter = 10_000)
```
The bridge-sampling estimate from the log ratios l1 = log q - log g at the posterior draws
(weights `w1`, clusters `clusters`; `nothing` = independent draws) and l2 = log q - log g at
the proposal draws (-Inf where q = 0). `bridge_combine` builds these from a `BridgeDraws`;
this method is for ratios computed elsewhere. See `bridge_combine` for the options and the
result.
"""
function bridge_estimate(l1::AbstractVector, w1::AbstractVector, l2::AbstractVector;
                         clusters::Union{Nothing, AbstractVector{<:Integer}} = nothing,
                         se_method::Symbol = :cluster, n_batches::Integer = 10,
                         batch_rng::AbstractRNG = Xoshiro(99), lognorm::Real = 0.0,
                         tol::Real = 1e-10, maxiter::Integer = 10_000)
    se_method in (:cluster, :iid, :ess) || throw(ArgumentError("se_method must be :cluster, :iid or :ess"))
    ok = [isfinite(l1[i]) && w1[i] > 0 for i in eachindex(l1)]
    l1, w1 = l1[ok], w1[ok]
    c = _relabel(clusters === nothing ? collect(1:length(ok)) : collect(Int, clusters)[ok])
    l2 = [isnan(x) ? -Inf : Float64(x) for x in l2]
    W = w1 ./ sum(w1)
    b = _bridge_iterate(l1, W, l2; tol = tol, maxiter = maxiter)
    b.converged || @warn "bridge sampling did not converge in $maxiter iterations"
    # Frühwirth-Schnatter (2004): RE² = Var_g(f2)/(n2 E_g(f2)²) + Var(Σ Wᵢ f1ᵢ)/E_p(f1)²
    m1 = sum(W .* b.f1); m2 = mean(b.f2)
    v_prop = var(b.f2) / (b.n2 * m2^2)
    v_ess = sum(W .* (b.f1 .- m1) .^ 2) / b.n1                     # i.i.d. with sample size ESS
    v_iid = _cluster_var(b.f1, W, collect(1:length(W)))            # independent weighted draws
    v_cl = _cluster_var(b.f1, W, c)                                 # clustered (duplicates / Eves)
    v_post = se_method === :cluster ? v_cl : se_method === :iid ? v_iid : v_ess
    nse(v) = sqrt(v_prop + v / m1^2)
    # Batch means: K random batches of posterior clusters paired with K batches of proposal draws
    K = n_batches
    batch = Float64[]
    if K >= 2
        G = maximum(c)
        p1 = randperm(batch_rng, G); p2 = randperm(batch_rng, length(l2))
        g1, g2 = G ÷ K, length(l2) ÷ K
        if g1 >= 1 && g2 >= 1
            members = [Int[] for _ in 1:G]
            for i in eachindex(c); push!(members[c[i]], i); end
            for k in 1:K
                rows = reduce(vcat, members[p1[(k-1)*g1+1:k*g1]])
                Wk = w1[rows] ./ sum(w1[rows])
                push!(batch, _bridge_iterate(l1[rows], Wk, l2[p2[(k-1)*g2+1:k*g2]]; tol = tol, maxiter = maxiter).logr)
            end
        end
    end
    # Importance sampling from the same proposal draws: log mean(q/g), unbiased in levels
    n = length(l2); lis = _logmeanexp_all(l2)
    r = exp.(l2 .- (lis + log(n)))
    return (log_mdd = b.logr + lognorm, nse = nse(v_post), se_method = se_method,
            nse_ess = nse(v_ess), nse_iid = nse(v_iid), nse_cluster = nse(v_cl),
            nse_batch = length(batch) >= 2 ? std(batch) / sqrt(length(batch)) : NaN,
            batch_estimates = batch .+ lognorm,
            rho_f1 = v_post / v_ess,
            iterations = b.iterations, converged = b.converged,
            ess_post = b.n1, n_post = length(l1), n_clusters_post = maximum(c),
            n_prop = n, n_prop_finite = count(isfinite, l2),
            is_log_mdd = lis + lognorm, is_nse = sqrt(n / (n - 1) * sum(abs2, r .- 1 / n)),
            is_max_share = maximum(r), lognorm = Float64(lognorm))
end

"""
```
bridge_combine(bd, logkernel_prop; lognorm = 0.0, se_method = :cluster, n_batches = 10,
               batch_rng = Xoshiro(99), tol = 1e-10, maxiter = 10_000)
```
Step 2 of bridge sampling: the Meng and Wong (1996) iterative optimal bridge estimate of
log ∫ kernel + `lognorm`, from `bd = bridge_draws(…)` and the log kernel at its proposal
draws (`logkernel_prop`, -Inf where the kernel is 0 or could not be evaluated). No likelihood
evaluations. `lognorm` is the prior's log normalizing constant (`effective_prior_lognorm`
for SMC's effective prior).

Returns a named tuple:
- `log_mdd`; `iterations`, `converged`;
- `nse`: Frühwirth-Schnatter's (2004) relative error, ≈ the SE of `log_mdd`, with the
  posterior-side variance chosen by `se_method`:
  - `:cluster` (default): the cluster-robust variance of the weighted posterior mean
    Σ Wᵢ f₁ᵢ, G/(G-1) Σ_c (Σ_{i∈c} Wᵢ (f₁ᵢ - m₁))², over the clusters of `bd` (duplicate
    particles, or Eves). This estimates the posterior-side variance, ρ_f₁(0) Var(f₁)/n₁,
    for a cloud whose particles are not independent: duplicates from resampling count once.
    Correlation beyond the clusters (ancestry shared by distinct particles) is not captured
    unless Eve indices are passed as clusters;
  - `:iid`: the same with every draw its own cluster (independent weighted draws);
  - `:ess`: Var(f₁)/n₁ with n₁ = ESS = 1/Σ Wᵢ² (the earlier formula, ρ_f₁(0) = 1);
  `nse_cluster`, `nse_iid` and `nse_ess` are all reported, and `rho_f1` = the chosen
  posterior-side variance / the `:ess` one;
- `nse_batch`, `batch_estimates`: `n_batches` random batches of posterior clusters, each
  paired with a batch of proposal draws, bridged separately; SE = sd/√K;
- `ess_post`, `n_post`, `n_clusters_post`, `n_prop`, `n_prop_finite`;
- `is_log_mdd`, `is_nse`, `is_max_share`: the importance-sampling estimate log mean(q/g)
  from the same proposal draws (unbiased in levels when g has heavier tails than the
  posterior), the delta-method SE of its log, and the largest normalized weight;
- `lognorm`.
"""
function bridge_combine(bd::BridgeDraws, logkernel_prop::AbstractVector; lognorm::Real = 0.0,
                        se_method::Symbol = :cluster, n_batches::Integer = 10,
                        batch_rng::AbstractRNG = Xoshiro(99), tol::Real = 1e-10,
                        maxiter::Integer = 10_000)
    length(logkernel_prop) == n_proposals(bd) ||
        throw(DimensionMismatch("$(length(logkernel_prop)) kernels for $(n_proposals(bd)) proposal draws"))
    est = bd.est
    θe = bd.θ[est, :]
    Ye = bd.real_line ? bounds_to_real(bd.transform, θe) : θe
    lje = bd.real_line ? bounds_logjac(bd.transform, θe) : zeros(length(est))
    l1 = bd.logkernel[est] .+ lje .- _logpdf_cols(bd.proposal, Matrix(Ye'))
    lk = [bd.in_support[j] && !isnan(logkernel_prop[j]) ? Float64(logkernel_prop[j]) : -Inf
          for j in eachindex(logkernel_prop)]
    l2 = lk .+ bd.logjac_prop .- bd.logg_prop
    l2[.!isfinite.(lk)] .= -Inf
    return bridge_estimate(l1, bd.weights[est], l2; clusters = bd.clusters[est], se_method = se_method,
                           n_batches = n_batches, batch_rng = batch_rng, lognorm = lognorm,
                           tol = tol, maxiter = maxiter)
end

"""
```
bridge_sampling(logkernel, θ, logkernel_post, weights; free_para_inds = nothing,
                bounds = nothing, real_line = true, ν = 7.0, M = size(θ, 1), split = :half,
                rng = Xoshiro(0), clusters = :duplicates, mapfun = map, lognorm = 0.0,
                se_method = :cluster, n_batches = 10, batch_rng = Xoshiro(99),
                tol = 1e-10, maxiter = 10_000)
bridge_sampling(logkernel, cloud::Cloud, free_para_inds; kwargs...)
```
Bridge-sampling estimate of log p(Y) = log ∫ kernel + `lognorm` from a weighted posterior
cloud (Meng and Wong 1996): `bridge_draws`, then `evaluate_proposals` (M likelihood
evaluations through `mapfun`, e.g. `pmap`), then `bridge_combine`; see those for the options
and the result. `logkernel(v)` returns log likelihood + log prior at a parameter vector v
(full, if `free_para_inds` is given), and -Inf where it cannot be evaluated.

Defaults: a multivariate t (ν = 7) fitted on the real line (logit / log on the bounds, plus
the log Jacobian) to a random half of the cloud, the other half the posterior side; on the
ss114 health cloud (d = 93) this matched the analytic MDD in conjugate tests to 0.01 and was
stable to ±0.25 across ν, splits and proposal halves. For production runs use the two-step
form, with the proposal draws evaluated as an SGE array.
"""
function bridge_sampling(logkernel::Function, θ::AbstractMatrix, logkernel_post::AbstractVector,
                         weights::AbstractVector; free_para_inds = nothing, bounds = nothing,
                         real_line::Bool = true, ν::Real = 7.0, M::Integer = size(θ, 1),
                         split = :half, rng::AbstractRNG = Xoshiro(0), clusters = :duplicates,
                         mapfun = map, lognorm::Real = 0.0, se_method::Symbol = :cluster,
                         n_batches::Integer = 10, batch_rng::AbstractRNG = Xoshiro(99),
                         tol::Real = 1e-10, maxiter::Integer = 10_000)
    bd = bridge_draws(θ, logkernel_post, weights; free_para_inds = free_para_inds, bounds = bounds,
                      real_line = real_line, ν = ν, M = M, split = split, rng = rng, clusters = clusters)
    lk = evaluate_proposals(logkernel, bd; mapfun = mapfun)
    return bridge_combine(bd, lk; lognorm = lognorm, se_method = se_method, n_batches = n_batches,
                          batch_rng = batch_rng, tol = tol, maxiter = maxiter)
end
function bridge_sampling(logkernel::Function, cloud::Cloud, free_para_inds::AbstractVector{<:Integer};
                         kwargs...)
    P = cloud.particles[:, 1:ind_para_end(size(cloud.particles, 2))]
    return bridge_sampling(logkernel, P, get_logpost(cloud), get_weights(cloud);
                           free_para_inds = free_para_inds, kwargs...)
end

########################################################################################
### Importance-sampling lower bound
########################################################################################

"""
```
ISBoundDraws
```
Step 1 of the IS lower bound (see `is_bound_draws`): fields `transform`, `real_line`, `τ`,
`post_mass` (the cloud's weighted share inside the τ-ellipsoid), `free_para_inds`,
`template`, and per draw `y_prop`, `θ_prop`, `logf`, `logjac_prop`, `in_support`.
"""
struct ISBoundDraws
    transform::BoundsTransform
    real_line::Bool
    τ::Float64
    post_mass::Float64
    free_para_inds::Vector{Int}
    template::Vector{Float64}
    y_prop::Matrix{Float64}
    θ_prop::Matrix{Float64}
    logf::Vector{Float64}
    logjac_prop::Vector{Float64}
    in_support::BitVector
end
n_proposals(bd::ISBoundDraws) = size(bd.y_prop, 1)
proposal_parameters(bd::ISBoundDraws) =
    (P = repeat(bd.template', n_proposals(bd)); P[:, bd.free_para_inds] = bd.θ_prop; P)
function evaluate_proposals(logkernel::Function, bd::ISBoundDraws; mapfun = map)
    P = proposal_parameters(bd); rows = findall(bd.in_support)
    vals = mapfun(j -> Float64(logkernel(P[j, :])), rows)
    out = fill(-Inf, n_proposals(bd))
    for (k, j) in enumerate(rows); out[j] = isnan(vals[k]) ? -Inf : vals[k]; end
    return out
end

"""
```
is_bound_draws(θ, weights; free_para_inds = nothing, bounds = nothing, real_line = true,
               τ = 0.9, M = 1_000, rng = Xoshiro(0), logkernel = nothing)
```
Step 1 of the importance-sampling lower bound: M draws from f_τ, the normal fitted to the
weighted cloud (on the real line by default) truncated to its τ-ellipsoid E_τ =
{q ≤ χ²_d(τ)} and renormalized. No likelihood evaluations. Pass `logkernel` (the cloud's log
kernels) only to drop rows with a non-finite kernel from the fit. Then evaluate the log
kernel at `proposal_parameters(bd)` and call `is_bound_combine`.
"""
function is_bound_draws(θ::AbstractMatrix, weights::AbstractVector; free_para_inds = nothing,
                        bounds = nothing, real_line::Bool = true, τ::Real = 0.9, M::Integer = 1_000,
                        rng::AbstractRNG = Xoshiro(0), logkernel = nothing)
    0 < τ <= 1 || throw(ArgumentError("τ must be in (0, 1]"))
    θf, template, free = _free_part(θ, free_para_inds)
    N, d = size(θf)
    bt = _resolve_bounds(bounds, d)
    _check_dim(bt, d)
    lk = logkernel === nothing ? zeros(N) : logkernel
    Y, _, usable = _usable_posterior(θf, lk, weights, bt, real_line)
    Yu, Wu = Y[usable, :], weights[usable] ./ sum(weights[usable])
    μ, Σ = weighted_mean_cov(Yu, Wu)
    C = cholesky(Symmetric(floored_cov(Σ)))
    L = C.L; logdetΣ = logdet(C)
    crit = τ == 1 ? Inf : quantile(Chisq(d), τ)
    qc = vec(sum(abs2, L \ (Yu .- μ')', dims = 1))
    post_mass = sum(Wu[qc .<= crit])
    Z = zeros(d, 0)
    while size(Z, 2) < M
        B = randn(rng, d, max(2 * (M - size(Z, 2)), 64))
        Z = hcat(Z, B[:, vec(sum(abs2, B, dims = 1)) .<= crit])
    end
    Z = Z[:, 1:M]
    q = vec(sum(abs2, Z, dims = 1))
    logf = (τ == 1 ? 0.0 : -log(τ)) .- d / 2 * log(2π) .- logdetΣ / 2 .- q ./ 2
    yp = Matrix((μ .+ L * Z)')
    θp = real_line ? bounds_to_model(bt, yp) : yp
    ljp = real_line ? bounds_logjac_real(bt, yp) : zeros(M)
    return ISBoundDraws(bt, real_line, Float64(τ), post_mass, free, template, yp, θp, logf, ljp,
                        bounds_in_support(bt, θp))
end

"""
```
is_bound_combine(bd, logkernel_prop; lognorm = 0.0, n_boot = 500, rng = Xoshiro(1))
```
Step 2 of the IS lower bound: with θⱼ ~ f_τ (see `is_bound_draws`), mean_j k(θⱼ)/f_τ(θⱼ) is
unbiased for ∫_{E_τ} k = p(Y) P_post(E_τ) ≤ p(Y) (k = 0 outside the support). So its log is a
lower bound on log p(Y) in expectation (Jensen), and by Markov's inequality
P(log bound > log p(Y) + t) ≤ exp(-t). On the real line the kernel includes the log Jacobian.
See `is_lower_bound(logratio; …)` for the result.
"""
function is_bound_combine(bd::ISBoundDraws, logkernel_prop::AbstractVector; lognorm::Real = 0.0,
                          n_boot::Integer = 500, rng::AbstractRNG = Xoshiro(1))
    length(logkernel_prop) == n_proposals(bd) ||
        throw(DimensionMismatch("$(length(logkernel_prop)) kernels for $(n_proposals(bd)) draws"))
    lr = [bd.in_support[j] && isfinite(logkernel_prop[j]) ?
          logkernel_prop[j] + bd.logjac_prop[j] - bd.logf[j] : -Inf for j in eachindex(logkernel_prop)]
    out = is_lower_bound(lr; post_mass = bd.post_mass, lognorm = lognorm, n_boot = n_boot, rng = rng)
    return merge(out, (τ = bd.τ, real_line = bd.real_line))
end

"""
```
is_lower_bound(logkernel, θ, logkernel_post, weights; free_para_inds = nothing,
               bounds = nothing, real_line = true, τ = 0.9, M = 1_000, rng = Xoshiro(0),
               mapfun = map, lognorm = 0.0, n_boot = 500, boot_rng = Xoshiro(1))
is_lower_bound(logratio; post_mass = NaN, lognorm = 0.0, n_boot = 500, rng = Xoshiro(1))
```
Importance-sampling lower bound on log p(Y) (a cheap check of an MDD estimate from above:
an estimate far below the bound is wrong). The first method runs `is_bound_draws`,
evaluates the M draws (`logkernel` as in `bridge_sampling`) and calls `is_bound_combine`.
The second takes the log ratios log k - log f_τ directly (-Inf where k = 0).

Returns `(log_bound, nse_boot, log_point, post_mass, max_share, n_zero, n_draws, lognorm)`
(plus `τ` and `real_line` from the first method): `log_bound` = log mean(k/f) + `lognorm`,
its bootstrap SE, `log_point` = `log_bound` - log `post_mass` (a point estimate of log p(Y)
when the cloud's share of E_τ is reliable), the largest term's share and the number of draws
with k = 0.
"""
function is_lower_bound(logratio::AbstractVector{<:Real}; post_mass::Real = NaN, lognorm::Real = 0.0,
                        n_boot::Integer = 500, rng::AbstractRNG = Xoshiro(1))
    lr = [isnan(x) ? -Inf : Float64(x) for x in logratio]
    n = length(lr)
    fin = Base.filter(isfinite, lr)
    lb = isempty(fin) ? -Inf : logsumexp(fin) - log(n)
    boot = [begin s = lr[rand(rng, 1:n, n)]; f = Base.filter(isfinite, s); isempty(f) ? -Inf : logsumexp(f) - log(n) end
            for _ in 1:n_boot]
    return (log_bound = lb + lognorm, nse_boot = n_boot >= 2 ? std(boot) : NaN,
            log_point = lb + lognorm - log(post_mass), post_mass = Float64(post_mass),
            max_share = isempty(fin) ? NaN : exp(maximum(fin) - logsumexp(fin)),
            n_zero = n - length(fin), n_draws = n, lognorm = Float64(lognorm))
end
function is_lower_bound(logkernel::Function, θ::AbstractMatrix, logkernel_post::AbstractVector,
                        weights::AbstractVector; free_para_inds = nothing, bounds = nothing,
                        real_line::Bool = true, τ::Real = 0.9, M::Integer = 1_000,
                        rng::AbstractRNG = Xoshiro(0), mapfun = map, lognorm::Real = 0.0,
                        n_boot::Integer = 500, boot_rng::AbstractRNG = Xoshiro(1))
    bd = is_bound_draws(θ, weights; free_para_inds = free_para_inds, bounds = bounds,
                        real_line = real_line, τ = τ, M = M, rng = rng, logkernel = logkernel_post)
    return is_bound_combine(bd, evaluate_proposals(logkernel, bd; mapfun = mapfun);
                            lognorm = lognorm, n_boot = n_boot, rng = boot_rng)
end

########################################################################################
### Normalizing constant of SMC's effective prior
########################################################################################

function _log_interval_mass(F_lo::Real, S_hi::Real, S_lo::Real, F_hi::Real)
    out = F_lo + S_hi
    out <= 0.5 && return log1p(-out)            # most of the mass inside: no cancellation
    m = S_lo <= F_hi ? S_lo - S_hi : F_hi - F_lo  # both bounds in one tail: difference there
    return m > 0 ? log(m) : -Inf
end

"""
```
prior_inbounds_logmass(prior, lb, ub; mc_draws = 100_000, rng = Xoshiro(0))
```
log P(lb < θ < ub) under a univariate prior, from its CDF. `RootInverseGamma(ν, τ)` uses
the `InverseGamma(ν/2, ντ²/2)` CDF of θ² (ModelConstructors' own `cdf` method for it is not
a CDF). Without a CDF method it falls back to `mc_draws` Monte Carlo draws, with a warning.
Returns `(logmass, method, se)`; `nothing` as the prior gives 0.
"""
function prior_inbounds_logmass(pr, lb::Real, ub::Real; mc_draws::Integer = 100_000,
                                rng::AbstractRNG = Xoshiro(0))
    pr === nothing && return (logmass = 0.0, method = :none, se = 0.0)
    if pr isa RootInverseGamma
        ig = InverseGamma(pr.ν / 2, pr.ν * pr.τ^2 / 2)
        lo, hi = max(lb, 0.0)^2, ub < 0 ? 0.0 : ub^2
        lm = _log_interval_mass(cdf(ig, lo), ccdf(ig, hi), ccdf(ig, lo), cdf(ig, hi))
        return (logmass = lm, method = :analytic, se = 0.0)
    end
    try
        lm = _log_interval_mass(cdf(pr, lb), ccdf(pr, ub), ccdf(pr, lb), cdf(pr, ub))
        return (logmass = lm, method = :analytic, se = 0.0)
    catch err
        err isa MethodError || rethrow(err)
        @warn "no CDF for $(typeof(pr)): Monte Carlo in-bounds mass from $mc_draws draws"
        k = count(_ -> lb < rand(rng, pr) < ub, 1:mc_draws)
        p = k / mc_draws
        return (logmass = log(p), method = :montecarlo, se = sqrt((1 - p) / (p * mc_draws)))
    end
end

_rand_prior(rng::AbstractRNG, pr::RootInverseGamma) = sqrt(rand(rng, InverseGamma(pr.ν / 2, pr.ν * pr.τ^2 / 2)))
_rand_prior(rng::AbstractRNG, pr) = rand(rng, pr)
function _rand_in_bounds(rng::AbstractRNG, pr, lb, ub)
    x = _rand_prior(rng, pr)
    while !(lb < x < ub); x = _rand_prior(rng, pr); end
    return x
end

function _p_valid_result(n_valid::Integer, n_draws::Integer)
    p = n_valid / n_draws
    return p, (n_valid > 0 ? sqrt((1 - p) / (p * n_draws)) : NaN)
end

"""
```
effective_prior_lognorm(priors, lb, ub; p_valid = nothing, loglikelihood = nothing,
                        n_draws = 1_000, rng = Xoshiro(0), mapfun = map, mc_draws = 100_000)
effective_prior_lognorm(parameters::ParameterVector; regime_switching = false, toggle = true,
                        p_valid = nothing, loglikelihood = nothing, data = nothing,
                        n_draws = 1_000, seed = 0)
```
The log normalizing constant of SMC's effective prior,

    lognorm = -Σⱼ log Pⱼ(lbⱼ < θⱼ < ubⱼ) - log P(valid | in bounds),

to add to an MDD estimate computed with the stored log prior (`logpdf`, not renormalized).
SMC's initial cloud draws each free slot from its prior until it is inside its value bounds
(`rand(::ParameterVector)`), and redraws the whole vector until the likelihood is finite
(`one_draw`), so the SMC log MDD is relative to the prior truncated to that set.

- The in-bounds masses use the priors' CDFs (`prior_inbounds_logmass`; RootInverseGamma via
  the InverseGamma CDF of θ²).
- P(valid): pass `p_valid` (a probability or a tuple `(n_valid, n_draws)`, e.g. summed over
  SGE tasks), or `loglikelihood` to estimate it from `n_draws` prior draws inside the bounds:
  the share with a finite log likelihood (errors count as invalid in the `ParameterVector`
  method, as in `one_draw`). Generic method: `loglikelihood(θ)` at the free-slot vector θ,
  through `mapfun`. `ParameterVector` method: `loglikelihood(parameters, data)` as in `smc`,
  draws from `rand(parameters; regime_switching)` with the task-local RNG seeded by `seed`
  (and restored afterwards). Neither given: P(valid) = 1, with a warning.

Returns `(lognorm, lognorm_se, sum_log_mass, log_mass, methods, p_valid, p_valid_se,
n_valid, n_draws)`, with `lognorm_se` the delta-method SE of log P(valid) (plus any Monte
Carlo mass terms).
"""
function effective_prior_lognorm(priors::AbstractVector, lb::AbstractVector{<:Real}, ub::AbstractVector{<:Real};
                                 p_valid = nothing, loglikelihood = nothing, n_draws::Integer = 1_000,
                                 rng::AbstractRNG = Xoshiro(0), mapfun = map, mc_draws::Integer = 100_000)
    length(priors) == length(lb) == length(ub) || throw(DimensionMismatch("priors and bounds differ in length"))
    masses = [prior_inbounds_logmass(priors[j], lb[j], ub[j]; mc_draws = mc_draws, rng = rng) for j in eachindex(priors)]
    pv, nv, nd = if p_valid !== nothing
        p_valid isa Tuple ? (p_valid[1] / p_valid[2], p_valid[1], p_valid[2]) : (Float64(p_valid), missing, missing)
    elseif loglikelihood !== nothing
        draws = [[_rand_in_bounds(rng, priors[j], lb[j], ub[j]) for j in eachindex(priors)] for _ in 1:n_draws]
        ll = mapfun(θ -> Float64(loglikelihood(θ)), draws)
        k = count(isfinite, ll)
        (k / n_draws, k, n_draws)
    else
        @warn "effective_prior_lognorm: neither p_valid nor loglikelihood given, so P(valid) = 1"
        (1.0, missing, missing)
    end
    return _lognorm_result(masses, pv, nv, nd)
end

function _lognorm_result(masses, pv, nv, nd)
    pv_se = (nv isa Integer && nd isa Integer) ? _p_valid_result(nv, nd)[2] : 0.0
    s = sum(m.logmass for m in masses; init = 0.0)
    se_mass2 = sum(m.se^2 for m in masses; init = 0.0)
    return (lognorm = -s - log(pv), lognorm_se = sqrt(pv_se^2 + se_mass2), sum_log_mass = s,
            log_mass = [m.logmass for m in masses], methods = [m.method for m in masses],
            p_valid = pv, p_valid_se = pv * pv_se, n_valid = nv, n_draws = nd)
end

function effective_prior_lognorm(parameters::ParameterVector; regime_switching::Bool = false,
                                 toggle::Bool = true, p_valid = nothing, loglikelihood = nothing,
                                 data = nothing, n_draws::Integer = 1_000, seed::Integer = 0,
                                 mc_draws::Integer = 100_000)
    slots = Base.filter(s -> !s.fixed, parameter_slots(parameters; regime_switching = regime_switching, toggle = toggle))
    masses = [prior_inbounds_logmass(s.prior, s.bounds...; mc_draws = mc_draws) for s in slots]
    pv, nv, nd = if p_valid !== nothing
        p_valid isa Tuple ? (p_valid[1] / p_valid[2], p_valid[1], p_valid[2]) : (Float64(p_valid), missing, missing)
    elseif loglikelihood !== nothing
        k = preserving_default_rng() do
            Random.seed!(seed)
            count(1:n_draws) do _
                draw = vec(rand(parameters, 1, regime_switching = regime_switching, toggle = toggle))
                try
                    update!(parameters, draw)
                    ll = loglikelihood(parameters, data)
                    toggle && toggle_regime!(parameters, 1)
                    isfinite(ll)
                catch err
                    (isa(err, ParamBoundsError) || isa(err, SingularException) ||
                     isa(err, LinearAlgebra.LAPACKException) || isa(err, PosDefException) ||
                     isa(err, DomainError)) || rethrow(err)
                    false
                end
            end
        end
        (k / n_draws, k, n_draws)
    else
        @warn "effective_prior_lognorm: neither p_valid nor loglikelihood given, so P(valid) = 1"
        (1.0, missing, missing)
    end
    return _lognorm_result(masses, pv, nv, nd)
end
