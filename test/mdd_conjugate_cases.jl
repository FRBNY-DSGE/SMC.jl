# Conjugate test cases with an analytic log MDD, for test/mdd_estimators.jl (included inside its
# module, so nothing here leaks into the other test files). Each case returns
# (truth, logk, draw, lb, ub, d): logk(θ) = log likelihood + log prior at a parameter vector,
# draw(rng, n) = n exact posterior draws (rows), lb/ub the parameter bounds.
using Distributions, LinearAlgebra, Random

# Gaussian mean with known covariance: scales 0.01 … 3, correlated likelihood, normal prior
function gauss_case(; d::Int = 12, seed::Int = 11)
    rng = Xoshiro(seed)
    sc = exp.(range(log(0.01), log(3.0), length = d))
    A = randn(rng, d, d) / sqrt(d); C = A * A' + 0.3I
    D = Diagonal(1 ./ sqrt.(diag(C))); Corr = D * C * D
    S = Matrix(Symmetric(Diagonal(sc) * Corr * Diagonal(sc)))
    V0 = Matrix(Diagonal((5 .* sc) .^ 2))
    y = rand(rng, MvNormal(zeros(d), Symmetric(S + V0)))
    Vn = Symmetric(inv(inv(S) + inv(V0))); μn = Vn * (S \ y)
    post = MvNormal(μn, Matrix(Vn))
    lik, pri = MvNormal(zeros(d), Symmetric(S)), MvNormal(zeros(d), Symmetric(V0))
    truth = logpdf(MvNormal(zeros(d), Symmetric(S + V0)), y)
    logk(θ) = logpdf(lik, y .- θ) + logpdf(pri, θ)
    draw(rng, n) = Matrix(rand(rng, post, n)')
    return (truth = truth, logk = logk, draw = draw, lb = fill(-Inf, d), ub = fill(Inf, d), d = d)
end

# Poisson-gamma rates: n_good well identified (200 obs) and the rest weakly identified (1 obs,
# posterior shape 2-4, mode near the lower bound 0, as the COVID σ's). Bounds (0, ∞).
function gamma_case(; d::Int = 15, n_good::Int = 5, seed::Int = 13)
    rng = Xoshiro(seed)
    α0 = fill(2.0, d); β0 = vcat(fill(1.0, n_good), fill(2.0, d - n_good))
    nobs = vcat(fill(200, n_good), fill(1, d - n_good))
    λ0 = vcat(0.5 .+ 2.5 .* rand(rng, n_good), fill(0.3, d - n_good))
    Ssum = zeros(d); Cfac = zeros(d)
    for j in 1:d
        yj = rand(rng, Poisson(λ0[j]), nobs[j]); Ssum[j] = sum(yj)
        Cfac[j] = sum(-1 .- logpdf.(Poisson(1.0), yj))          # Σ log yᵢ!
    end
    postd = [Gamma(α0[j] + Ssum[j], 1 / (β0[j] + nobs[j])) for j in 1:d]
    prid = [Gamma(α0[j], 1 / β0[j]) for j in 1:d]
    lk1(j, λ) = Ssum[j] * log(λ) - nobs[j] * λ - Cfac[j] + logpdf(prid[j], λ)
    truth = sum(lk1(j, mean(postd[j])) - logpdf(postd[j], mean(postd[j])) for j in 1:d)
    logk(θ) = any(x -> !(x > 0), θ) ? -Inf : sum(lk1(j, θ[j]) for j in 1:d)
    draw(rng, n) = reduce(hcat, [rand(rng, postd[j], n) for j in 1:d])
    return (truth = truth, logk = logk, draw = draw, lb = zeros(d), ub = fill(Inf, d), d = d)
end

# Beta-binomial: probabilities on (0, 1), few trials, several with 0 or n successes (posterior
# piled up against a bound). Bounds (0, 1), so the real-line map is a logit.
function beta_case(; d::Int = 10, seed::Int = 17)
    rng = Xoshiro(seed)
    a, b = 1.5, 1.5
    n = rand(rng, 2:6, d)
    k = [j % 3 == 0 ? 0 : j % 3 == 1 ? n[j] : rand(rng, 0:n[j]) for j in 1:d]
    postd = [Beta(a + k[j], b + n[j] - k[j]) for j in 1:d]
    truth = sum(logpdf(BetaBinomial(n[j], a, b), k[j]) for j in 1:d)
    logk(θ) = any(x -> !(0 < x < 1), θ) ? -Inf :
              sum(logpdf(Binomial(n[j], θ[j]), k[j]) + logpdf(Beta(a, b), θ[j]) for j in 1:d)
    draw(rng, n_) = reduce(hcat, [rand(rng, postd[j], n_) for j in 1:d])
    return (truth = truth, logk = logk, draw = draw, lb = zeros(d), ub = ones(d), d = d)
end

# An SMC-like cloud: N exact draws resampled multinomially (duplicates), equal weights
function resampled(rng, Θ::AbstractMatrix)
    idx = rand(rng, 1:size(Θ, 1), size(Θ, 1))
    return Θ[idx, :]
end

# A fixed, RNG-free input for the backward-compatibility check of modified_harmonic_mean:
# correlated non-Gaussian rows, one duplicate row, one -Inf kernel, one zero weight
function mhm_compat_input()
    N, d = 600, 5
    θ = [sin(0.37 * i * j) + 0.1 * j * cos(1.3 * i) for i in 1:N, j in 1:d]
    θ[20, :] = θ[19, :]
    logpost = [-0.5 * sum(abs2, θ[i, :]) + 0.3 * sin(2.1 * i) - 12.0 for i in 1:N]
    logpost[7] = -Inf
    w = [1.0 + 0.5 * sin(0.7 * i) for i in 1:N]
    w[11] = 0.0
    return θ, logpost, w
end
