# Toy targets for drift.jl, included on the main process and on every worker.
#
# A d-dimensional Gaussian "ridge" likelihood: loglh(x) = -(x - μ)' Q (x - μ) / 2 with
# Q = I - (1 - λ) v v', v = (1, -1, 1, -1, …)/√d. So the likelihood has precision 1 in every
# direction orthogonal to v and precision λ ≪ 1 along v (the ridge), and its centre μ lies far
# out along v. With the prior N(0, σ_p² I), the posterior is N(P \ (Q μ), inv(P)), P = Q + I/σ_p²,
# and as ϕ goes from 0 to 1 the tempered target's mean moves out along v. Q is built from
# elementary operations only (no LAPACK), so the fixture runs are the same on any machine.
using LinearAlgebra

drift_v(d) = [isodd(j) ? 1.0 : -1.0 for j in 1:d] ./ sqrt(d)
drift_Q(d, λ) = Matrix(1.0I, d, d) .- (1 - λ) .* (drift_v(d) * drift_v(d)')
drift_μ(d, along) = along .* drift_v(d) .+ [0.1 * ((j % 5) - 2) for j in 1:d]

const DRIFT_D     = 10
const DRIFT_SDP   = 100.0                        # prior sd
const DRIFT_Q     = drift_Q(DRIFT_D, 1e-3)
const DRIFT_MU    = drift_μ(DRIFT_D, 600.0)
drift_x(p, d) = [p[k].value for k in 1:d]
drift_ridge_loglh(p, data) = (x = drift_x(p, DRIFT_D) .- DRIFT_MU; -0.5 * dot(x, DRIFT_Q * x))

# Small, well-conditioned Gaussian for the well-mixed run and the fixture runs
const DRIFT_Q3  = [2.0 0.6 0.0; 0.6 1.0 0.3; 0.0 0.3 1.5]
const DRIFT_MU3 = [0.4, -0.7, 1.1]
drift_gauss3_loglh(p, data) = (x = drift_x(p, 3) .- DRIFT_MU3; -0.5 * dot(x, DRIFT_Q3 * x))

# Analytic posterior and log MDD for prior N(0, s² I) and loglh -(x - μ)'Q(x - μ)/2 (no constant)
function drift_gauss_truth(Q, μ, s)
    P = Q + I / s^2
    mean = P \ (Q * μ)
    Σ = inv(P); Σ = (Σ + Σ') / 2
    logZ = -0.5 * logdet(I + s^2 * Q) - 0.5 * dot(μ, (inv(Q) + s^2 * I) \ μ)
    return mean, Σ, logZ
end
