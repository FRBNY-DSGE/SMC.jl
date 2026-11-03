# Likelihoods for param_blocks_coverage.jl, included on the main process and on every worker.

# Gaussian toy: x1, x2 strongly correlated in the likelihood, x3 independent. Prior N(0, 2²) each.
# loglh(x) = -(x - μ)' Q (x - μ) / 2, so the posterior is N(P \ (Q μ), inv(P)) with P = Q + I / 4.
const GAUSS_Q  = [ 4.0 -3.6 0.0;
                  -3.6  4.0 0.0;
                   0.0  0.0 1.0]
const GAUSS_μ  = [0.5, -0.4, 1.0]
const GAUSS_μ0 = [-0.3, 0.6, 0.2]   # the old data's likelihood centre (bridge test)
gauss_loglh(x, μ) = -0.5 * dot(x .- μ, GAUSS_Q * (x .- μ))
blocks_gauss_likelihood(p, data) = gauss_loglh([p[k].value for k in 1:3], GAUSS_μ)
blocks_gauss_old_likelihood(p, data; new_model_params = false) =
    gauss_loglh([p[k].value for k in 1:3], GAUSS_μ0)

# Regime-switching toy: x1, x2 plain; a has two regimes, each with its own likelihood centre.
blocks_regime_likelihood(p, data) =
    -0.5 * ((p[1].value - 0.3)^2 + (p[2].value + 0.2)^2) -
     5.0 * ((regime_val(p[3], 1) - 0.2)^2 + (regime_val(p[3], 2) + 0.3)^2)
