# The test model has three independent linear regressions. Integrating out each
# intercept and slope gives this Gaussian; a one-dimensional grid integrates
# its remaining uniform sigma prior. The reference does not use SMC or MH.
function exact_log_mdd(data; prior_sd = 1e3, sigma_max = 1e3)
    total = 0.0
    T = size(data, 2)
    grid = range(0.3, 4.0, length = 8_001)
    for i in 1:3
        Z = hcat(ones(T), X[i, 1:T])
        logp(sigma) = logpdf(MvNormal(zeros(T),
                                     Symmetric(sigma^2 * I + prior_sd^2 * Z * Z')),
                              data[i, :])
        total += SMC.logsumexp(logp.(grid)) + log(step(grid)) - log(sigma_max)
    end
    return total
end
