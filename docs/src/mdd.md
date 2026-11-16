## Marginal Data Density

```@meta
CurrentModule = SMC
```

`smc` saves several log marginal data density (MDD) estimates in `log_mdd_estimates`. They
can disagree by tens of log points in large models, and only some of them can be trusted
there. Which one to use:

| Estimator | Cost | Use it for |
|:----------|:-----|:-----------|
| `bridge_sampling` (real line, t proposal) | one likelihood per proposal draw (M ≈ N) | **the MDD to report and compare models with** |
| `is_lower_bound` | ~1,000 likelihoods | a check: an estimate well below the bound is wrong |
| `modified_harmonic_mean` (`real_line = true`) | none | a cheap cross-check on the final cloud |
| SMC incremental weights (`smc_log_mdd`) | none | only once it agrees with the bridge |

Why:

- **SMC's sum of incremental weights is biased low when the mutation lags the target.** If
  the MH mutation mixes slowly, each stage's cloud sits in lower-likelihood regions than the
  tempered posterior, every increment is too small, and the shortfall adds up over the stages.
  The bookkeeping can be exactly right and the estimate still 50 log points too low (ss114,
  d = 93). Nothing in the run itself flags it.
- **Bridge sampling** (Meng and Wong 1996) uses only the final cloud and fresh draws from a
  proposal fitted to it, so it does not inherit the path's error. The defaults (parameters
  mapped to the real line by `bounds_transform`, a multivariate t with ν = 7 fitted to a
  random half of the cloud, the other half the posterior side) matched analytic MDDs to
  within 0.01 in conjugate tests at d = 93 and were stable to ±0.25 across ν, splits and
  proposal halves on the ss114 cloud. Fitting and estimating on the same particles
  (`split = :all`) is biased by about -0.15 there. Its standard error is Frühwirth-Schnatter's
  (2004), with the posterior-side variance cluster-robust over duplicate particles (or Eve
  indices), plus a batch-means SE.
- **The modified harmonic mean** (Geweke 1999) needs no likelihood evaluations, but in
  model space its normal f puts mass outside the parameter bounds when the posterior sits near
  a bound, which biases it **up** (by -log F_τ; `support_correction = true` estimates it). On the
  real line that bias disappears, but with a non-Gaussian posterior one particle can still
  carry much of the harmonic sum (`max_share`), and neither SE sees regions the cloud never
  visited.
- **Prior normalization.** All estimators use the stored log prior, which is not renormalized
  to the value bounds or to the region with a finite likelihood, whereas SMC's initial draws
  are. Add `effective_prior_lognorm(parameters; …)` (`lognorm` keyword) to put an estimate on
  SMC's footing; it is usually a fraction of a log point.

The estimation-AI note "MDD-TIEBREAK" has the ss114 case and the conjugate tests behind these
recommendations.

### Production use (likelihoods on SGE arrays)

```julia
bd = bridge_draws(θ, logpost, weights; free_para_inds = free, bounds = bounds_transform(m.parameters;
                  regime_switching = true), M = 15_000, rng = Xoshiro(1))
P  = proposal_parameters(bd)            # evaluate log likelihood + log prior at the rows of P
                                        # with bd.in_support (others: -Inf), e.g. as an SGE array
res = bridge_combine(bd, logkernel; lognorm = effective_prior_lognorm(m.parameters;
                     regime_switching = true, p_valid = (n_valid, n_draws)).lognorm)
res.log_mdd, res.nse, res.nse_batch, res.is_log_mdd
```

### Functions

```@docs
bridge_sampling
bridge_draws
bridge_combine
bridge_estimate
is_lower_bound
is_bound_draws
is_bound_combine
modified_harmonic_mean
effective_prior_lognorm
prior_inbounds_logmass
BoundsTransform
parameter_slots
duplicate_clusters
smc_log_mdd
```
