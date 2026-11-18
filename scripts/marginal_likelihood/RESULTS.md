# Marginal likelihood benchmark results

The linear test model has an independently integrated log marginal likelihood
of **−504.603704**. The primary accuracy metric is absolute log error,
`abs(log_mdd_estimate - truth)`, with the median across six seeds (11–16) to
reduce the effect of an occasional catastrophic run. RMSE retains sensitivity
to those failures. Proposal counts measure the extra likelihood work.

All runs used 1,000 particles, adaptive tempering with target 0.95, proposal
mixture `α = 0.9`, and the same seed for each configuration. The blocked runs
put each of the three independent regressions in its own fixed block and use
`blockwise_c = true`. Adaptive MH started at two steps, targeted 90% distinct
particles and maximum correlation 0.9, and was capped at ten steps.

| Method | Median absolute log error | RMSE | Mean proposals | Seeds improved vs fixed 2 |
| --- | ---: | ---: | ---: | ---: |
| Fixed 2 MH steps | 90.234 | 306.970 | 0.577 million | — |
| Adaptive 2–10 steps | 37.459 | 36.634 | 1.635 million | 6/6 |
| Fixed 8 steps | 35.317 | 37.523 | 1.796 million | 6/6 |
| Three blocks, fixed 2 steps | 32.736 | 31.352 | 1.271 million | 6/6 |
| Three blocks, adaptive 2–10 steps | **21.122** | **22.027** | 1.877 million | 6/6 |

The blocked adaptive configuration also improved all six seeds relative to
fixed eight-step MH at a similar mean proposal count. It is materially better
on this model, but its six errors remain between 16 and 28 log units: these
estimates are still unsuitable as accurate marginal likelihoods. All the
errors were negative. In seed 14, fixed two-step MH was 584.8 log units low;
the adaptive and blocked adaptive versions were 29.6 and 20.9 low.

Additional single-seed probes at seed 11 showed diminishing returns from more
steps and no improvement from more particles in that run:

| Configuration | Absolute log error | Proposals |
| --- | ---: | ---: |
| Three blocks, adaptive 2–10, 1,000 particles | 16.386 | 2.112 million |
| Three blocks, adaptive 2–30, 1,000 particles | 14.230 | 2.745 million |
| Three blocks, fixed 10, 1,000 particles | 12.770 | 4.770 million |
| Three blocks, adaptive 2–10, 5,000 particles | 20.134 | 9.780 million |

These probes are only one seed and cannot establish the effect of particle
count across runs. The six-seed comparison demonstrates improvement, not a
complete fix. It does not validate larger DSGE models or distinguish poor
mixing from a separate weighting or normalization error there.

The cluster jobs were 346513 (standard), 346516 (blocked), 346517 (high-step
probe), 346518 (larger-particle probe), and 346515 (focused diagnostics tests).
All completed with exit status 0. Reproduce the table with
`python scripts/marginal_likelihood/summarize.py`; per-run CSV and JLD2 outputs
are in the ignored `results/` directory.
