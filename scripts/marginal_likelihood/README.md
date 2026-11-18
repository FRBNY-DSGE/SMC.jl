# Marginal likelihood benchmark

`benchmark.jl` measures absolute error and root mean squared error of SMC's log
marginal likelihood against an independently integrated value for the repository's
linear test model. It uses the same six seeds for fixed two-step MH, adaptive MH
(two to ten steps), and fixed eight-step MH. Proposal counts and elapsed seconds
show the computational cost. The exact value is computed by analytically
integrating each regression's coefficients and numerically integrating its one
remaining scale parameter, as in `test/diagnostics.jl`.

Run on the cluster from the repository root:

```sh
julia1126-batch 16 1 scripts/marginal_likelihood/benchmark.jl
```

The script writes one row per completed run to `results/runs.csv` and a final
`results/summary.csv`. The JLD2 outputs retain each stage's health record. This
small model measures whether the policy can improve a known answer; the result
does not establish accuracy for a larger DSGE model.

`benchmark_blocks.jl` runs a second comparison with each independent regression
in its own fixed mutation block and a separately adapted proposal scale. It
writes to `results/blocks`:

```sh
julia1126-batch 16 1 scripts/marginal_likelihood/benchmark_blocks.jl
```

`probe_high_steps.jl` checks one seed with a 30-step adaptive cap and ten fixed
steps in three blocks, to see whether more mutation continues to reduce error.
`probe_more_particles.jl` compares the effect of 5,000 particles with blocked
adaptive mutation on that seed.

After both six-seed jobs finish, run `python scripts/marginal_likelihood/summarize.py`
for a combined table with median absolute error, root mean squared log error,
mean proposal count, and the number of seeds improved relative to fixed two-step MH.
