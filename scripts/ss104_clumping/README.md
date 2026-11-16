# SS104 clumping stress comparison

`run_acceptance.jl` and `run_esjd.jl` compare the bounded sigmoid scale update
with the direct ESJD pilot. They load the 2026Q3 SS104 stage-300 scratch
checkpoint (15,000 particles), select a common weighted sample of valid draws,
and duplicate them to make a 35%-distinct, equally weighted cloud. The number
of distinct draws is required to exceed the number of free parameters.
The local `env/` pins DSGE to the checkpoint's recorded commit
`e0341e919c1832dd57f10b70aecb5544bac9c9fd`, with its matching
StateSpaceRoutines and ModelConstructors sources, while loading SMC from this
checkout. Both arms verify a stored particle's likelihood against the
reconstructed model before mutation.

Each arm starts from that identical cloud, the same stored proposal scale, and
the same random seed. It runs three mutation sweeps at the checkpoint's fixed
tempering parameter using the SS104 likelihood and SMC's actual mutation kernel
(three MH steps, 0.975 mixture weight). Before each sweep, covariance is
computed from the current cloud. ESJD uses 32 pilot particles and trial scale
multipliers 0.5, 1, and 2. The independent diagnostic tests five scales on 32
common particles. It does not change either arm's trajectory.

The CSV outputs report acceptance, failed proposals, distinct particles,
covariance eigenvalues, spread relative to the source cloud, chosen scale, and
relative covariance-normalized ESJD efficiency. Per-sweep clouds are saved as
JLD2. `plot_results.jl` writes a comparison PDF.

Run on the cluster from this directory using the documented wrapper:

```sh
julia1126-batch 16 1 run_acceptance.jl
julia1126-batch 16 1 run_esjd.jl
julia1126-batch 8 1 plot_results.jl
```

These fixed-target sweeps isolate mutation recovery. They do not advance SMC
tempering, so they are not a full posterior estimation or evidence comparison.
