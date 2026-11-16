# An–Schorfheide SMC scale comparison

These scripts run the same full-data An–Schorfheide estimation with the same seed and
mutation settings. `run_bounded_sigmoid.jl` uses the existing acceptance-rate update;
`run_esjd.jl` chooses the scale by pilot ESJD. Both enable the full-cloud ESJD diagnostic.
The scripts call `SMC.smc` directly, so they use the development SMC code without needing
the DSGE `smc2` settings wrapper. Each run writes into its own result directory.

From this directory on `ranph1xn6`, submit the two independent jobs with the
documented HPC wrapper:

```sh
julia1126-batch 8 1 run_bounded_sigmoid.jl
julia1126-batch 8 1 run_esjd.jl
```

These request 8 GB plus the wrapper's 1 GB overhead and one CPU per job.
Inspect the returned job IDs with `qstat -j <jobid>`. Standard output and errors
are written beside the scripts as `run_*.o<jobid>` and `run_*.e<jobid>`.

Defaults: 300 particles, one SMC group, seed 1234, full reference data, three MH
steps, and an ESJD diagnostic every 10 stages with five candidate scales. The ESJD
adapter tests three relative scales on up to 64 pilot particles per stage. The
reference data come from DSGE's `test/estimate/smc/reference/smc.h5`.

Set these environment variables on **both** runs to change the comparison:

| Variable | Default | Meaning |
| --- | --- | --- |
| `SMC_COMPARE_N` | `300` | Particles in the single SMC group |
| `SMC_COMPARE_SEED` | `1234` | Random seed |
| `SMC_COMPARE_DIAGNOSTIC_EVERY` | `10` | Check every this many stages |
| `SMC_COMPARE_DIAGNOSTIC_SCALES` | `5` | Scales per diagnostic grid |
| `SMC_COMPARE_PILOT_PARTICLES` | `64` | Pilot particles for ESJD adaptation |
| `SMC_COMPARE_MAX_OBS` | `0` | Use first N observations; `0` means all |
| `SMC_COMPARE_DATA_PATH` | DSGE reference file | HDF5 input with a `data` dataset |
| `SMC_COMPARE_OUTDIR` | `results` beside scripts | Parent directory for both methods |

The output includes `*_stage_health.csv`, `*_esjd_diagnostic.csv`, stage snapshots,
and a final JLD2 cloud. After both jobs finish, create the two-panel PDF with:

```sh
julia1126-batch 8 1 plot_esjd_comparison.jl
```

It writes `esjd_scaler_comparison.pdf` beside the scripts. The top panel plots
relative covariance-normalized ESJD efficiency against tempering parameter,
with dashed 100% and 90% references. The bottom panel plots each method's
current and best tested scale. The diagnostic uses the same proposal covariance
and random streams for all candidate scales at a given stage. Its variance
normalization is fixed within each stage, although the reference covariance
changes as SMC progresses. Each curve's optimum is within its own tested grid.
The existing report samples every tenth stage; set
`SMC_COMPARE_DIAGNOSTIC_EVERY=1` on both estimation jobs to plot every stage.

When a results directory contains multiple runs, set `SMC_PLOT_SIGMOID_CSV`,
`SMC_PLOT_ESJD_CSV`, and optionally `SMC_PLOT_OUTPUT` to choose files. The script
also accepts three positional arguments when run directly.

`SMC_COMPARE_MAX_OBS` is for a quick check; use `0` for the full comparison.
