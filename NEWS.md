# SMC.jl 0.1.15 Release Notes
New Features:
- Option to use different likelihood functions with tempered updates
- Option to make near zero diagonals for proposal densities exactly zero to avoid errors

Bug Fixes:
- Make tempered updates ignore particles with `-Inf` log-likelihoods
- Ensure setting `tempered_update_prior_weight = 1` still works

# SMC.jl 0.1.14 Release Notes
Miscellaneous:
- Implement `add_parameters_to_cloud` to help construct
  a bridge distribution for a new model by using an estimation of a previous
  model, which uses only a subset of the new model's parameters (under some mild assumptions).
- Resolved `logpost` and `logprior` confusion in the source code

# SMC.jl 0.1.13 Release Notes
Miscelleanous:
- Moved `get_fixed_para_inds` and `get_free_para_inds` to ModelConstructors.jl
- Update syntax for HDF5 deprecations

# SMC.jl 0.1.12 Release Notes
New Features:
- Better print statements during estimation
- Parallel multinomial sampling

Bug Fixes:
- Estimation of regime switching parameters works correctly

# SMC.jl 0.1.11 Release Notes
New Features:
- Estimate regime switching parameters

# SMC.jl 0.1.10 Release Notes
New Features:
- Bug fixes from last release (save rejoined cloud ater rejoining)"

# SMC.jl 0.1.9 Release Notes
New Features:
- New functionality to split and join (after splitting) Clouds (allows users to split a large Cloud into smaller Clouds under the Github size limit, to be rejoined by user).

# SMC.jl 0.1.8 Release Notes
New Features:
- Bug fixes and cleanup:

# SMC.jl 0.1.7 Release Notes
New Features:
- Permit bridging between clouds with differing numbers of particles

Bug fixes and cleanup:
- Catching domain errors

# SMC.jl 0.1.6 Release Notes
Bug fixes and cleanup:
- Due to erroneous tagging situation, domain error catching included in this tag.

# SMC.jl 0.1.5 Release Notes
Bug fixes and cleanup; most notably, catching domain errors in mutation.

# SMC.jl 0.1.4 Release Notes
Bug fixes and cleanup, including:
- Turn off precompilation (causes issues on some heterogeneous clusters)
- Fix capitalization bug on Mac and Windows
- Fully unit-tested.

# SMC.jl 0.1.3 Release Notes
Bug fixes and cleanup

# SMC.jl 0.1.2 Release Notes
Bug fixes and cleanup

# SMC.jl 0.1.1 Release Notes
Bug fixes and cleanup
