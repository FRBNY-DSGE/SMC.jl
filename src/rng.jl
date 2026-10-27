# Reproducible random streams for SMC
#
# Every random operation in an SMC run draws from its own stream, whose seed depends only on
# the run seed and on the coordinates of the operation, never on the worker or process that
# runs it. That is what makes a run with a given seed bitwise identical serially and with any
# number of workers, and lets a run resume from a stage checkpoint.
#
#     stream_seed(run_seed, operation, stage, particle)
#
# Version 1: FNV-1a over the operation name, mixed with the run seed and the integer
# coordinates by SplitMix64. No process hash or worker ID is used. The streams are
#
#     operation                 stage   particle    used for
#     :initial_particle         0       i           prior draw of particle i (initial_draw!)
#     :bridge_prior_particle    0       row         prior draws mixed into a bridge (μ > 0)
#     :bridge_resample_1        0       0           resampling the old cloud for a bridge
#     :bridge_resample_2        0       0           resampling away -Inf likelihoods in a bridge
#     :prior_covariance         0       0           prior draws for the prior covariance
#     :stage_resample           i       0           selection step at stage i
#     :stage_blocks             i       0           parameter blocks at stage i
#     :mutation_particle        i       k           mutation of particle k at stage i
#     :added_parameter_particle 0       row         add_parameters_to_cloud(; smc_seed)
#     :group                    0       g           run seed of particle group g (n_smc_groups > 1);
#                                       0           split of a shared old cloud across the groups
#     :bridge_sampling          0       0           split and proposal draws for bridge sampling
#
# Resampling and blocks draw from an explicit Xoshiro(seed). Per-particle draws (which happen
# inside model code such as rand(::ParameterVector)) instead Random.seed! the task-local RNG
# of the task doing the draw, i.e. inside the @distributed body on the worker. smc() puts the
# caller's task-local RNG back when it returns (preserving_default_rng).
#
# A stream is Julia's Xoshiro seeded with the derived UInt64 (Random.seed! hashes it with
# SHA-256), so bitwise replay also needs the same Julia version and the same dependency
# versions. smc() saves both next to the seed (`run_info`). Changing the derivation, the
# operation names or their coordinates must bump SMC_SEED_VERSION; resuming refuses a
# checkpoint of another version.
const SMC_SEED_VERSION = 1

function splitmix64(x::UInt64)
    x += 0x9e3779b97f4a7c15
    x = (x ⊻ (x >> 30)) * 0xbf58476d1ce4e5b9
    x = (x ⊻ (x >> 27)) * 0x94d049bb133111eb
    return x ⊻ (x >> 31)
end

function stream_seed(run_seed::UInt64, operation::Symbol, stage::Integer = 0,
                     particle::Integer = 0)
    h = UInt64(0xcbf29ce484222325)
    for b in codeunits(String(operation))
        h = (h ⊻ UInt64(b)) * 0x100000001b3
    end
    return splitmix64(splitmix64(splitmix64(run_seed ⊻ h) ⊻ UInt64(stage)) ⊻ UInt64(particle))
end

function checked_smc_seed(seed::Integer)
    0 <= seed <= typemax(UInt64) || throw(ArgumentError("smc_seed must be a nonnegative UInt64 value"))
    return UInt64(seed)
end

"""
```
preserving_default_rng(f)
```
Run `f()` and then put the caller's task-local RNG back as it was. The per-particle streams
work by reseeding the task-local RNG, which must not leak into the caller's later draws.
"""
function preserving_default_rng(f)
    saved = copy(Random.default_rng())
    try
        return f()
    finally
        copy!(Random.default_rng(), saved)
    end
end

"""
```
package_commit(mod::Module)
```
The git commit of the checkout `mod` was loaded from, with a `-dirty` suffix if its code
(`src/`, `Project.toml`) differs from that commit, or `""` if it isn't a git checkout or git is
unavailable. `safe.directory=*` lets git read checkouts owned by another user (shared
directories); other changes in the tree, such as test reference files rewritten by the tests,
do not count as dirty.
"""
function package_commit(mod::Module)
    dir = pkgdir(mod)
    dir === nothing && return ""
    git(args...) = `git -c 'safe.directory=*' -C $dir $args`
    try
        commit = String(readchomp(pipeline(git("rev-parse", "HEAD"); stderr = devnull)))
        clean  = success(pipeline(git("diff", "--quiet", "HEAD", "--", "src", "Project.toml");
                                  stdout = devnull, stderr = devnull))
        return clean ? commit : commit * "-dirty"
    catch
        return ""
    end
end
#= Old: no safe.directory (blank on shared checkouts), and --dirty counted any change in the
   tree, e.g. test reference files rewritten by the tests
function package_commit(mod::Module)
    dir = pkgdir(mod)
    dir === nothing && return ""
    try
        return String(readchomp(pipeline(`git -C $dir describe --always --dirty --abbrev=40`; stderr = devnull)))
    catch
        return ""
    end
end
=#
