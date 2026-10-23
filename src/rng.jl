# Version 1: FNV-1a over the operation name, mixed with the run seed and
# integer coordinates by SplitMix64. No process hash or worker ID is used.
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
