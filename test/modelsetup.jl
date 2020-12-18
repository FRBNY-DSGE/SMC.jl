using ModelConstructors, HDF5

regenerate_data = false

###################################################################
# Set Up Linear Model
###################################################################

function setup_linear_model(; regime_switching::Bool = false)
    m = GenericModel()

    # Set up linear parameters
    m <= parameter(:α1, 0., (-1e5, 1e5), (-1e5, 1e5), Untransformed(), Normal(0, 1e3),
                  fixed = false)
    m <= parameter(:β1, 0., (-1e5, 1e5), (-1e5, 1e5), Untransformed(), Normal(0, 1e3),
                  fixed = false)
    m <= parameter(:σ1, 1., (1e-5, 1e5), (1e-5, 1e5), SquareRoot(), Uniform(0, 1e3),
                  fixed = false)
    m <= parameter(:α2, 0., (-1e5, 1e5), (-1e5, 1e5), Untransformed(), Normal(0, 1e3),
                  fixed = false)
    m <= parameter(:β2, 0., (-1e5, 1e5), (-1e5, 1e5), Untransformed(), Normal(0, 1e3),
                  fixed = false)
    m <= parameter(:σ2, 1., (1e-5, 1e5), (1e-5, 1e5), SquareRoot(), Uniform(0, 1e3),
                  fixed = false)
    m <= parameter(:α3, 0., (-1e5, 1e5), (-1e5, 1e5), Untransformed(), Normal(0, 1e3),
                  fixed = false)
    m <= parameter(:β3, 0., (-1e5, 1e5), (-1e5, 1e5), Untransformed(), Normal(0, 1e3),
                  fixed = false)
    m <= parameter(:σ3, 1., (1e-5, 1e5), (1e-5, 1e5), SquareRoot(), Uniform(0, 1e3),
                  fixed = false)
    m <= Setting(:n_particles, 400) #-> will get commented eventually
    m <= Setting(:n_Φ, 100)
    m <= Setting(:λ, 2.0)
    m <= Setting(:n_smc_blocks, 1)
    m <= Setting(:use_parallel_workers, false)
    m <= Setting(:step_size_smc, 0.5)
    m <= Setting(:n_mh_steps_smc, 1)
    m <= Setting(:resampler_smc, :polyalgo)
    m <= Setting(:target_accept, 0.25)

    m <= Setting(:mixture_proportion, .9)
    m <= Setting(:tempering_target, 0.95)
    m <= Setting(:resampling_threshold, .5)
    m <= Setting(:use_fixed_schedule, true)

    if regime_switching
        for i in 1:3
            ModelConstructors.set_regime_fixed!(m[Symbol("α$(i)")], 1, false)
            ModelConstructors.set_regime_fixed!(m[Symbol("α$(i)")], 2, false)
            ModelConstructors.set_regime_fixed!(m[Symbol("α$(i)")], 3, false) # Do not estimate this parameter, just to check this functionality
            ModelConstructors.set_regime_val!(m[Symbol("α$(i)")], 1, -.1 * i)
            ModelConstructors.set_regime_val!(m[Symbol("α$(i)")], 2, .1 * i)
            ModelConstructors.set_regime_val!(m[Symbol("α$(i)")], 3, float(i))

        end
        for i in 1:3
            ModelConstructors.set_regime_val!(m[Symbol("β$(i)")], 1, .2 * i)
            ModelConstructors.set_regime_prior!(m[Symbol("β$(i)")], 1, Normal(0, 1e3)) # regime-switching prior, just to check functionality
            ModelConstructors.set_regime_val!(m[Symbol("β$(i)")], 2, -.1 * i)
            ModelConstructors.set_regime_prior!(m[Symbol("β$(i)")], 2, Normal(0, 1e3))
            ModelConstructors.set_regime_val!(m[Symbol("β$(i)")], 3, .1 * i)
            ModelConstructors.set_regime_prior!(m[Symbol("β$(i)")], 3, Normal(0, 1e2))
        end
    end

    return m
end


# Generate Data
if regenerate_data
    Random.seed!(1793)

    X = randn(Float64, (3, 100))
    err = randn(Float64, (3, 100))
    β = collect(1:3)
    α = collect(1:3)

    data = β .* X .+ α .+ err

    # Regime-switching version
    Xrs = randn(Float64, (3, 300))
    err = randn(Float64, (3, 300))
    β₁ = collect(1:3)
    β₂ = collect(2:4)
    β₃ = collect(3:5)

    α₁ = collect(1:3)
    α₂ = collect(1:3)

    rsdata = similar(err)
    reg1   = 1:100
    reg2   = 101:200
    reg3   = 201:300
    rsdata[:, reg1] = β₁ .* Xrs[:, reg1] .+ α₁ .+ err[:, reg1]
    rsdata[:, reg2] = β₂ .* Xrs[:, reg2] .+ α₂ .+ err[:, reg2]
    rsdata[:, reg3] = β₃ .* Xrs[:, reg3] .+ α₂ .+ err[:, reg3]

    # Save Data

    h5open("reference/test_data.h5", "w") do file
        write(file, "data", data)
        write(file, "rsdata", rsdata)
        write(file, "X", X)
        write(file, "Xrs", Xrs)
    end
end

# Read Predictors from data
X = h5read("reference/test_data.h5", "X")
Xrs = h5read("reference/test_data.h5", "Xrs")

# Log Likelihood Function
N = 3
function loglik_fn(p, d)
    # we assume the ordering of (α_i, β_i, σ_i)
    Σ = zeros(N,N)
    α = Vector{Float64}(undef,N)
    β = Vector{Float64}(undef,N)
    for i in 1:N
        α[i]   = p[i * 3 - 2] # note p is a ParameterVector
        β[i]   = p[i * 3 - 1] # but since LHS are Vector{Float64},
        Σ[i,i] = p[i * 3]^2   # the value of p is autoamtically stored
    end
    det_Σ = det(Σ)
    inv_Σ = inv(Σ)
    term1 = -N / 2 * log(2 * π) - 1 /2 * log(det_Σ) # N should be number of equations
    logprob = 0.
    errors = d .- α .- β .* X[:, 1:size(d, 2)] # bridging test uses first 1/2 of sample so need to be robust there
    for t in 1:size(d, 2) # Number of time periods
        logprob += term1 - 1/2 * dot(errors[:, t], inv_Σ * errors[:, t]) # Should be error in each time period
    end
    return logprob
end

function rs_loglik_fn(p, d)
    # we assume the ordering of (α_i, β_i, σ_i)
    Σ = zeros(N,N)
    α = Vector{Float64}(undef, 3 * N)
    β = Vector{Float64}(undef, 3 * N)
    for i in 1:N
        α[i]   = regime_val(p[N * (i - 1) + 1], 1)
        β[i]   = regime_val(p[N * (i - 1) + 2], 1)
        Σ[i,i] = p[N * (i - 1) + 3]
    end
    for i in 1:N # Looping over each entry of the constant vector and betas in a given regime
        α[N + i] = regime_val(p[N * (i - 1) + 1], 2) # α1, α2, α3 each has 3 regimes. This line is for regime 2.
        α[(2 * N) + i] = regime_val(p[N * (i - 1) + 1], 3) # This line is for regime 3.
        β[N + i] = regime_val(p[N * (i - 1) + 2], 2) # β1, β2, β3 each has 3 regimes.
        β[(2 * N) + i] = regime_val(p[N * (i - 1) + 2], 3)
    end

    det_Σ = det(Σ)
    inv_Σ = inv(Σ)
    term1 = -N / 2 * log(2 * π) - 1 /2 * log(det_Σ)
    logprob = 0.
    errors = similar(d)
    errors[:, reg1] = d[:, reg1] .- α[1:N] .- β[1:N] .* Xrs[:, reg1]
    errors[:, reg2] = d[:, reg2] .- α[(N + 1):(2 * N)] .- β[(N + 1):(2 * N)] .* Xrs[:, reg2]
    errors[:, reg3] = d[:, reg3] .- α[(2 * N + 1):(3 * N)] .- β[(2 * N + 1):(3 * N)] .* Xrs[:, reg3]
    for t in 1:size(d, 2) # see above
        logprob += term1 - 1/2 * dot(errors[:, t], inv_Σ * errors[:, t])
    end
    return logprob
end
