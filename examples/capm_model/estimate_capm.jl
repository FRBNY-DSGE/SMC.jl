using ModelConstructors, FileIO, Random, SMC

### Estimate a single factor CAPM model
# R_{it} = α_i + β_i R_{Mt} + ϵ_{it}, i = 1,...,N; t = 1,...,T
# where R_{Mt} is the excess return on a market index in time period t,
# and ϵ_{it} is an i.i.d. normally distributed mean zero shock with variance σ_i^2

### Construct a generic model and populate it with parameters
capm = GenericModel()
fn = dirname(@__FILE__)
capm <= Setting(:dataroot, "$(fn)/../save/input_data/")
capm <= Setting(:saveroot, "$(fn)/../save/")

capm <= parameter(:α1, 0., (-1e5, 1e5), (-1e5, 1e5), Untransformed(), Normal(0, 1e3),
                  fixed = false)
capm <= parameter(:β1, 0., (-1e5, 1e5), (-1e5, 1e5), Untransformed(), Normal(0, 1e3),
                  fixed = false)
capm <= parameter(:σ1, 1., (1e-5, 1e5), (1e-5, 1e5), SquareRoot(), Uniform(0, 1e3),
                  fixed = false)
capm <= parameter(:α2, 0., (-1e5, 1e5), (-1e5, 1e5), Untransformed(), Normal(0, 1e3),
                  fixed = false)
capm <= parameter(:β2, 0., (-1e5, 1e5), (-1e5, 1e5), Untransformed(), Normal(0, 1e3),
                  fixed = false)
capm <= parameter(:σ2, 1., (1e-5, 1e5), (1e-5, 1e5), SquareRoot(), Uniform(0, 1e3),
                  fixed = false)
capm <= parameter(:α3, 0., (-1e5, 1e5), (-1e5, 1e5), Untransformed(), Normal(0, 1e3),
                  fixed = false)
capm <= parameter(:β3, 0., (-1e5, 1e5), (-1e5, 1e5), Untransformed(), Normal(0, 1e3),
                  fixed = false)
capm <= parameter(:σ3, 1., (1e-5, 1e5), (1e-5, 1e5), SquareRoot(), Uniform(0, 1e3),
                  fixed = false)

### Estimate with SMC

## Get data
N = 3 # number of asset returns
lik_data = load("../../../save/input_data/capm.jld2", "lik_data")
market_data = load("../../../save/input_data/capm.jld2", "market_data")

## Construct likelihood function:
# likelihood function is just R_{it} ∼ N(α_i + β_i R_{Mt}, σ_i)
# parameters to estimate are α_i, β_i, σ_i
# data is a time series of individual factor returns
# use S&P 500 data and a returns on a couple stocks. To get into returns,
# just take the log of the prices and regress those on each other.
function likelihood_fnct(p, d)
    # we assume the ordering of (α_i, β_i, σ_i)
    Σ = zeros(N,N)
    α = Vector{Float64}(undef,N)
    β = Vector{Float64}(undef,N)
    for i in 1:N
        α[i]   = p[i * 3 - 2]
        β[i]   = p[i * 3 - 2]
        Σ[i,i] = p[i * 3]^2
    end
    det_Σ = det(Σ)
    inv_Σ = inv(Σ)
    term1 = -N / 2 * log(2 * π) - 1 /2 * log(det_Σ)
    logprob = 0.
    errors = d .- α .- β .* market_data
    for t in 1:size(d,2)
        logprob += term1 - 1/2 * dot(errors, inv_Σ * errors)
    end
    return exp(logprob)
end

Random.seed!(1793)
println("Starting to estimate CAPM with SMC . . .")
@everywhere using SMC, OrderedCollections
smc(likelihood_fnct, capm.parameters, lik_data)
