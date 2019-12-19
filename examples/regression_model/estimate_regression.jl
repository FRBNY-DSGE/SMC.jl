using ModelConstructors, FileIO, Random, SMC, JLD2

### Construct a generic model and populate it with parameters
reg = GenericModel()
fn = dirname(@__FILE__)
reg <= Setting(:dataroot, "$(fn)/save/input_data/")
reg <= Setting(:saveroot, "$(fn)/save/")

reg <= parameter(:α1, 0., (-1e5, 1e5), (-1e5, 1e5), Untransformed(), Normal(0, 10), fixed = false)
reg <= parameter(:β1, 0., (-1e5, 1e5), (-1e5, 1e5), Untransformed(), Normal(0, 10), fixed = false)

### Estimate with SMC

## Make data
N = 100 # number observations
M = 1 # number regressors

X = rand(N)
β = 1.
α = 1.
σ2 = 1.
y = β*X .+ α #.+ randn(N) .* sqrt(σ2)
5B

jldopen(dataroot(reg)*"reg_data.jld2", "w") do file
    file["data"] = hcat(y, X)
end

function compute_coefficients(X::Vector, Y::Vector)
    @assert length(X) == length(Y)
    X_aug = hcat(ones(length(X)), X)
    β     = inv(X_aug'* X_aug)*X_aug'*Y
    return β
end

function compute_coefficients(X::Array, Y::Vector)
    @assert size(X)[1] == length(Y)
    X_aug = hcat(ones(size(X)[1], 1), X)
    β     = inv(X_aug'* X_aug)*X_aug'*Y
    return β
end

compute_coefficients(X, y)

## Construct log-likelihood function:
function log_likelihood_fnct(p, y)
    α = p[1]
    β = p[2]
    term1 = -(N/2)*log(2*π) - (N/2)*log(σ2)
    errors = y .- α .- β .* X
    logprob = term1 - (1/(2*σ2))*dot(errors, errors)
    return logprob
end

Random.seed!(1793)
println("Starting to estimate Regression with SMC . . .")
@everywhere using SMC
smc(log_likelihood_fnct, reg.parameters, Matrix(Matrix{Float64}((y'))'), n_parts = 100, use_fixed_schedule = true) #, tempering_target = 0.99)

cloud = load("smc_cloud.jld2", "cloud")
mean(SMC.get_vals(cloud), dims = 2)
