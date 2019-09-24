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
X = rand(100)
β = 1.
α = 1.
y = β*X + α
N = 1
data = hcat(y, X)
jldopen(dataroot(reg)*"reg_data.jld2", "w") do file
    file["data"] = data
end

## Construct likelihood function:
function likelihood_fnct(p, d)
    α = p[1]
    β = p[2]
    Σ = 1 #ones(N,N)
    det_Σ = det(Σ)
    inv_Σ = inv(Σ)
    term1 = -N / 2 * log(2 * π) - 1 /2 * log(det_Σ)
    logprob = 0.
    errors = d[:, 1] .- α .- β .* d[:, 2]
    for t in 1:size(d,1)
        logprob += term1 - 1/2 * dot(errors, inv_Σ * errors)
    end
    return logprob
end

Random.seed!(1793)
println("Starting to estimate Regression with SMC . . .")
@everywhere using SMC
smc(likelihood_fnct, reg.parameters, data, n_parts = 10, use_fixed_schedule = false, tempering_target = 0.97)
