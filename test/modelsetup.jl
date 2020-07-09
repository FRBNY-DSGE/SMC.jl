###################################################################
# Set Up Linear Model
###################################################################

function setup_linear_model()
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

    return m
end


# Generate Data
#
#  X = rand(Float64, (3, 100))
#  err = rand(Float64, (3, 100))
#  β = fill(2, 3) #(1,2,3)
#  β[1]=1
#  β[2]=2
#  β[3]=3
#
#  α = fill(2,3)
#  α[1]=1
#  α[2]=2
#  α[3]=3
#
#  data = β .* X .+ α .+ err

# Save Data

#  h5open("reference/test_data.h5", "w") do file
#      write(file, "data", data)
#      write(file, "X", X)
#  end

# Read Data
data = h5read("reference/test_data.h5", "data")
X = h5read("reference/test_data.h5", "X")

# Log Likelihood Function
N = 3
function loglik_fn(p, d)
    # we assume the ordering of (α_i, β_i, σ_i)
    Σ = zeros(N,N) 
    α = Vector{Float64}(undef,N)
    β = Vector{Float64}(undef,N)
    for i in 1:N
        α[i]   = p[i * 3 - 2]
        β[i]   = p[i * 3 - 1]
        Σ[i,i] = p[i * 3]^2
    end
    det_Σ = det(Σ)
    inv_Σ = inv(Σ)
    term1 = -size(d,2) / 2 * log(2 * π) - 1 /2 * log(det_Σ)
    logprob = 0.
    errors = d .- α .- β .* X[:, 1:size(sparse(d),2)] 
    for t in 1:size(d,2) #see above
        logprob += term1 - 1/2 * dot(errors, inv_Σ * errors)
    end
    return exp(logprob)
end

