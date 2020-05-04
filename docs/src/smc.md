## SMC Main Function

You can use SMC to estimate any Bayesian model. This requires (1) parameters and their associated prior distributions (2) data (3) a log-likelihood function. These three ingredients are the only inputs into the smc driver.

`smc(loglikelihood::Function, parameters::ParameterVector, data::Matrix)` function.

Let's look at an example. First, we'll make a model. We need the [ModelConstructors](https://frbny-dsge.github.io/ModelConstructors.jl) package to do that.

```@example
using ModelConstructors
ols_model = GenericModel()
```

Next, assign our intercept and coefficient parameters to the model.
```@example
reg <= parameter(:α1, 0., (-1e5, 1e5), (-1e5, 1e5), Untransformed(), Normal(0, 10), fixed = false)
reg <= parameter(:β1, 0., (-1e5, 1e5), (-1e5, 1e5), Untransformed(), Normal(0, 10), fixed = false)
```
The first argument is the name of the parameter, the second is its initial value, the third and fourther are bounds on the parameter (they should always be the same). If the sampler draws a value outside of these bounds, it'll try again. The fifth argument is whether you would like to perform a transformation on the parameter when using it (ignore this for now--it's for more advanced users). The sixth argument is the prior, ``N(0, 10)``, and last argument says the parameters aren't fixed parameters (i.e. we're trying to estimate them!)

We'll make some artifical data to test our model
```@example
X = rand(100) #hide
β = 1.
α = 1.
y = β*X + α
```

We need a log-likelihood function! Note that it's important you define a log-likelihood function rather than a likelihood function.
```@example
function likelihood_fnct(p, d)
    α = p[1]
    β = p[2]
    Σ = 1
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
```
And that's it! Now let's run SMC.

```@example
smc(likelihood_fnct, reg.parameters, data, n_parts = 10, use_fixed_schedule = false, tempering_target = 0.97)
```


```@docs
smc(loglikelihood::Function, parameters::ParameterVector{U}, data::Matrix{S};
         kwargs...) where {S<:AbstractFloat, U<:Number}
```
