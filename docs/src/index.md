# SMC.jl Documentation

```@docs
function smc(loglikelihood::Function, parameters::ParameterVector{U}, data::Matrix{S};
         kwargs...) where {S<:AbstractFloat, U<:Number}
```
