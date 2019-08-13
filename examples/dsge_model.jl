using DSGE, ModelConstructors, SMC
import ModelConstructors: ParameterVector
import SMC: smc

## Initialize model object
m = AnSchorfheide()
data = df_to_matrix(m, load_data(m))

## Define a likelihood function with correct input/output format
function my_likelihood(parameters::ParameterVector, data::Matrix{Float64})
    update!(m, parameters)
    likelihood(m, data; sampler = false, catch_errors = true)
end

smc(my_likelihood, m.parameters, data)
