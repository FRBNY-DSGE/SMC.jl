import Base: <=

Interval{T} = Tuple{T,T}
"""
```
Transform
```

Subtypes of the abstract Transform type indicate how a `Parameter`'s
value is transformed from model space (which can be bounded to a
limited section of the real line) to the entire real line (which is
necessary for mode-finding using csminwel). The transformation is
performed by the `transform_to_real_line` function, and is reversed by the
`transform_to_model_space` function.
"""
abstract type Transform end

struct Untransformed <: Transform end
struct SquareRoot    <: Transform end
struct Exponential   <: Transform end

Base.show(io::IO, t::Untransformed) = @printf io "x -> x\n"
Base.show(io::IO, t::SquareRoot)    = @printf io "x -> (a+b)/2 + (b-a)/2*c*x/sqrt(1 + c^2 * x^2)\n"
Base.show(io::IO, t::Exponential)   = @printf io "x -> b + (1/c) * log(x-a)\n"

"""
```
AbstractParameter{T<:Number}
```

The AbstractParameter type is the common supertype of all model
parameters, including steady-state values.  Its subtype structure is
as follows:

-`AbstractParameter{T<:Number}`: The common abstract supertype for all parameters.
    -`Parameter{T<:Number, U<:Transform}`: The abstract supertype for parameters that are directly estimated.
        -`UnscaledParameter{T<:Number, U:<Transform}`: Concrete type for parameters that do not need to be scaled for equilibrium conditions.
        -`ScaledParameter{T<:Number, U:<Transform}`: Concrete type for parameters that are scaled for equilibrium conditions.
    -`SteadyStateParameter{T<:Number}`: Concrete type for steady-state parameters.
"""
abstract type AbstractParameter{T<:Number} end

"""
```
Parameter{T<:Number, U<:Transform} <: AbstractParameter{T}
```

The Parameter type is the common supertype of time-invariant, non-steady-state model
parameters. It has 2 subtypes, `UnscaledParameter` and `ScaledParameter`.
`ScaledParameter`s are parameters whose values are scaled when used in the model's
equilibrium conditions. The scaled value is stored for convenience, and udpated when the
parameter's value is updated.
"""
abstract type Parameter{T,U<:Transform} <: AbstractParameter{T} end

ParameterVector{T} =  Vector{AbstractParameter{T}}
NullablePrior      =  Nullable{ContinuousUnivariateDistribution}

"""
```
UnscaledParameter{T<:Number,U<:Transform} <: Parameter{T,U}
```

Time-invariant model parameter whose value is used as-is in the model's equilibrium
conditions.

#### Fields
- `key::Symbol`: Parameter name. For maximum clarity, `key`
  should conform to the guidelines established in the DSGE Style Guide.
- `value::T`: Parameter value. Initialized in model space (guaranteed
  to be between `valuebounds`), but can be transformed between model
  space and the real line via calls to `transform_to_real_line` and
`transform_to_model_space`.
- `valuebounds::Interval{T}`: Bounds for the parameter's value in model space.
- `transform_parameterization::Interval{T}`: Parameters used to
  transform `value` between model space and the real line.
- `transform::U`: Transformation used to transform `value` between
  model space and real line.
- `prior::NullablePrior`: Prior distribution for parameter value.
- `fixed::Bool`: Indicates whether the parameter's value is fixed rather than estimated.
- `description::String`:  A short description of the parameter's economic
  significance.
- `tex_label::String`: String for printing the parameter name to LaTeX.
"""
mutable struct UnscaledParameter{T,U} <: Parameter{T,U}
    key::Symbol
    value::T                                # parameter value in model space
    valuebounds::Interval{T}                # bounds of parameter value
    transform_parameterization::Interval{T} # parameters for transformation
    transform::U                            # transformation between model space and real line for optimization
    prior::NullablePrior                    # prior distribution
    fixed::Bool                             # is this parameter fixed at some value?
    description::String
    tex_label::String               # LaTeX label for printing
end


"""
```
ScaledParameter{T,U} <: Parameter{T,U}
```

Time-invariant model parameter whose value is scaled for use in the model's equilibrium
conditions.

#### Fields

- `key::Symbol`: Parameter name. For maximum clarity, `key`
  should conform to the guidelines established in the DSGE Style Guide.
- `value::T`: The parameter's unscaled value. Initialized in model
  space (guaranteed to be between `valuebounds`), but can be
  transformed between model space and the real line via calls to
  `transform_to_real_line` and `transform_to_model_space`.
- `scaledvalue::T`: Parameter value scaled for use in `eqcond.jl`
- `valuebounds::Interval{T}`: Bounds for the parameter's value in model space.
- `transform_parameterization::Interval{T}`: Parameters used to
  transform `value` between model space and the real line.
- `transform::U`: The transformation used to convert `value` between model space and the
  real line, for use in optimization.
- `prior::NullablePrior`: Prior distribution for parameter value.
- `fixed::Bool`: Indicates whether the parameter's value is fixed rather than estimated.
- `scaling::Function`: Function used to scale parameter value for use in equilibrium
  conditions.
- `description::String`: A short description of the parameter's economic
  significance.
- `tex_label::String`: String for printing parameter name to LaTeX.
"""
mutable struct ScaledParameter{T,U} <: Parameter{T,U}
    key::Symbol
    value::T
    scaledvalue::T
    valuebounds::Interval{T}
    transform_parameterization::Interval{T}
    transform::U
    prior::NullablePrior
    fixed::Bool
    scaling::Function
    description::String
    tex_label::String
end

"""
```
SteadyStateParameter{T} <: AbstractParameter{T}
```

Steady-state model parameter whose value depends upon the value of other (non-steady-state)
`Parameter`s. `SteadyStateParameter`s must be constructed and added to an instance of a
model object `m` after all other model `Parameter`s have been defined. Once added to `m`,
`SteadyStateParameter`s are stored in `m.steady_state`. Their values are calculated and set
by `steadystate!(m)`, rather than being estimated directly. `SteadyStateParameter`s do not
require transformations from the model space to the real line or scalings for use in
equilibrium conditions.

#### Fields

- `key::Symbol`: Parameter name. Should conform to the guidelines
  established in the DSGE Style Guide.
- `value::T`: The parameter's steady-state value.
- `description::String`: Short description of the parameter's economic significance.
- `tex_label::String`: String for printing parameter name to LaTeX.
"""
mutable struct SteadyStateParameter{T} <: AbstractParameter{T}
    key::Symbol
    value::T
    description::String
    tex_label::String
end

"""
```
SteadyStateValueGrid{T} <: AbstractParameter{T}
```

Steady-state model parameter grid (for heterogeneous agent models) whose value is calculated by an
iterative procedure.
`SteadyStateParameterGrid`s must be constructed and added to an instance of a
model object `m` after all other model `Parameter`s have been defined. Once added to `m`,
`SteadyStateParameterGrid`s are stored in `m.steady_state`. Their values are calculated and set
by `steadystate!(m)`, rather than being estimated directly. `SteadyStateParameter`s do not
require transformations from the model space to the real line or scalings for use in
equilibrium conditions.

#### Fields

- `key::Symbol`: Parameter name. Should conform to the guidelines
  established in the DSGE Style Guide.
- `value::Array{T}`: The parameter's steady-state value grid.
- `description::String`: Short description of the parameter's economic significance.
- `tex_label::String`: String for printing parameter name to LaTeX.
"""
mutable struct SteadyStateParameterGrid{T} <: AbstractParameter{T}
    key::Symbol
    value::Array{T}
    description::String
    tex_label::String
end

function SteadyStateParameterGrid(key::Symbol,
                                  value::Array{T};
                                  description::String = "No description available",
                                  tex_label::String = "") where {T<:Number}

    return SteadyStateParameterGrid{T}(key, value, description, tex_label)
end

"""
```
SteadyStateParameterArray{T} <: AbstractParameter{T}
```
Steady-state model parameter whose value is an Array and
 depends upon the value of other (non-steady-state)
`Parameter`s. `SteadyStateParameterArray`s must be constructed and added to an instance of a
model object `m` after all other model `Parameter`s have been defined. Once added to `m`,
`SteadyStateParameterArray`s are stored in `m.steady_state`. Their values are calculated and set
by `steadystate!(m)`, rather than being estimated directly. `SteadyStateParameterArray`s do not
require transformations from the model space to the real line or scalings for use in
equilibrium conditions.

#### Fields

- `key::Symbol`: Parameter name. Should conform to the guidelines
  established in the DSGE Style Guide.
- `value::Array{T}`: The parameter's steady-state values.
- `description::String`: Short description of the parameter's economic significance.
- `tex_label::String`: String for printing parameter name to LaTeX.
"""
mutable struct SteadyStateParameterArray{T} <: AbstractParameter{T}
    key::Symbol
    value::Array{T}
    description::String
    tex_label::String
end

"""
```
SteadyStateParameterArray{T<:Number}(key::Symbol, value::Array{T};
                                description::String = "",
                                tex_label::String = "")
```

SteadyStateParameter constructor with optional `description` and `tex_label` arguments.
"""
function SteadyStateParameterArray(key::Symbol,
                                   value::Array{T};
                                   description::String = "No description available",
                                   tex_label::String = "") where {T<:Number}

    return SteadyStateParameterArray(key, value, description, tex_label)
end

# TypeError: non-boolean (BitArray{1}) used in boolean context
# gets thrown when we print the value.

function Base.show(io::IO, p::SteadyStateParameterArray{T}) where {T}
    @printf io "%s\n" typeof(p)
    @printf io "(:%s)\n%s\n"      p.key p.description
    @printf io "LaTeX label: %s\n"     p.tex_label
    @printf io "-----------------------------\n"
    @printf io "value:        [%+6f,...,%+6f]\n" p.value[1] p.value[end]
end

hasprior(p::Parameter) = !isnull(p.prior)

NullableOrPrior = Union{NullablePrior, ContinuousUnivariateDistribution}

# We want to use value field from UnscaledParameters and
# SteadyStateParameters in computation, so we alias their union here.
UnscaledOrSteadyState = Union{UnscaledParameter, SteadyStateParameter}

"""
```
ParamBoundsError <: Exception
```

A `ParamBoundsError` is thrown upon an attempt to assign a parameter value that is not
between `valuebounds`.
"""
mutable struct ParamBoundsError <: Exception
    msg::String
end
ParamBoundsError() = ParamBoundsError("Value not between valuebounds")
Base.showerror(io::IO, ex::ParamBoundsError) = print(io, ex.msg)

"""
```
parameter{T,U<:Transform}(key::Symbol, value::T, valuebounds = (value,value),
                          transform_parameterization = (value,value),
                          transform = Untransformed(), prior = NullablePrior();
                          fixed = true, scaling::Function = identity, description = "",
                          tex_label::String = "")
```

By default, returns a fixed `UnscaledParameter` object with key `key`
and value `value`. If `scaling` is given, a `ScaledParameter` object
is returned.
"""
function parameter(key::Symbol,
                   value::T,
                   valuebounds::Interval{T} = (value,value),
                   transform_parameterization::Interval{T} = (value,value),
                   transform::U             = DSGE.Untransformed(),
                   prior::NullableOrPrior   = NullablePrior();
                   fixed::Bool              = true,
                   scaling::Function        = identity,
                   description::String = "No description available.",
                   tex_label::String = "") where {T, U <:Transform}

    # If fixed=true, force bounds to match and leave prior as null.  We need to define new
    # variable names here because of lexical scoping.

    valuebounds_new = valuebounds
    transform_parameterization_new = transform_parameterization
    transform_new = transform
    U_new = U
    prior_new = prior

    if fixed
        transform_parameterization_new = (value,value)  # value is transformed already
        transform_new = Untransformed()                 # fixed priors should stay untransformed
        U_new = Untransformed

        if isa(transform, Untransformed)
            valuebounds_new = (value,value)
        end
    else
        transform_parameterization_new = transform_parameterization
    end

    # ensure that we have a Nullable{Distribution}, if not construct one
    prior_new = !isa(prior_new,NullablePrior) ? NullablePrior(prior_new) : prior_new

    if scaling == identity
        return UnscaledParameter{T,U_new}(key, value, valuebounds_new,
                                          transform_parameterization_new, transform_new,
                                          prior_new, fixed, description, tex_label)
    else
        return ScaledParameter{T,U_new}(key, value, scaling(value), valuebounds_new,
                                        transform_parameterization_new, transform_new,
                                        prior_new, fixed, scaling, description, tex_label)
    end
end

"""
```
SteadyStateParameter(key::Symbol, value::T; description::String = "",
                      tex_label::String = "") where {T <: Number}
```

SteadyStateParameter constructor with optional `description` and `tex_label` arguments.
"""
function SteadyStateParameter(key::Symbol, value::T;
                              description::String = "No description available",
                              tex_label::String = "") where {T <: Number}
    return SteadyStateParameter(key, value, description, tex_label)
end


"""
```
parameter(p::UnscaledParameter{T,U}, newvalue::T) where {T<:Number,U<:Transform}
```

Returns an UnscaledParameter with value field equal to `newvalue`. If `p` is a fixed
parameter, it is returned unchanged.
"""
function parameter(p::UnscaledParameter{T,U}, newvalue::T) where {T <: Number, U <: Transform}
    p.fixed && return p    # if the parameter is fixed, don't change its value
    a,b = p.valuebounds
    if !(a <= newvalue <= b)
        throw(ParamBoundsError("New value of $(string(p.key)) ($(newvalue)) is out of bounds ($(p.valuebounds))"))
    end
    UnscaledParameter{T,U}(p.key, newvalue, p.valuebounds, p.transform_parameterization,
                           p.transform, p.prior, p.fixed, p.description, p.tex_label)
end


"""
```
parameter(p::ScaledParameter{T,U}, newvalue::T) where {T<:Number,U<:Transform}
```

Returns a ScaledParameter with value field equal to `newvalue` and scaledvalue field equal
to `p.scaling(newvalue)`. If `p` is a fixed parameter, it is returned unchanged.
"""
function parameter(p::ScaledParameter{T,U}, newvalue::T) where {T <: Number, U <: Transform}
    p.fixed && return p    # if the parameter is fixed, don't change its value
    a,b = p.valuebounds
    if !(a <= newvalue <= b)
        throw(ParamBoundsError("New value of $(string(p.key)) ($(newvalue)) is out of bounds ($(p.valuebounds))"))
    end
    ScaledParameter{T,U}(p.key, newvalue, p.scaling(newvalue), p.valuebounds,
                         p.transform_parameterization, p.transform, p.prior, p.fixed,
                         p.scaling, p.description, p.tex_label)
end

function Base.show(io::IO, p::Parameter{T,U}) where {T, U}
    @printf io "%s\n" typeof(p)
    @printf io "(:%s)\n%s\n"      p.key p.description
    @printf io "LaTeX label: %s\n"     p.tex_label
    @printf io "-----------------------------\n"
    #@printf io "real value:        %+6f\n" transform_to_real_line(p)
    @printf io "unscaled, untransformed value:        %+6f\n" p.value
    isa(p,ScaledParameter) && @printf "scaled, untransformed value:        %+6f\n" p.scaledvalue
    #!isa(U(),Untransformed) && @printf io "transformed value: %+6f\n" p.value

    if hasprior(p)
        @printf io "prior distribution:\n\t%s\n" get(p.prior)
    else
        @printf io "prior distribution:\n\t%s\n" "no prior"
    end

    @printf io "transformation for csminwel:\n\t%s" U()
    @printf io "parameter is %s\n" p.fixed ? "fixed" : "not fixed"
end

function Base.show(io::IO, p::SteadyStateParameter{T}) where {T}
    @printf io "%s\n" typeof(p)
    @printf io "(:%s)\n%s\n"      p.key p.description
    @printf io "LaTeX label: %s\n"     p.tex_label
    @printf io "-----------------------------\n"
    @printf io "value:        %+6f\n" p.value
end

function Base.show(io::IO, p::SteadyStateParameterGrid{T}) where {T}
    @printf io "%s\n" typeof(p)
    @printf io "(:%s)\n%s\n"      p.key p.description
    @printf io "LaTeX label: %s\n"     p.tex_label
    @printf io "-----------------------------\n"
    @printf io "value:        [%f,...,%f]" p.value[1] p.value[end]
end

"""
```
transform_to_model_space{T<:Number, U<:Transform}(p::Parameter{T,U}, x::T)
```

Transforms `x` from the real line to lie between `p.valuebounds` without updating `p.value`.
The transformations are defined as follows, where (a,b) = p.transform_parameterization and c
a scalar (default=1):

- Untransformed: `x`
- SquareRoot:    `(a+b)/2 + (b-a)/2 * c * x/sqrt(1 + c^2 * x^2)`
- Exponential:   `a + exp(c*(x-b))`
"""
transform_to_model_space(p::Parameter{T,Untransformed}, x::T) where T = x
function transform_to_model_space(p::Parameter{T,SquareRoot}, x::T) where T
    (a,b), c = p.transform_parameterization, one(T)
    (a+b)/2 + (b-a)/2*c*x/sqrt(1 + c^2 * x^2)
end
function transform_to_model_space(p::Parameter{T,Exponential}, x::T) where T
    (a,b),c = p.transform_parameterization,one(T)
    a + exp(c*(x-b))
end

transform_to_model_space(pvec::ParameterVector{T}, values::Vector{T}) where T = map(transform_to_model_space, pvec, values)

"""
```
transform_to_real_line(p::Parameter{T,U}, x::T = p.value) where {T<:Number, U<:Transform}
```

Transforms `p.value` from model space (between `p.valuebounds`) to the real line, without updating
`p.value`. The transformations are defined as follows,
where (a,b) = p.transform_parameterization, c a scalar (default=1), and x = p.value:

- Untransformed: x
- SquareRoot:   (1/c)*cx/sqrt(1 - cx^2), where cx =  2 * (x - (a+b)/2)/(b-a)
- Exponential:   a + exp(c*(x-b))
"""
transform_to_real_line(p::Parameter{T,Untransformed}, x::T = p.value) where T = x
function transform_to_real_line(p::Parameter{T,SquareRoot}, x::T = p.value) where T
    (a,b), c = p.transform_parameterization, one(T)
    cx = 2. * (x - (a+b)/2.)/(b-a)
    if cx^2 >1
        println("Parameter is: $(p.key)")
        println("a is $a")
        println("b is $b")
        println("x is $x")
        println("cx is $cx")
        error("invalid paramter value")
    end
    (1/c)*cx/sqrt(1 - cx^2)
end
function transform_to_real_line(p::Parameter{T,Exponential}, x::T = p.value) where T
    (a,b),c = p.transform_parameterization,one(T)
    b + (1 ./ c) * log(x-a)
end

transform_to_real_line(pvec::ParameterVector{T}, values::Vector{T}) where T  = map(transform_to_real_line, pvec, values)
transform_to_real_line(pvec::ParameterVector{T}) where T = map(transform_to_real_line, pvec)


# define operators to work on parameters
Base.convert(::Type{T}, p::UnscaledParameter) where {T <: Number}     = convert(T,p.value)
Base.convert(::Type{T}, p::ScaledParameter) where {T <: Number}       = convert(T,p.scaledvalue)
Base.convert(::Type{T}, p::SteadyStateParameter) where {T <: Number}  = convert(T,p.value)

Base.promote_rule(::Type{AbstractParameter{T}}, ::Type{U}) where {T<:Number, U<:Number} = promote_rule(T,U)

# Define scalar operators on parameters
for op in (:(Base.:+),
           :(Base.:-),
           :(Base.:*),
           :(Base.:/),
           :(Base.:^))

    @eval ($op)(p::UnscaledOrSteadyState, q::UnscaledOrSteadyState) = ($op)(p.value, q.value)
    @eval ($op)(p::UnscaledOrSteadyState, x::Integer)            = ($op)(p.value, x)
    @eval ($op)(p::UnscaledOrSteadyState, x::Number)            = ($op)(p.value, x)
    @eval ($op)(x::Number, p::UnscaledOrSteadyState)            = ($op)(x, p.value)

    @eval ($op)(p::ScaledParameter, q::ScaledParameter) = ($op)(p.scaledvalue, q.scaledvalue)
    @eval ($op)(p::ScaledParameter, x::Integer)            = ($op)(p.scaledvalue, x)
    @eval ($op)(p::ScaledParameter, x::Number)            = ($op)(p.scaledvalue, x)
    @eval ($op)(x::Number, p::ScaledParameter)            = ($op)(x, p.scaledvalue)

    @eval ($op)(p::ScaledParameter, q::UnscaledOrSteadyState) = ($op)(p.scaledvalue, q.value)
    @eval ($op)(p::UnscaledOrSteadyState, q::ScaledParameter) = ($op)(p.value, q.scaledvalue)
end

# Define scalar functional mappings and comparisons
for f in (:(Base.exp),
          :(Base.log),
          :(Base.transpose),
          :(Base.:-),
          :(Base.:<),
          :(Base.:>),
          :(Base.:<=),
          :(Base.:>=))

    @eval ($f)(p::UnscaledOrSteadyState) = ($f)(p.value)
    @eval ($f)(p::ScaledParameter) = ($f)(p.scaledvalue)

    if f != :(Base.:-)
        @eval ($f)(p::UnscaledOrSteadyState, x::Number) = ($f)(p.value, x)
        @eval ($f)(p::ScaledParameter, x::Number) = ($f)(p.scaledvalue, x)
    end
end

# Define scalar operators on grids
for op in (:(Base.:+),
           :(Base.:-),
           :(Base.:*),
           :(Base.:/))

    @eval ($op)(g::SteadyStateParameterGrid, x::Integer)        = ($op)(g.value, x)
    @eval ($op)(g::SteadyStateParameterGrid, x::Number)         = ($op)(g.value, x)
    @eval ($op)(x::Integer, g::SteadyStateParameterGrid)        = ($op)(x, g.value)
    @eval ($op)(x::Number, g::SteadyStateParameterGrid)         = ($op)(x, g.value)
end

# Define vectorized arithmetic for Unscaled or Steady-State Parameters
for op in (:(Base.:+),
           :(Base.:-),
           :(Base.:*),
           :(Base.:/))

    @eval ($op)(p::UnscaledOrSteadyState, x::Vector)        = ($op)(p.value, x)
    @eval ($op)(p::UnscaledOrSteadyState, x::Matrix)        = ($op)(p.value, x)
    @eval ($op)(x::Vector, p::UnscaledOrSteadyState)        = ($op)(x, p.value)
    @eval ($op)(x::Matrix, p::UnscaledOrSteadyState)        = ($op)(x, p.value)
end

"""
```
update!(pvec::ParameterVector{T}, values::Vector{T}) where T
```

Update all parameters in `pvec` that are not fixed with
`values`. Length of `values` must equal length of `pvec`.
Function optimized for speed.
"""
function update!(pvec::ParameterVector{T}, values::Vector{T}) where T
    # this function is optimised for speed
    @assert length(values) == length(pvec) "Length of input vector (=$(length(values))) must match length of parameter vector (=$(length(pvec)))"
    map!(parameter, pvec, pvec, values)
end

"""
```
update(pvec::ParameterVector{T}, values::Vector{T}) where T
```

Returns a copy of `pvec` where non-fixed parameter values are updated
to `values`. `pvec` remains unchanged. Length of `values` must
equal length of `pvec`.

We define the non-mutating version like this because we need the type stability of map!
"""
update(pvec::ParameterVector{T}, values::Vector{T}) where T = update!(copy(pvec), values)

Distributions.pdf(p::AbstractParameter) = exp(logpdf(p))
# we want the unscaled value for ScaledParameters
Distributions.logpdf(p::Parameter{T,U}) where {T, U} = logpdf(get(p.prior),p.value)

# this function is optimised for speed
function Distributions.logpdf(pvec::ParameterVector{T}) where T
	x = zero(T)
	@inbounds for i = 1:length(pvec)
        if hasprior(pvec[i])
    		x += logpdf(pvec[i])
        end
	end
	x
end

# calculate logpdf at new values, without needing to allocate a temporary array with update
function Distributions.logpdf(pvec::ParameterVector{T}, values::Vector{T}) where T
    @assert length(values) == length(pvec) "Length of input vector (=$(length(values))) must match length of parameter vector (=$(length(pvec)))"

    x = zero(T)
    @inbounds for i = 1:length(pvec)
        if hasprior(pvec[i])
            x += logpdf(parameter(pvec[i], values[i]))
        end
    end
    x
end

Distributions.pdf(pvec::ParameterVector{T}) where T  = exp(logpdf(pvec))
Distributions.pdf(pvec::ParameterVector{T}, values::Vector{T}) where T = exp(logpdf(pvec, values))

"""
```
Distributions.rand(p::Vector{AbstractParameter{Float64}})
```

Generate a draw from the prior of each parameter in `p`.
"""
function Distributions.rand(p::Vector{AbstractParameter{Float64}})
    draw = zeros(length(p))
    for (i, para) in enumerate(p)
        draw[i] = if para.fixed
            para.value
        else
            # Resample until all prior draws are within the value bounds
            prio = rand(para.prior.value)
            while !(para.valuebounds[1] < prio < para.valuebounds[2])
                prio = rand(para.prior.value)
            end
            prio
        end
    end
    return draw
end

"""
```
Distributions.rand(p::Vector{AbstractParameter{Float64}}, n::Int)
```

Generate `n` draws from the priors of each parameter in `p`.This returns a matrix of size
`(length(p),n)`, where each column is a sample.
"""
function Distributions.rand(p::Vector{AbstractParameter{Float64}}, n::Int)
    priorsim = zeros(length(p), n)
    for i in 1:n
        priorsim[:, i] = rand(p)
    end
    return priorsim
end

function describe_prior(param::Parameter)
    if param.fixed
        return "fixed at " * string(param.value)

    elseif !param.fixed && !isnull(param.prior)
        (prior_mean, prior_std) = moments(param)

        prior_dist = string(typeof(get(param.prior)))
        prior_dist = replace(prior_dist, "Distributions." => "")
        prior_dist = replace(prior_dist, "DSGE." => "")
        prior_dist = replace(prior_dist, "{Float64}" => "")

        mom1, mom2 = if isa(prior, RootInverseGamma)
            "tau", "nu"
        else
            "mu", "sigma"
        end

        return prior_dist * "(" * mom1 * "=" * string(round(prior_mean, digits=4)) * ", " *
                                  mom2 * "=" * string(round(prior_std, digits=4)) * ")"
    else
        error("Parameter must either be fixed or have non-null prior: " * string(param.key))
    end
end
