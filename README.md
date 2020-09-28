<p align="center">
<img width="450px" src="https://github.com/FRBNY-DSGE/SMC.jl/blob/master/docs/smc_logo_thin_crop.png" alt="SMC.jl"/>
</p>

[![Build Status](https://travis-ci.com/FRBNY-DSGE/SMC.jl.svg?branch=master)](https://travis-ci.com/FRBNY-DSGE/SMC.jl)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://frbny-dsge.github.io/SMC.jl/latest)
[![Coverage Status](https://coveralls.io/repos/github/FRBNY-DSGE/SMC.jl/badge.svg?branch=master)](https://coveralls.io/github/FRBNY-DSGE/SMC.jl?branch=master)

# Sequential Monte Carlo

This package implements the Sequential Monte Carlo (SMC) sampling algorithm, an alternative to Metropolis Hastings Markov Chain Monte Carlo sampling for approximating posterior distributions. The SMC algorithm implemented here is based upon Edward Herbst and Frank Schorfheide's paper "[Sequential Monte Carlo Sampling for DSGE Models](http://dx.doi.org/10.1002/jae.2397)" and the code accompanying their book, *Bayesian Estimation of DSGE Models*. More information and the original MATLAB scripts from which this code was derived can be found at Frank Schorfheide's [website](https://sites.sas.upenn.edu/schorf/pages/bayesian-estimation-dsge-models).

Our implementation features what we term *generalized tempering* for "online" estimation, as outlined in our recent paper, "[Online Estimation of DSGE Models](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3426004)." For a broad overview of the algorithm, one may refer to the following *Liberty Street Economics* [article](https://libertystreeteconomics.newyorkfed.org/2019/08/online-estimation-of-dsge-models.html).

Comments and suggestions are welcome, and best submitted as either an issue or a pull request. :point_up:

## Installation and Versioning

`SMC.jl` is a registered Julia package in the [`General`](https://github.com/JuliaRegistries/General) registry, compatible with Julia `v1.x`. To install it, open your Julia REPL, type `]` to enter the package manager, and run

```julia
pkg> add SMC
```

## Usage

The package requires our auxiliary package, [ModelConstructors.jl](https://github.com/FRBNY-DSGE/ModelConstructors.jl), which contains useful data structures for creating custom models (e.g. `Parameter`, `State`, `Observable`, `Setting` types).

For examples of how to set up a model in the form SMC can estimate, see scripts in the [`examples/`](https://github.com/FRBNY-DSGE/SMC.jl/tree/master/examples) folder.

## Precompilation

The `SMC.jl` package is not precompiled by default because when running code in parallel, we want to re-compile
the copy of `SMC.jl` on each processor to guarantee the right version of the code is being used. If users do not
anticipate using parallelism, then users ought to change the first line of `src/SMC.jl` from

```
isdefined(Base, :__precompile__) && __precompile__(false)
```

to

```
isdefined(Base, :__precompile__) && __precompile__(true)
```

## Disclaimer

Copyright Federal Reserve Bank of New York. You may reproduce, use, modify, make derivative works of, and distribute and this code in whole or in part so long as you keep this notice in the documentation associated with any distributed works. Neither the name of the Federal Reserve Bank of New York (FRBNY) nor the names of any of the authors may be used to endorse or promote works derived from this code without prior written permission. Portions of the code attributed to third parties are subject to applicable third party licenses and rights. By your use of this code you accept this license and any applicable third party license.

THIS CODE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT ANY WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY WARRANTIES OR CONDITIONS OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, EXCEPT TO THE EXTENT THAT THESE DISCLAIMERS ARE HELD TO BE LEGALLY INVALID. FRBNY IS NOT, UNDER ANY CIRCUMSTANCES, LIABLE TO YOU FOR DAMAGES OF ANY KIND ARISING OUT OF OR IN CONNECTION WITH USE OF OR INABILITY TO USE THE CODE, INCLUDING, BUT NOT LIMITED TO DIRECT, INDIRECT, INCIDENTAL, CONSEQUENTIAL, PUNITIVE, SPECIAL OR EXEMPLARY DAMAGES, WHETHER BASED ON BREACH OF CONTRACT, BREACH OF WARRANTY, TORT OR OTHER LEGAL OR EQUITABLE THEORY, EVEN IF FRBNY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES OR LOSS AND REGARDLESS OF WHETHER SUCH DAMAGES OR LOSS IS FORESEEABLE.
