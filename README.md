<p align="center">
<img width="450px" src="https://github.com/FRBNY-DSGE/SMC.jl/blob/master/docs/smc_logo_thin_crop.png" alt="SMC.jl"/>
</p>

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://frbny-dsge.github.io/SMC.jl)

# Sequential Monte Carlo

This package implements the Sequential Monte Carlo (SMC) sampling algorithm, an alternative to Metropolis Hastings Markov Chain Monte Carlo sampling for approximating posterior distributions. The SMC algorithm implemented here is based upon Edward Herbst and Frank Schorfheide's paper "[Sequential Monte Carlo Sampling for DSGE Models](http://dx.doi.org/10.1002/jae.2397)" and the code accompanying their book, *Bayesian Estimation of DSGE Models*. More information and the original MATLAB scriptsthat this code replicates can be found at Frank Schorfheide's [website](https://sites.sas.upenn.edu/schorf/pages/bayesian-estimation-dsge-models).

Our implementation features what we term *generalized tempering* for "online" estimation, as outlined in our recent paper, "[Online Estimation of DSGE Models](https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr893.pdf)." For a broad overview of our paper, one may refer to the following *Liberty Street Economics* [article](https://libertystreeteconomics.newyorkfed.org/2019/08/online-estimation-of-dsge-models.html).

Comments and suggestions are welcome, and best submitted as either an issue or a pull request to this branch. :point_up:

Installation
------
To install, run the following commands in Julia 0.7 (support for Julia 1.0 and 1.1 soon!)
```julia
julia>]add ModelConstructors
julia>]add https://github.com/FRBNY-DSGE/SMC.jl.git
```

The package requires our auxiliary package, [ModelConstructors.jl](https://github.com/FRBNY-DSGE/ModelConstructors.jl), which contains useful data structures for creating custom models (e.g. `Parameter`, `State`, `Observable`, `Setting` types).

For examples of how to set up a model in the form SMC can estimate, see scripts in the [`examples/`](https://github.com/FRBNY-DSGE/SMC.jl/tree/master/examples) folder.

Disclaimer
------
Copyright Federal Reserve Bank of New York. You may reproduce, use, modify, make derivative works of, and distribute and this code in whole or in part so long as you keep this notice in the documentation associated with any distributed works. Neither the name of the Federal Reserve Bank of New York (FRBNY) nor the names of any of the authors may be used to endorse or promote works derived from this code without prior written permission. Portions of the code attributed to third parties are subject to applicable third party licenses and rights. By your use of this code you accept this license and any applicable third party license.

THIS CODE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT ANY WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY WARRANTIES OR CONDITIONS OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, EXCEPT TO THE EXTENT THAT THESE DISCLAIMERS ARE HELD TO BE LEGALLY INVALID. FRBNY IS NOT, UNDER ANY CIRCUMSTANCES, LIABLE TO YOU FOR DAMAGES OF ANY KIND ARISING OUT OF OR IN CONNECTION WITH USE OF OR INABILITY TO USE THE CODE, INCLUDING, BUT NOT LIMITED TO DIRECT, INDIRECT, INCIDENTAL, CONSEQUENTIAL, PUNITIVE, SPECIAL OR EXEMPLARY DAMAGES, WHETHER BASED ON BREACH OF CONTRACT, BREACH OF WARRANTY, TORT OR OTHER LEGAL OR EQUITABLE THEORY, EVEN IF FRBNY HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES OR LOSS AND REGARDLESS OF WHETHER SUCH DAMAGES OR LOSS IS FORESEEABLE.
