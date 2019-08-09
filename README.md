# Sequential Monte Carlo

This package implements the Sequential Monte Carlo (SMC) sampling algorithm,
an alternative to Metropolis Hastings Markov Chain Monte Carlo sampling for approximating
posterior distributions. The SMC algorithm implemented here is based upon Edward Herbst and Frank
Schorfheide's paper ["Sequential Monte Carlo Sampling for DSGE
Models"](http://dx.doi.org/10.1002/jae.2397) and the code accompanying
their book *Bayesian Estimation of DSGE Models*. Our implementation features
what we term *generalized tempering* for "online" estimation, as outlined in our recent paper, ["Online Estimation of DSGE Models"](https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr893.pdf).

More information and the original MATLAB scripts that this code replicates can be found at
Frank Schorfheide's [website](https://sites.sas.upenn.edu/schorf/pages/bayesian-estimation-dsge-models).

Comments and suggestions are welcome, and best submitted as
either an issue or a pull request to this branch. :point_up: