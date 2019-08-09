# Sequential Monte Carlo
# Herbst-Schorfheide Sequential Monte Carlo Implementation and Replication
[![Build Status](https://travis-ci.org/FRBNY-DSGE/SMC.jl.svg)](https://travis-ci.org/FRBNY-DSGE/SMC.jl)

This package implements Sequential Monte Carlo (SMC) sampling algorithm,
an alternative to Metropolis Hastings Markov Chain Monte Carlo sampling for approximating
posterior distributions. The SMC algorithm implemented here is based upon Edward Herbst and Frank
Schorfheide's paper ["Sequential Monte Carlo Sampling for DSGE
Models"](http://dx.doi.org/10.1002/jae.2397) and the code accompanying
their book *Bayesian Estimation of DSGE Models*. Our implementation features
what we term *generalized tempering* for "online" estimation, as outlined in our recent paper, ["Online Estimation of DSGE Models"](https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr893.pdf).

More information and the original MATLAB scripts that this code replicates can be found at
Frank Schorfheide's [website](https://sites.sas.upenn.edu/schorf/pages/bayesian-estimation-dsge-models).

Comments and suggestions are welcome, and best submitted as
either an issue or a pull request to this branch.

## Background

For further reading on the inn

The *DSGE.jl* package implements the New York Fed DSGE model and provides
general code to estimate many user-specified DSGE models. The package is
introduced in the Liberty Street Economics blog post
[The FRBNY DSGE Model Meets Julia](http://libertystreeteconomics.newyorkfed.org/2015/12/the-frbny-dsge-model-meets-julia.html).
(We previously referred to our model as the "FRBNY DSGE Model".)

This Julia-language implementation mirrors the MATLAB code included in the
Liberty Street Economics blog post
[The FRBNY DSGE Model Forecast](http://libertystreeteconomics.newyorkfed.org/2015/05/the-frbny-dsge-model-forecast-april-2015.html).

For the latest documentation on the *code*, click on the docs|latest button
above. Documentation for the most recent *model version* is available
[here](https://github.com/FRBNY-DSGE/DSGE.jl/blob/master/docs/DSGE_Model_Documentation_1002.pdf).

This Julia-language implementation mirrors the MATLAB code included in
the Liberty Street Economics blog post [The FRBNY DSGE Model
Forecast](http://libertystreeteconomics.newyorkfed.org/2015/05/the-frbny-dsge-model-forecast-april-2015.html).