using Documenter, SMC

makedocs(modules = [SMC],
         clean = false,
         format = Documenter.HTML(),
         sitename = "SMC.jl",
         authors = "FRBNY-DSGE",
         linkcheck = false,
         strict = false,
         pages = Any[
                     "Home"                                   => "index.md",
                     "License"                                => "license.md"
         ],
         doctest = false # for now
)

deploydocs(
    repo = "github.com/FRBNY-DSGE/SMC.jl.git",
    target = "build",
    deps = nothing,
    devbranch = "master",
    branch = "gh-pages",
    # versions = "v#",
    # julia = "0.7",
    # osname = "osx",
    make = nothing
)
