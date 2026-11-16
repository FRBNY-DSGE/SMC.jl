# Plot both completed An–Schorfheide scale diagnostics in one PDF.
# With no arguments, use the sole diagnostic CSV in each results directory.
# To select particular runs, pass sigmoid.csv esjd.csv output.pdf or set
# SMC_PLOT_SIGMOID_CSV, SMC_PLOT_ESJD_CSV, and SMC_PLOT_OUTPUT.

ENV["GKSwstype"] = "100"  # headless GR output on compute nodes
using CSV, Plots

function default_csv(method::String)
    directory = joinpath(@__DIR__, "results", method)
    isdir(directory) || error("No results directory: $directory")
    files = filter(f -> endswith(f, "_esjd_diagnostic.csv"), readdir(directory; join = true))
    length(files) == 1 || error("Expected one diagnostic CSV in $directory; found $(length(files)). " *
                                "Set SMC_PLOT_$(method == "esjd" ? "ESJD" : "SIGMOID")_CSV to select a run.")
    return only(files)
end

function read_diagnostic(path::String)
    isfile(path) || error("Diagnostic CSV not found: $path")
    stages = Dict{Int, Vector{NamedTuple}}()
    for row in CSV.File(path)
        entry = (phi = Float64(row.phi), trial_c = Float64(row.trial_c),
                 esjd = Float64(row.esjd), current_c = Float64(row.current_c),
                 current = lowercase(string(row.is_current)) == "true")
        all(isfinite, (entry.phi, entry.trial_c, entry.esjd, entry.current_c)) ||
            error("Nonfinite diagnostic value at stage $(row.stage) in $path")
        push!(get!(stages, Int(row.stage), NamedTuple[]), entry)
    end
    isempty(stages) && error("No diagnostic stages in $path")

    points = NamedTuple[]
    for stage in sort!(collect(keys(stages)))
        trials = stages[stage]
        current = filter(t -> t.current, trials)
        length(current) == 1 || error("Expected exactly one current scale at stage $stage in $path")
        all(t -> isapprox(t.phi, current[1].phi; atol = 1e-12), trials) ||
            error("Tempering parameter differs within stage $stage in $path")
        best = trials[argmax(getproperty.(trials, :esjd))]
        best.esjd > 0 || error("Best ESJD is not positive at stage $stage in $path")
        efficiency = 100 * current[1].esjd / best.esjd
        0 <= efficiency <= 100 + 1e-8 || error("Invalid efficiency at stage $stage in $path")
        push!(points, (phi = current[1].phi, efficiency = min(efficiency, 100.0),
                       current_c = current[1].current_c, best_c = best.trial_c))
    end
    sort!(points; by = p -> p.phi)
    return points
end

function main()
    length(ARGS) in (0, 3) || error("Usage: plot_esjd_comparison.jl [sigmoid.csv esjd.csv output.pdf]")
    sigmoid_csv = length(ARGS) == 3 ? ARGS[1] :
        (haskey(ENV, "SMC_PLOT_SIGMOID_CSV") ? ENV["SMC_PLOT_SIGMOID_CSV"] : default_csv("bounded_sigmoid"))
    esjd_csv = length(ARGS) == 3 ? ARGS[2] :
        (haskey(ENV, "SMC_PLOT_ESJD_CSV") ? ENV["SMC_PLOT_ESJD_CSV"] : default_csv("esjd"))
    output = length(ARGS) == 3 ? ARGS[3] :
        get(ENV, "SMC_PLOT_OUTPUT", joinpath(@__DIR__, "esjd_scaler_comparison.pdf"))
    endswith(lowercase(output), ".pdf") || error("Output must be a PDF file")

    sigmoid = read_diagnostic(sigmoid_csv)
    esjd = read_diagnostic(esjd_csv)
    gr()
    top = plot(; xlims = (0, 1), ylims = (0, 100),
               ylabel = "Relative ESJD efficiency (%)",
               title = "Mutation scale efficiency across tempering",
               legend = :bottomright, gridalpha = 0.25)
    plot!(top, getproperty.(sigmoid, :phi), getproperty.(sigmoid, :efficiency);
          label = "Bounded sigmoid", color = :steelblue, marker = :circle, markersize = 3)
    plot!(top, getproperty.(esjd, :phi), getproperty.(esjd, :efficiency);
          label = "ESJD adaptation", color = :darkorange, marker = :diamond, markersize = 3)
    hline!(top, [100]; label = "Best tested scale (100%)", color = :gray40,
           linestyle = :dash, linewidth = 1.5)
    hline!(top, [90]; label = "Near optimal (90%)", color = :gray60,
           linestyle = :dash, linewidth = 1.5)

    bottom = plot(; xlims = (0, 1), xlabel = "Tempering parameter ϕ",
                  ylabel = "Mutation scale c", legend = :topright, gridalpha = 0.25)
    plot!(bottom, getproperty.(sigmoid, :phi), getproperty.(sigmoid, :current_c);
          label = "Sigmoid current", color = :steelblue, linewidth = 2)
    plot!(bottom, getproperty.(sigmoid, :phi), getproperty.(sigmoid, :best_c);
          label = "Sigmoid best grid", color = :steelblue, linestyle = :dash, linewidth = 2)
    plot!(bottom, getproperty.(esjd, :phi), getproperty.(esjd, :current_c);
          label = "ESJD current", color = :darkorange, linewidth = 2)
    plot!(bottom, getproperty.(esjd, :phi), getproperty.(esjd, :best_c);
          label = "ESJD best grid", color = :darkorange, linestyle = :dash, linewidth = 2)

    figure = plot(top, bottom; layout = (2, 1), size = (1050, 850),
                  margin = 7 * Plots.mm)
    mkpath(dirname(output))
    savefig(figure, output)
    println("Saved $output")
    println("Bounded sigmoid: $(length(sigmoid)) checked stages; ESJD: $(length(esjd)) checked stages")
    println("Input CSVs: $sigmoid_csv; $esjd_csv")
end

main()
