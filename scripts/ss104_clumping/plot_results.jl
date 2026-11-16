ENV["GKSwstype"] = "100"
using CSV, DataFrames, Plots

root = joinpath(@__DIR__, "results")
sig = CSV.read(joinpath(root, "acceptance_metrics.csv"), DataFrame)
esjd = CSV.read(joinpath(root, "esjd_metrics.csv"), DataFrame)
all(sig.sweep .== esjd.sweep) || error("Comparison sweeps do not match")

gr()
styles = ((sig, "Bounded sigmoid", :steelblue, :circle),
          (esjd, "ESJD", :darkorange, :diamond))
distinct_plot = plot(; xlabel="Mutation sweep", ylabel="Distinct particles (%)",
    title="Recovery from 35% distinct cloud", ylim=(0, 100), legend=:bottomright)
accept_plot = plot(; xlabel="Mutation sweep", ylabel="Acceptance (%)",
    title="Mutation acceptance", ylim=(0, 100), legend=:topright)
scale_plot = plot(; xlabel="Mutation sweep", ylabel="Proposal scale c",
    title="Chosen scale and diagnostic optimum", legend=:topright)
eff_plot = plot(; xlabel="Mutation sweep", ylabel="Relative ESJD efficiency (%)",
    title="Scale efficiency at fixed SS104 target", ylim=(0, 100), legend=:bottomright)
hline!(eff_plot, [100]; color=:gray40, linestyle=:dash, label="Best tested")
hline!(eff_plot, [90]; color=:gray60, linestyle=:dot, label="90%")
for (table, label, color, marker) in styles
    plot!(distinct_plot, table.sweep, 100 .* table.distinct_share;
          label=label, color=color, marker=marker, linewidth=2)
    plot!(accept_plot, table.sweep, 100 .* table.acceptance;
          label=label, color=color, marker=marker, linewidth=2)
    plot!(scale_plot, table.sweep, table.c;
          label="$label chosen", color=color, marker=marker, linewidth=2)
    plot!(scale_plot, table.sweep, table.diagnostic_best_c;
          label="$label best grid", color=color, linestyle=:dash, linewidth=2)
    plot!(eff_plot, table.sweep, 100 .* table.diagnostic_efficiency;
          label=label, color=color, marker=marker, linewidth=2)
end
figure = plot(distinct_plot, accept_plot, scale_plot, eff_plot;
              layout=(2,2), size=(1250,850), margin=5 * Plots.mm)
path = joinpath(root, "ss104_clumping_comparison.pdf")
savefig(figure, path)
println("Saved $path")
