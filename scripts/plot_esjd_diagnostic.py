"""Plot current and best mutation scales from an SMC ESJD diagnostic CSV.

Usage: python scripts/plot_esjd_diagnostic.py run_esjd_diagnostic.csv -o scales.png
Requires matplotlib.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()

    with args.csv_path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    stages = {}
    for row in rows:
        stages[int(row["stage"])] = row
    if not stages:
        parser.error("the diagnostic CSV has no results")

    stage_numbers = sorted(stages)
    current = [float(stages[s]["current_c"]) for s in stage_numbers]
    best = [float(stages[s]["best_c"]) for s in stage_numbers]
    efficiency = [float(stages[s]["efficiency"]) for s in stage_numbers]

    figure, (scales_axis, efficiency_axis) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 6), constrained_layout=True
    )
    scales_axis.plot(stage_numbers, current, "o-", label="Current c")
    scales_axis.plot(stage_numbers, best, "s-", label="Best grid c")
    scales_axis.set_ylabel("Mutation scale c")
    scales_axis.legend()
    scales_axis.grid(alpha=0.3)

    efficiency_axis.plot(stage_numbers, efficiency, "o-", color="tab:green")
    efficiency_axis.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    efficiency_axis.set_xlabel("Tempering stage")
    efficiency_axis.set_ylabel("ESJD efficiency")
    efficiency_axis.set_ylim(bottom=0)
    efficiency_axis.grid(alpha=0.3)

    output = args.output or args.csv_path.with_name(
        args.csv_path.stem.replace("_esjd_diagnostic", "_esjd_scales") + ".png"
    )
    figure.savefig(output, dpi=180)
    print(output)


if __name__ == "__main__":
    main()
