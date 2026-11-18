"""Compare log marginal likelihood error across the paired benchmark seeds."""

import csv
import math
from pathlib import Path
from statistics import mean, median


HERE = Path(__file__).resolve().parent
paths = [HERE / "results" / "runs.csv", HERE / "results" / "blocks" / "runs.csv"]
by_method = {}
for path in paths:
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            by_method.setdefault(row["method"], {})[int(row["seed"])] = row

baseline = by_method["fixed_2"]
print("method,n,median_abs_error,rmse_log_mdd,mean_proposals,paired_improved")
for method, rows in by_method.items():
    seeds = sorted(set(rows) & set(baseline))
    errors = [float(rows[s]["error"]) for s in seeds]
    abs_errors = [abs(e) for e in errors]
    proposals = [float(rows[s]["proposals"]) for s in seeds]
    improved = sum(abs(float(rows[s]["error"])) < abs(float(baseline[s]["error"]))
                   for s in seeds)
    print(f"{method},{len(seeds)},{median(abs_errors):.3f},"
          f"{math.sqrt(mean(e * e for e in errors)):.3f},{mean(proposals):.0f},"
          f"{improved}/{len(seeds)}")
