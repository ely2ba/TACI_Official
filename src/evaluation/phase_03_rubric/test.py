#!/usr/bin/env python3
"""
Model-level matrix (Paralegal slice)

Rows  : models
Cols  :
    wrapper_strict_pass, wrapper_rescued_pass, wrapper_fail
    schema_strict_pass , schema_rescued_pass , schema_fail
    safety_mean
    accuracy_mean, coverage_mean, depth_mean,
    style_mean, utility_mean, specificity_mean
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ───────── paths ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]

MASTER = ROOT / "outputs/results/evaluated" / "master_per_output_paralegal.csv"
RUBRIC = ROOT / "outputs/results/evaluated" / "phase_03_rubric_per_output_paralegal.csv"

OUT   = ROOT / "outputs/results/evaluated" / "model_grader_matrix_paralegal.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

AXES = ["accuracy", "coverage", "depth", "style", "utility", "specificity"]

# ───────── load master (phase-0/1/2 metrics) ─────────────────────────
m = pd.read_csv(MASTER, dtype={"model": str})

# wrapper counts -------------------------------------------------------
wrap_counts = (
    m.assign(
        wrapper_strict_pass  = (m.wrapper_strict  == 1).astype(int),
        wrapper_rescued_pass = (m.wrapper_rescued == 1).astype(int),
        wrapper_fail         = (~((m.wrapper_strict == 1) |
                                  (m.wrapper_rescued == 1))).astype(int),
    )
    .groupby("model")[["wrapper_strict_pass",
                       "wrapper_rescued_pass",
                       "wrapper_fail"]]
    .sum()
    .reset_index()
)

# schema counts --------------------------------------------------------
schema_counts = (
    m.assign(
        schema_strict_pass  = (m.schema_strict  == 1).astype(int),
        schema_rescued_pass = (m.schema_rescued == 1).astype(int),
        schema_fail         = (~((m.schema_strict == 1) |
                                 (m.schema_rescued == 1))).astype(int),
    )
    .groupby("model")[["schema_strict_pass",
                       "schema_rescued_pass",
                       "schema_fail"]]
    .sum()
    .reset_index()
)

# safety mean ----------------------------------------------------------
def pick_safety(row):
    if not np.isnan(row.safety_strict):   # strict pass path
        return row.safety_strict
    if not np.isnan(row.safety_rescued):  # rescued path
        return row.safety_rescued
    return np.nan

m["safety_use"] = m.apply(pick_safety, axis=1)

safety_mean = (
    m.groupby("model").safety_use
      .mean()
      .round(3)
      .rename("safety_mean")
      .reset_index()
)

# ───────── load rubric axes (phase-3) ─────────────────────────────────
r_axis = (
    pd.read_csv(RUBRIC, usecols=["model", *AXES])
      .groupby("model")[AXES]
      .mean()
      .add_suffix("_mean")
      .round(2)
      .reset_index()
)

# ───────── merge all parts ────────────────────────────────────────────
matrix = wrap_counts.merge(schema_counts, on="model", how="outer") \
                    .merge(safety_mean,  on="model", how="outer") \
                    .merge(r_axis,       on="model", how="outer")

matrix.to_csv(OUT, index=False)
print(f"✓ wrote {OUT}")
