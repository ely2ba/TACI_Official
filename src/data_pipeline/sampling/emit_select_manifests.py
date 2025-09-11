#!/usr/bin/env python3
"""
Standalone post-processor to emit versioned manifests from the full manifest.

Inputs
- Reads exactly: data/manifests/full/manifest_full.csv
- Assumes canonical column names:
  Required: uid, SOC, TaskID, occupation_raw, occupation_canonical, task_statement, importance
  Optional (pass-through if present): onet_version, digitally_amenable, modality, modality_confidence

Outputs (created under a date-stamped folder using today's date)
- data/manifests/YYYYMMDD_v1/
  - manifest_v0.csv        (baseline, unfiltered)
  - manifest_mvp.csv       (filtered MVP slice)

Determinism
- Deterministic; no randomness.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np


IN_PATH = Path("data/manifests/full/manifest_full.csv")

# Pilot SOCs (hard-coded)
PILOT_SOCS = ["23-2011.00", "43-4051.00", "13-1031.00"]

# Deterministic, no randomness (coverage-based selection)


def ensure_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in input: {missing}")


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def make_output_dir() -> Path:
    stamp = date.today().strftime("%Y%m%d") + "_v1"
    out_dir = Path("data/manifests") / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def column_order(df: pd.DataFrame) -> list[str]:
    base = [
        "uid",
        "SOC",
        "TaskID",
        "occupation_raw",
        "occupation_canonical",
        "task_statement",
        "importance",
    ]
    optional = ["onet_version", "digitally_amenable", "modality", "modality_confidence"]
    cols = [c for c in base if c in df.columns]
    cols += [c for c in optional if c in df.columns]
    return cols


def build_manifest_v0(df: pd.DataFrame) -> pd.DataFrame:
    # Coerce importance and drop duplicates on key
    df = df.copy()
    df["importance"] = coerce_numeric(df["importance"]) if "importance" in df.columns else pd.to_numeric([])
    df = df.drop_duplicates(subset=["uid", "SOC", "TaskID"], keep="first")

    # Reorder columns to spec (include optional only if present)
    cols = column_order(df)
    out = df.loc[:, cols]

    # Stable sort: SOC, TaskID, uid
    sort_cols = [c for c in ["SOC", "TaskID", "uid"] if c in out.columns]
    out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return out


def per_soc_mvp_selection(g: pd.DataFrame, coverage_target: float, min_k: int, max_k: int):
    """Coverage-based selection per SOC.

    Steps:
    - Drop dupes on (uid,SOC,TaskID)
    - Coerce importance to numeric
    - Eligible = importance notna()
    - Sort by importance desc, uid asc
    - Smallest prefix reaching target coverage; pad to min_k; cap at max_k
    Returns: (selected_df, metrics_dict)
    """
    df = g.copy()
    df["importance"] = coerce_numeric(df["importance"]) if "importance" in df.columns else df.get("importance")
    df = df.drop_duplicates(subset=["uid", "SOC", "TaskID"], keep="first")

    # Eligible rows: non-NaN importance
    elig = df[df["importance"].notna()].copy()
    elig["importance"] = elig["importance"].fillna(0.0)
    # Sorting: importance desc, uid asc
    elig = elig.sort_values(["importance", "uid"], ascending=[False, True], kind="mergesort")

    total_imp = float(elig["importance"].sum()) if len(elig) else 0.0
    threshold = coverage_target * total_imp

    # Take smallest prefix reaching coverage
    cumsum = elig["importance"].cumsum()
    if len(elig) and total_imp > 0:
        idx = int((cumsum >= threshold).idxmax()) if (cumsum >= threshold).any() else elig.index[-1]
        # Compute position (inclusive) within sorted order
        pos = elig.index.get_indexer([idx])[0] + 1
        prefix_n = pos
    else:
        prefix_n = 0

    n_selected = max(prefix_n, min_k)
    if max_k is not None:
        n_selected = min(n_selected, max_k)

    sel = elig.head(n_selected).copy()
    selected_imp = float(sel["importance"].sum()) if len(sel) else 0.0
    achieved = (selected_imp / total_imp) if total_imp > 0 else 0.0

    metrics = {
        "SOC": g["SOC"].iloc[0] if len(g) else None,
        "eligible_count": int(len(elig)),
        "selected_count": int(len(sel)),
        "total_importance": float(total_imp),
        "selected_importance": float(selected_imp),
        "achieved_coverage": float(achieved),
        "capped": bool((max_k is not None) and (len(sel) == max_k) and (achieved < coverage_target)),
    }
    return sel, metrics


def build_manifest_mvp(df: pd.DataFrame, coverage_target: float = 0.80, min_k: int = 5, max_k: int = 25):
    # Filter to pilot SOCs
    m = df[df["SOC"].isin(PILOT_SOCS)].copy()

    # Coerce importance
    m["importance"] = coerce_numeric(m["importance"]) if "importance" in m.columns else m.get("importance")

    # Per-SOC selection and ordering; append in sorted SOC order for determinism
    out_list = []
    metrics_list = []
    for soc in sorted(m["SOC"].dropna().unique().tolist()):
        g = m[m["SOC"] == soc]
        sel, met = per_soc_mvp_selection(g, coverage_target=coverage_target, min_k=min_k, max_k=max_k)
        out_list.append(sel)
        metrics_list.append(met)
    out = pd.concat(out_list, axis=0) if out_list else m.iloc[0:0]

    # Reorder columns mirroring manifest_v0
    cols = column_order(out)
    out = out.loc[:, cols]
    return out.reset_index(drop=True), metrics_list


# Removed: parents_to_decompose emission per PROMPT A


def main() -> None:
    if not IN_PATH.exists():
        raise SystemExit(f"Input file not found: {IN_PATH}")

    df = pd.read_csv(IN_PATH)

    # Validate required columns and coerce importance
    required = ["uid", "SOC", "TaskID", "occupation_raw", "occupation_canonical", "task_statement", "importance"]
    ensure_columns(df, required)
    df["importance"] = coerce_numeric(df["importance"])  # keep NaN

    # Drop exact duplicates on the key
    df = df.drop_duplicates(subset=["uid", "SOC", "TaskID"], keep="first")

    # Build outputs
    out_dir = make_output_dir()
    v0 = build_manifest_v0(df)
    mvp, metrics_list = build_manifest_mvp(df)
    # Write outputs with deterministic settings
    v0_path = out_dir / "manifest_v0.csv"
    mvp_path = out_dir / "manifest_mvp.csv"

    v0.to_csv(v0_path, index=False)
    mvp.to_csv(mvp_path, index=False)

    # Prints (concise summaries)
    print(f"Wrote: {v0_path} rows={len(v0)}")
    print(f"Wrote: {mvp_path} rows={len(mvp)}")
    # Per-SOC MVP selection summary: eligible count, selected count, coverage stats
    if metrics_list:
        # Recompute eligibility after digital gate + non-NaN importance for counts
        for met in metrics_list:
            soc = met.get("SOC")
            eligible_count = met.get("eligible_count", 0)
            selected_count = met.get("selected_count", 0)
            total_imp = met.get("total_importance", 0.0)
            selected_imp = met.get("selected_importance", 0.0)
            achieved = met.get("achieved_coverage", 0.0)
            print(
                f"MVP {soc}: eligible={eligible_count}, selected={selected_count}, "
                f"coverage={achieved:.3f}, total_imp={total_imp:.6f}, selected_imp={selected_imp:.6f}"
            )
            if met.get("capped"):
                print(f"Warning: {soc} capped at max_k with coverage {achieved:.3f} < target")


if __name__ == "__main__":
    main()
