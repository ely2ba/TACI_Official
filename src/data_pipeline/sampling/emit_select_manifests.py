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
- Fixed selection seed: 137
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import date
import pandas as pd
import numpy as np


IN_PATH = Path("data/manifests/full/manifest_full.csv")

# Pilot SOCs (hard-coded)
PILOT_SOCS = ["23-2011.00", "43-4051.00", "51-9061.00"]

# No randomness required for PROMPT A (Top-7 deterministic selection)


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


def per_soc_mvp_selection(g: pd.DataFrame) -> pd.DataFrame:
    """Top-7 by importance (desc), tie-break by uid (asc). No randomness."""
    df = g.copy()
    df["importance"] = coerce_numeric(df["importance"]) if "importance" in df.columns else df.get("importance")
    df = df.drop_duplicates(subset=["uid", "SOC", "TaskID"], keep="first")
    # Sort and take top 7; NaNs in importance go last
    df_sorted = df.sort_values(["importance", "uid"], ascending=[False, True], na_position="last", kind="mergesort")
    return df_sorted.head(7)


def build_manifest_mvp(df: pd.DataFrame) -> pd.DataFrame:
    # Filter to pilot SOCs
    m = df[df["SOC"].isin(PILOT_SOCS)].copy()

    # Optional digitally_amenable gate
    if "digitally_amenable" in m.columns:
        m = m[m["digitally_amenable"] == True].copy()

    # Coerce importance
    m["importance"] = coerce_numeric(m["importance"]) if "importance" in m.columns else m.get("importance")

    # Per-SOC selection and ordering; append in sorted SOC order for determinism
    out_list = []
    for soc in sorted(m["SOC"].dropna().unique().tolist()):
        g = m[m["SOC"] == soc]
        sel = per_soc_mvp_selection(g)
        out_list.append(sel)
    out = pd.concat(out_list, axis=0) if out_list else m.iloc[0:0]

    # Reorder columns mirroring manifest_v0
    cols = column_order(out)
    out = out.loc[:, cols]
    return out.reset_index(drop=True)


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
    mvp = build_manifest_mvp(df)
    # Write outputs with deterministic settings
    v0_path = out_dir / "manifest_v0.csv"
    mvp_path = out_dir / "manifest_mvp.csv"

    v0.to_csv(v0_path, index=False)
    mvp.to_csv(mvp_path, index=False)

    # Prints (concise summaries)
    print(f"Wrote: {v0_path} rows={len(v0)}")
    print(f"Wrote: {mvp_path} rows={len(mvp)}")
    # Per-SOC MVP selection summary (eligible-after-digital and selected counts)
    if len(mvp):
        # Compute eligible counts after the digital gate per SOC
        df_pilot = df[df["SOC"].isin(PILOT_SOCS)].copy()
        if "digitally_amenable" in df_pilot.columns:
            eligible = df_pilot[df_pilot["digitally_amenable"] == True]
        else:
            eligible = df_pilot
        eligible_counts = eligible.groupby("SOC").size().to_dict()

        for soc in sorted(mvp["SOC"].dropna().unique().tolist()):
            selected_count = int((mvp["SOC"] == soc).sum())
            eligible_count = int(eligible_counts.get(soc, 0))
            print(f"MVP {soc}: eligible={eligible_count}, selected={selected_count}")


if __name__ == "__main__":
    main()
