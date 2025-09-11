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
  - parents_to_decompose.csv  (one column: parent_uid)

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
PILOT_SOCS = ["23-2011.00"]

# Fixed seed for deterministic behavior
SEED = 137


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
    # Work on a copy; g is already filtered to a single SOC
    df = g.copy()

    # Coerce and drop duplicates again locally (safety)
    df["importance"] = coerce_numeric(df["importance"]) if "importance" in df.columns else df.get("importance")
    df = df.drop_duplicates(subset=["uid", "SOC", "TaskID"], keep="first")

    # Exclude NaN from ranking/quartiles
    df_rankable = df[df["importance"].notna()].copy()

    # Top-K by importance
    K = 5
    topk = (
        df_rankable.sort_values(["importance", "uid"], ascending=[False, True], kind="mergesort")
        .head(K)
    )

    # Quartile labels (Q1 lowest .. Q4 highest) on rankable subset only
    # Handle edge case of fewer unique values than bins with duplicates='drop'
    try:
        q = pd.qcut(df_rankable["importance"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        df_rankable = df_rankable.assign(_quartile=q)
    except ValueError:
        # Not enough unique values to form quartiles → fall back to single-bin Q4
        df_rankable = df_rankable.assign(_quartile="Q4")

    # Sample N=2 from each quartile without replacement; deterministic
    samples = []
    for label in ["Q1", "Q2", "Q3", "Q4"]:
        qdf = df_rankable[df_rankable["_quartile"].astype(str) == label]
        if len(qdf) == 0:
            continue
        n = min(2, len(qdf))
        # Use a fixed random_state for determinism
        s = qdf.sample(n=n, random_state=SEED, replace=False)
        samples.append(s)
    sampled = pd.concat(samples, axis=0) if samples else df_rankable.iloc[0:0]

    # Union sets by uid
    sel_uids = pd.Index(topk["uid"]).union(sampled["uid"]) if (len(topk) or len(sampled)) else pd.Index([])
    sel = df[df["uid"].isin(sel_uids)].copy()

    # Stable order per spec:
    # 1) top-K block first (desc importance, tie uid asc)
    # 2) quartile samples in order Q4→Q3→Q2→Q1 (desc importance in each, tie uid asc)
    top_block = (
        sel[sel["uid"].isin(set(topk["uid"]))]
        .sort_values(["importance", "uid"], ascending=[False, True], kind="mergesort")
    )

    # Merge quartile labels onto sel for ordering
    q_map = df_rankable.set_index("uid")["_quartile"].to_dict()
    sel["_quartile"] = sel["uid"].map(q_map)

    quartile_blocks = []
    for label in ["Q4", "Q3", "Q2", "Q1"]:
        qb = sel[(sel["uid"].isin(set(sampled["uid"])) & sel["_quartile"].astype(str).eq(label))]
        if len(qb):
            qb = qb.sort_values(["importance", "uid"], ascending=[False, True], kind="mergesort")
            quartile_blocks.append(qb)

    ordered = pd.concat([top_block] + quartile_blocks, axis=0) if (len(top_block) or quartile_blocks) else sel.iloc[0:0]

    # De-duplicate by uid while preserving order
    ordered = ordered.drop_duplicates(subset=["uid"], keep="first")
    return ordered


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


def build_parents_to_decompose(mvp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for soc in sorted(mvp["SOC"].dropna().unique().tolist()):
        g = mvp[mvp["SOC"] == soc].copy()
        if len(g) == 0:
            continue
        # Top-1 by importance; ties broken by uid asc; NaN go last
        g = g.copy()
        g["importance"] = coerce_numeric(g["importance"]) if "importance" in g.columns else g.get("importance")
        g = g.sort_values(["importance", "uid"], ascending=[False, True], na_position="last", kind="mergesort")
        best_uid = g["uid"].iloc[0]
        rows.append({"parent_uid": best_uid, "SOC": soc, "_imp": g["importance"].iloc[0]})

    if not rows:
        return pd.DataFrame(columns=["parent_uid"])  # empty

    out = pd.DataFrame(rows)
    # Keep order grouped by SOC (sorted by SOC), then descending importance
    out = out.sort_values(["SOC", "_imp", "parent_uid"], ascending=[True, False, True], na_position="last", kind="mergesort")
    return out[["parent_uid"]].reset_index(drop=True)


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
    parents = build_parents_to_decompose(mvp)

    # Write outputs with deterministic settings
    v0_path = out_dir / "manifest_v0.csv"
    mvp_path = out_dir / "manifest_mvp.csv"
    parents_path = out_dir / "parents_to_decompose.csv"

    v0.to_csv(v0_path, index=False)
    mvp.to_csv(mvp_path, index=False)
    parents.to_csv(parents_path, index=False)

    # Prints (concise summaries)
    print(f"Wrote: {v0_path} rows={len(v0)}")
    print(f"Wrote: {mvp_path} rows={len(mvp)}")
    print(f"Wrote: {parents_path} rows={len(parents)}")

    # Per-SOC MVP selection summary
    if len(mvp):
        # Build quartile labels on the subset with importance notna for summary
        mvp_imp = mvp[mvp["importance"].notna()].copy()
        if len(mvp_imp):
            try:
                mvp_imp["_quartile"] = pd.qcut(mvp_imp["importance"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
            except ValueError:
                mvp_imp["_quartile"] = "Q4"

        for soc in sorted(mvp["SOC"].dropna().unique().tolist()):
            g = mvp[mvp["SOC"] == soc]
            total = len(g)
            # Approximate top-K count by comparing to top-K on full group
            orig_group = df[df["SOC"] == soc].copy()
            orig_group = orig_group[orig_group["importance"].notna()].copy()
            topk_ref = (
                orig_group.sort_values(["importance", "uid"], ascending=[False, True], kind="mergesort").head(5)
            )
            topk_count = int(g["uid"].isin(set(topk_ref["uid"])) .sum())
            q_counts = {q: int(mvp_imp[(mvp_imp["SOC"] == soc) & (mvp_imp["_quartile"].astype(str) == q)]["uid"].nunique()) for q in ["Q1","Q2","Q3","Q4"]}
            print(f"MVP {soc}: total={total}, topK={topk_count}, quartiles={q_counts}")

        # Parents summary
        for soc, row in zip(sorted(mvp["SOC"].dropna().unique().tolist()), parents.itertuples(index=False)):
            print(f"Parent for {soc}: {row.parent_uid}")


if __name__ == "__main__":
    main()
