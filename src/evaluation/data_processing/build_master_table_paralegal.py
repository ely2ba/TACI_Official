#!/usr/bin/env python3
"""
build_master_table_paralegal.py
Merges Phase-00→03 results for the paralegal slice and writes
graders/phase_00_to_03_condenser/master_per_output_paralegal.csv
"""
from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "evaluation"

# ── per-phase inputs (from evaluation directories) ──────────────────────
WRAP   = EVAL_DIR/"phase_00_wrapper/wrapper_per_output_paralegal.csv"
SCHEMA = EVAL_DIR/"phase_01_schema/schema_text_gui_per_output_paralegal.csv"
SAFETY = EVAL_DIR/"phase_02_safety/phase_02_safety_per_output_paralegal.csv"
RUBRIC = EVAL_DIR/"phase_03_rubric/phase_03_rubric_per_output_paralegal.csv"

# Catalogs from legacy graders location (static reference data)
MANUAL_CAT = ROOT/"outputs/results/catalogs/manual_stub_catalog.csv"
VISION_CAT = ROOT/"outputs/results/catalogs/vision_stub_catalog.csv"

MANIFEST   = ROOT/"data/manifests/paralegal_tasks.csv"
# Output to current data_processing directory
OUT_CSV    = Path(__file__).parent/"master_per_output_paralegal.csv"

# ── helpers ────────────────────────────────────────────────────────────
def safe_merge(df, csv_path, cols):
    if csv_path.exists():
        add = pd.read_csv(csv_path, dtype=str)[cols]
        return df.merge(add, on=["uid","variant","temp","file"], how="left")
    for c in cols:
        if c not in ("uid","variant","temp","file"):
            df[c] = np.nan
    return df

def main():
    # eligible uid set (paralegal only) ---------------------------------
    uid_ok = set(pd.read_csv(MANIFEST, usecols=["uid"], dtype=str)["uid"])

    # Phase-0 wrapper ---------------------------------------------------
    wrap = pd.read_csv(WRAP, dtype=str)
    wrap = wrap.rename(columns={"temperature":"temp"}) if "temperature" in wrap and "temp" not in wrap else wrap
    wrap["wrapper_strict"]  = wrap["strict"].astype(float)
    wrap["wrapper_rescued"] = wrap["rescued"].astype(float)
    base = wrap[["uid","variant","temp","model","modality","file",
                 "wrapper_strict","wrapper_rescued"]]

    # Phase-1 schema (TEXT/GUI) ----------------------------------------
    base = safe_merge(base, SCHEMA,
        ["uid","variant","temp","file","strict","rescued"])\
        .rename(columns={"strict":"schema_strict","rescued":"schema_rescued"})

    # IoU placeholder (no vision scored) -------------------------------
    base["iou"] = np.nan

    # Phase-2 safety ----------------------------------------------------
    base = safe_merge(base, SAFETY,
        ["uid","variant","temp","file","strict_safety","rescued_safety"])\
        .rename(columns={"strict_safety":"safety_strict",
                         "rescued_safety":"safety_rescued"})

    # Phase-3 rubric ----------------------------------------------------
    base = safe_merge(base, RUBRIC,
        ["uid","variant","temp","file","rubric_score"])

    # Manual stub catalogue (in-uid filter) ----------------------------
    if MANUAL_CAT.exists():
        man = pd.read_csv(MANUAL_CAT, dtype=str)
        man = man[man["uid"].isin(uid_ok)]          # keep only paralegal uids
        if len(man):
            man["rubric_score"] = np.nan
            man["is_manual_stub"] = "1"
            base = pd.concat([base, man[base.columns]], ignore_index=True)

    # Vision stub catalogue (non-vision models, filtered) --------------
    if VISION_CAT.exists():
        vs = pd.read_csv(VISION_CAT, dtype=str)
        vs = vs[vs["uid"].isin(uid_ok)]
        if len(vs):
            for col,val in {"wrapper_strict":0,"wrapper_rescued":0,
                            "schema_strict":0,"schema_rescued":0,
                            "safety_strict":1,"safety_rescued":1,
                            "rubric_score":np.nan,"iou":0,
                            "is_manual_stub":"0"}.items():
                vs[col] = vs.get(col, val)
            ks = ["uid","variant","temp","model"]
            vs = vs[~vs.set_index(ks).index.isin(base.set_index(ks).index)]
            base = pd.concat([base, vs[base.columns]], ignore_index=True)

    # Occupation label --------------------------------------------------
    occ = pd.read_csv(MANIFEST, usecols=["uid","OccTitleRaw"], dtype=str)\
            .rename(columns={"OccTitleRaw":"occupation"})
    base = base.merge(occ, on="uid", how="left")

    # Numeric coercion --------------------------------------------------
    for g in ["wrapper_strict","wrapper_rescued",
              "schema_strict","schema_rescued",
              "safety_strict","safety_rescued"]:
        base[g] = pd.to_numeric(base[g], errors="coerce").fillna(0)
    base["rubric_score"] = pd.to_numeric(base["rubric_score"], errors="coerce")
    base["iou"]          = pd.to_numeric(base["iou"], errors="coerce")
    if "is_manual_stub" not in base.columns:
        base["is_manual_stub"] = "0"

    # Save --------------------------------------------------------------
    cols = ["uid","occupation","variant","temp","model","modality","file",
            "wrapper_strict","wrapper_rescued",
            "schema_strict","schema_rescued",
            "safety_strict","safety_rescued",
            "rubric_score","iou","is_manual_stub"]
    base[cols].to_csv(OUT_CSV, index=False)
    print(f"✓ master table ({len(base):,} rows) → {OUT_CSV}")

if __name__ == "__main__":
    main()
