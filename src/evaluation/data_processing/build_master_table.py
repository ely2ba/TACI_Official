#!/usr/bin/env python3
"""
build_master_table.py   ·   Phase-00 → 03 condenser   (2025-06-12)

* Merges per-output CSVs from phases 0-3
* Adds manual-stub catalogue
* Adds vision-stub catalogue (for non-vision models)
* Writes  graders/phase_00_to_03_condenser/master_per_output.csv
"""

from pathlib import Path
import pandas as pd, numpy as np
import os

# ── repo paths ───────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
EVAL_DIR      = ROOT / "evaluation"

# Input from evaluation directories (where scripts now output)
WRAPPER_CSV   = EVAL_DIR / "phase_00_wrapper/wrapper_per_output.csv"
SCHEMA_CSV    = EVAL_DIR / "phase_01_schema/schema_text_gui_per_output.csv"
VISION_CSV    = EVAL_DIR / "phase_01_schema/vision_per_output.csv"
SAFETY_CSV    = EVAL_DIR / "phase_02_safety/phase_02_safety_per_output.csv"
RUBRIC_CSV    = EVAL_DIR / "phase_03_rubric/phase_03_rubric_per_output.csv"

# Catalogs from legacy graders location (static reference data)
MANUAL_CAT    = ROOT / "outputs/results/catalogs/manual_stub_catalog.csv"
VISION_CAT    = ROOT / "outputs/results/catalogs/vision_stub_catalog.csv"

# Manifest from data directory  
MANIFEST_CSV  = ROOT / "data/manifests/sampled_tasks_with_modality.csv"
# Output to current data_processing directory
OUT_CSV       = Path(__file__).parent / "master_per_output.csv"

# ── helper ───────────────────────────────────────────────────────────────
def safe_merge(base: pd.DataFrame, csv_path: Path, cols, how="left"):
    """Merge cols from csv_path if present; otherwise inject NaNs."""
    if csv_path.exists():
        add = pd.read_csv(csv_path, dtype=str)[cols]
        return base.merge(add, on=["uid","variant","temp","file"], how=how)
    for c in cols:
        if c not in ("uid","variant","temp","file"):
            base[c] = np.nan
    return base

# ── main ─────────────────────────────────────────────────────────────────
def main() -> None:
    # 1 ▸ Phase-0 wrapper --------------------------------------------------
    wrap = pd.read_csv(WRAPPER_CSV, dtype=str)
    if "temperature" in wrap.columns and "temp" not in wrap.columns:
        wrap = wrap.rename(columns={"temperature": "temp"})
    if "variant" not in wrap.columns:
        wrap["variant"] = "0"

    wrap["wrapper_strict"]  = wrap["strict"].astype(float)
    wrap["wrapper_rescued"] = wrap["rescued"].astype(float)

    base = wrap[["uid","variant","temp","model","modality","file",
                 "wrapper_strict","wrapper_rescued"]].copy()

    # 2 ▸ Phase-1 TEXT/GUI schema ----------------------------------------
    base = safe_merge(
        base, SCHEMA_CSV,
        ["uid","variant","temp","file","strict","rescued"]
    ).rename(columns={"strict":"schema_strict","rescued":"schema_rescued"})

    # 3 ▸ Phase-1 VISION schema + IoU ------------------------------------
    if VISION_CSV.exists():
        cols = ["uid","variant","temp","file",
                "strict","rescued",
                "iou_score" if "iou_score" in pd.read_csv(VISION_CSV,nrows=0).columns else "iou"]
        vis = (pd.read_csv(VISION_CSV, dtype=str)[cols]
                 .rename(columns={"strict":"schema_strict",
                                  "rescued":"schema_rescued",
                                  cols[-1]:"iou"}))
        base = (pd.concat([base, vis], ignore_index=True)
                  .drop_duplicates(["uid","variant","temp","file"], keep="first"))
    else:
        base["iou"] = np.nan

    # 4 ▸ Phase-2 safety --------------------------------------------------
    base = safe_merge(
        base, SAFETY_CSV,
        ["uid","variant","temp","file","strict_safety","rescued_safety"]
    ).rename(columns={"strict_safety":"safety_strict",
                      "rescued_safety":"safety_rescued"})

    # 5 ▸ Phase-3 rubric --------------------------------------------------
    base = safe_merge(
        base, RUBRIC_CSV,
        ["uid","variant","temp","file","rubric_score"]
    )

    # 6 ▸ manual-stub catalogue ------------------------------------------
    if MANUAL_CAT.exists():
        man = pd.read_csv(MANUAL_CAT, dtype=str)
        man["rubric_score"] = np.nan
        man["is_manual_stub"] = "1"
        base = pd.concat([base, man], ignore_index=True)

    # 7 ▸ vision-stub catalogue (non-vision models) -----------------------
    if VISION_CAT.exists():
        vs = pd.read_csv(VISION_CAT, dtype=str)
        # add default grading columns
        defaults = {
            "wrapper_strict":0,"wrapper_rescued":0,
            "schema_strict":0,"schema_rescued":0,
            "safety_strict":1,"safety_rescued":1,
            "rubric_score":np.nan,
            "iou":0,
            "strict_reason":"missing",
            "rescued_reason":"missing",
            "iou_reason":"missing",
            "is_manual_stub":"0"
        }
        for col,val in defaults.items():
            if col not in vs.columns:
                vs[col] = val
        # keep only rows not already in base
        key_cols = ["uid","variant","temp","model"]
        mask_new = ~vs.set_index(key_cols).index.isin(base.set_index(key_cols).index)
        new_rows = vs.loc[mask_new, base.columns]   # ensure identical column order
        base = pd.concat([base, new_rows], ignore_index=True)
        if len(new_rows):
            print(f"✓ merged {len(new_rows):,} vision-stub rows")

    # 8 ▸ occupation label -----------------------------------------------
    mani = pd.read_csv(MANIFEST_CSV, dtype=str)[["uid","OccTitleRaw"]]
    base = base.merge(mani.rename(columns={"OccTitleRaw":"occupation"}),
                      on="uid", how="left")

    # 9 ▸ numeric coercion ------------------------------------------------
    gates = ["wrapper_strict","wrapper_rescued",
             "schema_strict","schema_rescued",
             "safety_strict","safety_rescued"]
    for c in gates:
        base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0)

    base["rubric_score"] = pd.to_numeric(base["rubric_score"], errors="coerce")
    base["iou"]          = pd.to_numeric(base["iou"], errors="coerce")

    # 10 ▸ column order & save -------------------------------------------
    COLS = ["uid","occupation","variant","temp","model","modality","file",
            "wrapper_strict","wrapper_rescued",
            "schema_strict","schema_rescued",
            "safety_strict","safety_rescued",
            "rubric_score","iou","is_manual_stub"]
    base = base[COLS]

    # 11 ▸ append GUI stubs for skipped prompts (no selectors) -----------
    skip_path = ROOT / "outputs/prompts/skip_log.csv"
    if os.path.exists(skip_path):
        skip = pd.read_csv(skip_path, dtype=str)
        skip = skip[skip["reason"].isin(["no_selectors"])].copy()
        if len(skip):
            models = sorted(base["model"].dropna().unique().tolist())
            need = (skip.assign(key=1)
                        .merge(pd.DataFrame({"model": models, "key": [1]*len(models)}), on="key")
                        .drop(columns=["key"]))
            have = base[["uid","model"]].drop_duplicates()
            missing = (need.merge(have, on=["uid","model"], how="left", indicator=True)
                            .query("_merge=='left_only'").drop(columns=["_merge"]))
            if len(missing):
                # Build stub rows mirroring base columns
                stub_rows = []
                for _, r in missing.iterrows():
                    stub_rows.append({
                        "uid": r["uid"],
                        "occupation": r.get("occupation", np.nan),
                        "variant": "0",
                        "temp": "",
                        "model": r["model"],
                        "modality": "GUI",
                        "file": "",
                        "wrapper_strict": 0,
                        "wrapper_rescued": 0,
                        "schema_strict": 0,
                        "schema_rescued": 0,
                        "safety_strict": 0,
                        "safety_rescued": 0,
                        "rubric_score": np.nan,
                        "iou": 0,
                        "is_manual_stub": "0",
                    })
                stubs = pd.DataFrame(stub_rows, columns=COLS)
                base = pd.concat([base, stubs], ignore_index=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(OUT_CSV, index=False)
    print(f"✓ master table with {len(base):,} rows → {OUT_CSV}")

# ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
