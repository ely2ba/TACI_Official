#!/usr/bin/env python3
"""
phase_03_filter_paralegal.py
------------------------------------------------------------------
• Baseline “seen” rows      : graders/phase_00_wrapper/wrapper_per_output_paralegal.csv
• Phase-01 schema results   : graders/phase_01_schema/schema_text_gui_per_output_paralegal.csv
• Phase-02 safety results   : graders/phase_02_safety/phase_02_safety_per_output_paralegal.csv
• Prompts                   : prompts/one_occm/
• Run files                 : runs/openai/one_occ/**.json

Outputs (next to this script)
  1) phase_03_candidates_paralegal.csv
  2) model_flow_paralegal.csv   (phase-0 “seen” vs phase-3 “kept”)
"""
from __future__ import annotations
import argparse, csv, json, collections
from pathlib import Path
from typing import Dict, Tuple

# ── default paths ────────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent
WRAP_CSV    = Path("graders/phase_00_wrapper/wrapper_per_output_paralegal.csv")
PHASE1_CSV  = Path("graders/phase_01_schema/"
                   "schema_text_gui_per_output_paralegal.csv")
PHASE2_CSV  = Path("graders/phase_02_safety/"
                   "phase_02_safety_per_output_paralegal.csv")
PROMPTS_DIR = Path("prompts/one_occ")
RUNS_DIR    = Path("runs/openai/one_occ")

OUT_CANDS   = HERE / "phase_03_candidates_paralegal.csv"
OUT_FLOW    = HERE / "model_flow_paralegal.csv"

# ── helpers ───────────────────────────────────────────────────────────
def extract_answer(jpath: Path) -> str:
    try:
        rsp = json.loads(jpath.read_text()).get("response", {})
        return (
            rsp.get("content",    [{}])[0].get("text") or
            rsp.get("candidates", [{}])[0]
               .get("content", {}).get("parts", [{}])[0].get("text") or
            rsp.get("choices",    [{}])[0].get("message", {}).get("content") or
            rsp.get("choices",    [{}])[0].get("content") or ""
        ).strip()
    except Exception:
        return ""

def load_prompt(uid:str, variant:str, modality:str, root:Path) -> str:
    vtag = variant if str(variant).lower().startswith("v") else f"v{variant}"
    p = root / modality.lower() / f"{uid}_{vtag}.json"
    if not p.exists(): return ""
    try:
        blob = json.loads(p.read_text())
        if isinstance(blob, list) and len(blob) >= 2:
            return blob[1]["content"].strip()
        if isinstance(blob, str):
            return blob.strip()
        return json.dumps(blob, ensure_ascii=False)
    except Exception:
        return ""

# ── main builder ─────────────────────────────────────────────────────
def build(wrapper_csv:Path, phase1_csv:Path, phase2_csv:Path,
          runs_dir:Path, prompts_dir:Path) -> None:

    # ------------ phase-0 “seen” counts -------------------------------
    seen = collections.Counter()           # (model, modality) → count
    for w in csv.DictReader(wrapper_csv.open()):
        mod = w["modality"].upper()
        if mod not in ("TEXT","GUI"):               # ignore vision/manual
            continue
        seen[(w["model"], mod)] += 1

    # ------------ phase-1 schema lookup -------------------------------
    schema_map = {(r["uid"], r["variant"], r["temp"], r["file"]): r["schema_status"]
                  for r in csv.DictReader(phase1_csv.open())}

    # ------------ phase-3 filtering -----------------------------------
    kept = collections.Counter()
    candidates = []

    for r2 in csv.DictReader(phase2_csv.open()):
        mod = r2["modality"].upper()
        if mod not in ("TEXT","GUI"):
            continue

        schema_status = schema_map.get(
            (r2["uid"], r2["variant"], r2["temp"], r2["file"]), "fail"
        )
        safety_status = r2["safety_status"]

        if mod == "GUI"  and schema_status == "fail": continue
        if mod == "TEXT" and safety_status  == "fail": continue

        ans = extract_answer(runs_dir / r2["file"])
        if not ans: continue
        prm = load_prompt(r2["uid"], r2["variant"], mod, prompts_dir)
        if not prm: continue

        kept[(r2["model"], mod)] += 1

        candidates.append({
            "uid": r2["uid"], "variant": r2["variant"], "temp": r2["temp"],
            "model": r2["model"], "modality": mod, "file": r2["file"],
            "schema_status": schema_status, "safety_status": safety_status,
            "prompt": prm, "answer": ans
        })

    # ------------ write candidates ------------------------------------
    OUT_CANDS.parent.mkdir(parents=True, exist_ok=True)
    if candidates:
        with OUT_CANDS.open("w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=list(candidates[0].keys()))
            w.writeheader(); w.writerows(candidates)
        print(f"✓ {len(candidates)} rubric candidates → {OUT_CANDS}")
    else:
        OUT_CANDS.write_text("")
        print("No rows passed Phase-03 filter.")

    # ------------ flow table ------------------------------------------
    with OUT_FLOW.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["model","modality","phase0_seen","phase3_kept","pct_retained"])
        for (model, mod) in sorted(seen):
            s = seen[(model,mod)]
            k = kept[(model,mod)]
            pct = 100*k/s if s else 0.0
            w.writerow([model, mod, s, k, f"{pct:.1f}"])
    print(f"✓ model flow table → {OUT_FLOW}")

# ── CLI ----------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrapper-csv", default=WRAP_CSV, type=Path)
    ap.add_argument("--phase1-csv", default=PHASE1_CSV, type=Path)
    ap.add_argument("--phase2-csv", default=PHASE2_CSV, type=Path)
    ap.add_argument("--runs-dir",   default=RUNS_DIR,    type=Path)
    ap.add_argument("--prompts-dir",default=PROMPTS_DIR, type=Path)
    args = ap.parse_args()

    build(args.wrapper_csv, args.phase1_csv, args.phase2_csv,
          args.runs_dir, args.prompts_dir)
