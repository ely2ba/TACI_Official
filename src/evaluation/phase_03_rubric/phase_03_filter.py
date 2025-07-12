#!/usr/bin/env python3
"""
Phase-03 candidate filter (wrapper-preserving)

Keeps answers when:
 • GUI  → schema_status  != "fail"
 • TEXT → safety_status  != "fail"

Output → phase_03_candidates.csv with columns:
uid,variant,temp,model,modality,file,
schema_status,safety_status,prompt,answer   ← answer still contains <OUTPUT_…> tags
"""

from __future__ import annotations
import argparse, csv, json, re
from pathlib import Path
from typing import Dict

# ─── fixed default paths ────────────────────────────────────────────────
HERE        = Path(__file__).resolve().parent
PHASE1_CSV  = HERE.parent / "phase_01_schema" / "schema_text_gui_per_output.csv"
PHASE2_CSV  = HERE.parent / "phase_02_safety" / "phase_02_safety_per_output.csv"
PROMPTS_DIR = HERE.parent.parent / "prompts"
RUNS_DIR    = HERE.parent.parent / "runs"
OUT_CSV     = HERE / "phase_03_candidates.csv"

# ─── helpers ─────────────────────────────────────────────────────────────
def extract_answer(run_path: Path) -> str:
    """
    Return the *raw* answer exactly as generated, including wrapper tags.
    Falls back to empty string on error.
    """
    try:
        resp = json.loads(run_path.read_text()).get("response", {})
        return (
            resp.get("content", [{}])[0].get("text") or
            (resp.get("candidates", [{}])[0]
                 .get("content", {}).get("parts", [{}])[0].get("text")) or
            resp.get("choices", [{}])[0].get("message", {}).get("content") or
            resp.get("choices", [{}])[0].get("content") or ""
        ).strip()
    except Exception:
        return ""

def load_prompt(uid: str, variant: str, modality: str, pdir: Path) -> str:
    vtag = variant if variant.lower().startswith("v") else f"v{variant}"
    p   = pdir / modality.lower() / f"{uid}_{vtag}.json"
    if not p.exists():
        return ""
    try:
        data = json.loads(p.read_text())
        if isinstance(data, str):
            return data.strip()
        for k in ("prompt", "text", "task", "instruction"):
            if k in data and isinstance(data[k], str):
                return data[k].strip()
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, str) and v.strip():
                    return v.strip()
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return ""

# ─── main filter ────────────────────────────────────────────────────────
def build_candidates(p1_csv: Path, p2_csv: Path,
                     runs: Path, prompts: Path, out_csv: Path) -> None:

    schema_map = {
        (r["uid"], r["variant"], r["temp"], r["file"]): r["schema_status"]
        for r in csv.DictReader(p1_csv.open())
    }

    rows_out = []
    with p2_csv.open() as f:
        for r2 in csv.DictReader(f):
            key = (r2["uid"], r2["variant"], r2["temp"], r2["file"])
            mod   = r2["modality"].upper()
            sstat = schema_map.get(key, "fail")
            saf   = r2["safety_status"]

            if mod == "GUI"  and sstat == "fail": continue
            if mod == "TEXT" and saf   == "fail": continue
            if mod not in ("TEXT", "GUI"):       continue

            ans = extract_answer(runs / r2["file"])
            if not ans:
                continue

            prm = load_prompt(r2["uid"], r2["variant"], mod, prompts)
            if not prm:
                continue

            rows_out.append({
                "uid": r2["uid"], "variant": r2["variant"], "temp": r2["temp"],
                "model": r2["model"], "modality": mod, "file": r2["file"],
                "schema_status": sstat, "safety_status": saf,
                "prompt": prm, "answer": ans
            })

    if not rows_out:
        print("No rows passed filters – nothing to grade.")
        out_csv.write_text("")
        return

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline='', encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows_out[0].keys()))
        w.writeheader(); w.writerows(rows_out)
    print(f"✓ {len(rows_out)} candidates written → {out_csv}")

# ─── CLI entrypoint ──────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-csv", default=PHASE1_CSV, type=Path)
    ap.add_argument("--phase2-csv", default=PHASE2_CSV, type=Path)
    ap.add_argument("--runs-dir",   default=RUNS_DIR,   type=Path)
    ap.add_argument("--prompts-dir",default=PROMPTS_DIR,type=Path)
    ap.add_argument("--out-csv",    default=OUT_CSV,    type=Path)
    args = ap.parse_args()

    build_candidates(args.phase1_csv, args.phase2_csv,
                     args.runs_dir, args.prompts_dir, args.out_csv)
