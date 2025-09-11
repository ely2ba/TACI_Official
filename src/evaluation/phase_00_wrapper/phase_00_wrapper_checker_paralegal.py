#!/usr/bin/env python3
"""
phase_00_wrapper_checker_paralegal.py
Checks wrapper-tag compliance for the paralegal run set and writes
a single consolidated CSV:

    wrapper_per_output_paralegal.csv
        uid, model, variant, temperature, modality,
        strict, rescued, schema_status, file

The script scans *.json run-files under runs/openai/one_occ/ by default.
"""

from __future__ import annotations
import argparse, csv, json, pathlib, re, sys
from typing import Optional, Tuple

# ── wrapper-tag maps & regexes ─────────────────────────────────────────
WRAP_TAG = {"TEXT": "OUTPUT_TEXT", "GUI": "OUTPUT_JSON", "VISION": "OUTPUT_JSON"}
RESCUE_RE = {m: re.compile(rf"<{t}>(.*?)</{t}>", re.S | re.I)
             for m, t in WRAP_TAG.items()}

# ── helper to pull first text blob from various vendor formats ─────────
def extract_text(payload: dict) -> Optional[str]:
    rsp = payload.get("response", {})
    # Anthropic Claude
    if (txt := rsp.get("content", [{}])[0].get("text", "")):
        return txt
    # Gemini
    if (cands := rsp.get("candidates")):
        txt = cands[0].get("content", {}).get("parts", [{}])[0].get("text", "")
        if txt: return txt
    # OpenAI / Groq
    if (choices := rsp.get("choices")):
        txt = choices[0].get("message", {}).get("content") \
              or choices[0].get("content", "")
        if txt: return txt
    return None

# ── main ───────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate wrapper_per_output_paralegal.csv only")
    ap.add_argument("--runs-dir", default="runs/openai/one_occ",
                    help="folder with run *.json files")
    ap.add_argument("--out", default="wrapper_per_output_paralegal.csv",
                    help="output CSV filename")
    args = ap.parse_args()

    runs_dir = pathlib.Path(args.runs_dir).resolve()
    if not runs_dir.exists():
        sys.exit(f"[wrapper] no such folder: {runs_dir}")

    rows = []
    for f in runs_dir.rglob("*.json"):
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        if data.get("skipped"):
            continue

        mod = data.get("modality", "").upper()
        if mod not in WRAP_TAG:
            continue
        tag = WRAP_TAG[mod]

        raw = extract_text(data) or ""
        s = raw.replace("\r\n","\n").replace("\r","\n").lstrip("\ufeff")
        s_trim = s.strip(" \t\n\u200b\u200c\u200d\ufeff")
        txt = s_trim
        strict   = int(txt.startswith(f"<{tag}>") and txt.endswith(f"</{tag}>"))
        rescued  = int(bool(RESCUE_RE[mod].search(txt)))
        status   = "strict" if strict else "rescued" if rescued else "fail"

        rows.append([
            data.get("uid", ""),
            data.get("model") or f.parent.name,
            data.get("variant", ""),
            data.get("temperature", ""),
            mod,
            strict,
            rescued,
            status,
            str(f.relative_to(runs_dir))
        ])

    # write single consolidated CSV --------------------------------------
    out_path = pathlib.Path(args.out)
    with out_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow([
            "uid","model","variant","temperature","modality",
            "strict","rescued","schema_status","file"
        ])
        w.writerows(rows)

    print(f"✅  {len(rows)} records → {out_path}")

# entry-point
if __name__ == "__main__":
    main()
