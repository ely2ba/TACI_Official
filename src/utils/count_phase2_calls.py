#!/usr/bin/env python3
"""
Count how many answers Phase-2 safety grading will process.

• Reads the Phase-1 CSV and applies the same logic:
      – include STRICT    rows where strict == 1
      – include RESCUED   rows where rescued == 1  **and** strict != 1
• Optionally deduplicates by hashing the extracted text so you see
  roughly how many real OpenAI API calls you’ll pay for once caching
  kicks in.

Just edit the CONFIG section, press ▶ Run in VS Code, and read the printout.
"""

import csv, json, re, hashlib
from pathlib import Path
from typing import Set, Dict

# ------------------------------------------------------------------ #
# CONFIG – change these paths if your repo layout is different       #
# ------------------------------------------------------------------ #
PHASE1_CSV = Path("outputs/results/evaluated/schema_text_gui_per_output.csv")
RUNS_DIR   = Path("outputs")             # folder that holds raw JSON files
DO_DEDUP   = True                     # set False if you only want branch count

# ------------------------------------------------------------------ #
# Helpers                                                            #
# ------------------------------------------------------------------ #
TAG_RE_TEXT = re.compile(r"<OUTPUT_TEXT>(.*?)</OUTPUT_TEXT>", re.S | re.I)
TAG_RE_JSON = re.compile(r"<OUTPUT_JSON>(.*?)</OUTPUT_JSON>", re.S | re.I)

def extract_text(payload: str) -> str:
    m = TAG_RE_TEXT.search(payload) or TAG_RE_JSON.search(payload)
    return m.group(1).strip() if m else ""

def load_payload(path: Path) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        r = d.get("response", {})
        return (
            r.get("content", [{}])[0].get("text")
            or (r.get("candidates", [{}])[0]
                .get("content", {}).get("parts", [{}])[0].get("text"))
            or r.get("choices", [{}])[0].get("message", {}).get("content")
            or r.get("choices", [{}])[0].get("content")
            or ""
        )
    except Exception:
        return ""

sha256 = lambda txt: hashlib.sha256(txt.encode("utf-8")).hexdigest()

# ------------------------------------------------------------------ #
# Main logic                                                         #
# ------------------------------------------------------------------ #
def main():
    if not PHASE1_CSV.exists():
        raise FileNotFoundError(f"{PHASE1_CSV} not found")

    branch_total   = strict_hits = rescued_hits = 0
    unique_hashes: Set[str] = set()

    with open(PHASE1_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            strict  = float(row.get("strict", 0))
            rescued = float(row.get("rescued", 0))
            file_rel = row.get("file", "")

            # ---- STRICT branch ----
            if strict == 1.0:
                branch_total += 1
                strict_hits  += 1
                if DO_DEDUP:
                    text = extract_text(load_payload(RUNS_DIR / file_rel))
                    unique_hashes.add(sha256(text))

            # ---- RESCUED branch ----
            if strict != 1.0 and rescued == 1.0:
                branch_total += 1
                rescued_hits += 1
                if DO_DEDUP:
                    text = extract_text(load_payload(RUNS_DIR / file_rel))
                    unique_hashes.add(sha256(text))

    # ------------------ results ------------------ #
    print("\nPhase-2 moderation branches:")
    print(f"  strict   : {strict_hits}")
    print(f"  rescued  : {rescued_hits}")
    print(f"  total    : {branch_total}")

    if DO_DEDUP:
        print(f"\nUnique texts after dedup : {len(unique_hashes)}")
        print("≈ number of OpenAI API calls once caching is on")

if __name__ == "__main__":
    main()
