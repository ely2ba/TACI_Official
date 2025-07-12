#!/usr/bin/env python3
"""
build_vision_stub_catalog.py

Locate all *existing* Vision-stub JSON files produced by non-vision models
and list them in `vision_stub_catalog.csv`.

Output columns (like manual catalogue):
    uid , variant , temp , file , model , modality , is_digital
"""

from pathlib import Path
import pandas as pd, csv, re

# ------------------------------------------------------------------------
RUNS   = Path("runs")                                   # per-model folders
MANIF  = Path("data/manifests/sampled_tasks_with_modality.csv")
OUT    = Path("graders/phase_01_schema/vision_stub_catalog.csv")

NON_VISION = {
    "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo",
    "claude-3-5-sonnet-20240620", "claude-3-opus-20240229",
    "llama3-8b-8192"
}

# ------------------------------------------------------------------------
mf = pd.read_csv(MANIF, dtype=str)
vision_uids = set(
    mf.loc[mf.modality.str.upper() == "VISION", "uid"].dropna().unique()
)

if not vision_uids:
    print("No VISION rows in manifest – nothing to catalogue.")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["uid","variant","temp","file","model","modality","is_digital"]
        )
    exit(0)

# pattern: <uid>_v<var>_t<int>_<frac>.json   (same as your prompt files)
pat = re.compile(r"^(?P<uid>[a-f0-9]{8})_v(?P<var>\d+)_t(?P<int>\d)_(?P<frac>\d+)\.json$")

rows = []
for fp in RUNS.rglob("*.json"):
    model = fp.parent.name
    if model not in NON_VISION:
        continue

    m = pat.match(fp.name)
    if not m:
        continue

    uid = m.group("uid")
    if uid not in vision_uids:
        continue

    variant = m.group("var")              # "0" / "1" / "2"
    temp    = f"{m.group('int')}.{m.group('frac')}"   # "0.5" etc.

    rows.append({
        "uid": uid,
        "variant": variant,
        "temp": temp,
        "file": str(fp.relative_to(RUNS)),
        "model": model,
        "modality": "VISION",
        "is_digital": 1          # still a digital artefact; physical ≠ manual
    })

# ------------------------------------------------------------------------
OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(
        f,
        fieldnames=["uid","variant","temp","file","model","modality","is_digital"]
    )
    w.writeheader()
    w.writerows(rows)

print(f"✓ vision_stub_catalog.csv → {OUT}   ({len(rows)} rows)")
