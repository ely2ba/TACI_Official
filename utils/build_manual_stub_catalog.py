#!/usr/bin/env python3
"""
Build manual_stub_catalog.csv

• Looks up every UID whose modality == MANUAL in
  data/manifests/sampled_tasks_with_modality.csv
• Finds the corresponding *.json stub files under runs/*
• Outputs phase_00_wrapper/manual_stub_catalog.csv
  with columns: uid, variant, temp, file, model, modality, is_digital
"""

from pathlib import Path
import pandas as pd, csv, re

# ---------- paths -------------------------------------------------------
RUNS   = Path("runs")                                   # per-model folders
MANIF  = Path("data/manifests/sampled_tasks_with_modality.csv")
OUT    = Path("graders/phase_00_wrapper/manual_stub_catalog.csv")

# ---------- load manifest ----------------------------------------------
mf = pd.read_csv(MANIF, dtype=str)
manual_uids = set(
    mf[mf["modality"].str.upper() == "MANUAL"]["uid"].dropna().unique()
)

if not manual_uids:
    print("No MANUAL rows in manifest – nothing to catalogue.")
    header = ["uid","variant","temp","file","model","modality","is_digital"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w",newline='',encoding="utf-8") as f:
        csv.writer(f).writerow(header)
    exit(0)

# ---------- scan runs/ --------------------------------------------------
rows = []
pat = re.compile(r"^(?P<uid>[a-f0-9]{8})_(?P<variant>v\d+)_(?P<temp>t\d)_")

for fp in RUNS.rglob("*.json"):
    m = pat.match(fp.stem)
    if not m:
        continue
    uid = m.group("uid")
    if uid not in manual_uids:
        continue                     # not a manual task
    rows.append({
        "uid": uid,
        "variant": m.group("variant"),            # v0 / v1 / v2 …
        "temp":    m.group("temp"),               # t0 / t0_5 …
        "file":    str(fp.relative_to(RUNS)),
        "model":   fp.parent.name,                # folder name under runs/
        "modality": "MANUAL",
        "is_digital": 0
    })

# ---------- write catalogue --------------------------------------------
OUT.parent.mkdir(parents=True, exist_ok=True)
header = ["uid","variant","temp","file","model","modality","is_digital"]

with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=header)
    w.writeheader()
    w.writerows(rows)

print(f"✓ manual_stub_catalog.csv → {OUT}   ({len(rows)} rows)")
