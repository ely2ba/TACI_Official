#!/usr/bin/env python3
"""
assign_archetypes.py — TACI v1.1

Adds a `ui_archetype` column to sampled_tasks_with_modality.csv
using keyword rules from config/archetype_rules.yml.

Only modifies rows where modality == 'GUI'.
"""

import pandas as pd
import yaml
import pathlib

# Paths
MANIFEST = pathlib.Path("data/manifests/sampled_tasks_with_modality.csv")
RULE_FILE = pathlib.Path("config/archetype_rules.yml")

# Load manifest and rule file
df = pd.read_csv(MANIFEST, dtype=str).fillna("")
rules = yaml.safe_load(RULE_FILE.read_text())

# Rule matcher
def match_archetype(text: str) -> str:
    text = text.lower()
    for archetype, spec in rules.items():
        for keyword in spec.get("keywords", []):
            if keyword.lower() in text:
                return archetype
    return ""

# Assign archetypes to GUI rows only
df["ui_archetype"] = df.apply(
    lambda row: match_archetype(f"{row['Task']} {row['OccTitleClean']}")
    if row["modality"].strip().upper() == "GUI" else "",
    axis=1
)

# Overwrite file in place
df.to_csv(MANIFEST, index=False)
print(f"✅ ui_archetype column added to {MANIFEST}")
