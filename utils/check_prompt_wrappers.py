#!/usr/bin/env python3
"""
check_prompt_wrappers.py
Verifies that every non-MANUAL prompt under ./prompts/ contains the
correct opening *and* closing wrapper tags.

Manual prompts are intentionally wrapper-free and are skipped.
"""

import json, pathlib, re, sys

PROMPT_DIR = pathlib.Path("prompts")
bad_files  = []
checked    = 0

for f in PROMPT_DIR.rglob("*.json"):
    modality = f.parent.name.upper()
    if modality == "MANUAL":
        continue        # manual tasks have no wrapper by design

    msgs = json.loads(f.read_text())
    user = msgs[1]["content"]
    if "<OUTPUT_JSON>" in user:
        ok = re.search(r"<OUTPUT_JSON>.*?</OUTPUT_JSON>", user, re.S)
    elif "<OUTPUT_TEXT>" in user:
        ok = re.search(r"<OUTPUT_TEXT>.*?</OUTPUT_TEXT>", user, re.S)
    else:
        ok = False

    if not ok:
        bad_files.append(f)
    checked += 1

valid = checked - len(bad_files)
pct   = (valid / checked * 100) if checked else 100.0
print(f"Wrapper-valid non-manual prompts: {valid}/{checked} ({pct:.1f}%)")

if bad_files:
    print("\nFirst 20 offending files:")
    for p in bad_files[:20]:
        print(" •", p)
    sys.exit(1)
else:
    print("All good! ✅")
