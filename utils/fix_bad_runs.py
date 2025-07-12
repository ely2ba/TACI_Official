#!/usr/bin/env python3
"""
fix_bad_runs.py
Deletes OpenAI run files whose assistant `content` string lacks the
required wrapper tag. After running, simply relaunch run_batch_openai.py
and it will resend only the deleted jobs.
"""

import json, pathlib, re

RUNS = pathlib.Path("runs/openai")
WRAP_TAG = {"TEXT":"OUTPUT_TEXT", "GUI":"OUTPUT_JSON", "VISION":"OUTPUT_JSON"}

bad = 0
for f in RUNS.rglob("*.json"):
    data = json.loads(f.read_text())
    if data.get("skipped"):
        continue
    mod   = data["modality"]
    tag   = WRAP_TAG.get(mod)
    if not tag:
        continue
    content = data["response"]["choices"][0]["message"]["content"]
    if not re.search(fr"<{tag}>.*?</{tag}>", content, re.S):
        f.unlink()
        bad += 1
print(f"🗑️  Deleted {bad} malformed run files. Re-run the batch to regenerate them.")
