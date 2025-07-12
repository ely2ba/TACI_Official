#!/usr/bin/env python3
"""
sample_for_human_review.py
Randomly copies ≈5 % of OpenAI run files into runs/openai_human_sample/
grouped by model directory.
"""

import random, shutil, pathlib, math

SOURCE_DIR  = pathlib.Path("runs/openai")
DEST_DIR    = pathlib.Path("runs/openai_human_sample")
SAMPLE_RATE = 0.05    # 5 %

random.seed(42)       # reproducibility

DEST_DIR.mkdir(parents=True, exist_ok=True)

total = copied = 0
for model_dir in SOURCE_DIR.iterdir():
    if not model_dir.is_dir():
        continue
    files = list(model_dir.glob("*.json"))
    k = max(1, math.ceil(len(files) * SAMPLE_RATE))
    sample = random.sample(files, k)
    dest_model = DEST_DIR / model_dir.name
    dest_model.mkdir(parents=True, exist_ok=True)
    for f in sample:
        shutil.copy2(f, dest_model / f.name)
    total += len(files)
    copied += k
    print(f"{model_dir.name}: copied {k}/{len(files)}")

print(f"\n✅ Human-review sample created: {copied}/{total} files "
      f"({copied/total:.1%}) → {DEST_DIR}")
