# utils/auto_tag_vision.py

import yaml
import re
import pandas as pd
from pathlib import Path

# Adjust these paths for your actual file locations
CSV_PATH = Path("data/manifests/sampled_tasks_with_modality.csv")
YAML_PATH = Path("config/vision_archetypes.yml")

# 1. Load vision archetype matching rules
cfg = yaml.safe_load(YAML_PATH.open())

# 2. Compile regex patterns
patterns = [
    (name, re.compile("|".join(map(re.escape, props["match"])), re.IGNORECASE))
    for name, props in cfg.items()
]

# 3. Load tasks CSV
df = pd.read_csv(CSV_PATH)

# 4. Add column if missing
if "vision_archetype" not in df.columns:
    df["vision_archetype"] = None

# 5. Apply pattern matching to VISION tasks
for name, pat in patterns:
    mask = (df["modality"] == "VISION") & df["Task"].str.contains(pat)
    df.loc[mask, "vision_archetype"] = name

# 6. Save
df.to_csv(CSV_PATH, index=False)
print(f"✅ vision_archetype column updated in {CSV_PATH}")
