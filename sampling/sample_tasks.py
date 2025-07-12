# sampling/sample_tasks.py
"""
Build the 30-task pilot manifest.
Adds:
  • OccTitleRaw      – original O*NET title
  • OccTitleClean    – cleaned, singular title
  • uid              – deterministic 8-char MD5(SOC-TaskID)

Output → data/manifests/sampled_tasks.csv
"""

import hashlib
import re
from pathlib import Path

import inflect              # pip install inflect
import pandas as pd

RAW = Path("data/onet_raw")
OUT = Path("data/manifests"); OUT.mkdir(parents=True, exist_ok=True)

# ── 1. Load O*NET files ──────────────────────────────────────────────
ts  = pd.read_csv(RAW / "Task_Statements.txt", sep="\t", header=0)
tr  = pd.read_csv(RAW / "Task_Ratings.txt",   sep="\t", header=0)
occ = pd.read_csv(RAW / "Occupation_Data.txt", sep="\t", header=0,
                  usecols=["O*NET-SOC Code", "Title"])\
        .rename(columns={"O*NET-SOC Code": "SOC", "Title": "OccTitleRaw"})

# ── 2. Importance rows (Scale ID = IM) ───────────────────────────────
imp = tr[tr["Scale ID"] == "IM"][["O*NET-SOC Code", "Task ID", "Data Value"]]\
        .rename(columns={"O*NET-SOC Code": "SOC",
                         "Task ID": "TaskID",
                         "Data Value": "Importance"})

# ── 3. Merge & keep Core tasks ───────────────────────────────────────
df = ts.rename(columns={"O*NET-SOC Code": "SOC", "Task ID": "TaskID"})\
       .merge(imp, on=["SOC", "TaskID"], how="inner")
df = df[df["Task Type"] == "Core"].merge(occ, on="SOC", how="left")

# ── 4. Clean occupation titles (singularise every noun) ──────────────
infl = inflect.engine()

def clean_title(raw: str) -> str:
    # Replace commas & ampersands with spaces, collapse spaces
    txt = re.sub(r"[,&]", " ", raw)
    txt = re.sub(r"\s{2,}", " ", txt).strip()

    # Split on 'and'/'or' while keeping connectors
    parts = re.split(r"\s+(and|or)\s+", txt, flags=re.I)
    cleaned = []

    for part in parts:
        if part.lower() in ("and", "or"):
            cleaned.append(part.lower())
            continue
        # Singularise each plural noun in this chunk
        words = []
        for w in part.split():
            singular = infl.singular_noun(w) or w
            words.append(singular)
        cleaned.append(" ".join(words))

    title = " ".join(cleaned)
    title = re.sub(r"\s{2,}", " ", title).strip()
    return title.title()

df["OccTitleClean"] = df["OccTitleRaw"].apply(clean_title)

# ── 5. Deterministic uid ─────────────────────────────────────────────
df["uid"] = df.apply(
    lambda r: hashlib.md5(f"{r['SOC']}-{r['TaskID']}".encode()).hexdigest()[:8],
    axis=1,
)

# ── 6. Pilot SOC list ────────────────────────────────────────────────
PILOT_SOCs = [
    "23-2011.00","15-2051.01","43-4051.00","43-6013.00","25-2021.00",
    "29-2034.00","13-1081.00","51-9061.00","43-3031.00","43-5071.00"
]

# ── 7. Select top / median / bottom Importance per SOC ───────────────
rows = []
for soc in PILOT_SOCs:
    sub = df[df["SOC"] == soc].sort_values("Importance", ascending=False)
    if len(sub) < 3:
        rows.append(sub.head(3))
    else:
        rows.append(pd.DataFrame([sub.iloc[0], sub.iloc[len(sub)//2], sub.iloc[-1]]))

pilot = pd.concat(rows).reset_index(drop=True)

# ── 8. Save manifest ─────────────────────────────────────────────────
out_file = OUT / "sampled_tasks.csv"
pilot.to_csv(out_file, index=False)
print(f"✅  Generated {len(pilot)} tasks → {out_file}")
