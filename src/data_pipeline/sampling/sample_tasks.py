# sampling/sample_tasks.py
"""
Build comprehensive task manifest across 20 diverse occupations with 3-vote modality determination.
Covers TEXT, GUI, VISION, and MULTI-modal occupations for robust TACI evaluation.

Adds:
  • OccTitleRaw      – original O*NET title
  • OccTitleClean    – cleaned, singular title
  • uid              – deterministic 8-char MD5(SOC-TaskID)
  • modality         – TEXT/GUI/VISION via 3-vote GPT-4o-mini consensus

Output → data/manifests/sampled_tasks_comprehensive.csv
"""
from __future__ import annotations
import hashlib
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List

import inflect              # pip install inflect
import openai
import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm

RAW = Path("data/onet_raw")
OUT = Path("data/manifests"); OUT.mkdir(parents=True, exist_ok=True)
CACHE_JSON = OUT / "modality_cache_comprehensive.json"   # cache for comprehensive tasks

# ── Configuration ─────────────────────────────────────────────────────
MODEL_NAME = "gpt-4.1-mini-2025-04-14"
TEMPERATURE = 0.3
VOTES_PER_TASK = 3

SYSTEM_PROMPT = (
    "You are an evaluator for the Task-AI Capability Index. "
    "Return exactly one word — TEXT, GUI, or VISION — that best "
    "describes the modality a model must handle to complete the task."
)

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

# ── 3. Merge & keep Core and available tasks ─────────────────────────
df = ts.rename(columns={"O*NET-SOC Code": "SOC", "Task ID": "TaskID"})\
       .merge(imp, on=["SOC", "TaskID"], how="left")

# Include all task types (Core, Supplemental, and NaN) for comprehensive coverage
df = df.merge(occ, on="SOC", how="left")

# ── Helper Functions ─────────────────────────────────────────────────
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

def load_cache(path: Path) -> Dict[str, str]:
    return json.loads(path.read_text()) if path.exists() else {}

def save_cache(path: Path, data: Dict[str, str]) -> None:
    path.write_text(json.dumps(data, indent=2))

@retry(wait=wait_random_exponential(min=1, max=20),
       stop=stop_after_attempt(6))
def vote_once(client: openai.OpenAI, statement: str, seed: int) -> str:
    resp = client.chat.completions.create(
        model        = MODEL_NAME,
        temperature  = TEMPERATURE,
        seed         = seed,
        messages     = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": statement}
        ],
    )
    ans = resp.choices[0].message.content.strip().upper()
    if ans not in {"TEXT", "GUI", "VISION"}:
        raise ValueError(f"Unexpected label: {ans}")
    return ans

def majority(votes: List[str]) -> str:
    top, freq = Counter(votes).most_common(1)[0]
    return top if freq >= 2 else votes[0]        # deterministic tie-break

df["OccTitleClean"] = df["OccTitleRaw"].apply(clean_title)

# ── 5. Deterministic uid ─────────────────────────────────────────────
df["uid"] = df.apply(
    lambda r: hashlib.md5(f"{r['SOC']}-{r['TaskID']}".encode()).hexdigest()[:8],
    axis=1,
)

# ── 6. Comprehensive SOC list by modality ───────────────────────────
PILOT_SOCs = [
    # TEXT-Primary Occupations
    "23-2011.00",  # Paralegals and Legal Assistants
    "15-1252.00",  # Software Developers
    "43-4051.00",  # Customer Service Representatives
    "29-2072.00",  # Medical Records Specialists
    "27-3023.00",  # News Analysts, Reporters, and Journalists
    "27-3031.00",  # Public Relations Specialists
    "27-3091.00",  # Interpreters and Translators
    
    # GUI-Primary Occupations
    "15-1255.00",  # Web and Digital Interface Designers
    
    # VISION-Primary Occupations
    "29-1224.00",  # Radiologists
    "27-1024.00",  # Graphic Designers
    
    # MULTI-Modal Occupations
    "15-2051.00",  # Data Scientists
    "15-1212.00",  # Information Security Analysts
    "13-1082.00",  # Project Management Specialists
    "43-6011.00",  # Executive Secretaries and Admin Assistants
    "13-1071.00",  # Human Resources Specialists
    "13-2011.00",  # Accountants and Auditors
    "13-1161.00",  # Market Research Analysts and Marketing Specialists
    "13-1031.00",  # Claims Adjusters, Examiners, and Investigators
    "25-9031.00",  # Instructional Coordinators
    "17-2051.00",  # Civil Engineers
]

# ── 7. Select ALL tasks for specified occupations ───────────────────
comprehensive_tasks = df[df["SOC"].isin(PILOT_SOCs)].reset_index(drop=True)

if comprehensive_tasks.empty:
    raise SystemExit("No tasks found for specified occupations — check O*NET files and SOC codes.")

print(f"📋 Found tasks for {len(PILOT_SOCs)} occupations:")
for soc in PILOT_SOCs:
    soc_tasks = comprehensive_tasks[comprehensive_tasks["SOC"] == soc]
    occ_title = soc_tasks["OccTitleClean"].iloc[0] if len(soc_tasks) > 0 else "Unknown"
    print(f"   {soc}: {len(soc_tasks)} tasks ({occ_title})")

# ── 8. 3-Vote Modality Classification ────────────────────────────────
load_dotenv()
offline = bool(os.getenv("OFFLINE_GRADER")) or not os.getenv("OPENAI_API_KEY")
cache = load_cache(CACHE_JSON)
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if not offline else None

modalities: List[str] = []
print(f"🔄  Classifying modality for {len(comprehensive_tasks)} comprehensive tasks...")

for _, row in tqdm(comprehensive_tasks.iterrows(), total=len(comprehensive_tasks), desc="Classifying modality"):
    uid = row["uid"]
    stmt = row.get("Task Statement") or row.get("Description") or row.get("Task") or ""
    
    if uid in cache:                       # cached result
        modalities.append(cache[uid])
        continue
    if offline:
        modalities.append("TEXT")          # default stub
        continue

    # 3-vote consensus
    votes = [vote_once(client, stmt, seed) for seed in range(1, VOTES_PER_TASK+1)]
    label = majority(votes)
    cache[uid] = label
    modalities.append(label)

    time.sleep(0.1 + random.random()*0.2)  # gentle pacing

save_cache(CACHE_JSON, cache)
comprehensive_tasks["modality"] = modalities

# ── 9. Save manifest ─────────────────────────────────────────────────
out_file = OUT / "sampled_tasks_comprehensive.csv"
comprehensive_tasks.to_csv(out_file, index=False)

# Summary stats
print(f"✅  Generated {len(comprehensive_tasks)} tasks → {out_file}")
if modalities:
    modality_counts = Counter(modalities)
    print(f"📊  Modality distribution: {dict(modality_counts)}")
    
    # Show breakdown by occupation
    print(f"📈  Tasks by occupation:")
    for soc in PILOT_SOCs:
        soc_count = len(comprehensive_tasks[comprehensive_tasks["SOC"] == soc])
        occ_title = comprehensive_tasks[comprehensive_tasks["SOC"] == soc]["OccTitleClean"].iloc[0] if soc_count > 0 else "Unknown"
        print(f"   {soc}: {soc_count:2d} tasks ({occ_title})")
else:
    print("ℹ️   Running in offline mode - no modality classification")
