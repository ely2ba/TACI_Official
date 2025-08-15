#!/usr/bin/env python3
"""
Generate a manifest of *every* O*NET task for paralegals (SOC 23-2011.00)
and assign each a primary modality (TEXT | GUI | VISION) via a three-vote
self-consistency pass with GPT-4o-mini.

Output
------
data/manifests/paralegal_tasks.csv
    uid, SOC, TaskID, Task Statement, Importance,
    OccTitleRaw, OccTitleClean, modality
"""
from __future__ import annotations
import hashlib, os, re, time, random, json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import inflect                  # pip install inflect python-dotenv tenacity tqdm
import pandas as pd
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
from dotenv import load_dotenv
from tqdm import tqdm

# ───────── paths ──────────────────────────────────────────────────────
RAW        = Path("data/onet_raw")
OUT_DIR    = Path("data/manifests")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV    = OUT_DIR / "paralegal_tasks.csv"
CACHE_JSON = OUT_DIR / "modality_cache.json"   # optional persistent cache

PARALEGAL_SOC = "23-2011.00"
MODEL_NAME    = "gpt-4o-mini"
TEMPERATURE   = 0.3
VOTES_PER_TASK = 3

SYSTEM_PROMPT = (
    "You are an evaluator for the Task-AI Capability Index. "
    "Return exactly one word — TEXT, GUI, or VISION — that best "
    "describes the modality a model must handle to complete the task."
)

# ───────── helpers ────────────────────────────────────────────────────
infl = inflect.engine()

def clean_title(raw: str) -> str:
    txt = re.sub(r"[,&]", " ", raw)
    txt = re.sub(r"\s{2,}", " ", txt).strip()
    parts, out = re.split(r"\s+(and|or)\s+", txt, flags=re.I), []
    for part in parts:
        if part.lower() in ("and", "or"):
            out.append(part.lower());  continue
        out.append(" ".join(infl.singular_noun(w) or w for w in part.split()))
    return re.sub(r"\s{2,}", " ", " ".join(out)).title()

def md5_uid(soc: str, task_id: str) -> str:
    return hashlib.md5(f"{soc}-{task_id}".encode()).hexdigest()[:8]

def load_cache(path: Path) -> Dict[str, str]:
    return json.loads(path.read_text()) if path.exists() else {}

def save_cache(path: Path, data: Dict[str, str]) -> None:
    path.write_text(json.dumps(data, indent=2))

# ───────── OpenAI vote wrapper ────────────────────────────────────────
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

# ───────── main workflow ──────────────────────────────────────────────
def main():
    # 1 ▸ load O*NET tables ------------------------------------------------
    ts  = pd.read_csv(RAW / "Task_Statements.txt", sep="\t", header=0)
    tr  = pd.read_csv(RAW / "Task_Ratings.txt",   sep="\t", header=0)
    occ = pd.read_csv(
            RAW / "Occupation_Data.txt", sep="\t", header=0,
            usecols=["O*NET-SOC Code", "Title"]
          ).rename(columns={"O*NET-SOC Code": "SOC", "Title": "OccTitleRaw"})

    imp = tr[tr["Scale ID"] == "IM"][["O*NET-SOC Code", "Task ID", "Data Value"]] \
            .rename(columns={"O*NET-SOC Code": "SOC",
                             "Task ID": "TaskID",
                             "Data Value": "Importance"})

    df = ts.rename(columns={"O*NET-SOC Code": "SOC", "Task ID": "TaskID"}) \
           .merge(imp, on=["SOC", "TaskID"], how="left") \
           .merge(occ, on="SOC", how="left")

    df["OccTitleClean"] = df["OccTitleRaw"].apply(clean_title)
    df["uid"] = df.apply(lambda r: md5_uid(r["SOC"], r["TaskID"]), axis=1)

    para = df[df["SOC"] == PARALEGAL_SOC].reset_index(drop=True)
    if para.empty:
        raise SystemExit("No paralegal tasks found — check O*NET files.")

    # 2 ▸ modality classification -----------------------------------------
    load_dotenv()
    offline  = bool(os.getenv("OFFLINE_GRADER")) or not os.getenv("OPENAI_API_KEY")
    cache    = load_cache(CACHE_JSON)
    client   = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if not offline else None

    modalities: List[str] = []
    for _, row in tqdm(para.iterrows(), total=len(para), desc="Classifying modality"):
        uid   = row["uid"]
        stmt  = row.get("Task Statement") or row.get("Description") or ""
        if uid in cache:                       # cached result
            modalities.append(cache[uid]);   continue
        if offline:
            modalities.append("TEXT");       continue   # default stub

        votes = [vote_once(client, stmt, seed) for seed in range(1, VOTES_PER_TASK+1)]
        label = majority(votes)
        cache[uid] = label
        modalities.append(label)

        time.sleep(0.1 + random.random()*0.2)  # gentle pacing

    save_cache(CACHE_JSON, cache)
    para["modality"] = modalities

    # 3 ▸ save manifest ----------------------------------------------------
    para.to_csv(OUT_CSV, index=False)
    print(f"✅  Generated {len(para)} tasks with modality → {OUT_CSV}")

# ───────── CLI hook ────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
