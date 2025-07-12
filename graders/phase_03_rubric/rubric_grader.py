#!/usr/bin/env python3
"""
Phase-03 Rubric Grader  (GPT-4o-mini • 3-vote self-consistency • cached)

Inputs
------
graders/phase_03_rubric/phase_03_candidates.csv

Outputs
-------
graders/phase_03_rubric/phase_03_rubric_per_output.csv
graders/phase_03_rubric/rubric_cache.json   (persistent cache)

Each row gets:
  rubric_score        int 1-5
  contradiction_flag  0/1
  rubric_reason       optional short string
"""

from __future__ import annotations
import argparse, csv, json, hashlib, os, sys, time, collections
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv
import openai
from tqdm import tqdm

# ───────── paths / constants ─────────────────────────────────────────────
HERE           = Path(__file__).resolve().parent
CAND_CSV       = HERE / "phase_03_candidates.csv"
OUT_CSV        = HERE / "phase_03_rubric_per_output.csv"
CACHE_JSON     = HERE / "rubric_cache.json"

MODEL_NAME     = "gpt-4o-mini"
BATCH_SIZE     = 16
CALLS_PER_TEXT = 3           # self-consistency votes (seeds 1-3)
TEMPERATURE    = 0

SYSTEM_MSG = (
    "You are a strict grader for the Task-AI Capability Index (TACI).\n"
    "Return STRICT JSON with keys:\n"
    '  {"rubric_score":1-5,"contradiction_flag":0|1,"reason":string?}\n\n'
    "Rubric: 1 unusable • 2 poor • 3 adequate • 4 good • 5 expert.\n"
    "Set contradiction_flag=1 if the answer contradicts itself OR the task.\n"
    'Include \"reason\" only when rubric_score ≤ 2 (max 40 words).'
)
USER_TEMPLATE = (
    "### TASK PROMPT\n{prompt}\n\n"
    "### MODEL ANSWER\n{answer}\n"
)

# ───────── helpers ───────────────────────────────────────────────────────
def md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

def load_cache(path: Path) -> Dict[str, dict]:
    return json.loads(path.read_text()) if path.exists() else {}

def save_cache(path: Path, data: Dict[str, dict]) -> None:
    path.write_text(json.dumps(data, indent=2))

def grade_once(task: dict, seed: int, client) -> dict:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        seed=seed,                      # deterministic diversity
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user",
             "content": USER_TEMPLATE.format(prompt=task["prompt"],
                                             answer=task["answer"])},
        ],
    )
    return json.loads(resp.choices[0].message.content)

def majority_vote(results: List[dict]) -> dict:
    scores = [r["rubric_score"] for r in results]
    counter = collections.Counter(scores).most_common()
    voted_score = counter[0][0] if counter else 3
    flag = 1 if any(r["contradiction_flag"] for r in results) else 0
    # choose reason from the lowest score (if ≤2)
    low_reason = ""
    for r in results:
        if r["rubric_score"] <= 2 and r.get("reason"):
            low_reason = r["reason"].strip().replace("\n", " ")
            break
    return {"rubric_score": voted_score,
            "contradiction_flag": flag,
            "rubric_reason": low_reason}

# ───────── main workflow ─────────────────────────────────────────────────
def main(batch: int):
    if not CAND_CSV.exists():
        sys.exit("phase_03_candidates.csv missing. Run the filter first.")

    rows = list(csv.DictReader(CAND_CSV.open()))
    if not rows:
        sys.exit("Candidate CSV is empty.")

    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    offline = (not openai_key) or bool(os.getenv("OFFLINE_GRADER"))
    client = openai.OpenAI(api_key=openai_key) if not offline else None

    cache: Dict[str, dict] = load_cache(CACHE_JSON)
    new_tasks: Dict[str, dict] = {}

    # ---------- build task dict ------------------------------------------
    for r in rows:
        key = md5(r["prompt"] + r["answer"])
        if key not in cache:
            new_tasks[key] = {"prompt": r["prompt"], "answer": r["answer"]}

    # ---------- grade new tasks ------------------------------------------
    if not offline and new_tasks:
        tasks_list = list(new_tasks.items())
        for i in tqdm(range(0, len(tasks_list), batch)):
            sub = tasks_list[i:i+batch]
            for key, task in sub:
                results = []
                for seed in range(1, CALLS_PER_TEXT + 1):
                    try:
                        results.append(grade_once(task, seed, client))
                    except Exception as e:
                        time.sleep(2); results.append(grade_once(task, seed, client))
                cache[key] = majority_vote(results)
        save_cache(CACHE_JSON, cache)

    # ---------- attach scores to rows ------------------------------------
    for r in rows:
        key = md5(r["prompt"] + r["answer"])
        scored = cache.get(key, {"rubric_score":"NA",
                                 "contradiction_flag":"NA",
                                 "rubric_reason":""})
        r.update(scored)

    # ---------- write output ---------------------------------------------
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline='', encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"✓ rubric results → {OUT_CSV}")

# ───────── CLI -----------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = p.parse_args()
    main(args.batch_size)
