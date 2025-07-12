#!/usr/bin/env python3
"""
Phase-03 Rubric Grader — Paralegal slice
(GPT-4o-mini · 3-vote self-consistency · cached)

Reads
-----
phase_03_candidates_paralegal.csv   (from Phase-03 filter)

Writes
------
phase_03_rubric_per_output_paralegal.csv
rubric_cache.json   (persistent cache)

Each row gets:
  accuracy, coverage, depth, style, utility, specificity  (1-5 ints)
  rubric_score                                            (1-5 int; min-rule)
  weighted_avg                                            (0-5 float)
  explanation                                             (≤ 25 words for low scores)
"""
from __future__ import annotations
import argparse, csv, json, hashlib, os, sys, time
from pathlib import Path
from typing import Dict, List
import numpy as np

from dotenv import load_dotenv
import openai
from tqdm import tqdm

# ───────── paths / constants ─────────────────────────────────────────
HERE           = Path(__file__).resolve().parent
CAND_CSV       = HERE / "phase_03_candidates_paralegal.csv"
OUT_CSV        = HERE / "phase_03_rubric_per_output_paralegal.csv"
CACHE_JSON     = HERE / "rubric_cache_one_occ.json"

MODEL_NAME     = "gpt-4o-mini"
BATCH_SIZE     = 16
CALLS_PER_TEXT = 3           # self-consistency votes (seed 1-3)
TEMPERATURE    = 0           # deterministic for o-mini

AXES = ["accuracy", "coverage", "depth", "style", "utility", "specificity"]
WEIGHTS = dict(accuracy=0.30, coverage=0.20,
               depth=0.15, style=0.10,
               utility=0.15, specificity=0.10)

SYSTEM_MSG = (
    "You are a senior domain-expert grader for the Task-AI Capability Index "
    "(TACI). Structure, wrapper compliance, and policy safety have ALREADY "
    "been validated. Judge ONLY professional quality.\n\n"
    "Return STRICT JSON with keys exactly:\n"
    '{"accuracy":1-5,"coverage":1-5,"depth":1-5,"style":1-5,"utility":1-5,'
    '"specificity":1-5,"explanation":string}\n\n'
    "Scoring guide (1 = unusable, 3 = adequate draft, 5 = expert-grade):\n"
    "• Accuracy   – factual correctness, no contradictions\n"
    "• Coverage   – all sub-tasks answered\n"
    "• Depth      – reasoning quality, insight\n"
    "• Style      – correct professional tone/format\n"
    "• Utility    – ready for use without heavy edits\n"
    "• Specificity– task-relevant details; penalise boilerplate\n\n"
    "Start each axis at 3. If unsure, ROUND DOWN.\n"
    "Include \"explanation\" (≤ 25 words) whenever ANY axis ≠ 3.\n"
    "A score of 5 MUST be justified by a concrete detail."
)

USER_TEMPLATE = (
    "### TASK PROMPT\n{prompt}\n\n"
    "### MODEL ANSWER\n{answer}\n"
)

# ───────── helper functions ───────────────────────────────────────────
md5 = lambda s: hashlib.md5(s.encode()).hexdigest()

def load_cache(path: Path) -> Dict[str, dict]:
    return json.loads(path.read_text()) if path.exists() else {}

def save_cache(path: Path, data: Dict[str, dict]) -> None:
    path.write_text(json.dumps(data, indent=2))

def safe_int(v):
    try:
        iv = int(v)
        return iv if 1 <= iv <= 5 else None
    except Exception:
        return None

def validate_axes(d: dict) -> dict | None:
    # Ensure all axis scores are ints 1-5
    for ax in AXES:
        iv = safe_int(d.get(ax))
        if iv is None:
            return None
        d[ax] = iv

    d["explanation"] = d.get("explanation", "").strip()

    # Require justification for any 5; downgrade to 4 if missing/weak
    if any(d[ax] == 5 for ax in AXES):
        if len(d["explanation"]) < 10:
            for ax in AXES:
                if d[ax] == 5:
                    d[ax] = 4
            d["explanation"] = "5-score justification too short; downgraded"
    return d

def grade_once(task: dict, seed: int, client) -> dict:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        seed=seed,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": USER_TEMPLATE.format(**task)},
        ],
    )
    raw  = json.loads(resp.choices[0].message.content)
    parsed = validate_axes(raw)
    if parsed is None:
        # fallback to worst case
        return {ax: 1 for ax in AXES} | {"explanation": "bad schema"}
    return parsed

def aggregate_votes(votes: List[dict]) -> dict:
    # median per axis
    med = {ax: int(round(np.median([v[ax] for v in votes]))) for ax in AXES}
    med["rubric_score"] = min(med.values())  # strict min-rule

    # weighted average (research metric)
    med["weighted_avg"] = round(sum(WEIGHTS[ax]*med[ax] for ax in AXES), 2)

    # explanation (first low vote)
    med["explanation"] = ""
    if med["rubric_score"] <= 3:
        for v in votes:
            if any(v[ax] <= 3 for ax in AXES) and v.get("explanation"):
                med["explanation"] = v["explanation"].strip().replace("\n", " ")
                break
    return med

# ───────── main workflow ──────────────────────────────────────────────
def main(batch_size: int) -> None:
    if not CAND_CSV.exists():
        sys.exit("phase_03_candidates_paralegal.csv missing — run the filter first.")

    rows = list(csv.DictReader(CAND_CSV.open()))
    if not rows:
        sys.exit("Candidate CSV is empty — nothing to grade.")

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    offline = (not api_key) or bool(os.getenv("OFFLINE_GRADER"))
    client  = openai.OpenAI(api_key=api_key) if not offline else None

    cache = load_cache(CACHE_JSON)
    pending: Dict[str, dict] = {
        md5(r["prompt"] + r["answer"]): {"prompt": r["prompt"], "answer": r["answer"]}
        for r in rows if md5(r["prompt"] + r["answer"]) not in cache
    }

    # ── grade new tasks ------------------------------------------------
    if not offline and pending:
        items = list(pending.items())
        for i in tqdm(range(0, len(items), batch_size), desc="rubric batches"):
            for key, task in items[i:i+batch_size]:
                votes = []
                for seed in range(1, CALLS_PER_TEXT + 1):
                    try:
                        votes.append(grade_once(task, seed, client))
                    except Exception:
                        time.sleep(2)
                        votes.append(grade_once(task, seed, client))
                cache[key] = aggregate_votes(votes)
        save_cache(CACHE_JSON, cache)

    # ── attach scores --------------------------------------------------
    for r in rows:
        scored = cache.get(md5(r["prompt"] + r["answer"]), {
            **{ax: "NA" for ax in AXES},
            "rubric_score": "NA",
            "weighted_avg": "NA",
            "explanation": ""
        })
        r.update(scored)

    # ── write output ---------------------------------------------------
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline='', encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"✓ rubric results → {OUT_CSV}")

# ───────── CLI ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    main(parser.parse_args().batch_size)
