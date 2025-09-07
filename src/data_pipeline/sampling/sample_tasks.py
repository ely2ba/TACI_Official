# sampling/sample_tasks.py
"""
Build a comprehensive task manifest for TACI across pilot occupations,
with multi-vote (3x) modality determination and explicit handling of
MANUAL / INCONCLUSIVE outcomes so we can safely stub non-digital tasks.

What this script does
---------------------
1) Loads O*NET Task Statements, Task Ratings (Importance), and Occupation Data.
2) Selects ALL tasks for the specified SOCs (no cherry-picking).
3) Cleans & singularizes occupation titles for display (keeps raw title for joins).
4) Creates deterministic task `uid` (md5(SOC-TaskID)[:8]).
5) Classifies each task with a 3-vote LLM consensus into:
       TEXT | GUI | VISION | MANUAL | INCONCLUSIVE
   - Majority (≥2/3) wins; otherwise → REVIEW.
   - Stores all three raw votes.
6) Derives:
   - digital_amenable = True iff modality ∈ {TEXT, GUI, VISION}
   - needs_stub = True iff modality ∈ {MANUAL, INCONCLUSIVE, REVIEW, UNLABELED}
   - modality_agreement = {1,2,3}
   - modality_disagreement flag
7) Writes CSV to data/manifests/sampled_tasks_comprehensive.csv
   and caches per-task votes to data/manifests/modality_cache_comprehensive.json

Environment variables
---------------------
OPENAI_API_KEY    : If missing, runs in OFFLINE mode (labels become UNLABELED).
OFFLINE_GRADER    : If set to any truthy value, forces OFFLINE mode.
ONET_VERSION      : Optional provenance tag (e.g., "DB_28.1").
MODEL_NAME        : Optional override of the default LLM name.

Notes
-----
- Default pilot SOCs (3) are set below; swap/extend to 20+ SOCs as needed.
- We separate **modality** from **digital amenability** so reviewers can't
  accuse us of entangling label semantics with executability decisions.
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
from typing import Dict, List, Optional

import inflect              # pip install inflect
import openai               # pip install openai
import pandas as pd
import spacy                # pip install spacy
from dotenv import load_dotenv
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────
RAW = Path("data/onet_raw")
OUT = Path("data/manifests")
OUT.mkdir(parents=True, exist_ok=True)
CACHE_JSON = OUT / "modality_cache_comprehensive.json"  # stores votes per uid

# ── Config ────────────────────────────────────────────────────────────
DEFAULT_MODEL_NAME = "gpt-4.1-mini-2025-04-14"
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
TEMPERATURE = 0.3
VOTES_PER_TASK = 3
SLEEP_MIN, SLEEP_MAX = 0.1, 0.3

# Allowed normalized labels
ALLOWED = {"TEXT", "GUI", "VISION", "MANUAL", "INCONCLUSIVE"}
FALLBACK_LABEL = "REVIEW"
OFFLINE_LABEL = "UNLABELED"  # used when in offline mode

# Pilot occupations (extend to 20+ later)
PILOT_SOCs = [
    "23-2011.00",  # Paralegals and Legal Assistants (mostly TEXT)
    "43-4051.00",  # Customer Service Representatives (mostly GUI)
    "51-9061.00",  # Quality Control Inspectors (mostly VISION)
]

# ── System prompt (deterministic, one-token output) ───────────────────
SYSTEM_PROMPT = """You label the PRIMARY interface modality needed for an AI system to complete a task.

Return EXACTLY one token from: TEXT, GUI, VISION, MANUAL, INCONCLUSIVE.

Definitions:
- TEXT: The task can be completed using language-only I/O (reading/writing text). No on-screen navigation is essential.
- GUI: The task requires operating software interfaces (clicking, typing into forms, selecting menus, navigating apps/sites).
- VISION: The task requires visual inspection/recognition of images/objects (e.g., detect defects, read diagrams, identify components).
- MANUAL: The task requires physical/manual activity, on-site presence, or non-digital manipulation (e.g., lift/install/repair/assemble/operate machinery).
- INCONCLUSIVE: The task description is too ambiguous or lacks enough detail to decide.

Tie-breakers:
- If both reading/writing text AND navigating software are essential → GUI.
- If any essential step requires visual inspection/recognition → VISION.
- If physical/manual action is essential → MANUAL.
- If none of the above clearly apply, prefer TEXT.
- If still uncertain, label INCONCLUSIVE.

Answer with only one of: TEXT, GUI, VISION, MANUAL, INCONCLUSIVE.
"""

 

# ── Utilities: NLP setup for cleaning titles ──────────────────────────
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy model 'en_core_web_sm' not found. Install with:")
    print("   python -m spacy download en_core_web_sm")
    nlp = None

infl = inflect.engine()


def clean_title(raw: str) -> str:
    """
    Clean and singularize occupation titles using NLP-based approach.

    Examples:
        "Claims Adjusters, Examiners, and Investigators" → "Claim Adjuster Examiner And Investigator"
        "Human Resources Specialists" → "Human Resource Specialist"
        "News Analysts, Reporters, and Journalists" → "News Analyst Reporter And Journalist"
    """
    if not raw:
        return raw

    if not nlp:
        return _clean_title_fallback(raw)

    # Normalize punctuation: keep 'and' but remove commas and slashes
    clean = re.sub(r"[,&/]", " ", raw)
    clean = re.sub(r"\s{2,}", " ", clean).strip()

    doc = nlp(clean)
    tokens = []
    for tok in doc:
        if tok.tag_ in {"NNS", "NNPS"}:  # plural nouns
            singular = infl.singular_noun(tok.text)
            tokens.append(singular if singular else (tok.lemma_ if tok.lemma_ != "-PRON-" else tok.text))
        else:
            tokens.append(tok.text)

    result = " ".join(tokens)
    result = re.sub(r"\s{2,}", " ", result).strip()
    return result.title()


def _clean_title_fallback(raw: str) -> str:
    """Lightweight fallback if spaCy is not available."""
    txt = re.sub(r"[,&/]", " ", raw or "")
    txt = re.sub(r"\s{2,}", " ", txt).strip()
    words = txt.split()
    cleaned = []
    for i, word in enumerate(words):
        if (
            len(word) >= 4
            and word.lower() not in {"and", "or", "this", "analysis", "basis"}
            and word.endswith("s")
            and (i == 0 or words[i - 1].lower() not in {"and", "or"})
        ):
            singular = infl.singular_noun(word)
            cleaned.append(singular if singular else word)
        else:
            cleaned.append(word)
    result = " ".join(cleaned)
    result = re.sub(r"\s{2,}", " ", result).strip()
    return result.title()


# ── Cache helpers ─────────────────────────────────────────────────────
def load_cache(path: Path) -> Dict[str, List[str]]:
    if path.exists():
        try:
            data = json.loads(path.read_text())
            # ensure values are lists
            for k, v in list(data.items()):
                if isinstance(v, str):
                    data[k] = [v]
            return data
        except Exception:
            print(f"Warning: could not parse cache at {path}, starting fresh.")
    return {}


def save_cache(path: Path, data: Dict[str, List[str]]) -> None:
    path.write_text(json.dumps(data, indent=2))


# ── Label normalization ───────────────────────────────────────────────
def normalize_label(ans: str) -> str:
    """
    Normalize the model's answer to one of the canonical labels or raise.
    Accepts some common variants/synonyms.
    """
    if not ans:
        return FALLBACK_LABEL
    s = re.sub(r"[^A-Za-z]", "", ans).upper()

    # Synonym map
    syn = {
        "TEXTUAL": "TEXT",
        "LANGUAGE": "TEXT",
        "WRITING": "TEXT",
        "DOCUMENT": "TEXT",
        "INTERFACE": "GUI",
        "UI": "GUI",
        "WEB": "GUI",
        "BROWSER": "GUI",
        "APP": "GUI",
        "VISIONIMAGE": "VISION",
        "IMAGE": "VISION",
        "VISUAL": "VISION",
        "PHYSICAL": "MANUAL",
        "HANDS": "MANUAL",
        "ON SITE": "MANUAL",
        "ONSITE": "MANUAL",
        "FIELD": "MANUAL",
        "UNKNOWN": "INCONCLUSIVE",
        "UNCERTAIN": "INCONCLUSIVE",
        "AMBIGUOUS": "INCONCLUSIVE",
        "MULTI": "INCONCLUSIVE",
        "MULTIMODAL": "INCONCLUSIVE",
    }

    if s in ALLOWED:
        return s
    if s in syn:
        return syn[s]
    # try direct mapping if it exactly equals one known chunk
    for k, v in syn.items():
        if s == k:
            return v
    return FALLBACK_LABEL


# ── OpenAI call with retry ────────────────────────────────────────────
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def vote_once(client: openai.OpenAI, statement: str, seed: int) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        seed=seed,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": statement.strip() or "Label this task."},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    label = normalize_label(raw)
    if label not in (ALLOWED | {FALLBACK_LABEL}):
        raise ValueError(f"Unexpected normalized label: {label} (raw: {raw})")
    return label


# ── Main ──────────────────────────────────────────────────────────────
def main() -> None:
    # 1) Load O*NET files
    ts = pd.read_csv(RAW / "Task_Statements.txt", sep="\t", header=0)
    tr = pd.read_csv(RAW / "Task_Ratings.txt", sep="\t", header=0)
    occ = (
        pd.read_csv(RAW / "Occupation_Data.txt", sep="\t", header=0, usecols=["O*NET-SOC Code", "Title"])
        .rename(columns={"O*NET-SOC Code": "SOC", "Title": "OccTitleRaw"})
    )

    # 2) Importance rows (Scale ID = IM)
    # Include month if available for provenance
    keep_cols = ["O*NET-SOC Code", "Task ID", "Data Value"]
    if "N" in tr.columns:
        keep_cols.append("N")
    if "Date" in tr.columns:
        keep_cols.append("Date")
    imp = (
        tr[tr["Scale ID"] == "IM"][keep_cols]
        .rename(
            columns={
                "O*NET-SOC Code": "SOC",
                "Task ID": "TaskID",
                "Data Value": "Importance",
                "N": "ImportanceN",
                "Date": "ratings_month",
            }
        )
    )

    # 3) Merge & keep all tasks (Core/Supplemental/NaN)
    df = ts.rename(columns={"O*NET-SOC Code": "SOC", "Task ID": "TaskID"})
    # Standardize task text
    if "Task" in df.columns:
        df = df.rename(columns={"Task": "TaskText"})
    elif "Task Statement" in df.columns:
        df = df.rename(columns={"Task Statement": "TaskText"})
    elif "Description" in df.columns:
        df = df.rename(columns={"Description": "TaskText"})
    else:
        raise SystemExit("Could not find a task text column in Task_Statements.txt (expected 'Task', 'Task Statement', or 'Description').")

    # Task type if present
    if "Task Type" in df.columns:
        df = df.rename(columns={"Task Type": "TaskType"})
    elif "Category" in df.columns:
        df = df.rename(columns={"Category": "TaskType"})
    else:
        df["TaskType"] = None

    df = df.merge(imp, on=["SOC", "TaskID"], how="left")
    df = df.merge(occ, on="SOC", how="left")

    # 4) Clean titles and deterministic uid (longer for fewer collisions)
    df["OccTitleClean"] = df["OccTitleRaw"].apply(clean_title)
    df["uid"] = df.apply(lambda r: hashlib.md5(f"{r['SOC']}-{r['TaskID']}".encode()).hexdigest()[:12], axis=1)

    # Rename respondent counts to explicit name
    if "ImportanceN" in df.columns:
        df = df.rename(columns={"ImportanceN": "importance_n_respondents"})

    # Normalize ratings_month (MM/YYYY -> YYYY-MM)
    if "ratings_month" in df.columns:
        def _to_iso_month(val: str) -> str:
            if not isinstance(val, str):
                return ""
            m = re.match(r"^(\d{1,2})\/(\d{4})$", val.strip())
            if m:
                mm, yyyy = m.group(1), m.group(2)
                return f"{yyyy}-{int(mm):02d}"
            # pass through if already ISO-like
            m2 = re.match(r"^(\d{4})-(\d{2})(?:-(\d{2}))?$", val.strip())
            return val.strip() if m2 else ""
        df["ratings_month"] = df["ratings_month"].apply(_to_iso_month)

    # 5) Select ALL tasks for specified SOCs
    comprehensive_tasks = df[df["SOC"].isin(PILOT_SOCs)].reset_index(drop=True)
    if comprehensive_tasks.empty:
        raise SystemExit("No tasks found for specified occupations — check O*NET files and SOC codes.")

    print(f"📋 Found tasks for {len(PILOT_SOCs)} occupations:")
    for soc in PILOT_SOCs:
        soc_tasks = comprehensive_tasks[comprehensive_tasks["SOC"] == soc]
        occ_title = soc_tasks["OccTitleClean"].iloc[0] if len(soc_tasks) > 0 else "Unknown"
        print(f"   {soc}: {len(soc_tasks)} tasks ({occ_title})")

    # 6) LLM modality classification (3 votes)
    load_dotenv()
    offline = bool(os.getenv("OFFLINE_GRADER")) or not os.getenv("OPENAI_API_KEY")
    cache = load_cache(CACHE_JSON)
    client: Optional[openai.OpenAI] = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if not offline else None

    # Salt cache by model and prompt to avoid stale reuse across versions
    PROMPT_SALT = hashlib.md5(SYSTEM_PROMPT.encode()).hexdigest()[:8]
    def cache_key_for(uid: str) -> str:
        return f"{uid}:{MODEL_NAME}:{PROMPT_SALT}"

    print(f"🔄  Classifying modality for {len(comprehensive_tasks)} tasks...")
    vote_cols = [f"vote{i+1}" for i in range(VOTES_PER_TASK)]
    all_votes: List[List[str]] = []
    final_labels: List[str] = []

    for _, row in tqdm(comprehensive_tasks.iterrows(), total=len(comprehensive_tasks), desc="Classifying modality"):
        uid = row["uid"]
        stmt = (row.get("TaskText") or "").strip()

        key = cache_key_for(uid)

        # Load cached votes if present (salted)
        if key in cache:
            votes = cache[key]
            votes = votes if isinstance(votes, list) else [votes]
        elif offline:
            votes = [OFFLINE_LABEL] * VOTES_PER_TASK
        else:
            seeds = list(range(1, VOTES_PER_TASK + 1))
            votes = [vote_once(client, stmt, seed) for seed in seeds]
            cache[key] = votes
            # Gentle pacing
            time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

        # Reduce to final label
        c = Counter(votes)
        top, freq = c.most_common(1)[0]
        if freq >= 2 and top in (ALLOWED | {OFFLINE_LABEL}):
            label = top
        else:
            # If we had at least 1 allowed label and the rest are inconclusive-ish, prefer INCONCLUSIVE over REVIEW.
            allowed_votes = [v for v in votes if v in ALLOWED]
            if len(allowed_votes) >= 1 and len(set(allowed_votes)) > 1:
                label = "INCONCLUSIVE"
            else:
                label = FALLBACK_LABEL  # REVIEW

        final_labels.append(label)
        all_votes.append(votes)

    save_cache(CACHE_JSON, cache)

    # 7) Build output frame
    out = comprehensive_tasks.copy()

    # Per-vote columns
    for i, col in enumerate(vote_cols):
        out[col] = [vs[i] if len(vs) > i else None for vs in all_votes]

    out["modality"] = final_labels
    out["modality_agreement"] = out[vote_cols].apply(
        lambda r: Counter([x for x in r if x]).most_common(1)[0][1], axis=1
    )
    # Disagreement means votes not unanimous
    out["modality_disagreement"] = out["modality_agreement"] < VOTES_PER_TASK
    out["modality_confidence"] = out["modality_agreement"] / VOTES_PER_TASK

    # Derived flags
    out["digital_amenable"] = out["modality"].isin({"TEXT", "GUI", "VISION"})
    out["amenability_reason"] = out["modality"].map(
        {
            "TEXT": "Language-only I/O suffices.",
            "GUI": "Requires operating software UI.",
            "VISION": "Requires visual inspection/recognition.",
            "MANUAL": "Requires physical/manual action.",
            "INCONCLUSIVE": "Ambiguous task description.",
            "REVIEW": "No majority; needs human review.",
            "UNLABELED": "Offline mode; not labeled.",
        }
    )
    # Standardized amenability code enum for analysis
    code_map = {
        "TEXT": "LANGUAGE_ONLY",
        "GUI": "GUI_SOFTWARE",
        "VISION": "VISUAL_PERCEPTION",
        "MANUAL": "PHYSICAL_MANUAL",
        "INCONCLUSIVE": "AMBIGUOUS",
        "REVIEW": "REVIEW",
        "UNLABELED": "UNLABELED",
    }
    out["amenability_code"] = out["modality"].map(code_map)

    # Make stubs explicit
    out["needs_stub"] = out["modality"].isin({"MANUAL", "INCONCLUSIVE", "REVIEW", "UNLABELED"})
    out["stub_type"] = out["modality"].map({
        "MANUAL": "MANUAL",
        "INCONCLUSIVE": "AMBIGUOUS",
        "REVIEW": "REVIEW",
        "UNLABELED": "UNLABELED",
    }).fillna("NONE")

    # Normalize importance per SOC (weight for aggregation)
    if "Importance" in out.columns:
        out["Importance"] = out["Importance"].fillna(0)
        _soc_sum = out.groupby("SOC")["Importance"].transform("sum")
        out["importance_weight_norm"] = out["Importance"] / _soc_sum
        out.loc[_soc_sum == 0, "importance_weight_norm"] = 0
    out["needs_stub"] = out["modality"].isin({"MANUAL", "INCONCLUSIVE", "REVIEW", "UNLABELED"})

    # Provenance
    # O*NET source version from env or file (prefer explicit value)
    onet_version = os.getenv("ONET_VERSION")
    if not onet_version:
        version_file = Path("data/ONET_VERSION.txt")
        if version_file.exists():
            try:
                onet_version = version_file.read_text(encoding="utf-8").strip()
            except Exception:
                onet_version = None
    out["onetsrc_version"] = onet_version or "unspecified"
    out["model_name"] = MODEL_NAME
    out["votes_per_task"] = VOTES_PER_TASK
    out["vote_seeds"] = ",".join(map(str, range(1, VOTES_PER_TASK + 1)))
    out["generated_utc"] = pd.Timestamp.utcnow().isoformat(timespec="seconds")

    # 8) Save manifest
    out_file = OUT / "sampled_tasks_comprehensive.csv"
    out.to_csv(out_file, index=False)

    # 9) Summary stats
    print(f"✅  Generated {len(out)} tasks → {out_file}")
    modality_counts = Counter(out["modality"])
    print(f"📊  Modality distribution: {dict(modality_counts)}")
    print(f"📈  Tasks by occupation:")
    for soc in PILOT_SOCs:
        soc_count = (out["SOC"] == soc).sum()
        occ_title = out.loc[out["SOC"] == soc, "OccTitleClean"].iloc[0] if soc_count > 0 else "Unknown"
        print(f"   {soc}: {soc_count:3d} tasks ({occ_title})")
    # Per-SOC digital share and importance mass
    try:
        digital_share = (
            out.groupby("SOC")["digital_amenable"].mean().to_dict()
        )
        if "importance_weight_norm" in out.columns:
            imp_mass_digital = (
                out.assign(_dig=out["digital_amenable"].astype(bool))
                  .groupby("SOC")
                  .apply(lambda g: float(g.loc[g._dig, "importance_weight_norm"].sum()))
                  .to_dict()
            )
        else:
            imp_mass_digital = {}
        print("ℹ️  Digital share by SOC:", digital_share)
        if imp_mass_digital:
            print("ℹ️  Importance mass in digital tasks by SOC:", imp_mass_digital)
    except Exception:
        pass
    print("ℹ️  Flags/metrics: digital_amenable, amenability_code, needs_stub, stub_type, modality_agreement/confidence/disagreement.")


if __name__ == "__main__":
    main()
