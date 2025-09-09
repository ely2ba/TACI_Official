# sampling/sample_tasks.py
from __future__ import annotations

"""
Builds a comprehensive, single-file manifest of O*NET tasks for selected SOCs (v30.0, TXT),
enriching with ratings provenance, modality votes, digital amenability, DWA/IWA/GWA
linkages, Emerging Tasks flags, Technology/Tools with UNSPSC roll-ups (FAMILY, entropy in bits),
Job Zone/SVP, Work Context means (+6), Related Occupations, top Work Activities,
Alternate/Sample Titles (with canonicalized variants), and Education Typical (RL modal).

Inputs expected under data/onet_raw (TXT for v30.0):
- Task Statements.txt, Task Ratings.txt, Occupation Data.txt
- Tasks to DWAs.txt, DWA Reference.txt, (optional) IWA Reference.txt, (optional) Content Model Reference.txt
- Emerging Tasks.txt
- Technology Skills.txt, Tools Used.txt, UNSPSC Reference.txt
- Job Zones.txt
- Work Context.txt
- Related Occupations.txt
- Work Activities.txt
- Alternate Titles.txt, Sample of Reported Titles.txt
- (optional) Education, Training, and Experience.txt, Education, Training, and Experience Categories.txt

Environment:
- OPENAI_API_KEY (optional; if absent, offline labeling)
- MODEL_NAME (optional; default below)
- ONET_VERSION (optional; else read data/ONET_VERSION.txt)
- OFFLINE_GRADER (truthy to force offline)
- TITLES_LIMIT (cap titles per SOC; default 50)

Outputs:
- data/manifests/sampled_tasks_comprehensive.csv               (atomic write)
- data/manifests/sampled_tasks_comprehensive.csv.sha256        (csv hash)
- data/manifests/sampled_tasks_comprehensive.meta.json         (provenance)
- data/manifests/modality_cache_comprehensive.json             (atomic write)
"""

import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pandas.api.types import CategoricalDtype
from unicodedata import normalize as u_normalize

# Optional third-party helpers
try:
    import inflect  # pip install inflect
    infl = inflect.engine()
except Exception:
    infl = None

try:
    import spacy  # pip install spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

try:
    import openai  # pip install openai
except Exception:
    openai = None

from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────
RAW = Path("data/onet_raw")
OUT = Path("data/manifests")
OUT.mkdir(parents=True, exist_ok=True)
CACHE_JSON = OUT / "modality_cache_comprehensive.json"  # votes per (uid:model:prompt:ver:code)

# ── Config ────────────────────────────────────────────────────────────
DEFAULT_MODEL_NAME = "gpt-4.1-mini-2025-04-14"
MODEL_NAME = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
TEMPERATURE = 0.3
VOTES_PER_TASK = 3
SLEEP_MIN, SLEEP_MAX = 0.1, 0.3  # kept but deterministic due to fixed seed

# Allowed normalized labels
ALLOWED = {"TEXT", "GUI", "VISION", "MANUAL", "INCONCLUSIVE"}
FALLBACK_LABEL = "REVIEW"
OFFLINE_LABEL = "UNLABELED"  # used when in offline mode

# Pilot occupations (extend later if desired)
PILOT_SOCs = [
    "23-2011.00",  # Paralegals and Legal Assistants (TEXT)
    "43-4051.00",  # Customer Service Representatives (GUI)
    "51-9061.00",  # Inspectors, Testers, Sorters, Samplers, Weighers (VISION)
]

# UNSPSC roll-up preference (fixed to FAMILY per design) and entropy units
UNSPSC_LEVEL = "FAMILY"
UNSPSC_ENTROPY_UNIT = "bits"
TITLES_LIMIT = int(os.getenv("TITLES_LIMIT", "50"))
SCHEMA_VERSION = os.getenv("SCHEMA_VERSION", "tlc_manifest_schema_v0.4")

# Track which concrete files were read
SOURCE_FILES_USED: set[str] = set()

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

# Code fingerprint and ONET version for cache keys
CODE_FP = hashlib.md5(Path(__file__).read_bytes()).hexdigest()[:8]
ONET_VER_ENV = (os.getenv("ONET_VERSION") or "unspecified").strip()

# ── Robust file loading ───────────────────────────────────────────────
def _candidate_names(base: str) -> List[str]:
    bases = {base, base.replace("_", " "), base.replace(" ", "_")}
    out: set[str] = set()
    for b in bases:
        out.add(b)
        if b.lower().endswith(".txt"):
            out.add(b[:-4] + ".xlsx")
        elif b.lower().endswith(".xlsx"):
            out.add(b[:-5] + ".txt")
        out.add(b.replace("_", " "))
        out.add(b.replace(" ", "_"))
    return sorted(out, key=lambda x: (len(x), x))

def read_onet_table(candidates: List[str]) -> Optional[pd.DataFrame]:
    for base in candidates:
        for name in _candidate_names(base):
            path = RAW / name
            if not path.exists():
                continue
            try:
                if path.suffix.lower() == ".xlsx":
                    df = pd.read_excel(path)
                else:
                    df = pd.read_csv(
                        path, sep="\t", header=0, dtype=None,
                        encoding="utf-8", encoding_errors="replace"
                    )
                df.attrs["source_path"] = str(path)
                SOURCE_FILES_USED.add(path.name)
                return df
            except Exception as e:
                print(f"Warning: failed to read {path}: {e}")
                continue
    return None

# ── Atomic write helpers and hashing ──────────────────────────────────
def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)

def atomic_write_csv(df: pd.DataFrame, path: Path) -> str:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    sha = file_sha256(tmp)
    os.replace(tmp, path)
    return sha

# ── OpenAI call with drift guard ──────────────────────────────────────
def chat_complete(client, **kw):
    # Prefer Chat Completions; fallback to Responses if SDK changes
    try:
        return client.chat.completions.create(**kw)
    except AttributeError:
        return client.responses.create(**kw)

def extract_text(resp) -> str:
    # Try Chat Completions shape
    try:
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        pass
    # Try Responses-style generic extraction
    try:
        if hasattr(resp, "output") and resp.output:
            parts = []
            for block in resp.output:
                for seg in getattr(block, "content", []) or []:
                    if getattr(seg, "type", None) == "output_text":
                        parts.append(getattr(seg, "text", ""))
            if parts:
                return "".join(parts).strip()
    except Exception:
        pass
    return (str(resp) if resp is not None else "")

@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def vote_once(client, statement: str, seed: int) -> str:
    resp = chat_complete(
        client,
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        seed=seed,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": statement.strip() or "Label this task."},
        ],
    )
    raw = extract_text(resp)
    label = normalize_label(raw)
    if label not in (ALLOWED | {FALLBACK_LABEL}):
        raise ValueError(f"Unexpected normalized label: {label} (raw: {raw})")
    return label

# ── Title cleaning helpers ────────────────────────────────────────────
def clean_title(raw: str) -> str:
    if not raw:
        return raw
    txt = re.sub(r"[,&/]", " ", raw)
    txt = re.sub(r"\s{2,}", " ", txt).strip()
    if not nlp:
        words = txt.split()
        out = []
        for i, w in enumerate(words):
            if infl and len(w) >= 4 and w.lower() not in {"and","or"} and w.endswith("s") and (i == 0 or words[i-1].lower() not in {"and","or"}):
                s = infl.singular_noun(w)
                out.append(s if s else w)
            else:
                out.append(w)
        return " ".join(out).title()
    doc = nlp(txt)
    toks = []
    for t in doc:
        if t.tag_ in {"NNS","NNPS"}:
            s = infl.singular_noun(t.text) if infl else None
            toks.append(s if s else (t.lemma_ if t.lemma_ != "-PRON-" else t.text))
        else:
            toks.append(t.text)
    return re.sub(r"\s{2,}", " ", " ".join(toks)).strip().title()

def ascii_fold(s: str) -> str:
    s = s or ""
    return u_normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def canon(s: str) -> str:
    s = ascii_fold(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def unique_by_canon_keep_first(seq: List[str], limit: int = 50) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        t = (x or "").strip()
        if not t:
            continue
        c = canon(t)
        if c and c not in seen:
            seen.add(c)
            out.append(t)
        if len(out) >= limit:
            break
    return out

def semijoin(vals: List[str]) -> str:
    return "; ".join([v for v in vals if isinstance(v, str) and v.strip()])

# ── Label normalization ───────────────────────────────────────────────
def normalize_label(ans: str) -> str:
    if not ans:
        return FALLBACK_LABEL
    s = re.sub(r"[^A-Za-z]", "", ans).upper()
    syn = {
        "TEXTUAL": "TEXT", "LANGUAGE": "TEXT", "WRITING": "TEXT", "DOCUMENT": "TEXT",
        "INTERFACE": "GUI", "UI": "GUI", "WEB": "GUI", "BROWSER": "GUI", "APP": "GUI",
        "VISIONIMAGE": "VISION", "IMAGE": "VISION", "VISUAL": "VISION",
        "PHYSICAL": "MANUAL", "HANDS": "MANUAL", "ONSITE": "MANUAL", "ON SITE": "MANUAL", "FIELD": "MANUAL",
        "UNKNOWN": "INCONCLUSIVE", "UNCERTAIN": "INCONCLUSIVE", "AMBIGUOUS": "INCONCLUSIVE",
        "MULTI": "INCONCLUSIVE", "MULTIMODAL": "INCONCLUSIVE",
    }
    if s in ALLOWED:
        return s
    if s in syn:
        return syn[s]
    return FALLBACK_LABEL

# ── UNSPSC helpers ───────────────────────────────────────────────────
def unspsc_digits(x) -> Optional[str]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    d = re.sub(r"\D", "", str(x))
    return d if d else None

def rollup_family(code: Optional[str]) -> Optional[str]:
    if not code:
        return None
    d = re.sub(r"\D", "", str(code))
    return d[:4] if len(d) >= 4 else None

def entropy_from_counts(counts: Dict[str, int]) -> Optional[float]:
    if not counts:
        return None
    arr = np.array([v for v in counts.values() if v > 0], dtype=float)
    if arr.size == 0:
        return None
    p = arr / arr.sum()
    p = np.clip(p, 1e-12, 1.0)
    # bits for interpretability
    return float(-(p * (np.log2(p))).sum())

# ── Main ──────────────────────────────────────────────────────────────
def main() -> None:
    # Determinism
    random.seed(0)
    np.random.seed(0)

    # Load env late to allow ONET_VERSION override here too
    load_dotenv()
    onet_version_env = (os.getenv("ONET_VERSION") or ONET_VER_ENV or "unspecified").strip()

    # 1) Load core O*NET files
    ts = read_onet_table(["Task Statements.txt", "Task_Statements.txt"])
    tr = read_onet_table(["Task Ratings.txt", "Task_Ratings.txt"])
    occ = read_onet_table(["Occupation Data.txt", "Occupation_Data.txt"])
    if ts is None or tr is None or occ is None:
        raise SystemExit("Missing core O*NET files in data/onet_raw (Task Statements / Task Ratings / Occupation Data).")

    occ = occ.rename(columns={"O*NET-SOC Code": "SOC", "Title": "OccTitleRaw"})

    # Task statements + provenance
    df = ts.rename(columns={
        "O*NET-SOC Code": "SOC",
        "Task ID": "TaskID",
        "Task Statement": "TaskText",
        "Task": "TaskText",
        "Description": "TaskText",
        "Task Type": "TaskType",
        "Category": "TaskType",
        "Incumbents Responding": "task_incumbents_responding",
        "Date": "task_date_raw",
        "Domain Source": "task_domain_source",
        "Title": "task_occ_title_ts",
    })
    # Normalize task_date to YYYY-MM
    if "task_date_raw" in df.columns:
        dt1 = pd.to_datetime(df["task_date_raw"], format="%m/%Y", errors="coerce")
        dt2 = pd.to_datetime(df["task_date_raw"], errors="coerce")
        df["task_date"] = dt1.fillna(dt2).dt.to_period("M").dt.to_timestamp().dt.strftime("%Y-%m")
    else:
        df["task_date"] = None

    if "TaskText" not in df.columns:
        raise SystemExit("Could not find a task text column in Task Statements.")
    if "TaskType" not in df.columns:
        df["TaskType"] = None

    # 2) Task Ratings: keep IM (Importance) and RL (Relevance) only
    scales_present = sorted(tr.get("Scale ID", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    print(f"ℹ️  Task Ratings scales present: {scales_present}")

    # Map RT→RL if needed (defensive)
    if ("RL" not in scales_present) and ("RT" in scales_present):
        tr = tr.copy()
        tr["Scale ID"] = tr["Scale ID"].replace({"RT": "RL"})
        print("⚠️  Mapping RT → RL (Relevance) because RL is missing in this dataset.")

    keep_cols = ["O*NET-SOC Code", "Task ID", "Scale ID", "Data Value", "N", "Date", "Domain Source"]
    tr_sub = tr[tr["Scale ID"].isin(["IM", "RL"])][keep_cols].copy()
    tr_sub = tr_sub.rename(columns={
        "O*NET-SOC Code": "SOC",
        "Task ID": "TaskID",
        "Data Value": "DataValue",
        "N": "N_resp",
        "Date": "ratings_month",
        "Domain Source": "DomainSource",
    })
    _dt1 = pd.to_datetime(tr_sub["ratings_month"], format="%m/%Y", errors="coerce")
    _dt2 = pd.to_datetime(tr_sub["ratings_month"], errors="coerce")
    tr_sub["ratings_month_dt"] = _dt1.fillna(_dt2).dt.to_period("M").dt.to_timestamp()

    src_cat = CategoricalDtype(categories=["Incumbent", "Analyst", "Other"], ordered=True)
    tr_sub["DomainSource"] = tr_sub["DomainSource"].astype(str).str.title()
    tr_sub["DomainSourceCat"] = tr_sub["DomainSource"].where(
        tr_sub["DomainSource"].isin(src_cat.categories), "Other"
    ).astype(src_cat)

    tr_best = (
        tr_sub.sort_values(
            ["SOC", "TaskID", "Scale ID", "ratings_month_dt", "DomainSourceCat", "N_resp"],
            ascending=[True, True, True, False, True, False]
        )
        .drop_duplicates(["SOC", "TaskID", "Scale ID"], keep="first")
    )

    piv = (
        tr_best.assign(ratings_month_iso=tr_best["ratings_month_dt"].dt.strftime("%Y-%m"))
        .pivot_table(
            index=["SOC", "TaskID"],
            columns="Scale ID",
            values=["DataValue", "N_resp", "ratings_month_iso", "DomainSource"],
            aggfunc="first",
        )
    )
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index().rename(columns={
        "DataValue_IM": "Importance",
        "N_resp_IM": "importance_n_respondents",
        "ratings_month_iso_IM": "ratings_month_im",
        "DomainSource_IM": "ratings_source_im",
        "DataValue_RL": "Relevance",
        "N_resp_RL": "relevance_n_respondents",
        "ratings_month_iso_RL": "ratings_month_rl",
        "DomainSource_RL": "ratings_source_rl",
    })
    for c in ["Importance","Relevance","importance_n_respondents","relevance_n_respondents"]:
        if c in piv.columns:
            piv[c] = pd.to_numeric(piv[c], errors="coerce")

    for col in [
        "Importance","importance_n_respondents","ratings_month_im","ratings_source_im",
        "Relevance","relevance_n_respondents","ratings_month_rl","ratings_source_rl",
    ]:
        if col not in piv.columns:
            piv[col] = None

    # Merge statements + ratings + occupation titles
    df = df.merge(piv, on=["SOC","TaskID"], how="left").merge(occ, on="SOC", how="left")

    # Clean titles and deterministic uid (12 hex)
    df["OccTitleClean"] = df["OccTitleRaw"].apply(clean_title)
    df["uid"] = df.apply(lambda r: hashlib.md5(f"{r['SOC']}-{r['TaskID']}".encode()).hexdigest()[:12], axis=1)

    # Relevance retention (≥25 means task is “relevant” under O*NET rule)
    df["retained_by_relevance_rule"] = (df["Relevance"] >= 25).where(df["Relevance"].notna(), None)

    # 3) Select ALL tasks for pilot SOCs
    comprehensive_tasks = df[df["SOC"].isin(PILOT_SOCs)].reset_index(drop=True)
    if comprehensive_tasks.empty:
        raise SystemExit("No tasks found for specified occupations — check O*NET files and SOC codes.")
    print(f"📋 Found tasks for {len(PILOT_SOCs)} occupations:")
    for soc in PILOT_SOCs:
        sub = comprehensive_tasks[comprehensive_tasks["SOC"] == soc]
        t = sub["OccTitleClean"].iloc[0] if len(sub) else "Unknown"
        print(f"   {soc}: {len(sub)} tasks ({t})")

    # 4) Titles: Alternate and Sample (SOC-level; replicated per task row)
    alt = read_onet_table(["Alternate Titles.txt", "Alternate_Titles.txt", "Alternate Occupation Titles.txt", "Alternate_Occupation_Titles.txt"])
    samp = read_onet_table(["Sample of Reported Titles.txt", "Sample_Of_Reported_Titles.txt", "Sample of Reported Titles by Occupation.txt"])

    titles_by_soc = {}
    if alt is not None and "O*NET-SOC Code" in alt.columns:
        alt = alt.rename(columns={"O*NET-SOC Code": "SOC"})
        alt_name_col = None
        for c in ["Alternate Title", "Alternate Titles", "Alternate occupation title", "Alternate Occupation Title", "Alternate Job Title", "Alternate Job Titles", "Title"]:
            if c in alt.columns:
                alt_name_col = c
                break
        if alt_name_col:
            for soc, g in alt.groupby("SOC"):
                lst = unique_by_canon_keep_first([str(x) for x in g[alt_name_col].dropna().astype(str).tolist()], TITLES_LIMIT)
                titles_by_soc.setdefault(soc, {})["alt_titles_raw"] = semijoin(lst)
                titles_by_soc[soc]["alt_titles_canon"] = semijoin([canon(x) for x in lst])

    if samp is not None and "O*NET-SOC Code" in samp.columns:
        samp = samp.rename(columns={"O*NET-SOC Code": "SOC"})
        samp_name_col = None
        for c in ["Reported Job Title", "Reported Titles", "Sample of Reported Titles", "Title"]:
            if c in samp.columns:
                samp_name_col = c
                break
        if samp_name_col:
            for soc, g in samp.groupby("SOC"):
                lst = unique_by_canon_keep_first([str(x) for x in g[samp_name_col].dropna().astype(str).tolist()], TITLES_LIMIT)
                titles_by_soc.setdefault(soc, {})["sample_titles_raw"] = semijoin(lst)
                titles_by_soc[soc]["sample_titles_canon"] = semijoin([canon(x) for x in lst])

    if titles_by_soc:
        tb = pd.DataFrame(
            [{"SOC": k, **v} for k, v in titles_by_soc.items()]
        )
        comprehensive_tasks = comprehensive_tasks.merge(tb, on="SOC", how="left")
    else:
        comprehensive_tasks["alt_titles_raw"] = None
        comprehensive_tasks["alt_titles_canon"] = None
        comprehensive_tasks["sample_titles_raw"] = None
        comprehensive_tasks["sample_titles_canon"] = None

    # Title canon examples (up to 3)
    def pick_canon_examples(row) -> Optional[str]:
        parts = []
        for col in ["alt_titles_canon", "sample_titles_canon"]:
            s = row.get(col)
            if pd.isna(s) or not s:
                continue
            parts.extend([p.strip() for p in str(s).split(";") if p.strip()])
        # unique in order
        seen, out = set(), []
        for p in parts:
            if p not in seen:
                seen.add(p)
                out.append(p)
            if len(out) >= 3:
                break
        return "; ".join(out) if out else None

    comprehensive_tasks["title_canon_examples"] = comprehensive_tasks.apply(pick_canon_examples, axis=1)

    # 5) DWA→IWA→GWA linkages
    t2d = read_onet_table(["Tasks to DWAs.txt", "Task to DWA.txt", "Task_to_DWA.txt"])
    dwa_ref = read_onet_table(["DWA Reference.txt", "Detailed Work Activities.txt", "Detailed_Work_Activities.txt"])
    iwa_ref = read_onet_table(["IWA Reference.txt", "IWA_Reference.txt"])
    cm_ref = read_onet_table(["Content Model Reference.txt", "Content_Model_Reference.txt"])

    if t2d is not None:
        t2d = t2d.rename(columns={
            "O*NET-SOC Code": "SOC",
            "Task ID": "TaskID",
            "DWA ID": "DWA_ID",
            "DWA Title": "DWA_Title"
        })
        # If DWA_Title missing, try to backfill from dwa_ref
        if "DWA_Title" not in t2d.columns:
            if dwa_ref is not None:
                _dr = dwa_ref.rename(columns={"DWA ID": "DWA_ID", "DWA Title": "DWA_Title"})
                t2d = t2d.merge(_dr[["DWA_ID", "DWA_Title"]].drop_duplicates(), on="DWA_ID", how="left")
            else:
                t2d["DWA_Title"] = None

        # Build reference for IWA/GWA
        iwa_title_map = {}
        if iwa_ref is not None:
            _iw = iwa_ref.rename(columns={"IWA ID": "IWA_ID", "IWA Title": "IWA_Title"})
            if "IWA_ID" in _iw.columns and "IWA_Title" in _iw.columns:
                iwa_title_map = dict(
                    _iw[["IWA_ID", "IWA_Title"]].dropna().drop_duplicates().itertuples(index=False, name=None)
                )

        gwa_title_map = {}
        if cm_ref is not None:
            _cm = cm_ref.rename(columns={"Element ID": "GWA_ID", "Element Name": "GWA_Title"})
            if "GWA_ID" in _cm.columns and "GWA_Title" in _cm.columns:
                gwa_title_map = dict(
                    _cm[["GWA_ID", "GWA_Title"]].dropna().drop_duplicates().itertuples(index=False, name=None)
                )

        link = t2d[["SOC", "TaskID", "DWA_ID", "DWA_Title"]].dropna(subset=["DWA_ID"]).copy()

        # Attach IWA and GWA via DWA Reference
        if dwa_ref is not None:
            dr = dwa_ref.rename(columns={
                "DWA ID": "DWA_ID",
                "IWA ID": "IWA_ID",
                "Element ID": "GWA_ID",
                "DWA Title": "DWA_Title_DR"
            })
            keep_cols = [c for c in ["DWA_ID", "IWA_ID", "GWA_ID"] if c in dr.columns]
            dr = dr[keep_cols].drop_duplicates()
            link = link.merge(dr, on="DWA_ID", how="left")
        else:
            link["IWA_ID"] = None
            link["GWA_ID"] = None

        # Map titles for IWA/GWA if available
        link["IWA_Title"] = link["IWA_ID"].map(iwa_title_map) if iwa_title_map else None
        link["GWA_Title"] = link["GWA_ID"].map(gwa_title_map) if gwa_title_map else None

        # Aggregate lists per (SOC, TaskID)
        def uniq_list(s: pd.Series) -> List[str]:
            vals = [str(x) for x in s if pd.notna(x) and str(x) != ""]
            seen, out = set(), []
            for v in vals:
                if v not in seen:
                    seen.add(v)
                    out.append(v)
            return out

        agg = (link.groupby(["SOC", "TaskID"], dropna=False)
                    .agg(
                        dwa_ids=("DWA_ID", uniq_list),
                        dwa_titles=("DWA_Title", uniq_list),
                        iwa_id_list=("IWA_ID", uniq_list),
                        iwa_title_list=("IWA_Title", uniq_list),
                        gwa_id_list=("GWA_ID", uniq_list),
                        gwa_title_list=("GWA_Title", uniq_list),
                    )
                    .reset_index())

        for col in ["dwa_ids","dwa_titles","iwa_id_list","iwa_title_list","gwa_id_list","gwa_title_list"]:
            agg[col + "_count"] = agg[col].apply(lambda x: len(x) if isinstance(x, list) else 0)
            agg[col] = agg[col].apply(lambda lst: ";".join(lst) if isinstance(lst, list) else "")

        comprehensive_tasks = comprehensive_tasks.merge(agg, on=["SOC", "TaskID"], how="left")
    else:
        for c in ["dwa_ids","dwa_titles","dwa_ids_count","iwa_id_list","iwa_title_list","iwa_id_list_count","iwa_title_list_count","gwa_id_list","gwa_title_list","gwa_id_list_count","gwa_title_list_count"]:
            comprehensive_tasks[c] = None

    # 6) Emerging Tasks (SOC counts + per-task revision flag)
    emerg = read_onet_table(["Emerging Tasks.txt", "Emerging_Tasks.txt", "Emerging_Tasks.xlsx"])
    if emerg is not None:
        emerg = emerg.rename(columns={
            "O*NET-SOC Code": "SOC",
            "Category": "EmergingCategory",
            "Original Task ID": "OriginalTaskID",
            "Task": "EmergingTaskText"
        })
        soc_counts = emerg.groupby(["SOC","EmergingCategory"], dropna=False).size().unstack(fill_value=0)
        soc_counts.columns = [f"soc_emerging_{c.lower()}_count" for c in soc_counts.columns]
        soc_counts = soc_counts.reset_index()
        comprehensive_tasks = comprehensive_tasks.merge(soc_counts, on="SOC", how="left")
        for col in ["soc_emerging_new_count","soc_emerging_revision_count"]:
            if col not in comprehensive_tasks.columns:
                comprehensive_tasks[col] = 0
        rev_keys = set(emerg.loc[emerg["EmergingCategory"].astype(str).str.lower().eq("revision"),
                                  "OriginalTaskID"].dropna().astype(str).tolist())
        comprehensive_tasks["is_emerging_revision"] = comprehensive_tasks["TaskID"].astype(str).isin(rev_keys)
    else:
        comprehensive_tasks["soc_emerging_new_count"] = 0
        comprehensive_tasks["soc_emerging_revision_count"] = 0
        comprehensive_tasks["is_emerging_revision"] = False

    # 7) Technology Skills (SOC-level) + UNSPSC FAMILY roll-up
    tech = read_onet_table(["Technology Skills.txt", "Technology_Skills.txt"])
    unspsc_ref = read_onet_table(["UNSPSC Reference.txt", "UNSPSC_Reference.txt"])
    if tech is not None:
        tech = tech.rename(columns={
            "O*NET-SOC Code": "SOC",
            "Hot Technology": "HotTechnology",
            "In Demand": "InDemand",
            "Example": "TechnologyName",
            "Commodity Title": "Commodity Title",
            "Commodity Code": "Commodity Code",
            "Code": "Commodity Code"
        })
        tech["HotTechnology"] = tech.get("HotTechnology", False)
        tech["InDemand"] = tech.get("InDemand", False)
        tech["HotTechnology"] = tech["HotTechnology"].astype(str).str.upper().isin(["Y","YES","TRUE","1"])
        tech["InDemand"] = tech["InDemand"].astype(str).str.upper().isin(["Y","YES","TRUE","1"])
        tech["TechnologyName"] = tech.get("TechnologyName", tech.get("Commodity Title"))
        # Rank and aggregate top examples per SOC
        tech["weight"] = 1 + tech["HotTechnology"].astype(int) + tech["InDemand"].astype(int)
        tech_rank = (
            tech.groupby(["SOC", "TechnologyName"], dropna=False)["weight"].sum().reset_index()
                .sort_values(["SOC", "weight", "TechnologyName"], ascending=[True, False, True])
        )
        soc_counts = tech.groupby("SOC").agg(
            hot_tech_count=("HotTechnology", "sum"),
            in_demand_tech_count=("InDemand", "sum"),
        ).reset_index()
        top_examples = (
            tech_rank.groupby("SOC")
                .agg(top_hot_tech_examples=("TechnologyName", lambda s: "; ".join([x for x in s.head(3) if isinstance(x, str) and x])))
                .reset_index()
        )
        soc_agg = soc_counts.merge(top_examples, on="SOC", how="left")

        # UNSPSC FAMILY roll-up
        fam_labels = {}
        if unspsc_ref is not None:
            ref = unspsc_ref.rename(columns={
                "Commodity Code": "Commodity Code",
                "Commodity Title": "Commodity Title",
                "Family Code": "Family Code",
                "Family Title": "Family Title",
                "Class Code": "Class Code",
                "Class Title": "Class Title",
            })
            ref["Commodity Code"] = ref["Commodity Code"].apply(unspsc_digits)
            ref["Family Code"] = ref["Family Code"].apply(unspsc_digits)
            fam_labels = dict(
                ref.dropna(subset=["Family Code","Family Title"])[["Family Code","Family Title"]]
                   .drop_duplicates().itertuples(index=False, name=None)
            )

        tech_codes = tech.copy()
        tech_codes["Commodity Code"] = tech_codes.get("Commodity Code", None)
        tech_codes["Commodity Code"] = tech_codes["Commodity Code"].apply(unspsc_digits)
        if unspsc_ref is not None:
            fam_map = dict(
                unspsc_ref.rename(columns={"Commodity Code": "cc", "Family Code": "fc"})
                          .assign(cc=lambda d: d["cc"].apply(unspsc_digits),
                                  fc=lambda d: d["fc"].apply(unspsc_digits))
                          .dropna(subset=["cc","fc"])[["cc","fc"]].drop_duplicates()
                          .itertuples(index=False, name=None)
            )
            tech_codes["Family Code"] = tech_codes["Commodity Code"].map(fam_map)
        else:
            tech_codes["Family Code"] = tech_codes["Commodity Code"].apply(rollup_family)

        counts = (tech_codes.dropna(subset=["Family Code"])
                            .groupby(["SOC","Family Code"]).size().reset_index(name="cnt"))
        def fam_label(fc: str) -> str:
            name = fam_labels.get(fc, "")
            return f"{fc}:{name}".strip(":")
        if not counts.empty:
            tech_top3 = (counts.sort_values(["SOC","cnt","Family Code"], ascending=[True, False, True])
                              .groupby("SOC")
                              .apply(lambda g: "; ".join([fam_label(r["Family Code"]) for _, r in g.head(3).iterrows()]))
                              .reset_index(name="tech_family_top3"))
            ent = counts.groupby("SOC") \
                        .apply(lambda g: entropy_from_counts(dict(zip(g["Family Code"], g["cnt"])))) \
                        .reset_index(name="tech_family_entropy")
            soc_agg = soc_agg.merge(tech_top3, on="SOC", how="left").merge(ent, on="SOC", how="left")
        else:
            soc_agg["tech_family_top3"] = None
            soc_agg["tech_family_entropy"] = None

        comprehensive_tasks = comprehensive_tasks.merge(soc_agg, on="SOC", how="left")
    else:
        comprehensive_tasks["hot_tech_count"] = None
        comprehensive_tasks["in_demand_tech_count"] = None
        comprehensive_tasks["top_hot_tech_examples"] = None
        comprehensive_tasks["tech_family_top3"] = None
        comprehensive_tasks["tech_family_entropy"] = None

    # 8) Tools Used (SOC-level) + UNSPSC FAMILY
    tools = read_onet_table(["Tools Used.txt", "Tools_Used.txt"])
    if tools is not None:
        tools = tools.rename(columns={"O*NET-SOC Code": "SOC"})
        if "Example" in tools.columns:
            tools = tools.rename(columns={"Example": "ToolName"})
        elif "Tool" in tools.columns:
            tools = tools.rename(columns={"Tool": "ToolName"})
        elif "Commodity Title" in tools.columns:
            tools["ToolName"] = tools["Commodity Title"].astype(str)
        else:
            tools["ToolName"] = None

        tools["Commodity Code"] = tools.get("Commodity Code", tools.get("UNSPSC Code", tools.get("Code")))
        tools["Commodity Code"] = tools["Commodity Code"].apply(unspsc_digits)
        if unspsc_ref is not None:
            fam_map = dict(
                unspsc_ref.rename(columns={"Commodity Code": "cc", "Family Code": "fc"})
                          .assign(cc=lambda d: d["cc"].apply(unspsc_digits),
                                  fc=lambda d: d["fc"].apply(unspsc_digits))
                          .dropna(subset=["cc","fc"])[["cc","fc"]].drop_duplicates()
                          .itertuples(index=False, name=None)
            )
            tools["Family Code"] = tools["Commodity Code"].map(fam_map)
            fam_labels_tools = dict(
                unspsc_ref.rename(columns={"Family Code":"fc","Family Title":"ft"})
                          .dropna(subset=["fc","ft"])[["fc","ft"]].drop_duplicates()
                          .itertuples(index=False, name=None)
            )
        else:
            tools["Family Code"] = tools["Commodity Code"].apply(rollup_family)
            fam_labels_tools = {}

        counts = (tools.dropna(subset=["Family Code"])
                        .groupby(["SOC","Family Code"]).size().reset_index(name="cnt"))
        def fam_label_t(fc: str) -> str:
            name = fam_labels_tools.get(fc, "")
            return f"{fc}:{name}".strip(":")
        if not counts.empty:
            tools_top3 = (counts.sort_values(["SOC","cnt","Family Code"], ascending=[True, False, True])
                               .groupby("SOC")
                               .apply(lambda g: "; ".join([fam_label_t(r["Family Code"]) for _, r in g.head(3).iterrows()]))
                               .reset_index(name="tool_family_top3"))
            ent = counts.groupby("SOC") \
                        .apply(lambda g: entropy_from_counts(dict(zip(g["Family Code"], g["cnt"])))) \
                        .reset_index(name="tool_family_entropy")
            comprehensive_tasks = comprehensive_tasks.merge(tools_top3, on="SOC", how="left").merge(ent, on="SOC", how="left")
        else:
            comprehensive_tasks["tool_family_top3"] = None
            comprehensive_tasks["tool_family_entropy"] = None
    else:
        comprehensive_tasks["tool_family_top3"] = None
        comprehensive_tasks["tool_family_entropy"] = None

    # 9) Job Zones / SVP (SOC-level)
    jz = read_onet_table(["Job Zones.txt", "Job_Zones.txt"])
    if jz is not None:
        jz = jz.rename(columns={"O*NET-SOC Code":"SOC","Job Zone":"job_zone","SVP Range":"svp_range"})
        if "svp_range" not in jz.columns:
            jz["svp_range"] = None
        cols = [c for c in ["SOC","job_zone","svp_range"] if c in jz.columns]
        comprehensive_tasks = comprehensive_tasks.merge(jz[cols].drop_duplicates("SOC"), on="SOC", how="left")
        for c in ["job_zone","svp_range"]:
            if c not in comprehensive_tasks.columns:
                comprehensive_tasks[c] = None
    else:
        comprehensive_tasks["job_zone"] = None
        comprehensive_tasks["svp_range"] = None

    # 10) Work Context (means only, CT/CX), with +6 extras
    wc = read_onet_table(["Work Context.txt", "Work_Context.txt"])
    wc_mapped_counts_by_soc = {}
    wc_total_ctcx_by_soc = {}
    unmapped_wc = set()
    if wc is not None:
        wc = wc.rename(columns={
            "O*NET-SOC Code": "SOC",
            "Element Name": "ElementName",
            "Scale ID": "ScaleID",
            "Data Value": "DataValue",
            "Not Relevant": "NotRelevant",
            "Recommend Suppress": "RecommendSuppress",
        })
        wc = wc[wc["ScaleID"].astype(str).str.upper().isin(["CT","CX"])].copy()
        wc["NotRelevant"] = wc.get("NotRelevant", "N").astype(str).str.upper()
        wc["RecommendSuppress"] = wc.get("RecommendSuppress", "N").astype(str).str.upper()
        wc = wc[(wc["NotRelevant"] != "Y") & (wc["RecommendSuppress"] != "Y")]

        def map_wc(name):
            n = str(name or "").lower().replace("–", "-").replace("—", "-")
            if ("e-mail" in n) or ("electronic mail" in n) or ("email" in n):
                return "wc_electronic_mail"
            if "telephone" in n:
                return "wc_telephone"
            if ("face-to-face" in n) or ("face to face" in n):
                return "wc_face_to_face"
            if "physical proximity" in n:
                return "wc_physical_proximity"
            if "exact or accurate" in n:
                return "wc_exact_or_accurate"
            if ("handle, control, or feel objects" in n) or ("handle, control" in n) or ("using your hands" in n):
                return "wc_hands_on"
            # New +6
            if "consequence of error" in n:
                return "wc_consequence_of_error"
            if "time pressure" in n:
                return "wc_time_pressure"
            if "freedom to make decisions" in n:
                return "wc_freedom_to_make_decisions"
            if ("structured versus unstructured" in n) or ("structured vs unstructured" in n) or ("structured work" in n and "unstructured" in n):
                return "wc_structured_vs_unstructured"
            if "deal with external customers" in n or "deal with customers" in n:
                return "wc_deal_with_external_customers"
            if ("indoors" in n) and (("environmentally controlled" in n) or ("environmental control" in n)):
                return "wc_indoors_env_controlled"
            unmapped_wc.add(str(name))
            return None

        wc["col"] = wc["ElementName"].apply(map_wc)
        # coverage accounting
        wc_total = wc.groupby("SOC").size().to_dict()
        wc_total_ctcx_by_soc.update(wc_total)

        wc_sel = wc[wc["col"].notna()].copy()
        wc_mapped = wc_sel.groupby("SOC").size().to_dict()
        wc_mapped_counts_by_soc.update(wc_mapped)

        wc_piv = wc_sel.groupby(["SOC", "col"])['DataValue'].mean().unstack().reset_index()
        comprehensive_tasks = comprehensive_tasks.merge(wc_piv, on="SOC", how="left")

        for c in [
            "wc_electronic_mail","wc_telephone","wc_face_to_face","wc_physical_proximity","wc_hands_on","wc_exact_or_accurate",
            "wc_consequence_of_error","wc_time_pressure","wc_freedom_to_make_decisions",
            "wc_structured_vs_unstructured","wc_deal_with_external_customers","wc_indoors_env_controlled",
        ]:
            if c not in comprehensive_tasks.columns:
                comprehensive_tasks[c] = None
    else:
        for c in [
            "wc_electronic_mail","wc_telephone","wc_face_to_face","wc_physical_proximity","wc_hands_on","wc_exact_or_accurate",
            "wc_consequence_of_error","wc_time_pressure","wc_freedom_to_make_decisions",
            "wc_structured_vs_unstructured","wc_deal_with_external_customers","wc_indoors_env_controlled",
        ]:
            comprehensive_tasks[c] = None

    # 11) Related Occupations (optional) — rank by Relatedness Tier, then Index
    rel = read_onet_table(["Related Occupations.txt", "Related_Occupations.txt"])
    if rel is not None:
        rel = rel.rename(columns={
            "O*NET-SOC Code": "SOC",
            "Related O*NET-SOC Code": "RelatedSOC",
            "Related Title": "RelatedTitle",
            "Related Occupation": "RelatedTitle",
            "Relatedness Tier": "RelatednessTier",
            "Index": "RelatedIndex",
        })
        if "RelatedTitle" not in rel.columns:
            rel["RelatedTitle"] = ""

        occ_titles = occ[["SOC","OccTitleRaw"]].rename(columns={"SOC":"RelatedSOC","OccTitleRaw":"RelatedTitle_backfill"})
        rel = rel.merge(occ_titles, on="RelatedSOC", how="left")
        if "RelatedTitle_backfill" in rel.columns:
            rel["RelatedTitle"] = rel["RelatedTitle"].where(
                rel["RelatedTitle"].notna() & (rel["RelatedTitle"].astype(str).str.len() > 0),
                rel["RelatedTitle_backfill"]
            )
            rel = rel.drop(columns=["RelatedTitle_backfill"])

        tier_order = {"Primary-Short": 3, "Primary-Long": 2, "Supplemental": 1}
        rel["tier_rank"]  = rel["RelatednessTier"].map(tier_order).fillna(0)
        rel["idx_rank"]   = pd.to_numeric(rel.get("RelatedIndex"), errors="coerce").fillna(9999)
        rel_sorted = rel.sort_values(["SOC","tier_rank","idx_rank"], ascending=[True, False, True])

        def _top_titles(s: pd.Series) -> str:
            vals = [x for x in s.dropna().astype(str).tolist() if x]
            return "; ".join(vals[:3])

        top_by_soc = (rel_sorted.groupby("SOC")
                      .agg(related_occ_count=("RelatedSOC","nunique"),
                           top_related_titles=("RelatedTitle", _top_titles))
                      .reset_index())
        comprehensive_tasks = comprehensive_tasks.merge(top_by_soc, on="SOC", how="left")
    else:
        comprehensive_tasks["related_occ_count"] = None
        comprehensive_tasks["top_related_titles"] = None

    # 12) Work Activities (optional) — rank by Importance (IM) scale only
    wa = read_onet_table(["Work Activities.txt", "Work_Activities.txt"])
    if wa is not None:
        wa = wa.rename(columns={"O*NET-SOC Code":"SOC","Element Name":"WA_Name",
                                "Scale ID":"ScaleID","Data Value":"WA_Value"})
        wa = wa[wa["ScaleID"].astype(str).str.upper().eq("IM")]
        if ("WA_Value" not in wa.columns) or ("WA_Name" not in wa.columns):
            comprehensive_tasks["top_work_activities"] = None
        else:
            topwa = (wa.groupby(["SOC","WA_Name"])["WA_Value"].mean()
                       .reset_index()
                       .sort_values(["SOC","WA_Value"], ascending=[True, False]))
            wa_top3 = (topwa.groupby("SOC")
                       .agg(top_work_activities=("WA_Name", lambda s: "; ".join(s.head(3))))
                       .reset_index())
            comprehensive_tasks = comprehensive_tasks.merge(wa_top3, on="SOC", how="left")
    else:
        comprehensive_tasks["top_work_activities"] = None

    # 13) LLM modality classification (3 votes) with strong cache keys
    offline = bool(os.getenv("OFFLINE_GRADER")) or not os.getenv("OPENAI_API_KEY") or (openai is None)
    cache = {}
    if CACHE_JSON.exists():
        try:
            cache = json.loads(CACHE_JSON.read_text(encoding="utf-8"))
            for k, v in list(cache.items()):
                if isinstance(v, str):
                    cache[k] = [v]
        except Exception:
            print(f"Warning: could not parse cache at {CACHE_JSON}, starting fresh.")
            cache = {}

    client = (openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if not offline else None)

    PROMPT_MD5 = hashlib.md5(SYSTEM_PROMPT.encode()).hexdigest()
    PROMPT_SALT = PROMPT_MD5[:8]
    def cache_key_for(uid: str) -> str:
        return f"{uid}:{MODEL_NAME}:{PROMPT_SALT}:{onet_version_env}:{CODE_FP}"

    if offline:
        print("⚠️  Offline voting: UNLABELED * VOTES_PER_TASK for all tasks (no API key or client).")
    print(f"🔄  Classifying modality for {len(comprehensive_tasks)} tasks...")
    vote_cols = [f"vote{i+1}" for i in range(VOTES_PER_TASK)]
    all_votes: List[List[str]] = []
    final_labels: List[str] = []

    # itertuples for performance
    for row in tqdm(comprehensive_tasks.itertuples(index=False), total=len(comprehensive_tasks), desc="Classifying modality"):
        uid = getattr(row, "uid")
        stmt = (getattr(row, "TaskText", "") or "").strip()
        key = cache_key_for(uid)

        votes = None
        if key in cache:
            vv = cache[key]
            if isinstance(vv, list) and len(vv) == VOTES_PER_TASK:
                votes = vv

        if votes is None:
            if offline:
                votes = [OFFLINE_LABEL] * VOTES_PER_TASK
            else:
                seeds = list(range(1, VOTES_PER_TASK + 1))
                votes = [vote_once(client, stmt, seed) for seed in seeds]
            cache[key] = votes
            # deterministic sleep (seeded)
            time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

        c = Counter(votes)
        top, freq = c.most_common(1)[0]
        if freq >= 2 and top in (ALLOWED | {OFFLINE_LABEL}):
            label = top
        else:
            allowed_votes = [v for v in votes if v in ALLOWED]
            label = "INCONCLUSIVE" if (len(allowed_votes) >= 1 and len(set(allowed_votes)) > 1) else FALLBACK_LABEL
        final_labels.append(label)
        all_votes.append(votes)

    # persist cache atomically
    try:
        atomic_write_text(CACHE_JSON, json.dumps(cache, indent=2))
    except Exception as e:
        print(f"Warning: could not write cache: {e}")

    for i, col in enumerate(vote_cols):
        comprehensive_tasks[col] = [vs[i] if len(vs) > i else None for vs in all_votes]

    comprehensive_tasks["modality"] = final_labels
    comprehensive_tasks["modality_agreement"] = comprehensive_tasks[vote_cols].apply(lambda r: Counter([x for x in r if x]).most_common(1)[0][1], axis=1)
    comprehensive_tasks["modality_confidence"] = comprehensive_tasks["modality_agreement"] / VOTES_PER_TASK
    modality_cat = CategoricalDtype(
        categories=["TEXT", "GUI", "VISION", "MANUAL", "INCONCLUSIVE", "REVIEW", "UNLABELED"],
        ordered=False,
    )
    comprehensive_tasks["modality"] = comprehensive_tasks["modality"].astype(modality_cat)
    UNCERTAIN = {"INCONCLUSIVE", "REVIEW", "UNLABELED"}
    comprehensive_tasks["modality_uncertain"] = comprehensive_tasks["modality"].isin(UNCERTAIN)
    comprehensive_tasks["modality_disagreement"] = (comprehensive_tasks["modality_agreement"] < VOTES_PER_TASK) | comprehensive_tasks["modality_uncertain"]

    # Derived flags
    comprehensive_tasks["digital_amenable"] = comprehensive_tasks["modality"].isin({"TEXT", "GUI", "VISION"})
    comprehensive_tasks["amenability_reason"] = comprehensive_tasks["modality"].map({
        "TEXT": "Language-only I/O suffices.",
        "GUI": "Requires operating software UI.",
        "VISION": "Requires visual inspection/recognition.",
        "MANUAL": "Requires physical/manual action.",
        "INCONCLUSIVE": "Ambiguous task description.",
        "REVIEW": "No majority; needs human review.",
        "UNLABELED": "Offline mode; not labeled.",
    })
    code_map = {
        "TEXT": "LANGUAGE_ONLY",
        "GUI": "GUI_SOFTWARE",
        "VISION": "VISUAL_PERCEPTION",
        "MANUAL": "PHYSICAL_MANUAL",
        "INCONCLUSIVE": "AMBIGUOUS",
        "REVIEW": "REVIEW",
        "UNLABELED": "UNLABELED",
    }
    comprehensive_tasks["amenability_code"] = comprehensive_tasks["modality"].map(code_map)

    # Explicit stubs
    comprehensive_tasks["needs_stub"] = comprehensive_tasks["modality"].isin({"MANUAL", "INCONCLUSIVE", "REVIEW", "UNLABELED"})
    comprehensive_tasks["stub_type"] = comprehensive_tasks["modality"].map({
        "MANUAL": "MANUAL",
        "INCONCLUSIVE": "AMBIGUOUS",
        "REVIEW": "REVIEW",
        "UNLABELED": "UNLABELED",
    }).fillna("NONE")

    # Normalize importance per SOC (weight for aggregation)
    if "Importance" in comprehensive_tasks.columns:
        comprehensive_tasks["Importance"] = pd.to_numeric(comprehensive_tasks["Importance"], errors="coerce").fillna(0.0)
        _soc_sum = comprehensive_tasks.groupby("SOC")["Importance"].transform("sum")
        comprehensive_tasks["importance_weight_norm"] = 0.0
        mask = _soc_sum > 0
        comprehensive_tasks.loc[mask, "importance_weight_norm"] = comprehensive_tasks.loc[mask, "Importance"] / _soc_sum[mask]
        # Relevance-gated weights (RL >= 25); treat None as True when RL missing
        if "retained_by_relevance_rule" in comprehensive_tasks.columns:
            mask_rl = comprehensive_tasks["retained_by_relevance_rule"].fillna(True)
            comprehensive_tasks["importance_weight_norm_rl"] = 0.0
            soc_sum_rl = comprehensive_tasks.loc[mask_rl].groupby("SOC")["Importance"].transform("sum")
            ok = mask_rl & soc_sum_rl.gt(0)
            comprehensive_tasks.loc[ok, "importance_weight_norm_rl"] = comprehensive_tasks.loc[ok, "Importance"] / soc_sum_rl[ok]

    # 14) Education Typical (modal RL category mapped to description) — SOC-level
    ete = read_onet_table(["Education, Training, and Experience.txt", "Education_Training_and_Experience.txt", "Education.txt"])
    etec = read_onet_table(["Education, Training, and Experience Categories.txt", "Education_Training_and_Experience_Categories.txt", "ETE Categories.txt"])
    education_map = {}
    if etec is not None:
        etec = etec.rename(columns={"Element ID":"ElementID","Element Name":"ElementName","Scale ID":"ScaleID",
                                    "Category":"Category","Category Description":"CategoryDescription"})
        rl_cat = etec[etec["ScaleID"].astype(str).str.upper().eq("RL")][["Category","CategoryDescription"]].dropna()
        education_map = dict(rl_cat.drop_duplicates().itertuples(index=False, name=None))
    education_by_soc = None
    if ete is not None:
        ete = ete.rename(columns={"O*NET-SOC Code":"SOC","Scale ID":"ScaleID","Category":"Category","Data Value":"DataValue"})
        rl = ete[ete["ScaleID"].astype(str).str.upper().eq("RL")].copy()
        if not rl.empty:
            rl["DataValue"] = pd.to_numeric(rl["DataValue"], errors="coerce")
            rl["Category"] = pd.to_numeric(rl["Category"], errors="coerce")
            # Stable tie-break: higher DataValue, then higher Category
            rl = rl.sort_values(["SOC","DataValue","Category"], ascending=[True, False, False]).dropna(subset=["Category"])
            modal = rl.drop_duplicates(["SOC"], keep="first")[["SOC","Category","DataValue"]].copy()
            modal["education_typical_category"] = modal["Category"].astype("Int64")
            modal["education_typical_share"] = modal["DataValue"].astype(float)
            modal["education_typical"] = modal["Category"].map(education_map).fillna(modal["Category"].astype(str))
            education_by_soc = modal[["SOC","education_typical_category","education_typical","education_typical_share"]]
    if education_by_soc is not None:
        comprehensive_tasks = comprehensive_tasks.merge(education_by_soc, on="SOC", how="left")
    else:
        comprehensive_tasks["education_typical_category"] = pd.Series([pd.NA]*len(comprehensive_tasks), dtype="Int64")
        comprehensive_tasks["education_typical"] = None
        comprehensive_tasks["education_typical_share"] = None

    # Provenance columns (filled later for final values)
    manifest_version = pd.Timestamp.utcnow().strftime("%Y%m%d") + "-v1"
    comprehensive_tasks["onetsrc_version"] = onet_version_env or "unspecified"
    comprehensive_tasks["model_name"] = MODEL_NAME
    comprehensive_tasks["votes_per_task"] = VOTES_PER_TASK
    comprehensive_tasks["vote_seeds"] = ",".join(map(str, range(1, VOTES_PER_TASK + 1)))
    comprehensive_tasks["generated_utc"] = pd.Timestamp.utcnow().isoformat(timespec="seconds")
    comprehensive_tasks["schema_version"] = SCHEMA_VERSION
    comprehensive_tasks["manifest_version"] = manifest_version
    comprehensive_tasks["source_files_used"] = ";".join(sorted(SOURCE_FILES_USED))
    comprehensive_tasks["pipeline_stage"] = "sample_tasks_comprehensive"
    comprehensive_tasks["unspsc_rollup_level"] = UNSPSC_LEVEL
    comprehensive_tasks["unspsc_entropy_unit"] = UNSPSC_ENTROPY_UNIT
    comprehensive_tasks["modality_prompt_md5"] = PROMPT_MD5
    comprehensive_tasks["code_fingerprint"] = CODE_FP

    # 15) Save manifest atomically and write meta/hash
    out_file = OUT / "sampled_tasks_comprehensive.csv"
    csv_sha = atomic_write_csv(comprehensive_tasks, out_file)
    # write sha sidecar
    atomic_write_text(out_file.with_suffix(out_file.suffix + ".sha256"), csv_sha)

    # Meta sidecar
    meta = {
        "manifest_path": str(out_file),
        "rows": int(len(comprehensive_tasks)),
        "schema_version": SCHEMA_VERSION,
        "manifest_version": manifest_version,
        "onetsrc_version": onet_version_env or "unspecified",
        "unspsc_rollup_level": UNSPSC_LEVEL,
        "unspsc_entropy_unit": UNSPSC_ENTROPY_UNIT,
        "source_files_used": sorted(SOURCE_FILES_USED),
        "model_name": MODEL_NAME,
        "votes_per_task": VOTES_PER_TASK,
        "vote_seeds": list(range(1, VOTES_PER_TASK + 1)),
        "prompt_md5": PROMPT_MD5,
        "code_fingerprint": CODE_FP,
        "generated_utc": pd.Timestamp.utcnow().isoformat(timespec="seconds"),
        "csv_sha256": csv_sha,
    }
    atomic_write_text(OUT / "sampled_tasks_comprehensive.meta.json", json.dumps(meta, indent=2))

    # 16) Summary stats
    print(f"✅  Generated {len(comprehensive_tasks)} tasks → {out_file}")
    print(f"🔐  CSV SHA-256: {csv_sha}")
    modality_counts = Counter(comprehensive_tasks["modality"])
    print(f"📊  Modality distribution: {dict(modality_counts)}")
    print("📈  Tasks by occupation:")
    for soc in PILOT_SOCs:
        soc_count = (comprehensive_tasks["SOC"] == soc).sum()
        occ_title = comprehensive_tasks.loc[comprehensive_tasks["SOC"] == soc, "OccTitleClean"].iloc[0] if soc_count > 0 else "Unknown"
        print(f"   {soc}: {soc_count:3d} tasks ({occ_title})")
    try:
        digital_share = comprehensive_tasks.groupby("SOC")["digital_amenable"].mean().to_dict()
        imp_mass_digital = (
            comprehensive_tasks.assign(_dig=comprehensive_tasks["digital_amenable"].astype(bool))
              .groupby("SOC")
              .apply(lambda g: float(g.loc[g._dig, "importance_weight_norm"].sum() if "importance_weight_norm" in g else 0.0))
              .to_dict()
        )
        imp_mass_digital_rl = (
            comprehensive_tasks.assign(_dig=comprehensive_tasks["digital_amenable"].astype(bool))
              .groupby("SOC")
              .apply(lambda g: float(g.loc[g._dig, "importance_weight_norm_rl"].sum() if "importance_weight_norm_rl" in g else 0.0))
              .to_dict()
        )
        print("ℹ️  Digital share by SOC:", digital_share)
        print("ℹ️  Importance mass in digital tasks (plain) by SOC:", imp_mass_digital)
        print("ℹ️  Importance mass in digital tasks (RL≥25) by SOC:", imp_mass_digital_rl)
    except Exception:
        pass

    try:
        if "importance_weight_norm" in comprehensive_tasks.columns:
            wsum = comprehensive_tasks.groupby("SOC")["importance_weight_norm"].sum().round(6).to_dict()
            print("🧮  Weight sums by SOC:", wsum)

        dup_count = int(comprehensive_tasks.duplicated(["SOC", "TaskID", "uid"]).sum())
        print(f"🔁  Duplicate (SOC,TaskID,uid) rows: {dup_count}")

        cols_im = {"ratings_month_im", "ratings_source_im"}
        cols_rl = {"ratings_month_rl", "ratings_source_rl"}
        if cols_im.issubset(comprehensive_tasks.columns) and cols_rl.issubset(comprehensive_tasks.columns):
            print("🧾  Ratings month/source by SOC:")
            for soc, g in comprehensive_tasks.groupby("SOC"):
                ims = sorted({f"{m}/{s}" for m, s in zip(g["ratings_month_im"], g["ratings_source_im"]) if (pd.notna(m) or pd.notna(s))})
                rls = sorted({f"{m}/{s}" for m, s in zip(g["ratings_month_rl"], g["ratings_source_rl"]) if (pd.notna(m) or pd.notna(s))})
                print(f"   {soc} IM: {'; '.join(ims)} | RL: {'; '.join(rls)}")

        if wc_total_ctcx_by_soc:
            print("ℹ️  Work Context mapping coverage by SOC (mapped/total CT/CX rows):",
                  {k: f"{wc_mapped_counts_by_soc.get(k,0)}/{v}" for k, v in wc_total_ctcx_by_soc.items()})
        if unmapped_wc:
            sample = list(sorted(unmapped_wc))[:10]
            print(f"ℹ️  Work Context unmapped names (sample): {sample} (+{max(0,len(unmapped_wc)-10)} more)")
    except Exception as e:
        print("⚠️  Sanity report error:", e)

if __name__ == "__main__":
    main()