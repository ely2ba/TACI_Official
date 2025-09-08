# sampling/sample_tasks.py
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
import pandas as pd
import spacy                # pip install spacy
from dotenv import load_dotenv
from pandas.api.types import CategoricalDtype
from tenacity import retry, wait_random_exponential, stop_after_attempt
from tqdm import tqdm

# Optional import; handle offline if unavailable or no key
try:
    import openai           # pip install openai
except Exception:
    openai = None

# ── Paths ─────────────────────────────────────────────────────────────
RAW = Path("data/onet_raw")
OUT = Path("data/manifests")
OUT.mkdir(parents=True, exist_ok=True)
CACHE_JSON = OUT / "modality_cache_comprehensive.json"  # votes per (uid:model:prompt)

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
    "23-2011.00",  # Paralegals and Legal Assistants (TEXT)
    "43-4051.00",  # Customer Service Representatives (GUI)
    "51-9061.00",  # Inspectors, Testers, Sorters, Samplers, Weighers (VISION via MVTec)
    # swap to "43-3021.00" for Billing & Posting Clerks if you choose doc-vision later
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

# ── Robust file loading ───────────────────────────────────────────────
def _candidate_names(base: str) -> List[str]:
    """Build a deterministic, de-duplicated list of filename candidates.
    Tries space/underscore variants and .txt/.xlsx swaps.
    """
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
    # Prefer shorter, exact names first; then alphabetical
    return sorted(out, key=lambda x: (len(x), x))

def read_onet_table(candidates: List[str]) -> Optional[pd.DataFrame]:
    for base in candidates:
        for name in _candidate_names(base):
            path = RAW / name
            if not path.exists():
                continue
            try:
                if path.suffix.lower() == ".xlsx":
                    return pd.read_excel(path)
                return pd.read_csv(path, sep="\t", header=0, dtype=None, encoding_errors="ignore")
            except Exception as e:
                print(f"Warning: failed to read {path}: {e}")
                continue
    return None

# ── Title cleaning ────────────────────────────────────────────────────
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None
infl = inflect.engine()

def clean_title(raw: str) -> str:
    if not raw:
        return raw
    txt = re.sub(r"[,&/]", " ", raw)
    txt = re.sub(r"\s{2,}", " ", txt).strip()
    if not nlp:
        words = txt.split()
        out = []
        for i, w in enumerate(words):
            if len(w) >= 4 and w.lower() not in {"and","or"} and w.endswith("s") and (i == 0 or words[i-1].lower() not in {"and","or"}):
                s = infl.singular_noun(w)
                out.append(s if s else w)
            else:
                out.append(w)
        return " ".join(out).title()
    doc = nlp(txt)
    toks = []
    for t in doc:
        if t.tag_ in {"NNS","NNPS"}:
            s = infl.singular_noun(t.text)
            toks.append(s if s else (t.lemma_ if t.lemma_ != "-PRON-" else t.text))
        else:
            toks.append(t.text)
    return re.sub(r"\s{2,}", " ", " ".join(toks)).strip().title()

# ── Cache helpers ─────────────────────────────────────────────────────
def load_cache(path: Path) -> Dict[str, List[str]]:
    if path.exists():
        try:
            data = json.loads(path.read_text())
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

# ── OpenAI call with retry ────────────────────────────────────────────
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
def vote_once(client, statement: str, seed: int) -> str:
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
    # 1) Load core O*NET files
    ts = read_onet_table(["Task Statements.txt", "Task_Statements.txt"])
    tr = read_onet_table(["Task Ratings.txt", "Task_Ratings.txt"])
    occ = read_onet_table(["Occupation Data.txt", "Occupation_Data.txt"])
    if ts is None or tr is None or occ is None:
        raise SystemExit("Missing core O*NET files in data/onet_raw (Task Statements / Task Ratings / Occupation Data).")

    # Occupation titles
    occ = occ.rename(columns={"O*NET-SOC Code": "SOC", "Title": "OccTitleRaw"})

    # Task statements + provenance
    df = ts.rename(columns={
        "O*NET-SOC Code": "SOC",
        "Task ID": "TaskID",
        "Task": "TaskText",
        "Task Statement": "TaskText",
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

    # Ensure required task fields exist
    if "TaskText" not in df.columns:
        raise SystemExit("Could not find a task text column in Task Statements.")

    if "TaskType" not in df.columns:
        df["TaskType"] = None

    # 2) Task Ratings: keep IM (Importance) and RL (Relevance) only
    scales_present = sorted(tr.get("Scale ID", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    print(f"ℹ️  Task Ratings scales present: {scales_present}")

    # Always map RT->RL if RL is missing (O*NET 30.0 sometimes uses RT)
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
    # Parse month; prefer %m/%Y but fall back generously
    _dt1 = pd.to_datetime(tr_sub["ratings_month"], format="%m/%Y", errors="coerce")
    _dt2 = pd.to_datetime(tr_sub["ratings_month"], errors="coerce")
    tr_sub["ratings_month_dt"] = _dt1.fillna(_dt2).dt.to_period("M").dt.to_timestamp()

    # Source preference: Incumbent > Analyst > Other
    src_cat = CategoricalDtype(categories=["Incumbent", "Analyst", "Other"], ordered=True)
    tr_sub["DomainSource"] = tr_sub["DomainSource"].astype(str).str.title()
    tr_sub["DomainSourceCat"] = tr_sub["DomainSource"].where(
        tr_sub["DomainSource"].isin(src_cat.categories), "Other"
    ).astype(src_cat)

    # Pick best row per (SOC, TaskID, Scale): latest month, then source pref, then higher N
    tr_best = (
        tr_sub.sort_values(
            ["SOC", "TaskID", "Scale ID", "ratings_month_dt", "DomainSourceCat", "N_resp"],
            ascending=[True, True, True, False, True, False]
        )
        .drop_duplicates(["SOC", "TaskID", "Scale ID"], keep="first")
    )

    # Pivot to IM/RL, keeping month & source provenance
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
    # Coerce numeric where applicable
    for c in ["Importance","Relevance","importance_n_respondents","relevance_n_respondents"]:
        if c in piv.columns:
            piv[c] = pd.to_numeric(piv[c], errors="coerce")

    # Ensure expected columns exist
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

    # Relevance retention (29.0+): ≥25 means task is “relevant” under O*NET rule
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

    # 4) DWA linkages
    t2d = read_onet_table(["Tasks to DWAs.txt", "Task to DWA.txt", "Task_to_DWA.txt"])
    dwa_ref = read_onet_table(["DWA Reference.txt", "Detailed Work Activities.txt", "Detailed_Work_Activities.txt"])
    if t2d is not None:
        t2d = t2d.rename(columns={
            "O*NET-SOC Code": "SOC",
            "Task ID": "TaskID",
            "DWA ID": "DWA_ID",
            "DWA Title": "DWA_Title"
        })
        if "DWA_Title" not in t2d.columns and dwa_ref is not None:
            dwa_ref = dwa_ref.rename(columns={"DWA ID":"DWA_ID","DWA Title":"DWA_Title"})
            t2d = t2d.merge(dwa_ref[["DWA_ID","DWA_Title"]].drop_duplicates(), on="DWA_ID", how="left")
        key = ["SOC","TaskID"] if "SOC" in t2d.columns else ["TaskID"]
        g = (t2d.groupby(key, dropna=False)
              .agg(dwa_ids=("DWA_ID", lambda x: ";".join(sorted({str(i) for i in x if pd.notna(i)}))),
                   dwa_titles=("DWA_Title", lambda x: ";".join(sorted({str(i) for i in x if pd.notna(i)}))))
              .reset_index())
        g["dwa_count"] = g["dwa_ids"].apply(lambda s: 0 if not isinstance(s,str) or s=="" else len(s.split(";")))
        comprehensive_tasks = comprehensive_tasks.merge(g, on=key, how="left")
    else:
        comprehensive_tasks["dwa_ids"] = None
        comprehensive_tasks["dwa_titles"] = None
        comprehensive_tasks["dwa_count"] = 0

    # 5) Emerging Tasks (SOC counts + per-task revision flag)
    emerg = read_onet_table(["Emerging Tasks.txt", "Emerging_Tasks.xlsx"])
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

    # 6) Technology Skills (SOC-level)
    tech = read_onet_table(["Technology Skills.txt", "Technology_Skills.txt"])
    if tech is not None:
        tech = tech.rename(columns={
            "O*NET-SOC Code": "SOC",
            "Hot Technology": "HotTechnology",
            "In Demand": "InDemand",
        })
        # Name column: Example preferred; fallback to Commodity Title
        if "Example" in tech.columns:
            tech = tech.rename(columns={"Example": "TechnologyName"})
        elif "Technology Skill" in tech.columns:
            tech = tech.rename(columns={"Technology Skill": "TechnologyName"})
        elif "Commodity Title" in tech.columns:
            tech["TechnologyName"] = tech["Commodity Title"].astype(str)
        else:
            tech["TechnologyName"] = None
        tech["HotTechnology"] = tech.get("HotTechnology", False)
        tech["InDemand"] = tech.get("InDemand", False)
        tech["HotTechnology"] = tech["HotTechnology"].astype(str).str.upper().isin(["Y","YES","TRUE","1"])
        tech["InDemand"] = tech["InDemand"].astype(str).str.upper().isin(["Y","YES","TRUE","1"])
        soc_agg = tech.groupby("SOC").agg(
            hot_tech_count=("HotTechnology","sum"),
            in_demand_tech_count=("InDemand","sum"),
            top_hot_tech_examples=("TechnologyName", lambda s: "; ".join(sorted({x for x in s.dropna().astype(str)})[:3]))
        ).reset_index()
        comprehensive_tasks = comprehensive_tasks.merge(soc_agg, on="SOC", how="left")
    else:
        comprehensive_tasks["hot_tech_count"] = None
        comprehensive_tasks["in_demand_tech_count"] = None
        comprehensive_tasks["top_hot_tech_examples"] = None

    # 7) Tools Used (SOC-level)
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
        soc_tools = tools.groupby("SOC").agg(
            top_tools_examples=("ToolName", lambda s: "; ".join(sorted({x for x in s.dropna().astype(str)})[:3]))
        ).reset_index()
        comprehensive_tasks = comprehensive_tasks.merge(soc_tools, on="SOC", how="left")
    else:
        comprehensive_tasks["top_tools_examples"] = None

    # 8) Job Zones / SVP (SOC-level)
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

    # 9) Work Context (SOC-level) — keep only mean scales (CT or CX)
    wc = read_onet_table(["Work Context.txt", "Work_Context.txt"])
    if wc is not None:
        wc = wc.rename(columns={"O*NET-SOC Code":"SOC","Element Name":"ElementName",
                                "Scale ID":"ScaleID","Data Value":"DataValue"})
        wc = wc[wc["ScaleID"].astype(str).str.upper().isin(["CT","CX"])]
        wanted = {
            "Electronic Mail": "wc_electronic_mail",
            "Telephone": "wc_telephone",
            "Face-to-Face Discussions": "wc_face_to_face",
            "Physical Proximity": "wc_physical_proximity",
            "Spend Time Using Your Hands to Handle, Control, or Feel Objects": "wc_hands_on",
            "Importance of Being Exact or Accurate": "wc_exact_or_accurate",
        }
        wc_sel = wc[wc["ElementName"].isin(wanted.keys())].copy()
        wc_sel["col"] = wc_sel["ElementName"].map(wanted)
        wc_piv = wc_sel.groupby(["SOC","col"])["DataValue"].mean().unstack().reset_index()
        comprehensive_tasks = comprehensive_tasks.merge(wc_piv, on="SOC", how="left")
        for c in wanted.values():
            if c not in comprehensive_tasks.columns:
                comprehensive_tasks[c] = None
    else:
        for c in ["wc_electronic_mail","wc_telephone","wc_face_to_face",
                  "wc_physical_proximity","wc_hands_on","wc_exact_or_accurate"]:
            comprehensive_tasks[c] = None

    # 10) Related Occupations (optional) — rank by Relatedness Tier, then Index
    rel = read_onet_table(["Related Occupations.txt", "Related_Occupations.txt"])
    if rel is not None:
        rel = rel.rename(columns={
            "O*NET-SOC Code": "SOC",
            "Related O*NET-SOC Code": "RelatedSOC",
            "Related Title": "RelatedTitle",
            "Related Occupation": "RelatedTitle",   # handle alternate header
            "Relatedness Tier": "RelatednessTier",
            "Index": "RelatedIndex",
        })
        # Ensure RelatedTitle exists
        if "RelatedTitle" not in rel.columns:
            rel["RelatedTitle"] = ""

        # Backfill titles if missing using occupation data
        occ_titles = occ[["SOC","OccTitleRaw"]].rename(columns={"SOC":"RelatedSOC","OccTitleRaw":"RelatedTitle_backfill"})
        rel = rel.merge(occ_titles, on="RelatedSOC", how="left")
        if "RelatedTitle_backfill" in rel.columns:
            rel["RelatedTitle"] = rel["RelatedTitle"].where(
                rel["RelatedTitle"].notna() & (rel["RelatedTitle"].astype(str).str.len() > 0),
                rel["RelatedTitle_backfill"]
            )
            rel = rel.drop(columns=["RelatedTitle_backfill"])

        # Rank: Primary-Short > Primary-Long > Supplemental, then lowest Index
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
        # If titles are still empty, fall back to codes
        if top_by_soc["top_related_titles"].isna().all() or (top_by_soc["top_related_titles"].astype(str) == "").all():
            top_codes = (rel_sorted.groupby("SOC")
                         .agg(top_related_titles=("RelatedSOC", _top_titles))
                         .reset_index())
            top_by_soc = top_by_soc.drop(columns=["top_related_titles"]).merge(top_codes, on="SOC", how="left")

        comprehensive_tasks = comprehensive_tasks.merge(top_by_soc, on="SOC", how="left")
    else:
        comprehensive_tasks["related_occ_count"] = None
        comprehensive_tasks["top_related_titles"] = None

    # 11) Work Activities (optional) — rank by Importance (IM) scale only
    wa = read_onet_table(["Work Activities.txt", "Work_Activities.txt"])
    if wa is not None:
        wa = wa.rename(columns={"O*NET-SOC Code":"SOC","Element Name":"WA_Name",
                                "Scale ID":"ScaleID","Data Value":"WA_Value"})
        wa = wa[wa["ScaleID"].astype(str).str.upper().eq("IM")]
        if ("WA_Value" not in wa.columns) or ("WA_Name" not in wa.columns):
            print("⚠️  Work Activities missing WA_Value/WA_Name; skipping aggregation.")
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

    # 12) LLM modality classification (3 votes)
    load_dotenv()
    offline = bool(os.getenv("OFFLINE_GRADER")) or not os.getenv("OPENAI_API_KEY") or (openai is None)
    cache = load_cache(CACHE_JSON)
    client = (openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if not offline else None)

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

        if key in cache:
            votes = cache[key]
            votes = votes if isinstance(votes, list) else [votes]
        elif offline:
            votes = [OFFLINE_LABEL] * VOTES_PER_TASK
        else:
            seeds = list(range(1, VOTES_PER_TASK + 1))
            votes = [vote_once(client, stmt, seed) for seed in seeds]
            cache[key] = votes
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

    save_cache(CACHE_JSON, cache)

    # 13) Build output frame
    out = comprehensive_tasks.copy()
    for i, col in enumerate(vote_cols):
        out[col] = [vs[i] if len(vs) > i else None for vs in all_votes]

    out["modality"] = final_labels
    out["modality_agreement"] = out[vote_cols].apply(lambda r: Counter([x for x in r if x]).most_common(1)[0][1], axis=1)
    out["modality_disagreement"] = out["modality_agreement"] < VOTES_PER_TASK
    out["modality_confidence"] = out["modality_agreement"] / VOTES_PER_TASK

    # Derived flags
    out["digital_amenable"] = out["modality"].isin({"TEXT", "GUI", "VISION"})
    out["amenability_reason"] = out["modality"].map({
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
    out["amenability_code"] = out["modality"].map(code_map)

    # Explicit stubs
    out["needs_stub"] = out["modality"].isin({"MANUAL", "INCONCLUSIVE", "REVIEW", "UNLABELED"})
    out["stub_type"] = out["modality"].map({
        "MANUAL": "MANUAL",
        "INCONCLUSIVE": "AMBIGUOUS",
        "REVIEW": "REVIEW",
        "UNLABELED": "UNLABELED",
    }).fillna("NONE")

    # Normalize importance per SOC (weight for aggregation)
    if "Importance" in out.columns:
        out["Importance"] = pd.to_numeric(out["Importance"], errors="coerce").fillna(0.0)
        _soc_sum = out.groupby("SOC")["Importance"].transform("sum")
        out["importance_weight_norm"] = 0.0
        mask = _soc_sum > 0
        out.loc[mask, "importance_weight_norm"] = out.loc[mask, "Importance"] / _soc_sum[mask]

    # Provenance
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

    # 14) Save manifest
    out_file = OUT / "sampled_tasks_comprehensive.csv"
    out.to_csv(out_file, index=False)

    # 15) Summary stats
    print(f"✅  Generated {len(out)} tasks → {out_file}")
    modality_counts = Counter(out["modality"])
    print(f"📊  Modality distribution: {dict(modality_counts)}")
    print("📈  Tasks by occupation:")
    for soc in PILOT_SOCs:
        soc_count = (out["SOC"] == soc).sum()
        occ_title = out.loc[out["SOC"] == soc, "OccTitleClean"].iloc[0] if soc_count > 0 else "Unknown"
        print(f"   {soc}: {soc_count:3d} tasks ({occ_title})")
    try:
        digital_share = out.groupby("SOC")["digital_amenable"].mean().to_dict()
        imp_mass_digital = (
            out.assign(_dig=out["digital_amenable"].astype(bool))
              .groupby("SOC")
              .apply(lambda g: float(g.loc[g._dig, "importance_weight_norm"].sum() if "importance_weight_norm" in g else 0.0))
              .to_dict()
        )
        print("ℹ️  Digital share by SOC:", digital_share)
        print("ℹ️  Importance mass in digital tasks by SOC:", imp_mass_digital)
    except Exception:
        pass
    print("ℹ️  Fields included: Task provenance, IM/RL ratings + sources/months, DWA linkages, Emerging counts/revisions, Tech/Tools, Job Zone, Work Context (mean only), Related Occupations (tier+index), optional Work Activities (IM only).")

if __name__ == "__main__":
    main()
