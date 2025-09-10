Here’s a clean, meticulous replacement README that matches the **current TACI architecture** and your **preferred skeleton** (as reflected in the new `sample_tasks.py` and the gameplan). It’s written to be copy-paste ready.

---

# TACI (Task-AI Capability Index) — MVP v0.4

A reproducible research pipeline to measure model capability on real occupational tasks derived from O\*NET.
This MVP focuses on a **comprehensive manifest** for pilot SOCs, phase-gated validation, and publication-grade provenance.

> Scope: This repository builds a **single, comprehensive task manifest** for pilot SOCs, then feeds that manifest into prompt generation and Phase 0–4 evaluation. The manifest is the source of truth for sampling, modality, and rich task/occupation enrichments.

---

## Table of Contents

1. Quick Start
2. Repository Structure (Preferred Skeleton)
3. Data Requirements (O\*NET v30.0 TXT preferred)
4. Building the Comprehensive Manifest
5. Manifest Outputs & Provenance
6. Column Glossary (Comprehensive Manifest)
7. Prompt Generation & Evaluation Phases (v0.4)
8. Reproducibility, Caching & Determinism
9. Acceptance Checks & Sanity Tests
10. Known Limitations (MVP)
11. Changelog from “Alpha Showcase”
12. Citation & License

---

## 1) Quick Start

Install and prepare:

```bash
# Python
pip install -r requirements.txt
python -m spacy download en_core_web_sm  # optional; improves title cleaning

# Minimal env (MODEL_NAME is optional; OPENAI_API_KEY optional for offline mode)
export ONET_VERSION="30.0"
export MODEL_NAME="gpt-4.1-mini-2025-04-14"   # default in code
# export OPENAI_API_KEY="..."                  # omit => offline labeling
# export OFFLINE_GRADER=1                      # force offline voting if desired
```

Build the manifest (pilot SOCs only):

```bash
python src/data_pipeline/sampling/sample_tasks.py
```

Artifacts are written to `data/manifests/` (CSV + SHA256 + meta JSON, and optional edge views).

---

## 2) Repository Structure (Preferred Skeleton)

```
TACI/
├── README.md
├── requirements.txt
├── LICENSE
│
├── config/
│   ├── archetype_rules.yml         # GUI archetype rules (used post-manifest)
│   ├── vision_archetypes.yml       # Vision archetype rules (used post-manifest)
│   └── gui_selectors.json          # GUI selectors (by archetype; used post-manifest)
│
├── data/
│   ├── onet_raw/                   # Place O*NET TXT files here (see §3)
│   ├── manifests/
│   │   ├── sampled_tasks_comprehensive.csv
│   │   ├── sampled_tasks_comprehensive.csv.sha256
│   │   ├── sampled_tasks_comprehensive.meta.json
│   │   └── modality_cache_comprehensive.json     # voting cache (auto)
│   └── panel/                      # Downstream aggregation outputs (Phase 4+)
│
├── outputs/
│   ├── edges/                      # Optional edge views (task→DWA/IWA/GWA; UNSPSC counts)
│   └── results/                    # Phase outputs (00–04) and master tables
│
├── prompts/
│   ├── text/   # generated prompts
│   ├── gui/
│   ├── vision/
│   └── manual/
│
├── src/
│   ├── data_pipeline/
│   │   ├── sampling/
│   │   │   └── sample_tasks.py     # builds comprehensive manifest (this MVP’s core)
│   │   └── prompt_gen/
│   │       └── generate_prompts.py # consumes manifest → prompts/* (per modality)
│   │
│   ├── evaluation/                 # Phase gates (00–04)
│   │   ├── phase_00_wrapper/
│   │   ├── phase_01_schema/
│   │   ├── phase_02_safety/
│   │   ├── phase_03_rubric/
│   │   └── phase_04_composite/
│   │
│   ├── execution/
│   │   └── llm_client/             # API runners (OpenAI / Anthropic / Gemini / etc.)
│   │
│   └── analysis/
│       ├── insights_analyzer.py
│       └── visualizations.py
│
└── docs/
    └── methods/                    # notes, brief, figures (optional)
```

---

## 3) Data Requirements (O\*NET v30.0; TXT preferred)

Place the following under `data/onet_raw/` (TXT format preferred; the loader will fall back to XLSX if needed):

* **Core (required)**
  `Task Statements.txt`, `Task Ratings.txt`, `Occupation Data.txt`
* **Linkages**
  `Tasks to DWAs.txt`, `DWA Reference.txt`, `IWA Reference.txt` (optional), `Content Model Reference.txt` (optional)
* **Emerging**
  `Emerging Tasks.txt`
* **Tech/Tools + UNSPSC**
  `Technology Skills.txt`, `Tools Used.txt`, `UNSPSC Reference.txt`
* **Job Zones**
  `Job Zones.txt`
* **Work Context**
  `Work Context.txt`
* **Related Occupations**
  `Related Occupations.txt`
* **Work Activities**
  `Work Activities.txt`
* **Titles**
  `Alternate Titles.txt`, `Sample of Reported Titles.txt`
* **Education (optional but preferred)**
  `Education, Training, and Experience.txt`,
  `Education, Training, and Experience Categories.txt`

> The loader does robust name normalization (underscores/spaces/case) and prefers TXT to avoid Excel coercions.

---

## 4) Building the Comprehensive Manifest

**Pilot SOCs (fixed in code):**

* `23-2011.00` — Paralegals and Legal Assistants (TEXT-leaning)
* `43-4051.00` — Customer Service Representatives (GUI-leaning)
* `51-9061.00` — Inspectors/Testers/Sorters/Samplers/Weighers (VISION-leaning)

Environment knobs:

* `MODEL_NAME` (default: `gpt-4.1-mini-2025-04-14`)
* `OPENAI_API_KEY` (omit ⇒ offline labeling)
* `OFFLINE_GRADER` (truthy ⇒ force offline)
* `ONET_VERSION` (string recorded in outputs)
* `TITLES_LIMIT` (default 50; caps alt/sample titles stored per SOC)
* `EMIT_EDGE_VIEWS` (set to `1` to write edges into `outputs/edges/`)

Run:

```bash
python src/data_pipeline/sampling/sample_tasks.py
```

---

## 5) Manifest Outputs & Provenance

* `data/manifests/sampled_tasks_comprehensive.csv` — **single-file source of truth**
* `data/manifests/sampled_tasks_comprehensive.csv.sha256` — atomic write + integrity hash
* `data/manifests/sampled_tasks_comprehensive.meta.json` — generation metadata
* `data/manifests/modality_cache_comprehensive.json` — vote cache (idempotence)
* Optional edge views (if `EMIT_EDGE_VIEWS=1`):

  * `outputs/edges/task_dwa.csv`, `task_iwa.csv`, `task_gwa.csv`
  * `outputs/edges/soc_unspsc_counts.csv`

Provenance fields (also embedded as columns in the CSV):

* `schema_version` = `tlc_manifest_schema_v0.4`
* `manifest_version` = `YYYYMMDD-v1` (UTC date)
* `onetsrc_version`, `model_name`, `votes_per_task`, `vote_seeds`
* `modality_prompt_md5`, `code_fingerprint` (MD5 of script)
* `generated_utc`, `source_files_used`, `unspsc_rollup_level` (`FAMILY`), `unspsc_entropy_unit` (`bits`)

---

## 6) Column Glossary (Comprehensive Manifest)

**Core task identity**

* `SOC` — O\*NET SOC code
* `TaskID` — O\*NET task ID
* `uid` — deterministic MD5( `SOC-TaskID` ) first 12 hex chars
* `TaskText` — task statement
* `TaskType` — task category if available
* `task_date_raw` / `task_date` — normalized to `YYYY-MM` where possible
* `OccTitleRaw` / `OccTitleClean` — raw vs cleaned occupation title

**Ratings & sources (IM=Importance, RL=Relevance)**

* `Importance`, `Relevance` — numeric data values (coerced to float where possible)
* `importance_n_respondents`, `relevance_n_respondents`
* `ratings_month_im`, `ratings_month_rl` (ISO `YYYY-MM`)
* `ratings_source_im`, `ratings_source_rl` (Incumbent/Analyst/Other)
* `retained_by_relevance_rule` — boolean (`Relevance ≥ 25` per O\*NET)

**Titles (SOC-level; replicated per row)**

* `alt_titles_raw`, `alt_titles_canon` — deduped (ASCII fold, lower, strip punct), limit = `TITLES_LIMIT`
* `sample_titles_raw`, `sample_titles_canon` — same treatment
* `title_canon_examples` — first three canonical exemplars across alt+sample

**DWA / IWA / GWA linkages**

* `dwa_ids`, `dwa_titles`, `iwa_id_list`, `iwa_title_list`, `gwa_id_list`, `gwa_title_list` (semicolon-joined)
* `_count` companions for each list
* Degree & rarity features (per task, normalized within SOC):

  * `task_dwa_degree_soc_sum`, `task_iwa_degree_soc_sum`, `task_gwa_degree_soc_sum`
  * `task_dwa_degree_soc_norm`, `task_iwa_degree_soc_norm`, `task_gwa_degree_soc_norm`
  * `task_dwa_idf_sum`, `task_iwa_idf_sum`, `task_gwa_idf_sum`
  * `task_dwa_degree_soc_z`, `task_iwa_degree_soc_z`, `task_gwa_degree_soc_z`

**Emerging tasks**

* `soc_emerging_new_count`, `soc_emerging_revision_count` (SOC-level)
* `is_emerging_revision` (per task, if original task flagged as revision)

**Technology / Tools (UNSPSC roll-up to FAMILY)**

* `hot_tech_count`, `in_demand_tech_count` (SOC-level)
* `top_hot_tech_examples` (up to 3)
* `tech_family_top3`, `tool_family_top3` — `"FamilyCode:FamilyTitle; ..."`
* `tech_family_entropy`, `tool_family_entropy` — Shannon entropy in **bits**

**Job requirements**

* `job_zone`, `svp_range` (SOC-level)

**Work Context (CT/CX means; selected indicators)**

* `wc_electronic_mail`, `wc_telephone`, `wc_face_to_face`, `wc_physical_proximity`, `wc_hands_on`, `wc_exact_or_accurate`
* `wc_consequence_of_error`, `wc_time_pressure`, `wc_freedom_to_make_decisions`,
  `wc_structured_vs_unstructured`, `wc_deal_with_external_customers`, `wc_indoors_env_controlled`
* Coverage diagnostics: `wc_ctcx_rows_total_soc`, `wc_ctcx_rows_mapped_soc`

**Related Occupations (SOC-level)**

* `related_occ_count`, `top_related_titles` (up to 3)

**Work Activities (SOC-level)**

* `top_work_activities` (top-3 by mean IM)

**Modality classification (vote-based)**

* `vote1`, `vote2`, `vote3` ∈ `{TEXT, GUI, VISION, MANUAL, INCONCLUSIVE}` or `{REVIEW, UNLABELED}` in edge cases
* `modality` — majority with tie-break rules; categorical with levels:
  `TEXT, GUI, VISION, MANUAL, INCONCLUSIVE, REVIEW, UNLABELED`
* `modality_agreement` (1–3), `modality_confidence` (= agreement / 3)
* Diagnostics: `modality_uncertain` (INCONCLUSIVE/REVIEW/UNLABELED), `modality_disagreement`
* **Digital amenability**:

  * `digital_amenable` = `modality ∈ {TEXT, GUI, VISION}`
  * `amenability_reason` (human-readable mapping)
  * `amenability_code` ∈ `{LANGUAGE_ONLY, GUI_SOFTWARE, VISUAL_PERCEPTION, PHYSICAL_MANUAL, AMBIGUOUS, REVIEW, UNLABELED}`
  * Stub flags: `needs_stub`, `stub_type` (for MANUAL/AMBIGUOUS/REVIEW/UNLABELED)

**Aggregation weights**

* `importance_weight_norm` — IM normalized within SOC (sums to 1 per SOC)
* `importance_weight_norm_rl` — IM normalized within SOC **after RL≥25 filter**

**Education (SOC-level modal RL category)**

* `education_typical_category` (int), `education_typical` (label), `education_typical_share` (modal share)

**Provenance & constants**

* `onetsrc_version`, `model_name`, `votes_per_task`, `vote_seeds`, `generated_utc`,
  `schema_version`, `manifest_version`, `source_files_used`,
  `pipeline_stage="sample_tasks_comprehensive"`,
  `unspsc_rollup_level="FAMILY"`, `unspsc_entropy_unit="bits"`,
  `modality_prompt_md5`, `code_fingerprint`

---

## 7) Prompt Generation & Evaluation Phases (v0.4)

**Prompt Generation**

```bash
python src/data_pipeline/prompt_gen/generate_prompts.py \
  --manifest data/manifests/sampled_tasks_comprehensive.csv \
  --outdir prompts/
```

* Uses `modality` to route to `prompts/text|gui|vision|manual/` and attaches archetype-specific extras in later stages (selectors/images/stubs).

**Evaluation Phases (00–04)**

* **Phase 00 — Wrapper**: strict formatting checks per provider (reject malformed JSON/fields).
* **Phase 01 — Schema**: JSON schema compliance; IoU scoring for vision where applicable.
* **Phase 02 — Safety**: content moderation thresholds (configurable).
* **Phase 03 — Rubric**: task-specific rubric grading with 3-vote self-consistency (deterministic seeds).
* **Phase 04 — Composite**: aggregate to task/occupation/model summaries (capability excludes price/context).

> Minimal viable run: pick a subset of tasks from the manifest, generate prompts, run one provider, and push through 00→04 to validate the end-to-end measurement.

---

## 8) Reproducibility, Caching & Determinism

* **Deterministic seeds**: RNG seeded (Python/NumPy), vote seeds = `1..V` (default V=3).
* **Atomic writes**: CSV + `.sha256` hash; meta JSON captures exact fingerprinting.
* **LLM drift guard**: `modality_prompt_md5` and `code_fingerprint` (MD5 of script) embedded.
* **Idempotent voting**: `modality_cache_comprehensive.json` keyed by `(uid:model:prompt_md5:ONET_VERSION:code_fp)`.

Offline behavior:

* If `OPENAI_API_KEY` missing or `OFFLINE_GRADER` set ⇒ votes are `UNLABELED`; `modality` falls back to `UNLABELED`/`REVIEW` per tie-break rules, and downstream flags (`modality_uncertain`, `needs_stub`) reflect that.

---

## 9) Acceptance Checks & Sanity Tests

After building the manifest:

* **Row count > 0** for each pilot SOC; no empty occupations.
* **Weights sum**: `importance_weight_norm` sums to **1.0 per SOC** (±1e-6).
* **Digital share**: `digital_amenable` mean reported per SOC (inspect console summary).
* **Uncertainty**: `modality_uncertain` share per SOC is reported; investigate spikes.
* **Edges present** (if `EMIT_EDGE_VIEWS=1`): non-empty `task_*` edge CSVs.
* **Provenance present**: `schema_version`, `manifest_version`, `code_fingerprint`, `modality_prompt_md5` populated.
* **No duplicate keys**: `(SOC, TaskID, uid)` duplicates count = 0.

---

## 10) Known Limitations (MVP)

* **Pilot SOCs only** (3 codes). Expansion is trivial but intentionally deferred until Phase 00–04 validation is stable.
* **Work Context mapping** uses curated keyword heuristics for selected indicators; unmapped names are logged.
* **Education typical** requires ETE files; absent files yield nulls with robust defaults.
* **Offline modality** produces `UNLABELED` and marks tasks as uncertain; do not aggregate capabilities from offline labels.

---

## 11) Changelog from “Alpha Showcase”

* **Manifest first**: replaced ad-hoc samplers with a **single comprehensive manifest** (pilot SOCs), enriched with DWA/IWA/GWA, UNSPSC family entropy, titles canon, work context, related occupations, work activities, education modal RL.
* **Voting and cache**: modality via **3 votes** with agreement metrics and persistent cache keyed by prompt/model/code/ONET version.
* **Provenance hardening**: atomic writes, CSV hash, meta JSON, code fingerprint, prompt MD5.
* **Edge exports**: optional task→(DWA/IWA/GWA) and UNSPSC counts for graph analytics.
* **Aggregation weights**: normalized IM weights with RL≥25 optional gating for capability aggregation.
* **Phase 4 policy**: capability excludes price/context (kept out of capability scoring by design).

---

## 12) Citation & License

If you use TACI in academic or policy work, please cite the repository and the working paper (forthcoming).
License: see `LICENSE`.

---

