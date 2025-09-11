#!/usr/bin/env python3
"""
generate_prompts.py — TACI v2.0 (deterministic prompts + sidecars)

Goals per prompt:
- Exactly one wrapper block (<OUTPUT_TEXT>...</OUTPUT_TEXT> or <OUTPUT_JSON>...</OUTPUT_JSON>) with nothing outside.
- Strict schemas (GUI action list; VISION bbox [x,y,w,h]).
- No chain-of-thought leakage.
- Valid JSON only (double quotes, no extra keys, no trailing commas).
- Deterministic and hashable; write sha256 and meta sidecars per file.
"""

from __future__ import annotations
import json, textwrap, pathlib, html, hashlib
from typing import List, Dict
import pandas as pd
try:
    import yaml  # optional for VISION labels
except Exception:
    yaml = None

# ── paths & config ─────────────────────────────────────────
# Latest dated manifest discovery (data/manifests/YYYYMMDD_v1/manifest_v0.csv)
MANIFEST            = pathlib.Path("data/manifests/paralegal_tasks.csv")  # legacy fallback; unused if dated exists
SELECTOR_FILE       = pathlib.Path("config/gui_selectors.json")
VISION_LABELS_FILE  = pathlib.Path("config/vision_archetypes.yml")
PROMPT_ROOT         = pathlib.Path("prompts/one_occ")

MODALITIES = ["TEXT", "GUI", "VISION", "MANUAL", "INCONCLUSIVE"]
for m in MODALITIES:
    (PROMPT_ROOT / m.lower()).mkdir(parents=True, exist_ok=True)

SELECTOR_MAP = (
    json.loads(SELECTOR_FILE.read_text()) if SELECTOR_FILE.exists() else {}
)

TEXT_START, TEXT_END   = "<OUTPUT_TEXT>",  "</OUTPUT_TEXT>"
JSON_START, JSON_END   = "<OUTPUT_JSON>",  "</OUTPUT_JSON>"

# Formatting guidance and bounds
FORMAT_WARN_TEXT = "Use <OUTPUT_TEXT> ... </OUTPUT_TEXT>."
FORMAT_WARN_JSON = "Use <OUTPUT_JSON> ... </OUTPUT_JSON>."
TEXT_WORD_MIN = 200
TEXT_WORD_MAX = 300
EXPLANATION_WORD_MAX = 15

selectors_for = lambda arc: SELECTOR_MAP.get(arc, [])

def load_vision_labels_map() -> Dict[str, List[str]]:
    if VISION_LABELS_FILE.exists() and yaml is not None:
        try:
            data = yaml.safe_load(VISION_LABELS_FILE.read_text())
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}

VISION_LABELS_MAP = load_vision_labels_map()

# Skip logging for problematic rows (e.g., empty task text, missing selectors)
SKIP_LOG = pathlib.Path("outputs/prompts/skip_log.csv")

def log_skip(uid: str, occ: str, task: str, reason: str, archetype: str = "") -> None:
    SKIP_LOG.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not SKIP_LOG.exists()
    with open(SKIP_LOG, "a", encoding="utf-8", newline="\n") as f:
        if header_needed:
            f.write("uid,occupation,reason,archetype,task\n")
        safe_task = (task or "").replace("\n", " ").replace("\r", " ")
        f.write(f"{uid},{occ},{reason},{archetype},{safe_task}\n")

def get_task_text(row: dict) -> str:
    for k in [
        "task_statement", "Task", "TaskText", "task_text", "TaskDescription", "task_desc"
    ]:
        v = (row.get(k, "") or "").strip()
        if v:
            return html.escape(v)
    return ""

def find_latest_manifest() -> pathlib.Path:
    base = pathlib.Path("data/manifests")
    candidates = []
    if base.exists():
        for p in base.iterdir():
            if p.is_dir() and len(p.name) >= 9 and p.name.endswith("_v1") and p.name[:8].isdigit():
                m = p / "manifest_v0.csv"
                if m.exists():
                    candidates.append((p.name[:8], m))
    if candidates:
        candidates.sort(key=lambda t: t[0], reverse=True)
        return candidates[0][1]
    return MANIFEST  # fallback legacy

# ── helper to insert format spec ───────────────────────────
def fmt_block(output_tag: str) -> str:
    # Exact phrasing per spec; select tag externally
    if output_tag == TEXT_START:
        return (
            "<<FORMAT>> "
            "Use <OUTPUT_TEXT> ... </OUTPUT_TEXT>. "
            "Your response must consist of exactly one such block with no characters (including whitespace) before or after it. "
            "Do not include code fences, examples, or commentary. Do not echo instructions. "
            "If any instruction conflicts, the FORMAT and schema rules override all other text. "
            "Your first output character must be '<' and your final character must be '>'. <</FORMAT>>"
        )
    else:
        return (
            "<<FORMAT>> "
            "Use <OUTPUT_JSON> ... </OUTPUT_JSON>. "
            "Your response must consist of exactly one such block with no characters (including whitespace) before or after it. "
            "Do not include code fences, examples, or commentary. Do not echo instructions. "
            "If any instruction conflicts, the FORMAT and schema rules override all other text. "
            "Your first output character must be '<' and your final character must be '>'. <</FORMAT>>"
        )

# ── TEXT variants ──────────────────────────────────────────
def make_text_prompt(occ, task_html, v):
    system = (
        f"You are a deterministic assistant for {occ}. Think internally but do not reveal your reasoning. "
        f"Follow the required format exactly. Do not write anything outside the required output block."
    )
    fmt = fmt_block(TEXT_START)
    if v == 0:
        user = f"""{fmt}
Task: {task_html}
Content requirements: Answer directly and concretely for a {occ}. Be accurate, specific, and useful. Write approximately {TEXT_WORD_MIN}–{TEXT_WORD_MAX} words. Do not include preambles or meta commentary.
Output only the final answer enclosed in {TEXT_START}...{TEXT_END}."""
    elif v == 1:
        user = f"""{fmt}
Task: {task_html}
Content requirements: Answer directly and concretely for a {occ}. Be accurate, specific, and useful. Write approximately {TEXT_WORD_MIN}–{TEXT_WORD_MAX} words. Do not include preambles or meta commentary.
Persona: You are a senior {occ} preparing a client-ready memo with clear structure and actionable recommendations.
Output only the final answer enclosed in {TEXT_START}...{TEXT_END}."""
    else:
        user = f"""{fmt}
Task: {task_html}
Content requirements: Start with a short title line (plain text), then use numbered sections. Be specific and cite any assumptions briefly. {TEXT_WORD_MIN}–{TEXT_WORD_MAX} words total.
Output only the final answer enclosed in {TEXT_START}...{TEXT_END}."""
    return system, textwrap.dedent(user).strip()

# ── GUI variants ───────────────────────────────────────────
def make_gui_prompt(occ, task_html, v, sels: List[str]):
    system = (
        f"You are a deterministic assistant for {occ}. Think internally but do not reveal your reasoning. "
        f"Follow the required format exactly. Do not write anything outside the required output block."
    )
    fmt = fmt_block(JSON_START)

    shared = (
        f"Task: {task_html}\n"
        f"Required selectors (use only these; include all that are necessary): {json.dumps(sels, ensure_ascii=False)}\n"
        "Output schema: A JSON array of action objects, each with exactly these keys:\n"
        "\"action\": one of [\"click\",\"type\",\"select\",\"wait\",\"scroll\"]\n"
        "\"selector\": string (e.g., \"css:#email\", \"role:button[name='Submit']\", \"text:'Ticket Created'\")\n"
        "\"value\": string or null (required for \"type\" and \"select\"; null otherwise)\n"
        "\"wait_for\": optional string selector to wait for after the action\n"
        "Rules:\n"
        "Use only the required selectors list; do not invent new selectors.\n"
        "Include each selector only if needed to complete the task.\n"
        "If you cannot complete the task deterministically, return an empty array [].\n"
        "No extra keys; valid JSON only; double quotes for all strings; no comments; no trailing commas.\n"
        "Numbers must be integers; booleans as true/false; nulls as null.\n"
        "Before finalizing, silently verify your JSON parses and matches the schema; do not print the check.\n"
        f"Output only the JSON array enclosed in {JSON_START}...{JSON_END}."
    )

    if v == 0:
        user = f"""{fmt}
{shared}"""
    elif v == 1:
        exsel = sels[0] if sels else "css:#example"
        example_actions = [
            {"action": "type", "selector": exsel, "value": "alice@example.com", "wait_for": None},
            {"action": "click", "selector": "role:button[name='Submit']", "value": None, "wait_for": None},
            {"action": "wait", "selector": "text:'Ticket Created'", "value": None, "wait_for": None},
        ]
        # Validate strict keys and double quotes via JSON round-trip
        example_json = json.dumps(example_actions, ensure_ascii=False)
        # Simple structural assert
        for obj in example_actions:
            assert set(obj.keys()) == {"action", "selector", "value", "wait_for"}
        user = f"""{fmt}
{shared}

Example (do not include in your answer):
{JSON_START}{example_json}{JSON_END}"""
    else:
        user = f"""{fmt}
{shared}

If any required selector is missing or ambiguous, return {JSON_START}[]{JSON_END}."""
    return system, textwrap.dedent(user).strip()

# ── VISION variants ────────────────────────────────────────
def allowed_labels_for(archetype: str) -> List[str]:
    if archetype and isinstance(VISION_LABELS_MAP, dict):
        labels = VISION_LABELS_MAP.get(archetype)
        if isinstance(labels, list) and labels:
            return labels
    return ["normal", "defect"]

def make_vis_prompt(occ, task_html, v, arc: str):
    system = (
        f"You are a deterministic assistant for {occ}. Think internally but do not reveal your reasoning. "
        f"Follow the required format exactly. Do not write anything outside the required output block."
    )
    fmt = fmt_block(JSON_START)
    labels = allowed_labels_for(arc)
    assert isinstance(labels, list) and len(labels) > 0

    shared = (
        "Images: img_000 (cid:img_000), img_001 (cid:img_001)\n"
        f"Task: {task_html}. Compare the two images and detect any anomaly.\n"
        f"Allowed 'finding' labels: {json.dumps(labels, ensure_ascii=False)}\n"
        "Output schema (single JSON object):\n"
        "\"finding\": one of the allowed labels\n"
        "\"image_id\": \"img_000\" | \"img_001\" | null\n"
        "\"bbox\": [x,y,w,h] with integer pixel coordinates, or null\n"
        f"\"explanation\": string, <= {EXPLANATION_WORD_MAX} words\n"
        "Rules:\n"
        "If finding = \"normal\": image_id must be null and bbox must be null.\n"
        "If finding ≠ \"normal\": image_id must be the abnormal image and bbox must be provided.\n"
        "Coordinates use top-left origin; integers only.\n"
        "No extra keys; valid JSON only; double quotes; no comments.\n"
        "Before finalizing, silently verify your JSON parses and matches the schema; do not print the check.\n"
        f"Output only the JSON object enclosed in {JSON_START}...{JSON_END}."
    )

    if v == 0:
        user = f"""{fmt}
{shared}"""
    elif v == 1:
        pos = {"finding": labels[-1] if len(labels) > 1 else "defect", "image_id": "img_001", "bbox": [30,60,120,210], "explanation": "localized surface anomaly near seam"}
        neg = {"finding": "normal", "image_id": None, "bbox": None, "explanation": ""}
        user = f"""{fmt}
{shared}

Positive example (do not include):
{JSON_START}{json.dumps(pos, ensure_ascii=False)}{JSON_END}

Negative example (do not include):
{JSON_START}{json.dumps(neg, ensure_ascii=False)}{JSON_END}"""
    else:
        user = f"""{fmt}
{shared}

If you are uncertain, prefer "normal" with null image_id and bbox."""
    return system, textwrap.dedent(user).strip(), labels

# ── MANUAL variant ─────────────────────────────────────────
def make_manual_prompt(occ):
    system = f"As a veteran {occ}, you recognize this is a physical task not digitally executable today."
    user = "No AI output required. Do not generate any response. This prompt is a stub for coverage accounting."
    return system, user

# ── main loop ──────────────────────────────────────────────
def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def write_prompt_with_sidecars(path: pathlib.Path, payload: List[Dict], *, uid: str, modality: str, variant: int, format_tag: str, selectors: List[str] | None = None, allowed_findings: List[str] | None = None):
    # Ensure folder exists
    path.parent.mkdir(parents=True, exist_ok=True)
    # Deterministic JSON for prompt file (readable but stable newlines)
    json_text = json.dumps(payload, indent=2, ensure_ascii=False)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json_text)
    b = path.read_bytes()
    digest = sha256_bytes(b)
    # .sha256 sidecar: hex only, no newline
    (path.parent / f"{path.stem}.sha256").write_bytes(digest.encode("ascii"))
    # .meta.json sidecar
    if format_tag == "OUTPUT_TEXT":
        stop = [TEXT_END]
    elif format_tag == "OUTPUT_JSON":
        stop = [JSON_END]
    else:
        stop = []
    meta = {
        "uid": uid,
        "modality": modality,
        "variant": variant,
        "format_tag": format_tag,
        "required_selectors": selectors or [],
        "allowed_findings": allowed_findings or [],
        "stop": stop,
        "prompt_sha256": digest,
    }
    # Add prompt length for sanity/caching stats
    meta["prompt_chars"] = len(json_text)
    meta_path = path.parent / f"{path.stem}.meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, sort_keys=True, separators=(",", ":")))

def main():
    manifest_path = find_latest_manifest()
    df = pd.read_csv(manifest_path, dtype=str).fillna("")
    total = 0
    for _, row in df.iterrows():
        uid = row["uid"]
        # Manifest uses legacy fields here
        occ = row.get("OccTitleClean", "").strip() or row.get("occupation_canonical", "").strip()
        task_html = get_task_text(row)
        arc = row.get("ui_archetype", "")
        if not task_html:
            log_skip(uid, occ, "(empty)", reason="empty_task_text", archetype=arc)
            continue
        modality = (row.get("modality", "").strip().upper())
        assert modality in {"TEXT", "GUI", "VISION", "MANUAL", "INCONCLUSIVE"}
        sels = selectors_for(arc) if modality == "GUI" else []

        for v in (0, 1, 2):
            if modality == "TEXT":
                sys, usr = make_text_prompt(occ, task_html, v)
                payload = [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
                out = PROMPT_ROOT / modality.lower() / f"{uid}_v{v}.json"
                write_prompt_with_sidecars(out, payload, uid=uid, modality=modality, variant=v, format_tag="OUTPUT_TEXT")
            elif modality == "GUI":
                sys, usr = make_gui_prompt(occ, task_html, v, sels)
                payload = [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
                out = PROMPT_ROOT / modality.lower() / f"{uid}_v{v}.json"
                write_prompt_with_sidecars(out, payload, uid=uid, modality=modality, variant=v, format_tag="OUTPUT_JSON", selectors=sels)
            elif modality == "VISION":
                sys, usr, labels = make_vis_prompt(occ, task_html, v, row.get("vision_archetype", ""))
                payload = [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
                out = PROMPT_ROOT / modality.lower() / f"{uid}_v{v}.json"
                write_prompt_with_sidecars(out, payload, uid=uid, modality=modality, variant=v, format_tag="OUTPUT_JSON", allowed_findings=labels)
            elif modality == "MANUAL":
                sys, usr = make_manual_prompt(occ)
                payload = [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
                out = PROMPT_ROOT / "manual" / f"{uid}_v{v}.json"
                write_prompt_with_sidecars(out, payload, uid=uid, modality=modality, variant=v, format_tag="NONE")
            elif modality == "INCONCLUSIVE":
                sys, usr = make_manual_prompt(occ)
                payload = [{"role": "system", "content": sys}, {"role": "user", "content": usr}]
                out = PROMPT_ROOT / "inconclusive" / f"{uid}_v{v}.json"
                write_prompt_with_sidecars(out, payload, uid=uid, modality=modality, variant=v, format_tag="NONE")
            total += 1
    print(f"✅ Generated {total} prompt files under {PROMPT_ROOT}")

if __name__ == "__main__":
    main()
