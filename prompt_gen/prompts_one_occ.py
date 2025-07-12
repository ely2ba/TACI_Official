#!/usr/bin/env python3
"""
generate_prompts.py — TACI v1.5  (FORMAT spec, literal wrappers)

Adds an explicit <<FORMAT>> block to every TEXT / GUI / VISION variant to
raise wrapper-tag compliance to ≈ 98 %+ on GPT-4-class models.
"""

from __future__ import annotations
import json, textwrap, pathlib, html
import pandas as pd

# ── paths & config ─────────────────────────────────────────
MANIFEST      = pathlib.Path("data/manifests/paralegal_tasks.csv")
SELECTOR_FILE = pathlib.Path("config/gui_selectors.json")
PROMPT_ROOT   = pathlib.Path("prompts/one_occ")

MODALITIES = ["TEXT", "GUI", "VISION", "MANUAL"]
for m in MODALITIES:
    (PROMPT_ROOT / m.lower()).mkdir(parents=True, exist_ok=True)

SELECTOR_MAP = (
    json.loads(SELECTOR_FILE.read_text()) if SELECTOR_FILE.exists() else {}
)

REASON_TAG = "<REASONING>You may think internally; DO NOT reveal your reasoning.</REASONING>"
TEXT_START, TEXT_END   = "<OUTPUT_TEXT>",  "</OUTPUT_TEXT>"
JSON_START, JSON_END   = "<OUTPUT_JSON>",  "</OUTPUT_JSON>"

selectors_for = lambda arc: SELECTOR_MAP.get(arc, [])

# ── helper to insert format spec ───────────────────────────
def fmt_block(tag: str) -> str:
    return f"<<FORMAT>> Wrap your final answer exactly once inside {tag} … {tag.replace('<','</')} . " \
           f"Do not output anything outside these tags. <</FORMAT>>"

# ── TEXT variants ──────────────────────────────────────────
def make_text_prompt(occ, task_html, v):
    system = f"You are an expert {occ}. {REASON_TAG}"
    fmt    = fmt_block("<OUTPUT_TEXT>")
    if v == 0:
        user = f"""{fmt}

{TEXT_START}
TASK – {task_html}
{TEXT_END}"""
    elif v == 1:
        user = f"""{fmt}

You are a senior {occ} preparing a client-ready memo.

TASK – {task_html}

Respond inside {TEXT_START}…{TEXT_END}."""
    else:
        user = f"""{fmt}

TASK – {task_html}

OUTPUT RULES:
1. Wrap answer in {TEXT_START}…{TEXT_END}
2. Start with an H1 heading, then numbered sections"""
    return system, textwrap.dedent(user).strip()

# ── GUI variants ───────────────────────────────────────────
def make_gui_prompt(occ, task_html, v, sels):
    system = f"You are an RPA bot for {occ}. {REASON_TAG}"
    fmt    = fmt_block("<OUTPUT_JSON>")
    head   = f"TASK – {task_html}\nREQUIRED_SELECTORS = {json.dumps(sels, ensure_ascii=False)}"
    if v == 0:
        user = f"""{fmt}

{head}

Return a JSON array wrapped in {JSON_START}…{JSON_END}.
Each action: {{'action':'click|type|select|wait','selector':'CSS','text':''}}"""
    elif v == 1:
        exsel = sels[0] if sels else "#selector"
        example = f"[{{'action':'click','selector':'{exsel}','text':''}}]"
        user = f"""{fmt}

{head}

Example:
{JSON_START}{example}{JSON_END}

Now produce your own JSON array."""
    else:
        user = f"""{fmt}

{head}

If unable to complete the task, return:
{JSON_START}{{"error":"unable"}}{JSON_END}

Otherwise return the normal action array."""
    return system, textwrap.dedent(user).strip()

# ── VISION variants ────────────────────────────────────────
def make_vis_prompt(occ, task_html, v):
    system = f"You are a vision analyst for {occ}. {REASON_TAG}"
    fmt    = fmt_block("<OUTPUT_JSON>")
    imgs   = "![img_000](cid:img_000)\n![img_001](cid:img_001)"

    if v == 0:
        user = f"""{fmt}

{imgs}

TASK – {task_html}

Compare IMAGE 1 vs IMAGE 2. Identify any anomaly.

Return only:
{JSON_START}{{"finding":"<label>","image_id":"<img_000|img_001|null>","bbox":[x1,y1,x2,y2],"explanation":"<≤15 words>"}}{JSON_END}"""
    elif v == 1:
        user = f"""{fmt}

{imgs}

TASK – {task_html}

Positive example:
{JSON_START}{{"finding":"fracture","image_id":"img_001","bbox":[30,60,120,210],"explanation":"visible break"}}{JSON_END}

Negative example:
{JSON_START}{{"finding":"normal","image_id":null,"bbox":null,"explanation":""}}{JSON_END}

Now respond using the same schema."""
    else:
        user = f"""{fmt}

{imgs}

TASK – {task_html}

Detect anomaly if present; otherwise return the canonical *normal* stub:
{JSON_START}{{"finding":"normal","image_id":null,"bbox":null,"explanation":""}}{JSON_END}"""
    return system, textwrap.dedent(user).strip()

# ── MANUAL variant ─────────────────────────────────────────
def make_manual_prompt(occ):
    system = f"As a veteran {occ}, you recognise this physical task. {REASON_TAG}"
    return system, "Physical task—no AI response required."

# ── main loop ──────────────────────────────────────────────
def main():
    df = pd.read_csv(MANIFEST, dtype=str).fillna("")
    total = 0
    for _, row in df.iterrows():
        uid, occ = row["uid"], row["OccTitleClean"]
        task_html = html.escape(row["Task"])
        modality  = row["modality"].strip().upper()
        arc       = row.get("ui_archetype", "")
        sels      = selectors_for(arc) if modality == "GUI" else []

        for v in (0, 1, 2):
            if modality == "TEXT":
                sys, usr = make_text_prompt(occ, task_html, v)
            elif modality == "GUI":
                sys, usr = make_gui_prompt(occ, task_html, v, sels)
            elif modality == "VISION":
                sys, usr = make_vis_prompt(occ, task_html, v)
            else:  # MANUAL
                sys, usr = make_manual_prompt(occ)

            out = PROMPT_ROOT / modality.lower() / f"{uid}_v{v}.json"
            out.write_text(json.dumps(
                [{"role":"system","content":sys},{"role":"user","content":usr}],
                indent=2, ensure_ascii=False))
            total += 1
    print(f"✅ Generated {total} prompt files under {PROMPT_ROOT}")

if __name__ == "__main__":
    main()
