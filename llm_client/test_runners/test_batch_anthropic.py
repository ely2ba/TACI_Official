#!/usr/bin/env python3
# llm_client/test_three_prompts_anthropic_chat.py
# Smoke‐test v0 TEXT/GUI prompts via Anthropic’s Messages API
# (Claude 3 Opus & 3.5 Sonnet), stub VISION.

import os
import re
import json
import time
import hashlib
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from tqdm import tqdm

# ─── CONFIG ─────────────────────────────────────────────────────────
load_dotenv()  # expects ANTHROPIC_API_KEY

API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set ANTHROPIC_API_KEY in your environment")

# Chat‐capable Claude 3 models
MODELS     = ["claude-3-opus-20240229", "claude-3-5-sonnet-20240620"]
TEMPS      = [0.0]
MAX_TOKENS = 1024

client = Anthropic(api_key=API_KEY)

# ─── SAMPLE PROMPTS ─────────────────────────────────────────────────
PROMPTS_ROOT = Path("prompts")
SAMPLES = {
    "TEXT":   next(PROMPTS_ROOT.glob("text/*_v0.json")),
    "GUI":    next(PROMPTS_ROOT.glob("gui/*_v0.json")),
    "VISION": next(PROMPTS_ROOT.glob("vision/*_v0.json")),
}

# ─── OUTPUT ──────────────────────────────────────────────────────────
OUT_ROOT = Path("runs") / "anthropic_smoketest_chat"
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ─── HELPERS ────────────────────────────────────────────────────────
def sha256_of_messages(msgs: list[dict]) -> str:
    # Stable SHA256 over the JSON messages payload
    return hashlib.sha256(
        json.dumps(msgs, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()

def extract_wrapper(raw: str, tag: str) -> str:
    pat = rf"<{tag}>([\s\S]*?)(?:</{tag}>|$)"
    m = re.search(pat, raw)
    if not m:
        return raw.strip()
    inner = m.group(1).strip()
    return f"<{tag}>\n{inner}\n</{tag}>"

# ─── SMOKE-TEST ──────────────────────────────────────────────────────
new = 0
for model in MODELS:
    model_dir = OUT_ROOT / model.replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Testing Anthropic model: {model} ===")

    for mod, prompt_path in tqdm(SAMPLES.items(), desc="modalities", ncols=60):
        uid_variant = prompt_path.stem           # e.g. "abc123_v0"
        uid, _      = uid_variant.split("_v", 1)

        # Stub VISION: no image support in this smoke test
        if mod == "VISION":
            for temp in TEMPS:
                out = model_dir / f"{uid}_{mod}_v0_t{str(temp).replace('.', '_')}.json"
                if out.exists():
                    continue
                record = {
                    "uid":        uid,
                    "variant":    0,
                    "modality":   mod,
                    "temperature": temp,
                    "model":      model,
                    "skipped":    True,
                    "reason":     "vision stub—no image support",
                    "endpoint":   "Anthropic messages",
                    "timestamp":  datetime.utcnow().isoformat() + "Z"
                }
                out.write_text(json.dumps(record, indent=2))
                new += 1
            continue

        # Load system + user messages
        sys_user   = json.loads(prompt_path.read_text())
        system_msg = sys_user[0]["content"]
        user_msg   = sys_user[1]["content"]

        # Build Messages API payload: separate system & user
        messages = [{"role": "user", "content": user_msg}]
        prompt_sha = sha256_of_messages(messages + [{"role":"system","content":system_msg}])

        for temp in TEMPS:
            tslug = str(temp).replace(".", "_")
            out = model_dir / f"{uid}_{mod}_v0_t{tslug}.json"
            if out.exists():
                continue

            try:
                resp = client.messages.create(
                    model=model,
                    system=system_msg,
                    messages=messages,
                    max_tokens=MAX_TOKENS,
                    temperature=temp,
                )
            except Exception as e:
                print(f"❌ {model} {mod}_{uid}@{temp}: {e}")
                continue

            # Anthropic Messages API returns resp.content[...]    
            raw = resp.content[0].text

            # Extract the correct wrapper
            tag = "OUTPUT_TEXT" if mod == "TEXT" else "OUTPUT_JSON"
            clean = extract_wrapper(raw, tag)

            # Build record
            record = {
                "uid":           uid,
                "variant":       0,
                "modality":      mod,
                "temperature":   temp,
                "model":         model,
                "endpoint":      "Anthropic messages",
                "timestamp":     datetime.utcnow().isoformat() + "Z",
                "prompt_sha256": prompt_sha,
                "response": {
                    "raw":       raw,
                    "extracted": clean
                }
            }
            out.write_text(json.dumps(record, indent=2, ensure_ascii=False))
            new += 1
            time.sleep(1.0)

print(f"\n🏁 Anthropic messages smoketest complete: {new} files in {OUT_ROOT}")
