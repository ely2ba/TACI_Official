#!/usr/bin/env python3
# llm_client/run_batch_anthropic.py — TACI v1.2 Anthropic batch runner (raw full responses)
# Sweeps all prompt variants through Claude 3 Opus & 3.5 Sonnet via Messages API,
# stubbing VISION and MANUAL tasks, logging full response dicts.

import os
import time
import json
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from tqdm import tqdm

# ─── 1. Environment & constants ───────────────────────────────────────
load_dotenv()  # expects ANTHROPIC_API_KEY
API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing ANTHROPIC_API_KEY in environment")

client = Anthropic(api_key=API_KEY)

MODELS     = ["claude-3-opus-20240229", "claude-3-5-sonnet-20240620"]
TEMPS      = [0.0, 0.5]
MAX_TOKENS = 1024

# ─── 2. Paths & manifest ─────────────────────────────────────────────
PROMPTS_ROOT = Path("prompts")
PROMPT_FILES = list(PROMPTS_ROOT.rglob("*/*.json"))

RUNS_ROOT = Path("outputs") / "anthropic"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

# ─── 3. Provenance ────────────────────────────────────────────────────
try:
    COMMIT_SHA = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
    ).decode().strip()
except Exception:
    COMMIT_SHA = None

# ─── 4. Helpers ───────────────────────────────────────────────────────
def sha256_of_messages(msgs: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps(msgs, separators=(",", ":"), ensure_ascii=False)
        .encode("utf-8")
    ).hexdigest()

def exponential_backoff(fn, *args, max_tries=5, **kwargs):
    backoff = 1.0
    for attempt in range(1, max_tries+1):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if attempt == max_tries:
                raise
            time.sleep(backoff)
            backoff *= 2

# ─── 5. Batch loop ────────────────────────────────────────────────────
for model in MODELS:
    slug      = model.replace("/", "_").replace(".", "-")
    model_dir = RUNS_ROOT / slug
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Running Anthropic model: {model} ===")

    for prompt_path in tqdm(PROMPT_FILES, desc=slug, ncols=80):
        uid_variant = prompt_path.stem           # e.g. "abc123_v1"
        uid, v_str  = uid_variant.rsplit("_v", 1)
        variant     = int(v_str)
        modality    = prompt_path.parent.name.upper()

        for temp in TEMPS:
            temp_slug = str(temp).replace(".", "_")
            out_path  = model_dir / f"{uid}_v{variant}_t{temp_slug}.json"
            if out_path.exists():
                continue

            # MANUAL → stub
            if modality == "MANUAL":
                stub = {
                    "uid":         uid,
                    "variant":     variant,
                    "modality":    modality,
                    "temperature": temp,
                    "model":       model,
                    "skipped":     True,
                    "reason":      "manual task – auto-scored 0",
                    "endpoint":    "Anthropic messages",
                    "timestamp":   datetime.utcnow().isoformat() + "Z",
                    "commit_sha":  COMMIT_SHA,
                }
                out_path.write_text(json.dumps(stub, indent=2))
                continue

            # VISION → stub (no image support)
            if modality == "VISION":
                stub = {
                    "uid":         uid,
                    "variant":     variant,
                    "modality":    modality,
                    "temperature": temp,
                    "model":       model,
                    "skipped":     True,
                    "reason":      "vision stub—no image support",
                    "endpoint":    "Anthropic messages",
                    "timestamp":   datetime.utcnow().isoformat() + "Z",
                    "commit_sha":  COMMIT_SHA,
                }
                out_path.write_text(json.dumps(stub, indent=2))
                continue

            # Load system & user messages
            msgs       = json.loads(prompt_path.read_text())
            system_msg = msgs[0]["content"]
            user_msg   = msgs[1]["content"]

            # Prepare for prompt hashing
            payload_msgs = [
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": user_msg},
            ]
            prompt_hash = sha256_of_messages(payload_msgs)

            # Call Anthropic Messages API
            try:
                resp = exponential_backoff(
                    client.messages.create,
                    model=model,
                    system=system_msg,
                    messages=[{"role":"user","content":user_msg}],
                    max_tokens=MAX_TOKENS,
                    temperature=temp
                )
            except Exception as e:
                tqdm.write(f"❌ {model} {uid}_v{variant}@{temp}: {e}")
                continue

            # Convert full response object to dict
            resp_dict = resp.model_dump()

            # Build record with full response dict
            record = {
                "uid":           uid,
                "variant":       variant,
                "modality":      modality,
                "temperature":   temp,
                "model":         model,
                "endpoint":      "Anthropic messages",
                "timestamp":     datetime.utcnow().isoformat() + "Z",
                "commit_sha":    COMMIT_SHA,
                "prompt_sha256": prompt_hash,
                "response":      resp_dict
            }
            out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))

    print(f"✅ Completed runs for {model}, outputs in {model_dir}")

print(f"\n🏁 All Anthropic batch runs complete — see {RUNS_ROOT}")
