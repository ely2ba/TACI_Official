#!/usr/bin/env python3
# llm_client/run_batch_llama3_groq_rate_limited.py
# Groq llama3-8b-8192 batch runner with 30 RPM throttle + retry/backoff,
# and without manual extraction—raw responses only.

import os
import re
import json
import time
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

# ─── CONFIG ─────────────────────────────────────────────────────────
load_dotenv()  # expects GROQ_API_KEY

API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set GROQ_API_KEY in your environment")

ENDPOINT   = "https://api.groq.com/openai/v1/chat/completions"
HEADERS    = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type":  "application/json",
}
MODEL      = "llama3-8b-8192"
TEMPS      = [0.0, 0.5]
MAX_TOKENS = 1024

# 30 requests per minute → one every 2.0s
MIN_INTERVAL = 60.0 / 30.0
_last_request = 0.0

def throttle():
    global _last_request
    now = time.time()
    elapsed = now - _last_request
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)
    _last_request = time.time()

# ─── PROMPTS ─────────────────────────────────────────────────────────
PROMPT_ROOT  = Path("prompts")
PROMPT_FILES = list(PROMPT_ROOT.rglob("*/*_v[0-2].json"))

# ─── OUTPUT ──────────────────────────────────────────────────────────
OUT_ROOT = Path("runs") / "groq_batch_rate_limited" / MODEL.replace("/", "_")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

# ─── PROVENANCE ──────────────────────────────────────────────────────
try:
    COMMIT_SHA = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
    ).decode().strip()
except Exception:
    COMMIT_SHA = None

# ─── HELPERS ────────────────────────────────────────────────────────
def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def call_with_backoff(payload: dict, max_tries: int = 5) -> dict:
    backoff = 1.0
    for attempt in range(1, max_tries + 1):
        throttle()
        try:
            r = requests.post(ENDPOINT, headers=HEADERS, json=payload, timeout=30)
            if r.status_code == 429:
                retry_after = r.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else backoff
                print(f"⚠️  429, attempt {attempt}, sleeping {wait:.1f}s")
                time.sleep(wait)
                backoff *= 2
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            # retry on 5xx or network errors
            if attempt == max_tries or (hasattr(e, 'response') and e.response is not None and e.response.status_code < 500):
                raise
            print(f"⚠️  error on attempt {attempt}: {e}, retrying in {backoff:.1f}s")
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("Exceeded maximum retry attempts")

# ─── MAIN LOOP ───────────────────────────────────────────────────────
new = 0
for prompt_path in tqdm(PROMPT_FILES, desc="prompts", ncols=80):
    modality    = prompt_path.parent.name.upper()  # TEXT, GUI, VISION, MANUAL
    uid_variant = prompt_path.stem                  # e.g. "dc35566b_v1"
    uid, var    = uid_variant.split("_v", 1)
    variant     = int(var)
    messages    = json.loads(prompt_path.read_text())

    for temp in TEMPS:
        tslug    = str(temp).replace(".", "_")
        out_file = OUT_ROOT / f"{uid}_v{variant}_t{tslug}.json"
        if out_file.exists():
            continue

        # stub manual & vision
        if modality in ("MANUAL", "VISION"):
            record = {
                "uid":           uid,
                "variant":       variant,
                "modality":      modality,
                "temperature":   temp,
                "model":         MODEL,
                "skipped":       True,
                "reason":        f"{modality.lower()} stub—text model",
                "endpoint":      ENDPOINT,
                "timestamp":     datetime.utcnow().isoformat() + "Z",
                "commit_sha":    COMMIT_SHA,
            }
            out_file.write_text(json.dumps(record, indent=2))
            new += 1
            continue

        payload = {
            "model":      MODEL,
            "messages":   messages,
            "temperature":temp,
            "max_tokens": MAX_TOKENS,
        }

        try:
            data = call_with_backoff(payload)
        except Exception as e:
            print(f"❌ {MODEL} {modality}_{uid}@{temp}: {e}")
            continue

        # no extraction: leave raw content in place
        record = {
            "uid":           uid,
            "variant":       variant,
            "modality":      modality,
            "temperature":   temp,
            "model":         MODEL,
            "endpoint":      ENDPOINT,
            "timestamp":     datetime.utcnow().isoformat() + "Z",
            "prompt_sha256": sha256(prompt_path.read_text()),
            "commit_sha":    COMMIT_SHA,
            "response":      data
        }
        out_file.write_text(json.dumps(record, indent=2, ensure_ascii=False))
        new += 1

        # small extra pause to smooth bursts
        time.sleep(0.5)

print(f"\n🏁 Groq batch (rate-limited) complete: {new} files in {OUT_ROOT}")
