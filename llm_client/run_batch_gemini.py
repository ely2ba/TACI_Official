#!/usr/bin/env python3
# run_batch_gemini.py — TACI v1.3 (restricted to supported Gemini variants)

import os
import time
import json
import base64
import random
import traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google.auth import default
from google.auth.transport.requests import Request
import requests
from tqdm import tqdm
import pandas as pd

# ─── 1. Configuration ─────────────────────────────────────────────────
load_dotenv()
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
REGION     = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")

# Only the two confirmed‐working variants
GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash-preview-05-20",
]
TEMPS = [0.0, 0.5]

# Paths
PROMPTS_ROOT = Path("prompts")
PROMPT_FILES = list(PROMPTS_ROOT.rglob("*/*_v*.json"))

RUNS_ROOT = Path("runs") / "gemini"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

# Manifest & assets for VISION
MANIFEST    = Path("data/manifests/sampled_tasks_with_modality.csv")
df_manifest = pd.read_csv(MANIFEST, dtype=str).set_index("uid", drop=False)
ASSETS_ROOT = Path("assets/images")

# ─── 2. Authentication ────────────────────────────────────────────────
def get_auth_token() -> str:
    creds, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(Request())
    return creds.token

AUTH_TOKEN = get_auth_token()

# ─── 3. Helpers ───────────────────────────────────────────────────────
def encode_img(p: Path) -> dict:
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    return {"inline_data": {"mime_type": mime, "data": b64}}

def exponential_backoff(fn, *args, max_tries=5, **kwargs):
    for i in range(1, max_tries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if i == max_tries:
                raise
            time.sleep(random.uniform(1, 2 ** i))

# ─── 4. Batch Loop ────────────────────────────────────────────────────
for model_id in GEMINI_MODELS:
    endpoint = (
        f"https://{REGION}-aiplatform.googleapis.com/v1/"
        f"projects/{PROJECT_ID}/locations/{REGION}/"
        f"publishers/google/models/{model_id}:generateContent"
    )
    headers = {
        "Authorization": f"Bearer {AUTH_TOKEN}",
        "Content-Type": "application/json",
    }
    model_dir = RUNS_ROOT / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Running Gemini model: {model_id} ===")
    for prompt_path in tqdm(PROMPT_FILES, desc=model_id, ncols=80):
        uid, variant = prompt_path.stem.split("_v")
        variant = int(variant)
        modality = prompt_path.parent.name.upper()

        # MANUAL stubs
        if modality == "MANUAL":
            for temp in TEMPS:
                out = model_dir / f"{uid}_v{variant}_t{str(temp).replace('.','_')}.json"
                if not out.exists():
                    out.write_text(json.dumps({
                        "uid": uid, "variant": variant, "temperature": temp,
                        "model": model_id, "modality": modality,
                        "skipped": True, "reason": "manual task – auto-scored 0"
                    }, indent=2))
            continue

        # Only TEXT, GUI, VISION supported
        if modality not in {"TEXT", "GUI", "VISION"}:
            continue

        # Load prompt messages
        messages = json.loads(prompt_path.read_text())  # [{"role":..., "content":...}, ...]

        # Build 'contents' payload
        contents = [{"role": "user", "parts": []}]
        for msg in messages:
            contents[0]["parts"].append({"text": msg["content"]})

        # Inline two images for VISION
        if modality == "VISION":
            arche = df_manifest.at[uid, "vision_archetype"]
            img_dir = ASSETS_ROOT / arche
            imgs = sorted(img_dir.glob("*.[pj][pn]g"))[:2]
            if len(imgs) < 2:
                tqdm.write(f"⚠️ {uid}: missing images for archetype {arche}")
                continue
            for img in imgs:
                contents[0]["parts"].append(encode_img(img))

        # Call for each temperature
        for temp in TEMPS:
            tslug = str(temp).replace(".", "_")
            out = model_dir / f"{uid}_v{variant}_t{tslug}.json"
            if out.exists():
                continue

            payload = {
                "contents": contents,
                "generation_config": {
                    "temperature": temp,
                    "max_output_tokens": 4096
                }
            }

            try:
                resp = exponential_backoff(
                    requests.post, endpoint,
                    headers=headers, data=json.dumps(payload)
                )
                resp.raise_for_status()
            except Exception as e:
                tqdm.write(f"❌ {model_id} {uid}_v{variant}@{temp}: {e}")
                traceback.print_exc(limit=1)
                continue

            data = resp.json()
            record = {
                "uid": uid,
                "variant": variant,
                "temperature": temp,
                "model": model_id,
                "modality": modality,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "response": data
            }
            out.write_text(json.dumps(record, indent=2))
            tqdm.write(f"✅ {model_id} {uid}_v{variant}@{temp}")
            time.sleep(1.0)

print("\n🏁 Gemini batch runs complete — outputs under", RUNS_ROOT)
