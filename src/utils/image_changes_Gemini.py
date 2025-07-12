#!/usr/bin/env python3
# rerun_vision_task_gemini.py — TACI targeted VISION rerun for Gemini

import os, time, json, base64, random, traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google.auth import default
from google.auth.transport.requests import Request
import requests
import pandas as pd

# ─── 1. Config & Setup ────────────────────────────────────────────────
load_dotenv()
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
REGION     = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")

GEMINI_MODELS = [
    "gemini-2.0-flash",
    "gemini-2.5-flash-preview-05-20",
]
TEMPS = [0.0, 0.5]
VARIANTS = [0, 1, 2]
TARGET_UID = "ce0be4e8"

PROMPTS_ROOT = Path("prompts/vision")
RUNS_ROOT = Path("runs/gemini")
MANIFEST    = Path("data/manifests/sampled_tasks_with_modality.csv")
df_manifest = pd.read_csv(MANIFEST, dtype=str).set_index("uid", drop=False)
ASSETS_ROOT = Path("assets/images")

def get_auth_token() -> str:
    creds, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(Request())
    return creds.token

AUTH_TOKEN = get_auth_token()

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

# ─── 2. Patch Rerun for VISION ────────────────────────────────────────
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

    print(f"\n=== Rerunning VISION UID {TARGET_UID} for Gemini model: {model_id} ===")
    for variant in VARIANTS:
        prompt_path = PROMPTS_ROOT / f"{TARGET_UID}_v{variant}.json"
        if not prompt_path.exists():
            print(f"Prompt not found: {prompt_path}")
            continue

        # Build 'contents' payload with two images
        messages = json.loads(prompt_path.read_text())
        contents = [{"role": "user", "parts": []}]
        for msg in messages:
            contents[0]["parts"].append({"text": msg["content"]})

        # Add the two (current) images
        arche = df_manifest.at[TARGET_UID, "vision_archetype"]
        img_dir = ASSETS_ROOT / arche
        imgs = sorted(img_dir.glob("*.[pj][pn]g"))[:2]
        if len(imgs) < 2:
            print(f"⚠️ {TARGET_UID}: missing images for archetype {arche}")
            continue
        print(f"[INFO] {model_id} {TARGET_UID}_v{variant}: img_000 = {imgs[0].name}, img_001 = {imgs[1].name}")
        for img in imgs:
            contents[0]["parts"].append(encode_img(img))

        # For each temperature
        for temp in TEMPS:
            tslug = str(temp).replace(".", "_")
            out = model_dir / f"{TARGET_UID}_v{variant}_t{tslug}.json"

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
                print(f"❌ {model_id} {TARGET_UID}_v{variant}@{temp}: {e}")
                traceback.print_exc(limit=1)
                continue

            data = resp.json()
            record = {
                "uid": TARGET_UID,
                "variant": variant,
                "temperature": temp,
                "model": model_id,
                "modality": "VISION",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "response": data
            }
            out.write_text(json.dumps(record, indent=2))
            print(f"✔️  Replaced {out}")

print("\n🏁 Gemini VISION patch runs complete — see runs/gemini/")
