#!/usr/bin/env python3
# rerun_vision_task_openai.py — TACI patch utility for VISION reruns

import os, time, json, base64, hashlib, random, subprocess, traceback
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIConnectionError, APIError
from tqdm import tqdm
import pandas as pd

# ─── 1. Environment & constants ───────────────────────────────────────
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")

client = OpenAI(api_key=OPENAI_KEY)

TARGET_UID = "ce0be4e8"
MODELS = ["gpt-4o", "gpt-4.1"]
TEMPERATURES = [0.0, 0.5]
VARIANTS = [0, 1, 2]
MAX_TOKENS = 2048

MANIFEST    = Path("data/manifests/sampled_tasks_with_modality.csv")
PROMPT_DIR  = Path("prompts/vision")
RUNS_ROOT   = Path("runs/openai")
ASSETS_ROOT = Path("assets/images")

df_manifest = pd.read_csv(MANIFEST, dtype=str).fillna("").set_index("uid", drop=False)
try:
    COMMIT_SHA = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
except Exception:
    COMMIT_SHA = None

h256 = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()

def load_messages(path: Path):
    return json.loads(path.read_text())

def load_vision_images(uid: str):
    if uid not in df_manifest.index:
        raise ValueError(f"No manifest row for uid {uid}")
    archetype = df_manifest.at[uid, "vision_archetype"]
    if not archetype:
        raise ValueError(f"vision_archetype blank for {uid}")
    img_dir = ASSETS_ROOT / archetype
    paths = sorted(img_dir.glob("*.[pj][np]g"))[:2]
    if len(paths) < 2:
        raise FileNotFoundError(f"Need ≥2 images in {img_dir}")
    parts = []
    for p in paths:
        b64  = base64.b64encode(p.read_bytes()).decode("ascii")
        mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
        parts.append({"type": "image_url",
                      "image_url": {"url": f"data:{mime};base64,{b64}"}})
    return parts

def exponential_backoff(fn, *args, max_tries=6, **kwargs):
    for attempt in range(1, max_tries + 1):
        try:
            return fn(*args, **kwargs)
        except (RateLimitError, APIConnectionError):
            if attempt == max_tries:
                raise
            sleep = random.uniform(1, 2 ** attempt)
            print(f"⏳ retry {attempt}/{max_tries}, sleeping {sleep:.1f}s")
            time.sleep(sleep)
        except APIError:
            raise

# ─── 2. Main patch loop ──────────────────────────────────────────────
print(f"=== Rerunning VISION task {TARGET_UID} for gpt-4o and gpt-4.1 (all variants/temps) ===")

for model in MODELS:
    model_slug = model.replace("/", "_").replace(".", "-")
    model_dir  = RUNS_ROOT / model_slug
    model_dir.mkdir(parents=True, exist_ok=True)

    for variant in VARIANTS:
        prompt_path = PROMPT_DIR / f"{TARGET_UID}_v{variant}.json"
        if not prompt_path.exists():
            print(f"Prompt not found: {prompt_path}")
            continue

        for temp in TEMPERATURES:
            temp_slug = str(temp).replace(".", "_")
            out_path  = model_dir / f"{TARGET_UID}_v{variant}_t{temp_slug}.json"

            messages = load_messages(prompt_path)
            try:
                parts = load_vision_images(TARGET_UID)
            except Exception as e:
                print(f"⚠️  {TARGET_UID} vision load error: {e}")
                continue
            messages[1]["content"] = [
                {"type": "text", "text": messages[1]["content"]},
                *parts
            ]

            prompt_hash = h256(json.dumps(messages, separators=(",", ":"), ensure_ascii=False))

            try:
                resp = exponential_backoff(
                    client.chat.completions.create,
                    model=model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=MAX_TOKENS
                )
            except Exception as e:
                print(f"❌ {TARGET_UID}_v{variant}@{model}@{temp}: {e}")
                traceback.print_exc(limit=1)
                continue

            record = {
                "uid": TARGET_UID,
                "variant": variant,
                "temperature": temp,
                "model": model,
                "modality": "VISION",
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "commit_sha": COMMIT_SHA,
                "prompt_sha256": prompt_hash,
                "response": resp.model_dump()
            }
            out_path.write_text(json.dumps(record, indent=2))
            print(f"✔️  Replaced {out_path}")

print("\n🏁 Done patching VISION task", TARGET_UID)
