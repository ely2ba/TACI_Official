#!/usr/bin/env python3
# run_batch_openai_o3.py — TACI v1.2.5  (o3-only, seed + max_completion_tokens)

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

# single target model
MODELS = ["o3-2025-04-16"]
SUPPORT = {"o3-2025-04-16": {"TEXT", "GUI", "VISION"}}

# “temperatures” kept only to preserve the old filename scheme
TEMPERATURES = [0.0, 0.5]
SEEDS        = {0.0: 0, 0.5: 1}          # deterministic replicas

MAX_COMPLETION_TOKENS = {
    "TEXT":   2048,
    "GUI":    2048,
    "VISION": 2048,
}

# ─── 2. Paths ─────────────────────────────────────────────────────────
MANIFEST    = Path("data/manifests/paralegal_tasks.csv")
PROMPT_DIR  = Path("prompts/one_occ")
RUNS_ROOT   = Path("runs/openai/one_occ")
ASSETS_ROOT = Path("assets/images")
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

# ─── 3. Manifest & provenance ────────────────────────────────────────
df_manifest = pd.read_csv(MANIFEST, dtype=str).fillna("").set_index("uid", drop=False)
try:
    COMMIT_SHA = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
    ).decode().strip()
except Exception:
    COMMIT_SHA = None

# ─── 4. Helpers ───────────────────────────────────────────────────────
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
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"}
        })
    return parts

def exponential_backoff(fn, *args, max_tries=6, **kwargs):
    for attempt in range(1, max_tries + 1):
        try:
            return fn(*args, **kwargs)
        except (RateLimitError, APIConnectionError):
            if attempt == max_tries:
                raise
            time.sleep(random.uniform(1, 2 ** attempt))
        except APIError:
            raise

# ─── 5. Batch loop ────────────────────────────────────────────────────
prompt_files = list(PROMPT_DIR.rglob("*.json"))

for model in tqdm(MODELS, desc="models", ncols=80, leave=False):
    model_dir = RUNS_ROOT / model
    model_dir.mkdir(parents=True, exist_ok=True)
    supported = SUPPORT[model]

    print(f"\n=== Running {model} over {len(prompt_files)} prompts ===")
    for prompt_file in tqdm(prompt_files, desc=model, ncols=80):
        uid_variant = prompt_file.stem
        uid, v_str  = uid_variant.rsplit("_v", 1)
        variant     = int(v_str)
        modality    = prompt_file.parent.name.upper()  # TEXT / GUI / VISION / MANUAL

        for temp in TEMPERATURES:
            temp_slug = str(temp).replace(".", "_")
            out_path  = model_dir / f"{uid}_v{variant}_t{temp_slug}.json"
            if out_path.exists():
                continue

            # stub MANUAL tasks
            if modality == "MANUAL":
                out_path.write_text(json.dumps({
                    "uid": uid, "variant": variant, "temperature": temp,
                    "model": model, "modality": modality,
                    "skipped": True, "reason": "manual task – auto-scored 0"
                }, indent=2))
                continue

            # skip unsupported modality (won’t happen for o3)
            if modality == "VISION" and modality not in supported:
                out_path.write_text(json.dumps({
                    "uid": uid, "variant": variant, "temperature": temp,
                    "model": model, "modality": modality,
                    "skipped": True, "reason": "vision not supported"
                }, indent=2))
                continue

            if modality not in supported:
                continue

            # build messages
            messages = load_messages(prompt_file)
            if modality == "VISION":
                try:
                    parts = load_vision_images(uid)
                except Exception as e:
                    tqdm.write(f"⚠️  {uid} vision load error: {e}")
                    continue
                messages[1]["content"] = [
                    {"type": "text", "text": messages[1]["content"]},
                    *parts
                ]

            prompt_hash = h256(json.dumps(messages, separators=(",", ":"), ensure_ascii=False))
            seed        = SEEDS[temp]

            # API call (no temperature; seed for determinism)
            try:
                resp = exponential_backoff(
                    client.chat.completions.create,
                    model=model,
                    messages=messages,
                    max_completion_tokens=MAX_COMPLETION_TOKENS[modality],
                    seed=seed,
                )
            except Exception as e:
                tqdm.write(f"❌ {uid}_v{variant}@{model}@{temp}: {e}")
                traceback.print_exc(limit=1)
                continue

            # save record
            record = {
                "uid": uid,
                "variant": variant,
                "temperature": temp,      # retained for file trace
                "seed": seed,
                "model": model,
                "modality": modality,
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "commit_sha": COMMIT_SHA,
                "prompt_sha256": prompt_hash,
                "response": resp.model_dump()
            }
            out_path.write_text(json.dumps(record, indent=2))

    print(f"✅ Completed runs for {model}, results in {model_dir}")

print("\n🏁 All o3 batch runs complete — see", RUNS_ROOT)
