#!/usr/bin/env python3
# rerun_o3_failures.py — retry only the failed o3 jobs (self-contained)

import os, json, random, time, traceback, base64, hashlib
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIConnectionError, APIError
from tqdm import tqdm
import pandas as pd

# ─── 0. config & constants ────────────────────────────────────────────
load_dotenv()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")

client = OpenAI(api_key=OPENAI_KEY)

MODEL_ID   = "o3-2025-04-16"
MAX_COMPLETION_TOKENS = {"TEXT": 4000, "GUI": 2048, "VISION": 2048}
TEMPERATURES = [0.0, 0.5]
SEEDS        = {0.0: 0, 0.5: 1}           # deterministic mapping

# failed jobs (uid, variant, temperature)
FAILED = [
    ("9cebf902", 1, 0.0),
    ("7dda7be5", 1, 0.0),
    ("8b2ab848", 0, 0.0),
    ("7dda7be5", 0, 0.0),
    ("3db6d320", 0, 0.5),
    ("3db6d320", 0, 0.0),
]
FAILED_SET = {(u, v, t) for u, v, t in FAILED}

# paths
MANIFEST    = Path("data/manifests/paralegal_tasks.csv")
PROMPT_DIR  = Path("prompts/one_occ")
RUNS_ROOT   = Path("runs/openai/one_occ")
ASSETS_ROOT = Path("assets/images")

# provenance
try:
    COMMIT_SHA = (
        __import__("subprocess")
        .check_output(["git", "rev-parse", "HEAD"], stderr=__import__("subprocess").DEVNULL)
        .decode()
        .strip()
    )
except Exception:
    COMMIT_SHA = None

# helpers
h256 = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()

def load_messages(path: Path):
    return json.loads(path.read_text())

def load_vision_images(uid: str, df_manifest) -> list:
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
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
        parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
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

# manifest
df_manifest = pd.read_csv(MANIFEST, dtype=str).fillna("").set_index("uid", drop=False)

# ─── main loop ────────────────────────────────────────────────────────
for uid, variant, temp in tqdm(FAILED, desc="reruns", ncols=80):
    temp_slug = str(temp).replace(".", "_")
    out_path  = RUNS_ROOT / MODEL_ID / f"{uid}_v{variant}_t{temp_slug}.json"

    # delete old/truncated file if present
    if out_path.exists():
        out_path.unlink()

    # locate prompt
    prompt_file = next((p for p in PROMPT_DIR.rglob(f"{uid}_v{variant}.json")), None)
    if prompt_file is None:
        tqdm.write(f"⚠️  prompt not found for {uid}_v{variant}")
        continue

    modality = prompt_file.parent.name.upper()
    messages = load_messages(prompt_file)

    if modality == "VISION":
        try:
            parts = load_vision_images(uid, df_manifest)
        except Exception as e:
            tqdm.write(f"⚠️  skip {uid}_v{variant}@t{temp_slug}: {e}")
            continue
        messages[1]["content"] = [
            {"type": "text", "text": messages[1]["content"]},
            *parts,
        ]

    prompt_hash = h256(json.dumps(messages, separators=(",", ":"), ensure_ascii=False))
    seed        = SEEDS[temp]

    try:
        resp = exponential_backoff(
            client.chat.completions.create,
            model=MODEL_ID,
            messages=messages,
            max_completion_tokens=MAX_COMPLETION_TOKENS[modality],
            seed=seed,
        )
    except Exception as e:
        tqdm.write(f"❌ retry failed {uid}_v{variant}@t{temp_slug}: {e}")
        traceback.print_exc(limit=1)
        continue

    record = {
        "uid": uid,
        "variant": variant,
        "temperature": temp,
        "seed": seed,
        "model": MODEL_ID,
        "modality": modality,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "commit_sha": COMMIT_SHA,
        "prompt_sha256": prompt_hash,
        "response": resp.model_dump(),
    }
    out_path.write_text(json.dumps(record, indent=2))
    tqdm.write(f"✅ redone {uid}_v{variant}@t{temp_slug}")

print("🏁 selective o3 retries complete.")
