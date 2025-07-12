#!/usr/bin/env python3
# test_three_prompts.py — smoke‐test 3 sample prompts through your runner logic

import json, base64, hashlib, random, time, subprocess
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIConnectionError, APIError
import pandas as pd

# ─── Configuration ────────────────────────────────────────────────
load_dotenv()
client = OpenAI(api_key=__import__("os").environ["OPENAI_API_KEY"])

MODELS    = ["gpt-3.5-turbo","gpt-4","gpt-4-turbo","gpt-4o","gpt-4.1"]
SUPPORT   = {
    "gpt-3.5-turbo": {"TEXT","GUI"},
    "gpt-4":         {"TEXT","GUI"},
    "gpt-4-turbo":   {"TEXT","GUI"},
    "gpt-4o":        {"TEXT","GUI","VISION"},
    "gpt-4.1":       {"TEXT","GUI","VISION"},
}
MAX_TOKENS = {"TEXT":2048,"GUI":2048,"VISION":2048}

MANIFEST    = Path("data/manifests/sampled_tasks_with_modality.csv")
PROMPT_DIR  = Path("prompts")
RUNS_ROOT   = Path("runs") / "openai_smoketest"
ASSETS_ROOT = Path("assets/images")

df_manifest = pd.read_csv(MANIFEST, dtype=str).set_index("uid")
try:
    COMMIT_SHA = subprocess.check_output(["git","rev-parse","HEAD"], stderr=subprocess.DEVNULL).decode().strip()
except:
    COMMIT_SHA = None

# ─── Helpers ─────────────────────────────────────────────────────────
def h256(s): return hashlib.sha256(s.encode("utf-8")).hexdigest()

def backoff(fn,*a,max_tries=5,**kw):
    for i in range(max_tries):
        try: return fn(*a,**kw)
        except (RateLimitError,APIConnectionError):
            time.sleep(random.uniform(1, 2**(i+1)))
        except APIError:
            raise

# pick one prompt per modality (v0)
samples = {
    mod.upper(): next(PROMPT_DIR.glob(f"{mod}/*_v0.json"))
    for mod in ["text","gui","vision"]
}

# ─── Run mini-batch ────────────────────────────────────────────────
RUNS_ROOT.mkdir(parents=True, exist_ok=True)

for model in MODELS:
    supported = SUPPORT[model]
    mslug = model.replace("/","_").replace(".","-")
    model_dir = RUNS_ROOT / mslug
    model_dir.mkdir(parents=True, exist_ok=True)

    for mod, pf0 in samples.items():
        if mod not in supported:
            print(f"❌ {model} doesn’t support {mod}, skipping")
            continue

        uid = pf0.stem.split("_v")[0]

        for v in (0,1,2):
            pf = pf0.with_name(f"{uid}_v{v}.json")
            if not pf.exists():
                print(f"⚠️ missing {pf.name}, skipping")
                continue

            msgs = json.loads(pf.read_text())

            if mod=="VISION":
                # inline the two images
                arche = df_manifest.at[uid,"vision_archetype"]
                imgs = sorted((ASSETS_ROOT/arche).glob("*.png"))[:2]
                parts = [{"type":"text","text":msgs[1]["content"]}]
                for img in imgs:
                    b64 = base64.b64encode(img.read_bytes()).decode()
                    mime = "image/png" if img.suffix==".png" else "image/jpeg"
                    parts.append({"type":"image_url","image_url":{"url":f"data:{mime};base64,{b64}"}})
                msgs[1]["content"] = parts

            phash = h256(json.dumps(msgs, separators=(",",":"), ensure_ascii=False))

            try:
                resp = backoff(
                    client.chat.completions.create,
                    model=model,
                    messages=msgs,
                    temperature=0.0,
                    max_tokens=MAX_TOKENS[mod]
                )
            except Exception as e:
                print(f"❌ {model} {uid}_v{v}: {e}")
                continue

            record = {
                "uid": uid,
                "variant": v,
                "temperature": 0.0,
                "model": model,
                "modality": mod,
                "timestamp": datetime.utcnow().isoformat()+"Z",
                "commit_sha": COMMIT_SHA,
                "prompt_sha256": phash,
                "response": resp.model_dump()
            }

            out = model_dir / f"{uid}_v{v}_t0_0.json"
            out.write_text(json.dumps(record, indent=2))
            print(f"✅ {model} {uid}_v{v}")

print("\n🏁 Smoketest complete; outputs in", RUNS_ROOT)
