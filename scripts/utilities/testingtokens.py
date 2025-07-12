import json, pathlib

RUNS_DIR = pathlib.Path("runs/openai")
VISION_PROMPTS = pathlib.Path("prompts/vision")  # adjust if needed

TARGET_MODELS = ["gpt-3-5-turbo", "gpt-4", "gpt-4-turbo"]
TEMPS = [0.0, 0.5]  # Add all temps used in your batch runs

def slugify_temp(temp):
    return str(temp).replace('.', '_')

for model in TARGET_MODELS:
    model_dir = RUNS_DIR / model
    model_dir.mkdir(parents=True, exist_ok=True)
    for prompt_file in VISION_PROMPTS.glob("*.json"):
        uid_variant = prompt_file.stem  # e.g., "abc123_v0"
        uid, v_str = uid_variant.split("_v")
        variant = int(v_str)
        for temp in TEMPS:
            temp_slug = slugify_temp(temp)
            stub_path = model_dir / f"{uid}_v{variant}_t{temp_slug}.json"
            if not stub_path.exists():
                stub = {
                    "uid": uid,
                    "variant": variant,
                    "modality": "VISION",
                    "model": model,
                    "temperature": temp,
                    "skipped": True,
                    "reason": "vision not supported",
                }
                stub_path.write_text(json.dumps(stub, indent=2))
print("✅ Vision stubs created for all missing files/temps for:", ", ".join(TARGET_MODELS))
