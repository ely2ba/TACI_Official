import pathlib
import re

MODEL_DIR = pathlib.Path("runs/groq_batch_rate_limited/llama3-8b-8192")
pattern = re.compile(r"^([a-f0-9]+)_[A-Z]+_v(\d+)_t([0-9_]+)\.json$")

n_renamed = 0
for f in MODEL_DIR.glob("*.json"):
    m = pattern.match(f.name)
    if m:
        uid, variant, tslug = m.groups()
        new_name = f"{uid}_v{variant}_t{tslug}.json"
        new_path = f.parent / new_name
        if new_path.exists():
            print(f"SKIP: {new_name} already exists.")
            continue
        print(f"Renaming {f.name} -> {new_name}")
        f.rename(new_path)
        n_renamed += 1

print(f"Done. Renamed {n_renamed} files.")


