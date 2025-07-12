import json, pathlib, re, os

def robust_extract(content, tag):
    """
    Extracts content inside <TAG>...</TAG>, forgiving to extra junk/preamble.
    If tag is not found, returns whole content (stripped) wrapped in tag.
    Returns (fixed_content, rescue_type)
    """
    pattern = re.compile(fr"<{tag}>(.*?)</{tag}>", re.S | re.I)
    match = pattern.search(content)
    if match:
        inner = match.group(1).strip()
        # Only return the inside, but preserve tags around it
        if content.strip().startswith(f"<{tag}>") and content.strip().endswith(f"</{tag}>"):
            return (content.strip(), "perfect")  # Already good
        else:
            return (f"<{tag}>{inner}</{tag}>", "rescued")  # Tag found but not at boundaries
    # No tag at all, wrap everything
    if content.strip():
        return (f"<{tag}>{content.strip()}</{tag}>", "fallback")
    return ("", "empty")

def extract_response_content(data):
    # Add your universal extraction logic here as before
    if 'response' in data and isinstance(data['response'], dict):
        if 'raw' in data['response']:
            return data['response']['raw']
        if 'extracted' in data['response']:
            return data['response']['extracted']
    if 'response' in data and 'choices' in data['response']:
        choices = data['response']['choices']
        if choices and 'message' in choices[0]:
            return choices[0]['message']['content']
    if 'response' in data and 'candidates' in data['response']:
        candidates = data['response']['candidates']
        if candidates and 'content' in candidates[0]:
            parts = candidates[0]['content'].get('parts', [])
            if parts and 'text' in parts[0]:
                return parts[0]['text']
    return None

def fix_all_runs(runs_dir="runs", out_dir="runs_fixed"):
    runs_path = pathlib.Path(runs_dir)
    out_path = pathlib.Path(out_dir)
    logs = []
    for model_dir in runs_path.iterdir():
        if not model_dir.is_dir():
            continue
        for f in model_dir.rglob("*.json"):
            try:
                data = json.loads(f.read_text())
            except Exception as e:
                print(f"Failed to read {f}: {e}")
                continue
            modality = data.get("modality")
            tag = {"TEXT":"OUTPUT_TEXT", "GUI":"OUTPUT_JSON", "VISION":"OUTPUT_JSON"}.get(modality)
            if not tag:
                continue
            content = extract_response_content(data)
            if not content:
                continue
            fixed, rescue_type = robust_extract(content, tag)
            # Log what had to be rescued/fixed
            if rescue_type != "perfect":
                logs.append((str(f), rescue_type, content[:100]))

            # Place fixed file in parallel folder structure
            rel_path = f.relative_to(runs_path)
            out_file = out_path / rel_path
            out_file.parent.mkdir(parents=True, exist_ok=True)

            # Write fixed content back into the same field
            if "choices" in data.get("response", {}):
                data["response"]["choices"][0]["message"]["content"] = fixed
            elif "candidates" in data.get("response", {}):
                data["response"]["candidates"][0]["content"]["parts"][0]["text"] = fixed
            elif "raw" in data.get("response", {}):
                data["response"]["raw"] = fixed

            with open(out_file, "w", encoding="utf-8") as fout:
                json.dump(data, fout, ensure_ascii=False, indent=2)

    # Log rescue actions
    with open(out_path / "rescue_log.txt", "w", encoding="utf-8") as logf:
        for fname, rescue_type, preview in logs:
            logf.write(f"{rescue_type.upper()}: {fname}\n{preview}\n{'-'*40}\n")

if __name__ == "__main__":
    fix_all_runs("runs", "runs_fixed")
    print("Fix-up complete. Log of rescued files written to runs_fixed/rescue_log.txt")
