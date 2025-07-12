#!/usr/bin/env python3
"""
phase_01_schema_checker_paralegal.py
Validates OUTPUT_TEXT / OUTPUT_JSON wrappers for the paralegal batch.

Output (saved alongside this script)
------------------------------------
schema_text_gui_per_output_paralegal.csv
    uid, variant, temp, model, modality, file,
    strict, strict_reason, rescued, rescued_reason, schema_status
"""
from __future__ import annotations
import argparse, ast, csv, json, pathlib, re
from jsonschema import Draft7Validator

# ── constants ─────────────────────────────────────────────────────────
TAG        = "OUTPUT_JSON"
RESCUE_RE  = re.compile(rf"<{TAG}>(.*?)</{TAG}>", re.S | re.I)
TEXT_RE    = re.compile(r"<OUTPUT_TEXT>(.*?)</OUTPUT_TEXT>", re.S | re.I)
MIN_TEXT   = 5

SCHEMA_DIR = pathlib.Path(__file__).parent / "schemas"
GUI_SCHEMA = Draft7Validator(json.loads((SCHEMA_DIR / "GUI.json").read_text()))
FNAME_RE   = re.compile(r"_v(?P<var>[^_]+)_t(?P<int>-?\d+)_(?P<frac>\d+)\.json$")

# ── helpers ───────────────────────────────────────────────────────────
def extract_payload(d: dict) -> str:
    r = d.get("response", {})
    return (
        r.get("content", [{}])[0].get("text") or
        (r.get("candidates", [{}])[0]
            .get("content", {}).get("parts", [{}])[0].get("text")) or
        r.get("choices", [{}])[0].get("message", {}).get("content") or
        r.get("choices", [{}])[0].get("content", "")
    )

def load_pred(payload: str, strict: bool):
    if strict and payload.strip().startswith(f"<{TAG}>") and payload.strip().endswith(f"</{TAG}>"):
        body = payload[len(f"<{TAG}>"):-len(f"</{TAG}>")].strip()
    else:
        m = RESCUE_RE.search(payload); body = m.group(1).strip() if m else None
    if not body:
        return None
    try:
        return json.loads(body)
    except Exception:
        try:
            return ast.literal_eval(body)
        except Exception:
            return None

# ── main ──────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase-01 schema checker (paralegal slice only)")
    parser.add_argument("--runs-dir", default="runs/openai/one_occ",
                        help="folder containing run *.json files")
    parser.add_argument("--manifest", default="data/manifests/paralegal_tasks.csv",
                        help="task manifest giving modality")
    args = parser.parse_args()

    # uid ➜ modality (TEXT / GUI)
    uid2mod = {}
    with open(args.manifest, newline='', encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid2mod[row["uid"]] = row["modality"].upper()

    runs_dir = pathlib.Path(args.runs_dir).resolve()
    if not runs_dir.exists():
        raise SystemExit(f"No runs found in {runs_dir}")

    per_rows = []

    for file in runs_dir.rglob("*.json"):
        uid = file.stem.split("_")[0]
        modality = uid2mod.get(uid, "UNKNOWN").upper()

        m = FNAME_RE.search(file.name)
        if not m:
            raise ValueError(f"Bad filename pattern: {file.name}")
        variant = m.group("var")
        temp    = f"{m.group('int')}.{m.group('frac')}"

        data   = json.loads(file.read_text())
        model  = data.get("model") or file.parent.name
        rel    = str(file.relative_to(runs_dir))
        payload = extract_payload(data)

        # ---------- TEXT --------------------------------------------------
        if modality == "TEXT":
            m_strict  = TEXT_RE.fullmatch(payload.strip())
            strict    = 1.0 if (m_strict and len(m_strict.group(1).strip()) >= MIN_TEXT) else 0.0
            strict_r  = "ok" if strict else "empty or missing block (strict)"

            m_rescue  = TEXT_RE.search(payload)
            rescued   = 1.0 if (m_rescue and len(m_rescue.group(1).strip()) >= MIN_TEXT) else 0.0
            rescued_r = "ok" if rescued else "empty or missing block (rescued)"

            schema_status = "strict" if strict else ("rescued" if rescued else "fail")

            per_rows.append([uid, variant, temp, model, modality, rel,
                             strict, strict_r, rescued, rescued_r, schema_status])
            continue

        # ---------- GUI ---------------------------------------------------
        if modality == "GUI":
            pred_s = load_pred(payload, True)
            if not pred_s:
                strict, strict_r = 0.0, "no parsable JSON"
            else:
                errs = list(GUI_SCHEMA.iter_errors(pred_s))
                strict, strict_r = (1.0, "ok") if not errs else (0.0, errs[0].message)

            pred_r = load_pred(payload, False)
            if not pred_r:
                rescued, rescued_r = 0.0, "no parsable JSON"
            else:
                errs = list(GUI_SCHEMA.iter_errors(pred_r))
                rescued, rescued_r = (1.0, "ok") if not errs else (0.0, errs[0].message)

            schema_status = "strict" if strict else ("rescued" if rescued else "fail")

            per_rows.append([uid, variant, temp, model, modality, rel,
                             strict, strict_r, rescued, rescued_r, schema_status])
            continue

    # ── write single consolidated CSV (next to this script) ────────────
    out_path = pathlib.Path(__file__).parent / "schema_text_gui_per_output_paralegal.csv"
    with out_path.open("w", newline='', encoding="utf-8") as fp:
        csv.writer(fp).writerows(
            [["uid","variant","temp","model","modality","file",
              "strict","strict_reason","rescued","rescued_reason","schema_status"],
             *per_rows])
    print(f"✅  {len(per_rows)} rows → {out_path}")

# entry-point
if __name__ == "__main__":
    main()
