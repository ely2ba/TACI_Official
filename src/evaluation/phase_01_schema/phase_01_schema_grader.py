#!/usr/bin/env python3
"""
Phase-01 TEXT / GUI format validator  (filename-driven variant & temp)

Adds:
    • schema_status column  strict | rescued | fail

Outputs
    schema_text_gui_per_output.csv
    schema_model_summary_strict.csv
    schema_model_summary_rescued.csv
    schema_gui_failures.csv
    schema_text_failures.csv
"""

import argparse, ast, csv, json, pathlib, re
from collections import defaultdict
from jsonschema import Draft7Validator

# ───────────────────────── constants ──────────────────────────
TAG        = "OUTPUT_JSON"
RESCUE_RE  = re.compile(rf"<{TAG}>(.*?)</{TAG}>", re.S | re.I)
TEXT_RE    = re.compile(r"<OUTPUT_TEXT>(.*?)</OUTPUT_TEXT>", re.S | re.I)
MIN_TEXT   = 5
SCHEMA_DIR = pathlib.Path(__file__).parent / "schemas"
GUI_SCHEMA = Draft7Validator(json.loads((SCHEMA_DIR / "GUI.json").read_text()))
FNAME_RE   = re.compile(r"_v(?P<var>[^_]+)_t(?P<int>-?\d+)_(?P<frac>\d+)\.json$")

# ───────────────────────── helpers ────────────────────────────
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

# ───────────────────────── main ───────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--manifest",
                    default="data/manifests/sampled_tasks_with_modality.csv")
    args = ap.parse_args()

    # manifest only gives modality
    uid2mod = {}
    with open(args.manifest, newline='', encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid2mod[row["uid"]] = row["modality"].upper()

    runs = pathlib.Path(args.runs_dir).resolve()
    out  = pathlib.Path(__file__).parent

    per_rows, model_stat = [], defaultdict(lambda: {"ok": 0, "n": 0})

    for file in runs.rglob("*.json"):
        uid      = file.stem.split("_")[0]
        modality = uid2mod.get(uid, "UNKNOWN").upper()

        m = FNAME_RE.search(file.name)
        if not m:
            raise ValueError(f"Bad filename pattern: {file.name}")
        variant = m.group("var")
        temp    = f"{m.group('int')}.{m.group('frac')}"

        data   = json.loads(file.read_text())
        model  = data.get("model") or file.parent.name
        rel    = str(file.relative_to(runs))
        payload = extract_payload(data)

        # ---------- TEXT ---------------------------------------------------
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
            model_stat[(model, "TEXT")]["n"] += 1
            if strict: model_stat[(model, "TEXT")]["ok"] += 1
            continue

        # ---------- GUI ----------------------------------------------------
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
            model_stat[(model, "GUI")]["n"] += 1
            if strict: model_stat[(model, "GUI")]["ok"] += 1
            continue

    # ---------- write per-output CSV --------------------------------------
    per_csv = out / "schema_text_gui_per_output.csv"
    with per_csv.open("w", newline='', encoding="utf-8") as fp:
        csv.writer(fp).writerows(
            [["uid","variant","temp","model","modality","file",
              "strict","strict_reason","rescued","rescued_reason",
              "schema_status"],                    # <-- NEW column header
             *per_rows])
    print(f"[schema] wrote {per_csv}")

    # ---------- strict summary --------------------------------------------
    strict_csv = out / "schema_model_summary_strict.csv"
    with strict_csv.open("w", newline='', encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["model","modality","ok","total","pct_ok"])
        for (model, mod), d in sorted(model_stat.items()):
            pct = 100*d["ok"]/d["n"] if d["n"] else 0
            w.writerow([model, mod, d["ok"], d["n"], f"{pct:.1f}"])
    print(f"[schema] wrote {strict_csv}")

    # ---------- rescued summary -------------------------------------------
    rescued_stat = defaultdict(lambda: {"ok": 0, "n": 0})
    for row in per_rows:
        model, modality, rescued = row[3], row[4], row[8]
        rescued_stat[(model, modality)]["n"] += 1
        if rescued == 1.0:
            rescued_stat[(model, modality)]["ok"] += 1

    rescued_csv = out / "schema_model_summary_rescued.csv"
    with rescued_csv.open("w", newline='', encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["model","modality","ok","total","pct_ok"])
        for (model, mod), d in sorted(rescued_stat.items()):
            pct = 100*d["ok"]/d["n"] if d["n"] else 0
            w.writerow([model, mod, d["ok"], d["n"], f"{pct:.1f}"])
    print(f"[schema] wrote {rescued_csv}")

    # ---------- failure logs ----------------------------------------------
    gui_fail = out / "schema_gui_failures.csv"
    with gui_fail.open("w", newline='', encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(
            ["uid","variant","temp","model","modality","file",
             "strict","strict_reason","rescued","rescued_reason","schema_status"])
        for row in per_rows:
            if row[4]=="GUI" and row[10]=="fail":
                w.writerow(row)

    text_fail = out / "schema_text_failures.csv"
    with text_fail.open("w", newline='', encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(
            ["uid","variant","temp","model","modality","file",
             "strict","strict_reason","rescued","rescued_reason","schema_status"])
        for row in per_rows:
            if row[4]=="TEXT" and row[10]=="fail":
                w.writerow(row)
    print(f"[schema] wrote {gui_fail} and {text_fail}")

# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
