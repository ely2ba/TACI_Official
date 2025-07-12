#!/usr/bin/env python3
"""
Phase-01 Vision Schema + IoU Grader   ·   Python ≥ 3.8

✓ Validates JSON schema (strict / rescued)
✓ Computes IoU-based score
✓ Outputs columns compatible with TEXT / GUI graders
     ─ uid, variant, temp, model, modality, file,
       strict, strict_reason, rescued, rescued_reason,
       schema_status, iou, iou_reason
"""

from __future__ import annotations
import argparse, csv, json, pathlib, re, sys
from collections import defaultdict
from typing import List, Optional, Tuple

# ─────────── configuration ───────────────────────────────────────────────
ALLOWED_MODELS = {
    "gpt-4o", "gpt-4-1",
    "gemini-2.5-flash-preview-05-20", "gemini-2.0-flash"
}

TAG        = "OUTPUT_JSON"
RESCUE_RE  = re.compile(rf"<{TAG}>(.*?)</{TAG}>", re.S | re.I)
NORMAL     = {"normal","healthy","good","no anomaly","no defect","none","negative"}
IOU_TAU    = 0.50
REQ_KEYS   = {"finding","image_id","bbox","explanation"}

# ─────────── helpers ─────────────────────────────────────────────────────
def load_json(fp: pathlib.Path) -> Optional[dict]:
    try: return json.loads(fp.read_text())
    except Exception: return None

def extract_payload(d: dict) -> str:
    r = d.get("response", {})
    return (
        r.get("content", [{}])[0].get("text") or
        (r.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text")) or
        r.get("choices", [{}])[0].get("message", {}).get("content") or
        r.get("choices", [{}])[0].get("content", "")
    )

def inner_json(payload: str, strict: bool) -> Optional[str]:
    if strict and payload.strip().startswith(f"<{TAG}>") and payload.strip().endswith(f"</{TAG}>"):
        return payload[len(f"<{TAG}>"):-len(f"</{TAG}>")].strip()
    m = RESCUE_RE.search(payload)
    return m.group(1).strip() if m else None

safe_load = lambda txt: json.loads(txt) if txt else None

def schema_valid(obj: dict) -> Tuple[bool,str]:
    if not isinstance(obj, dict):
        return False, "not a JSON object"
    if set(obj.keys()) != REQ_KEYS:
        return False, "missing/extra keys"
    if not isinstance(obj["finding"], str) or not obj["finding"].strip():
        return False, "empty finding"
    if obj["image_id"] not in ("img_000", "img_001", None):
        return False, "invalid image_id"
    bbox = obj["bbox"]
    if bbox is not None:
        if (not isinstance(bbox, list) or len(bbox) != 4 or
            not all(isinstance(x, (int, float)) for x in bbox) or
            bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
            return False, "bad bbox"
    if not isinstance(obj["explanation"], str) or len(obj["explanation"].split()) > 15:
        return False, "explanation >15 words"
    return True, "ok"

is_normal = lambda lbl: lbl.strip().lower() in NORMAL

def iou(box_a: List[float], box_b: List[float]) -> float:
    if not box_a or not box_b: return 0.0
    xA, yA = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xB, yB = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter  = max(0, xB - xA) * max(0, yB - yA)
    areaA  = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]) + 1e-6
    areaB  = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]) + 1e-6
    return inter / (areaA + areaB - inter)

def score_pred(pred: dict, gt: dict) -> Tuple[float, str]:
    img_map  = gt.get("image_id_map", {})
    abs_imgs = [i for i, t in img_map.items() if t == "abnormal"]
    norm_imgs= [i for i, t in img_map.items() if t == "normal"]
    gt_box   = gt.get("bbox")

    lbl = (pred.get("finding") or "").strip().lower()
    img = pred.get("image_id")
    box = pred.get("bbox") if pred.get("bbox") not in ([0,0,0,0], None) else None

    # normal cases
    if is_normal(lbl) and img is None:
        return (0.0, "missed abnormal") if abs_imgs else (1.0, "both images normal")
    if is_normal(lbl):
        if abs_imgs: return 0.0, "missed abnormal"
        if img in norm_imgs and not box: return 1.0, "correct normal"
        return 0.0, "false bbox on normal"

    # abnormal cases
    if not abs_imgs:         return 0.0, "false positive (GT all normal)"
    if img not in abs_imgs:  return 0.0, "wrong image_id"
    if not box:              return 0.0, "missing bbox"
    if not gt_box:           return 0.0, "GT bbox missing"

    val = iou(gt_box, box)
    if val < IOU_TAU:
        return 0.0, f"IoU {val:.2f} < {IOU_TAU}"
    return (val - IOU_TAU) / (1 - IOU_TAU), f"IoU {val:.2f}"

# ─────────── main ────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--gts-dir",  default="vision_refs")
    args = ap.parse_args()

    runs = pathlib.Path(args.runs_dir).resolve()
    gts  = pathlib.Path(args.gts_dir).resolve()
    if not runs.exists(): sys.exit("[vision] runs dir missing")

    gt_cache = {p.stem.replace("_GT", ""): load_json(p) for p in gts.glob("*_GT.json")}

    per_rows: list[list] = []
    strict_stat = defaultdict(lambda: {"ok": 0, "n": 0})
    resc_stat   = defaultdict(lambda: {"ok": 0, "n": 0})

    for fp in runs.rglob("*.json"):
        data = load_json(fp)
        if not data: continue
        model = data.get("model") or fp.parent.name
        if model not in ALLOWED_MODELS: continue

        uid = fp.stem.split("_")[0]
        gt  = gt_cache.get(uid)
        if not gt: continue

        payload  = extract_payload(data)
        strict_s = inner_json(payload, True)
        resc_s   = inner_json(payload, False)

        # -------- schema validation --------------------------------------
        strict = rescued = 0
        strict_r = "no JSON"; resc_r = "no JSON"
        pred_for_iou = None

        if strict_s:
            obj = safe_load(strict_s)
            ok, strict_r = schema_valid(obj)
            if ok:
                strict = 1
                pred_for_iou = obj

        if resc_s:
            obj = safe_load(resc_s)
            ok, resc_r = schema_valid(obj)
            if ok:
                rescued = 1
                if pred_for_iou is None:
                    pred_for_iou = obj

        schema_status = ("strict" if strict else
                         "rescued" if rescued else "fail")

        # -------- IoU score ----------------------------------------------
        if pred_for_iou is not None:
            iou_val, iou_reason = score_pred(pred_for_iou, gt)
        else:
            iou_val, iou_reason = 0.0, "schema fail"

        per_rows.append([
            uid, data.get("variant", ""), data.get("temperature", ""),
            model, "VISION",                       # modality column
            str(fp.relative_to(runs)),
            strict, strict_r,
            rescued, resc_r,
            schema_status,                        # unified label
            f"{iou_val:.4f}", iou_reason
        ])

        strict_stat[model]["n"] += 1
        resc_stat[model]["n"]   += 1
        if strict:  strict_stat[model]["ok"] += 1
        if rescued: resc_stat[model]["ok"]  += 1

    # ─── write CSVs ──────────────────────────────────────────────────────
    out = pathlib.Path(__file__).parent
    hdr = ["uid","variant","temp","model","modality","file",
           "strict","strict_reason","rescued","rescued_reason",
           "schema_status","iou","iou_reason"]

    with (out / "vision_per_output.csv").open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([hdr, *per_rows])

    # strict summary
    with (out / "vision_model_summary_strict.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["model","ok","total","pct_ok"])
        for m, d in sorted(strict_stat.items()):
            pct = 100 * d["ok"] / d["n"] if d["n"] else 0
            w.writerow([m, d["ok"], d["n"], f"{pct:.1f}"])

    # rescued summary
    with (out / "vision_model_summary_rescued.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["model","ok","total","pct_ok"])
        for m, d in sorted(resc_stat.items()):
            pct = 100 * d["ok"] / d["n"] if d["n"] else 0
            w.writerow([m, d["ok"], d["n"], f"{pct:.1f}"])

    print(f"[vision] CSVs saved to {out}")

# ─────────── entry point ────────────────────────────────────────────────
if __name__ == "__main__":
    main()
