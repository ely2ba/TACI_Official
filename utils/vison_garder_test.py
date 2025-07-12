#!/usr/bin/env python3
# phase_01_vision_grader.py  –  Python ≥3.8

import argparse, csv, json, pathlib, re, sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# ---------------- config ----------------
IOU_TAU = 0.50               # IoU threshold (COCO style)
TAG = "OUTPUT_JSON"
RESCUE_RE = re.compile(rf"<{TAG}>(.*?)</{TAG}>", re.S | re.I)
NORMAL = {"normal","healthy","good","no anomaly","no defect","none","negative"}

# -------------- helpers -----------------
def load_json(path: pathlib.Path) -> Optional[dict]:
    try:  return json.loads(path.read_text())
    except Exception:  return None

def is_normal(lbl: str) -> bool:
    return lbl.strip().lower() in NORMAL

def iou(box_a: List[float], box_b: List[float]) -> float:
    if not box_a or not box_b: return 0.0
    xA, yA = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    xB, yB = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0,xB-xA)*max(0,yB-yA)
    areaA = (box_a[2]-box_a[0])*(box_a[3]-box_a[1])+1e-6
    areaB = (box_b[2]-box_b[0])*(box_b[3]-box_b[1])+1e-6
    return inter/(areaA+areaB-inter)

def extract_payload(d: dict) -> str:
    r = d.get("response", {})
    return (
        r.get("content", [{}])[0].get("text") or
        (r.get("candidates", [{}])[0]
           .get("content", {}).get("parts", [{}])[0].get("text")) or
        r.get("choices", [{}])[0].get("message", {}).get("content") or
        r.get("choices", [{}])[0].get("content","")
    )

def inner_json(payload: str, strict: bool) -> Optional[dict]:
    if strict and payload.strip().startswith(f"<{TAG}>") and payload.strip().endswith(f"</{TAG}>"):
        return load_json_text(payload[len(f"<{TAG}>"):-len(f"</{TAG}>")])
    m = RESCUE_RE.search(payload)
    return load_json_text(m.group(1)) if m else None

def load_json_text(txt: str) -> Optional[dict]:
    try:  return json.loads(txt.strip())
    except Exception:  return None

# ---------- scoring ---------------------
def score_pred(pred: dict, gt: dict) -> Tuple[float,str]:
    img_map = gt.get("image_id_map", {})
    abs_imgs = [i for i,t in img_map.items() if t=="abnormal"]
    norm_imgs = [i for i,t in img_map.items() if t=="normal"]
    gt_box  = gt.get("bbox")

    lbl = (pred.get("finding") or "").strip().lower()
    img = pred.get("image_id")
    box = pred.get("bbox") if pred.get("bbox") not in ([0,0,0,0],None) else None

    # all normal
    if is_normal(lbl) and img is None:
        if abs_imgs: return 0.0,"missed abnormal"
        return 1.0,"both images normal"

    # labelled normal but specific image
    if is_normal(lbl):
        if abs_imgs: return 0.0,"missed abnormal"
        if img in norm_imgs and not box: return 1.0,"correct normal image"
        return 0.0,"false bbox on normal"

    # abnormal claim
    if not abs_imgs: return 0.0,"false positive (GT all normal)"
    if img not in abs_imgs: return 0.0,"wrong image id"
    if not box: return 0.0,"missing bbox"
    if not gt_box: return 0.0,"GT bbox missing"
    val = iou(gt_box, box)
    if val < IOU_TAU: return 0.0,f"IoU {val:.2f} < {IOU_TAU}"
    score = (val-IOU_TAU)/(1-IOU_TAU)
    return score, f"IoU {val:.2f}"

# ----------------------------------------
def main()->None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--runs-dir",default="runs"); ap.add_argument("--gts-dir",default="vision_refs")
    ap.add_argument("--quiet",action="store_true")
    args=ap.parse_args()
    runs=pathlib.Path(args.runs_dir).resolve()
    gts =pathlib.Path(args.gts_dir).resolve()
    if not runs.exists(): print("[vision] runs dir missing"); sys.exit(0)

    # cache GT
    gts_dict={p.stem.replace("_GT",""):load_json(p) for p in gts.glob("*_GT.json")}

    per_rows=[]
    model_sum=defaultdict(lambda:{"strict_sum":0,"strict_n":0,"resc_sum":0,"resc_n":0})

    for f in runs.rglob("*.json"):
        uid=f.stem.split("_")[0]; gt=gts_dict.get(uid)
        if not gt: continue
        data=load_json(f); 
        if not data: continue
        payload=extract_payload(data)

        strict_p=inner_json(payload,True)
        res_p  =inner_json(payload,False)

        for mode,pred in [("strict",strict_p),("rescued",res_p)]:
            if pred:
                sc,reason=score_pred(pred,gt)
            else:
                sc,reason=(0.0,"no parsable JSON")
            per_rows.append([
                uid, data.get("variant",""), data.get("temperature",""),
                data.get("model") or f.parent.name,
                str(f.relative_to(runs)), mode, f"{sc:.2f}", reason
            ])
            m=data.get("model") or f.parent.name
            if mode=="strict":
                model_sum[m]["strict_sum"]+=sc; model_sum[m]["strict_n"]+=1
            else:
                model_sum[m]["resc_sum"]+=sc; model_sum[m]["resc_n"]+=1

    out=pathlib.Path(__file__).parent
    # per-file csv
    with (out/"vision_per_output.csv").open("w",newline="",encoding="utf-8") as fp:
        csv.writer(fp).writerows([
            ["uid","variant","temp","model","file","mode","score","reason"],*per_rows])

    # model summary csv
    with (out/"vision_model_summary.csv").open("w",newline="",encoding="utf-8") as fp:
        w=csv.writer(fp); w.writerow(["model","mean_strict","mean_rescued","n_strict","n_rescued"])
        for m,d in sorted(model_sum.items()):
            ms   = d["strict_sum"]/d["strict_n"] if d["strict_n"] else 0
            mr   = d["resc_sum"]/d["resc_n"] if d["resc_n"] else 0
            w.writerow([m,f"{ms:.2f}",f"{mr:.2f}",d["strict_n"],d["resc_n"]])

    if not args.quiet:
        print(f"[vision] CSVs saved to {out}")

if __name__=="__main__":
    main()
