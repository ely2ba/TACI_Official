#!/usr/bin/env python3
"""
Phase-04 Composite Scorer · Paralegal slice  (v3.6.1)

* Fix: metric-filter now skips entries whose value is None before unpacking.
"""
from pathlib import Path
import json, numpy as np, pandas as pd

# ───────── explicit repo paths ────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
HERE          = Path(__file__).parent

# Input data from legacy graders location
MASTER_CSV    = ROOT / "outputs/results/evaluated" / "master_per_output_paralegal.csv"
PRICE_CSV     = ROOT / "outputs/results/evaluated"       / "model_price_ctx.csv"
MANIFEST_CSV  = ROOT / "data/manifests/paralegal_tasks.csv"
# Config file now local to evaluation
WEIGHT_JS     = HERE / "weights.json"

# Output to current evaluation directory
OUT_DIR       = HERE
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PER       = OUT_DIR / "phase_04_composite_per_output_paralegal.csv"
OUT_MODEL     = OUT_DIR / "composite_model_summary_paralegal.csv"
OUT_DIGITAL   = OUT_DIR / "digital_automation_summary_paralegal.csv"
OUT_OCC       = OUT_DIR / "occupation_summary_paralegal.csv"

# ───────── constants / helpers ────────────────────────────────────────
PRICE_CEIL, CTX_REQ = 0.002, 128_000
SAFE_THR,  MILD_THR = 0.30, 0.80

def ci_boot(arr, reps=1_000, seed=42):
    if arr.size == 0: return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, (reps, arr.size), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])

def ci_boot_w(arr, w, reps=1_000, seed=42):
    if arr.size == 0 or w.sum() == 0:
        return (np.nan, np.nan)
    p = w / w.sum()
    rng = np.random.default_rng(seed)
    idx = np.arange(arr.size)
    means = [np.average(arr[rng.choice(idx, arr.size, True, p)], weights=w)
             for _ in range(reps)]
    return np.percentile(means, [2.5, 97.5])

# ───────── load & merge ───────────────────────────────────────────────
df = pd.read_csv(MASTER_CSV, dtype=str)
price = pd.read_csv(PRICE_CSV, dtype=str)\
           .rename(columns={"price_per_k_usd":"price_per_k"})
df = df.merge(price, on="model", how="left")

imp = pd.read_csv(MANIFEST_CSV, usecols=["uid","Importance"], dtype=str)
imp["Importance"] = pd.to_numeric(imp["Importance"], errors="coerce")
df = df.merge(imp, on="uid", how="left")
df["w_task"] = (df["Importance"].astype(float)
                  .fillna(df["Importance"].astype(float).mean())
                  .clip(lower=1) / 5.0)

# numeric casting (skip missing) --------------------------------------
for col in ["schema_strict","schema_rescued",
            "rubric_score","iou",
            "price_per_k","ctx_tokens",
            "risk_strict","risk_rescued",
            "w_task"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ───────── primitives ────────────────────────────────────────────────
df["norm_schema"] = np.select(
    [df.schema_strict == 1, df.schema_rescued == 1], [1.0, 0.5], 0.0)
df["norm_rubric"] = df.rubric_score.fillna(0) / 5
df["norm_iou"]    = df.iou.fillna(0)

def norm_price(row):
    if row.modality.upper() not in {"TEXT","GUI"}: return np.nan
    if pd.isna(row.price_per_k) or pd.isna(row.ctx_tokens): return 0.0
    if row.ctx_tokens < CTX_REQ:                            return 0.0
    return float(np.clip((PRICE_CEIL - row.price_per_k) / PRICE_CEIL, 0, 1))
df["norm_price"] = df.apply(norm_price, axis=1)

def cap_safety(row):
    if "risk_strict" in row:
        risk = row.risk_strict if row.schema_strict==1 else \
               row.risk_rescued if row.schema_rescued==1 else np.nan
        if not pd.isna(risk):
            return 1.0 if risk < SAFE_THR else 0.75 if risk < MILD_THR else 0.0
    return 1.0
df["norm_safety"] = df.apply(cap_safety, axis=1)

# ───────── composite score ───────────────────────────────────────────
weights = json.loads(Path(WEIGHT_JS).read_text())
if isinstance(weights, list):
    weights = dict(zip(["rubric","schema","iou","price_ctx"], weights))

def composite(row):
    vis   = "VISION" in row.modality.upper()
    price = not pd.isna(row.norm_price)
    metrics = {
        "rubric":    (weights["rubric"],    row.norm_rubric),
        "schema":    (weights["schema"],    row.norm_schema),
        "iou":       (weights["iou"],       row.norm_iou)    if vis   else None,
        "price_ctx": (weights["price_ctx"], row.norm_price)  if price else None
    }
    active = {}
    for k, val in metrics.items():
        if val is None: continue
        w, m = val
        if m is not None and not np.isnan(m):
            active[k] = (w, m)
    if not active: return np.nan
    W = sum(w for w,_ in active.values())
    raw = sum(w*m for w,m in active.values()) / W
    return raw * row.norm_safety

df["composite_score"] = df.apply(composite, axis=1) * 100

# ───────── per-output table ───────────────────────────────────────────
df.to_csv(OUT_PER, index=False)
print(f"✓ per-output composite → {OUT_PER}")

# model summary --------------------------------------------------------
grp = df.groupby("model").composite_score
mod = grp.mean().to_frame("mean_score").reset_index()
mod["N"] = grp.size().values
mod[["ci_low","ci_high"]] = grp.apply(lambda x: ci_boot(x.dropna().values))\
                               .apply(pd.Series)
mod.to_csv(OUT_MODEL, index=False)
print(f"✓ model summary → {OUT_MODEL}")

# digital summary ------------------------------------------------------
dig = df[df.modality != "MANUAL"]
if len(dig):
    grp = dig.groupby("model").composite_score
    dig_sum = grp.mean().to_frame("mean_score").reset_index()
    dig_sum["N"] = grp.size().values
    dig_sum[["ci_low","ci_high"]] = grp.apply(
        lambda x: ci_boot(x.dropna().values)).apply(pd.Series)
    dig_sum.to_csv(OUT_DIGITAL, index=False)
    print(f"✓ digital summary → {OUT_DIGITAL}")

# occupation summary (weighted) ---------------------------------------
rows = []
for (occ, model), sub in df.groupby(["occupation","model"]):
    w = sub.w_task.values
    x = sub.composite_score.values
    rows.append([
        occ, model,
        np.average(x, weights=w),
        len(sub),
        *ci_boot_w(x, w)
    ])

pd.DataFrame(rows, columns=[
    "occupation","model","mean_score","N","ci_low","ci_high"
]).to_csv(OUT_OCC, index=False)
print(f"✓ occupation summary → {OUT_OCC}")
