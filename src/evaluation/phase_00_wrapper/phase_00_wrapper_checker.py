#!/usr/bin/env python3
"""
phase_00_wrapper_checker.py
Strict + rescued wrapper-tag compliance (TEXT / GUI / VISION).

Outputs
  wrapper_strict.csv
  wrapper_rescued.csv
  bad_outputs_strict.csv
  rescued_non_strict.csv
  per_output_flags.csv
  wrapper_per_output.csv      # unified results (NOW WITH schema_status)
  wrapper_metrics.json        # SHA-256 hashes for reproducibility
"""

import argparse
import csv
import hashlib
import json
import pathlib
import re
import sys
from collections import defaultdict
from typing import Optional, Tuple

# ---------- wrapper tag regexes ---------------------------------------------
WRAP_TAG = {
    "TEXT": "OUTPUT_TEXT",
    "GUI": "OUTPUT_JSON",
    "VISION": "OUTPUT_JSON",
}
RESCUED_REGEX = {
    mod: re.compile(rf"<{tag}>(.*?)</{tag}>", re.S | re.I)
    for mod, tag in WRAP_TAG.items()
}

# ---------- helpers ---------------------------------------------------------
def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_total_tokens(resp: dict) -> int:
    """Vendor-agnostic token counter (OpenAI / Anthropic / Gemini)."""
    if not resp:
        return 0
    # OpenAI / Groq
    if "usage" in resp and "prompt_tokens" in resp["usage"]:
        u = resp["usage"]
        return u.get("prompt_tokens", 0) + u.get("completion_tokens", 0)
    # Anthropic
    if "usage" in resp and "input_tokens" in resp["usage"]:
        u = resp["usage"]
        return u.get("input_tokens", 0) + u.get("output_tokens", 0)
    # Gemini
    if "usageMetadata" in resp:
        meta = resp["usageMetadata"]
        if "totalTokenCount" in meta:
            return meta["totalTokenCount"]
        pt = meta.get("promptTokenCount", 0)
        ct = meta.get("candidatesTokenCount", 0)
        return pt + (sum(ct) if isinstance(ct, list) else ct)
    return 0


def extract_text(data: dict) -> Optional[str]:
    """Return first text payload across vendor formats, else None."""
    resp = data.get("response", {})
    # Anthropic Claude 3
    if "content" in resp and resp["content"]:
        txt = resp["content"][0].get("text")
        if txt:
            return txt
    # Gemini
    if "candidates" in resp and resp["candidates"]:
        txt = resp["candidates"][0].get("content", {}).get("parts", [{}])[0].get("text")
        if txt:
            return txt
    # OpenAI / Groq
    if "choices" in resp and resp["choices"]:
        txt = resp["choices"][0].get("message", {}).get("content")
        if txt:
            return txt
        txt = resp["choices"][0].get("content")
        if txt:
            return txt
    return None


def write_summary_csv(path: pathlib.Path, stats: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "ok", "bad", "%ok", "N", "mean_tok"])
        for model, d in sorted(stats.items()):
            ok, bad, n = d["ok"], d["bad"], d["n"]
            mean_tok = d["tok"] / d["tok_n"] if d["tok_n"] else 0
            pct_ok = 100 * ok / n if n else 0
            w.writerow([model, ok, bad, f"{pct_ok:.1f}", n, f"{mean_tok:.1f}"])


# ---------- main ------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Wrapper-tag compliance checker")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    runs_dir = pathlib.Path(args.runs_dir).resolve()
    if not runs_dir.exists():
        print(f"[wrapper] directory {runs_dir} not found; nothing to grade.")
        sys.exit(0)

    strict = defaultdict(lambda: {"ok": 0, "bad": 0, "n": 0, "tok": 0, "tok_n": 0})
    rescued = defaultdict(lambda: {"ok": 0, "bad": 0, "n": 0, "tok": 0, "tok_n": 0})

    bad_strict, rescued_nonstrict, per_output = [], [], []
    manifest_hashes = {}
    per_phase_rows = []  # unified per-output table

    # -------- iterate over all run files -----------------------------------
    for file in runs_dir.rglob("*.json"):
        try:
            data = json.loads(file.read_text())
        except Exception:
            continue
        if data.get("skipped"):
            continue

        mod = data.get("modality", "").upper()
        if mod not in WRAP_TAG:
            continue
        tag = WRAP_TAG[mod]
        model_ver = data.get("model") or file.parent.name
        rel_path = file.relative_to(runs_dir)

        manifest_hashes[str(rel_path)] = sha256(file)

        raw = extract_text(data) or ""
        # Normalize and trim per Phase-0 tolerant gating
        s = raw.replace("\r\n", "\n").replace("\r", "\n")
        s = s.lstrip("\ufeff")  # BOM
        outer_len_before = len(s)
        # zero-width + whitespace trim
        s_trim = s.strip(" \t\n\u200b\u200c\u200d\ufeff")
        outer_len_after = len(s_trim)

        def extract(tag_open: str, tag_close: str, s0: str) -> Optional[Tuple[str,str,str]]:
            i = s0.find(tag_open)
            j = s0.rfind(tag_close)
            if i == -1 or j == -1 or j < i:
                return None
            inner = s0[i+len(tag_open):j]
            prefix = s0[:i]
            suffix = s0[j+len(tag_close):]
            return inner, prefix, suffix

        # Try both tag pairs; pick the one with the longer inner
        candidates = []
        for tmod, tname in WRAP_TAG.items():
            open_t, close_t = f"<{tname}>", f"</{tname}>"
            res = extract(open_t, close_t, s_trim)
            if res is not None:
                inner, prefix, suffix = res
                candidates.append((len(inner), tname, inner, prefix, suffix))
        if candidates:
            candidates.sort(reverse=True)
            _, tagname_used, inner, prefix, suffix = candidates[0]
            has_block = 1
            # strict: exactly one well-formed block, no extra content after trim
            is_strict = int(prefix == "" and suffix == "")
            is_rescued = int(not is_strict)
            reason = "ok_strict" if is_strict else "rescued_outer_junk"
            # check multiple blocks by counting occurrences
            occ_open = s_trim.count(f"<{tagname_used}>")
            occ_close = s_trim.count(f"</{tagname_used}>")
            if occ_open > 1 or occ_close > 1:
                reason = "multiple_blocks"
        else:
            has_block = 0
            is_strict = 0
            is_rescued = 0
            reason = "no_tags"
        tokens = extract_total_tokens(data.get("response", {}))

        # default stats increment
        for d in (strict, rescued):
            d[model_ver]["n"] += 1

        # meta for traceability
        uid      = data.get("uid", "")
        variant  = data.get("variant", "")
        temp     = data.get("temperature", "")

        # ---------- handle empty answer ------------------------------------
        if not text:
            strict[model_ver]["bad"] += 1
            rescued[model_ver]["bad"] += 1
            per_output.append([rel_path, model_ver, 0, 0])
            # schema_status = fail
            per_phase_rows.append([
                uid, model_ver, variant, temp, mod,
                0, 0, "fail",           # <-- new column value
                str(rel_path)
            ])
            continue

        # wrapper checks (legacy metrics maintained on normalized+trimmed text)
        text = s_trim
        is_strict_legacy  = text.startswith(f"<{tag}>") and text.endswith(f"</{tag}>")
        is_rescued_legacy = bool(RESCUED_REGEX[mod].search(text))

        # strict stats
        if is_strict_legacy:
            strict[model_ver]["ok"] += 1
            if tokens:
                strict[model_ver]["tok"] += tokens
                strict[model_ver]["tok_n"] += 1
        else:
            strict[model_ver]["bad"] += 1
            bad_strict.append(str(rel_path))

        # rescued stats
        if is_rescued_legacy:
            rescued[model_ver]["ok"] += 1
            if tokens:
                rescued[model_ver]["tok"] += tokens
                rescued[model_ver]["tok_n"] += 1
            if not is_strict_legacy:
                rescued_nonstrict.append(str(rel_path))
        else:
            rescued[model_ver]["bad"] += 1

        per_output.append([rel_path, model_ver, int(is_strict_legacy), int(is_rescued_legacy)])

        # ---------- schema_status logic ------------------------------------
        if is_strict_legacy:
            schema_status = "strict"
        elif is_rescued_legacy:
            schema_status = "rescued"
        else:
            schema_status = "fail"

        per_phase_rows.append([
            uid, model_ver, variant, temp, mod,
            int(is_strict_legacy), int(is_rescued_legacy),
            schema_status,                # <-- NEW COLUMN
            str(rel_path)
        ])

        # Phase-0 gating diagnostics CSV row
        # uid, file_id, has_block, is_strict, is_rescued, reason, outer_len_before_trim, outer_len_after_trim
        # collect later; write per file in loop aggregation list
        pass

    # ---------- write summary files ---------------------------------------
    out = pathlib.Path(__file__).parent
    write_summary_csv(out / "wrapper_strict.csv", strict)
    write_summary_csv(out / "wrapper_rescued.csv", rescued)

    (out / "bad_outputs_strict.csv").write_text("\n".join(bad_strict))
    (out / "rescued_non_strict.csv").write_text("\n".join(rescued_nonstrict))
    with (out / "per_output_flags.csv").open("w", newline="", encoding="utf-8") as fp:
        csv.writer(fp).writerows([["file", "model", "strict", "rescued"], *per_output])

    # unified per-output CSV (now includes schema_status)
    with (out / "wrapper_per_output.csv").open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow([
            "uid","model","variant","temperature","modality",
            "strict","rescued","schema_status","file"   # <-- header updated
        ])
        w.writerows(per_phase_rows)

    # Phase-0 gating diagnostics CSV
    diag_rows = []
    for file in runs_dir.rglob("*.json"):
        try:
            data = json.loads(file.read_text())
        except Exception:
            continue
        if data.get("skipped"):
            continue
        mod = data.get("modality", "").upper()
        if mod not in WRAP_TAG:
            continue
        tag = WRAP_TAG[mod]
        raw = extract_text(data) or ""
        s = raw.replace("\r\n", "\n").replace("\r", "\n")
        s = s.lstrip("\ufeff")
        before = len(s)
        s_trim = s.strip(" \t\n\u200b\u200c\u200d\ufeff")
        after = len(s_trim)
        # detect blocks on trimmed
        def extract2(tag_open, tag_close, s0):
            i = s0.find(tag_open); j = s0.rfind(tag_close)
            if i == -1 or j == -1 or j < i: return None
            inner = s0[i+len(tag_open):j]
            prefix = s0[:i]; suffix = s0[j+len(tag_close):]
            return inner, prefix, suffix
        best = None
        for tmod, tname in WRAP_TAG.items():
            res = extract2(f"<{tname}>", f"</{tname}>", s_trim)
            if res is not None:
                inner, prefix, suffix = res
                tup = (len(inner), tname, prefix, suffix)
                if (best is None) or (tup[0] > best[0]):
                    best = tup
        has_block = int(best is not None)
        if best is None:
            strict_f = 0; rescued_f = 0; reason = "no_tags"
        else:
            _, tname, prefix, suffix = best
            strict_f = int(prefix == "" and suffix == "")
            rescued_f = int(not strict_f)
            # multiple blocks heuristic
            open_ct = s_trim.count(f"<{tname}>"); close_ct = s_trim.count(f"</{tname}>")
            reason = "ok_strict" if strict_f else ("multiple_blocks" if (open_ct>1 or close_ct>1) else "rescued_outer_junk")
        diag_rows.append([
            data.get("uid",""), str(file.relative_to(runs_dir)), has_block, strict_f, rescued_f, reason, before, after
        ])
    with (out / "wrapper_phase0_gating.csv").open("w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["uid","file_id","has_block","is_strict","is_rescued","reason","outer_len_before_trim","outer_len_after_trim"])
        w.writerows(diag_rows)

    (out / "wrapper_metrics.json").write_text(
        json.dumps({"wrapper_schema": 1, "prompt_hashes": manifest_hashes}, indent=2)
    )

    if not args.quiet:
        print(f"[wrapper] summaries + manifest saved in {out}")

# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
