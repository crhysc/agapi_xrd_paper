#!/usr/bin/env python3
"""
Compile reviewer-facing replication metrics for AGAPI-XRD runs.

This script reads a text file whose non-empty, non-comment lines point to
subdirectories of a runs/ directory (for example: bmgn, gsas2, no_refinement).
For each listed run directory, it loads the saved outputs from the RRUFF AGAPI-XRD
pipeline and recomputes manuscript-style metrics into a single JSON summary.

Primary use:
    python compile_agapi_replication_metrics.py \
        --run-list runs_to_compile.txt \
        --runs-root runs \
        --output replication_summary.json

Accepted line formats inside --run-list:
    bmgn
    no_refinement
    runs/gsas2
    /absolute/path/to/runs/bmgn

The script prefers a CSV summary inside each run directory and falls back to
all_results_live.json when needed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

PARAMS_ALL = ["a", "b", "c", "alpha", "beta", "gamma", "volume"]
PARAMS_LENGTHS = ["a", "b", "c"]
PARAMS_METHOD = ["a", "b", "c", "volume"]
CRYSTAL_SYSTEM_ORDER = [
    "cubic",
    "orthorhombic",
    "monoclinic",
    "hexagonal",
    "tetragonal",
    "triclinic",
    "rhombohedral",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compile AGAPI-XRD replication metrics from one or more run directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--run-list", required=True, help="Text file listing run subdirectories to compile.")
    p.add_argument("--runs-root", default="runs", help="Root directory containing run subdirectories.")
    p.add_argument("--output", default="replication_summary.json", help="Output JSON path.")
    p.add_argument(
        "--jsd-bins",
        type=int,
        default=30,
        help="Number of histogram bins for Jensen-Shannon divergence calculations.",
    )
    p.add_argument(
        "--laplace-alpha",
        type=float,
        default=1e-8,
        help="Laplace smoothing constant for histogram-based JSD calculations.",
    )
    p.add_argument(
        "--relative-error-mode",
        choices=["entrywise_mean", "pooled_lengths"],
        default="entrywise_mean",
        help=(
            "How to compute the <10%% and <30%% relative-error fractions. "
            "'entrywise_mean' uses the mean relative error across a,b,c per entry; "
            "'pooled_lengths' pools all individual a,b,c relative errors together."
        ),
    )
    p.add_argument(
        "--raw-rruff-count",
        type=int,
        default=None,
        help="Optional study-context count for raw RRUFF entries before filtering.",
    )
    p.add_argument(
        "--unique-valid-chemistry-count",
        type=int,
        default=None,
        help="Optional study-context count for unique minerals with valid chemistry.",
    )
    return p.parse_args()


def read_run_list(path: Path) -> List[str]:
    lines: List[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    if not lines:
        raise ValueError(f"No usable run paths found in {path}")
    return lines


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out):
        return None
    return out


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        x = float(value)
        return None if math.isnan(x) else x
    if isinstance(value, Path):
        return str(value)
    return value


def normalize_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    lowered = series.astype(str).str.strip().str.lower()
    return lowered.isin({"true", "1", "yes", "y", "t"})


def normalize_best_method(value: Any, query_used: Any = None) -> Optional[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        if isinstance(query_used, str) and query_used.startswith("elements:"):
            return "pattern_matching_elements"
        return None
    s = str(value).strip().lower()
    if not s or s == "nan":
        return None
    if "diffract" in s:
        return "diffractgpt"
    if "pattern" in s and "element" in s:
        return "pattern_matching_elements"
    if "pattern" in s:
        if isinstance(query_used, str) and query_used.startswith("elements:"):
            return "pattern_matching_elements"
        return "pattern_matching"
    if isinstance(query_used, str) and query_used.startswith("elements:"):
        return "pattern_matching_elements"
    return s.replace(" ", "_")


CS_MAP = {
    "trigonal": "rhombohedral",
    "rhombohedral": "rhombohedral",
    "hexagonal": "hexagonal",
    "cubic": "cubic",
    "monoclinic": "monoclinic",
    "orthorhombic": "orthorhombic",
    "tetragonal": "tetragonal",
    "triclinic": "triclinic",
}


def normalize_crystal_system(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    s = str(value).strip().lower()
    if not s or s == "nan":
        return None
    s = re.sub(r"[^a-z]", "", s)
    return CS_MAP.get(s, s if s else None)


REQUIRED_NUMERIC_COLUMNS = [
    "rruff_a",
    "rruff_b",
    "rruff_c",
    "rruff_alpha",
    "rruff_beta",
    "rruff_gamma",
    "rruff_volume",
    "pred_a",
    "pred_b",
    "pred_c",
    "pred_alpha",
    "pred_beta",
    "pred_gamma",
    "pred_volume",
    "pm_similarity",
    "dg_similarity",
    "best_similarity",
    "refinement_rwp",
    "time_s",
]


ESSENTIAL_COLUMNS = [
    "mineral_name",
    "rruff_id",
    "formula",
    "elements",
    "query_used",
    "pm_success",
    "dg_success",
    "best_method",
    "best_similarity",
    "rruff_crystal_system",
    "error",
]


def flatten_live_json(entries: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for entry in entries:
        row: Dict[str, Any] = {
            "mineral_name": entry.get("mineral_name"),
            "rruff_id": entry.get("rruff_id"),
            "formula": entry.get("formula"),
            "elements": entry.get("elements"),
            "query_used": entry.get("query_used"),
            "num_points": entry.get("num_points"),
            "time_s": entry.get("time_s"),
            "error": entry.get("error"),
        }
        rc = entry.get("rruff_cell") or {}
        for k in ["a", "b", "c", "alpha", "beta", "gamma", "volume"]:
            row[f"rruff_{k}"] = rc.get(k)
        row["rruff_crystal_system"] = rc.get("crystal_system")

        pm = entry.get("pattern_matching") or {}
        row["pm_success"] = pm.get("success")
        row["pm_jid"] = pm.get("jid")
        row["pm_similarity"] = pm.get("similarity")
        row["pm_formula"] = pm.get("formula")
        row["pm_spacegroup"] = pm.get("spacegroup")
        row["pm_num_atoms"] = pm.get("num_atoms")

        dg = entry.get("diffractgpt") or {}
        row["dg_success"] = dg.get("success")
        row["dg_similarity"] = dg.get("similarity")
        row["dg_formula"] = dg.get("formula")
        row["dg_spacegroup"] = dg.get("spacegroup")
        row["dg_num_atoms"] = dg.get("num_atoms")

        bm = entry.get("best_match") or {}
        row["best_method"] = bm.get("method")
        row["best_similarity"] = bm.get("similarity")

        pc = entry.get("predicted_cell") or {}
        for k in ["a", "b", "c", "alpha", "beta", "gamma", "volume"]:
            row[f"pred_{k}"] = pc.get(k)
        row["cell_type"] = pc.get("cell_type")

        ref = entry.get("refinement") or {}
        row["refinement_rwp"] = ref.get("rwp")
        row["refinement_engine"] = ref.get("engine")
        rows.append(row)
    return pd.DataFrame(rows)


def load_run_dataframe(run_dir: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    provenance: Dict[str, str] = {}

    # Prefer a summary CSV in the run directory itself.
    root_csvs = sorted(run_dir.glob("*_results.csv")) + sorted(run_dir.glob("results.csv"))
    if root_csvs:
        df = pd.read_csv(root_csvs[0])
        provenance["primary_file"] = str(root_csvs[0])
        provenance["format"] = "csv"
        return finalize_dataframe(df, run_dir), provenance

    nested_csvs = sorted(run_dir.glob("rruff_results_*/results.csv"))
    if nested_csvs:
        df = pd.read_csv(nested_csvs[-1])
        provenance["primary_file"] = str(nested_csvs[-1])
        provenance["format"] = "csv"
        return finalize_dataframe(df, run_dir), provenance

    live_json = run_dir / "all_results_live.json"
    if live_json.exists():
        with open(live_json) as f:
            payload = json.load(f)
        df = flatten_live_json(payload)
        provenance["primary_file"] = str(live_json)
        provenance["format"] = "all_results_live.json"
        return finalize_dataframe(df, run_dir), provenance

    nested_jsons = sorted(run_dir.glob("rruff_results_*/all_results.json"))
    if nested_jsons:
        with open(nested_jsons[-1]) as f:
            payload = json.load(f)
        df = flatten_live_json(payload)
        provenance["primary_file"] = str(nested_jsons[-1])
        provenance["format"] = "all_results.json"
        return finalize_dataframe(df, run_dir), provenance

    raise FileNotFoundError(
        f"Could not find a results CSV or live/all_results JSON inside {run_dir}"
    )


def finalize_dataframe(df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    df = df.copy()
    for col in ESSENTIAL_COLUMNS:
        if col not in df.columns:
            df[col] = None
    for col in REQUIRED_NUMERIC_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["pm_success", "dg_success"]:
        if col not in df.columns:
            df[col] = False
        df[col] = normalize_bool_series(df[col])

    if "query_used" not in df.columns:
        df["query_used"] = None

    df["best_method_normalized"] = [
        normalize_best_method(v, q) for v, q in zip(df["best_method"], df["query_used"])
    ]
    df["rruff_crystal_system_normalized"] = df["rruff_crystal_system"].map(normalize_crystal_system)

    if "rruff_id" not in df.columns:
        df["rruff_id"] = None
    if "mineral_name" not in df.columns:
        df["mineral_name"] = None

    df["entry_key"] = df.apply(make_entry_key, axis=1)
    df["run_name"] = run_dir.name
    return df


def make_entry_key(row: pd.Series) -> str:
    rid = row.get("rruff_id")
    if rid is not None and str(rid).strip() and str(rid).strip().lower() != "nan":
        return f"rruff:{str(rid).strip()}"
    name = row.get("mineral_name")
    formula = row.get("formula")
    return f"name:{str(name).strip()}|formula:{str(formula).strip()}"


def complete_case_mask(df: pd.DataFrame, params: Iterable[str]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for p in params:
        mask &= df[f"rruff_{p}"].notna() & df[f"pred_{p}"].notna()
    return mask


def mae(exp: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - exp)))


def rmse(exp: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - exp) ** 2)))


def r2(exp: np.ndarray, pred: np.ndarray) -> Optional[float]:
    denom = float(np.sum((exp - np.mean(exp)) ** 2))
    if denom <= 1e-14:
        return None
    return float(1.0 - np.sum((pred - exp) ** 2) / denom)


def mad_exp(exp: np.ndarray) -> float:
    mu = float(np.mean(exp))
    return float(np.mean(np.abs(exp - mu)))


def median_abs_error(exp: np.ndarray, pred: np.ndarray) -> float:
    return float(np.median(np.abs(pred - exp)))


def js_divergence(exp: np.ndarray, pred: np.ndarray, bins: int, alpha: float) -> Optional[float]:
    if len(exp) == 0 or len(pred) == 0:
        return None
    lo = float(min(np.min(exp), np.min(pred)))
    hi = float(max(np.max(exp), np.max(pred)))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    if abs(hi - lo) < 1e-14:
        return 0.0
    hist_exp, edges = np.histogram(exp, bins=bins, range=(lo, hi))
    hist_pred, _ = np.histogram(pred, bins=edges)
    p = (hist_exp.astype(float) + alpha)
    q = (hist_pred.astype(float) + alpha)
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    return float(0.5 * (kl_pm + kl_qm))


def metric_block(exp: np.ndarray, pred: np.ndarray, bins: int, alpha: float) -> Dict[str, Any]:
    _mae = mae(exp, pred)
    _mad = mad_exp(exp)
    return {
        "n": int(len(exp)),
        "mae": _mae,
        "rmse": rmse(exp, pred),
        "r2": r2(exp, pred),
        "median_abs_error": median_abs_error(exp, pred),
        "mad_exp": _mad,
        "skill": (1.0 - (_mae / _mad)) if _mad > 1e-14 else None,
        "jsd": js_divergence(exp, pred, bins=bins, alpha=alpha),
    }


def simple_metric_block(exp: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    _mae = mae(exp, pred)
    _mad = mad_exp(exp)
    return {
        "n": int(len(exp)),
        "mae": _mae,
        "skill": (1.0 - (_mae / _mad)) if _mad > 1e-14 else None,
    }


def summarize_overall_lattice(df: pd.DataFrame, bins: int, alpha: float) -> Dict[str, Any]:
    mask = complete_case_mask(df, PARAMS_ALL)
    sub = df.loc[mask].copy()
    out: Dict[str, Any] = {
        "n": int(len(sub)),
        "parameter_order": list(PARAMS_ALL),
    }
    for p in PARAMS_ALL:
        if len(sub) == 0:
            out[p] = empty_metric_block(full=True)
            continue
        exp = sub[f"rruff_{p}"].to_numpy(dtype=float)
        pred = sub[f"pred_{p}"].to_numpy(dtype=float)
        out[p] = metric_block(exp, pred, bins=bins, alpha=alpha)
    return out


def empty_metric_block(full: bool = False) -> Dict[str, Any]:
    base: Dict[str, Any] = {"n": 0, "mae": None, "skill": None}
    if full:
        base.update({
            "rmse": None,
            "r2": None,
            "median_abs_error": None,
            "mad_exp": None,
            "jsd": None,
        })
    return base


def summarize_method_breakdown(df: pd.DataFrame) -> Dict[str, Any]:
    mask = complete_case_mask(df, PARAMS_METHOD)
    sub = df.loc[mask].copy()
    out: Dict[str, Any] = {}
    labels = ["pattern_matching", "diffractgpt", "pattern_matching_elements"]
    for label in labels:
        group = sub.loc[sub["best_method_normalized"] == label].copy()
        bucket: Dict[str, Any] = {"n": int(len(group))}
        for p in PARAMS_METHOD:
            if len(group) == 0:
                bucket[p] = empty_metric_block(full=False)
                continue
            exp = group[f"rruff_{p}"].to_numpy(dtype=float)
            pred = group[f"pred_{p}"].to_numpy(dtype=float)
            bucket[p] = simple_metric_block(exp, pred)
        out[label] = bucket
    return out


def summarize_crystal_system_breakdown(df: pd.DataFrame) -> Dict[str, Any]:
    mask = complete_case_mask(df, ["a", "volume"])
    sub = df.loc[mask].copy()
    out: Dict[str, Any] = {}
    for cs in CRYSTAL_SYSTEM_ORDER:
        group = sub.loc[sub["rruff_crystal_system_normalized"] == cs].copy()
        if len(group) == 0:
            out[cs] = {
                "n": 0,
                "a": empty_metric_block(full=False),
                "volume": empty_metric_block(full=False),
            }
            continue
        out[cs] = {
            "n": int(len(group)),
            "a": simple_metric_block(
                group["rruff_a"].to_numpy(dtype=float),
                group["pred_a"].to_numpy(dtype=float),
            ),
            "volume": simple_metric_block(
                group["rruff_volume"].to_numpy(dtype=float),
                group["pred_volume"].to_numpy(dtype=float),
            ),
        }
    return out


def summarize_relative_error_thresholds(df: pd.DataFrame, mode: str) -> Dict[str, Any]:
    mask = complete_case_mask(df, PARAMS_LENGTHS)
    sub = df.loc[mask].copy()
    if len(sub) == 0:
        return {
            "mode": mode,
            "n_entries": 0,
            "fraction_predictions_within_10pct": None,
            "fraction_predictions_within_30pct": None,
        }

    exp = sub[[f"rruff_{p}" for p in PARAMS_LENGTHS]].to_numpy(dtype=float)
    pred = sub[[f"pred_{p}" for p in PARAMS_LENGTHS]].to_numpy(dtype=float)
    rel = np.abs(pred - exp) / np.maximum(np.abs(exp), 1e-12)

    if mode == "pooled_lengths":
        vals = rel.reshape(-1)
        n = int(vals.size)
    else:
        vals = rel.mean(axis=1)
        n = int(len(vals))

    return {
        "mode": mode,
        "n_entries": int(len(sub)),
        "n_values": n,
        "fraction_predictions_within_10pct": float(np.mean(vals <= 0.10)),
        "fraction_predictions_within_30pct": float(np.mean(vals <= 0.30)),
    }


def summarize_coverage(df: pd.DataFrame) -> Dict[str, Any]:
    total = int(len(df))
    pm_success = df["pm_success"].fillna(False)
    dg_success = df["dg_success"].fillna(False)
    element_pm = pm_success & df["query_used"].fillna("").astype(str).str.startswith("elements:")
    formula_pm = pm_success & ~element_pm
    any_match = pm_success | dg_success

    best_counts = Counter(x for x in df["best_method_normalized"].tolist() if x)

    def count_pct(mask: pd.Series) -> Dict[str, Any]:
        c = int(mask.sum())
        return {"count": c, "pct": (100.0 * c / total) if total else None}

    return {
        "total_entries": total,
        "pattern_matching_formula": count_pct(formula_pm),
        "pattern_matching_elements": count_pct(element_pm),
        "pattern_matching_total": count_pct(pm_success),
        "diffractgpt": count_pct(dg_success),
        "combined_any_method": count_pct(any_match),
        "unmatched": count_pct(~any_match),
        "best_method_counts": {
            "pattern_matching": int(best_counts.get("pattern_matching", 0)),
            "diffractgpt": int(best_counts.get("diffractgpt", 0)),
            "pattern_matching_elements": int(best_counts.get("pattern_matching_elements", 0)),
        },
    }


def summarize_runtime_and_failures(df: pd.DataFrame) -> Dict[str, Any]:
    times = pd.to_numeric(df["time_s"], errors="coerce").dropna()
    errors = df["error"].dropna().astype(str)
    error_types: Counter[str] = Counter()
    for e in errors:
        key = e.split(":", 1)[0].strip() if ":" in e else e.strip()
        error_types[key] += 1
    return {
        "num_errors": int(errors.shape[0]),
        "error_type_counts": dict(error_types),
        "mean_time_s": float(times.mean()) if len(times) else None,
        "median_time_s": float(times.median()) if len(times) else None,
        "min_time_s": float(times.min()) if len(times) else None,
        "max_time_s": float(times.max()) if len(times) else None,
    }


def summarize_refinement_quality(df: pd.DataFrame) -> Dict[str, Any]:
    rwp = pd.to_numeric(df["refinement_rwp"], errors="coerce").dropna()
    engines = Counter(df["refinement_engine"].dropna().astype(str)) if "refinement_engine" in df.columns else Counter()
    return {
        "n_refined_success": int(len(rwp)),
        "mean_rwp": float(rwp.mean()) if len(rwp) else None,
        "median_rwp": float(rwp.median()) if len(rwp) else None,
        "std_rwp": float(rwp.std(ddof=0)) if len(rwp) else None,
        "engine_counts": dict(engines),
    }


def summarize_similarity(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for prefix, success_col, sim_col in [
        ("pattern_matching", "pm_success", "pm_similarity"),
        ("diffractgpt", "dg_success", "dg_similarity"),
    ]:
        ok = df.loc[df[success_col].fillna(False), sim_col].dropna().astype(float)
        out[prefix] = {
            "n": int(len(ok)),
            "mean": float(ok.mean()) if len(ok) else None,
            "median": float(ok.median()) if len(ok) else None,
            "min": float(ok.min()) if len(ok) else None,
            "max": float(ok.max()) if len(ok) else None,
        }
    return out


def summarize_element_errors(df: pd.DataFrame) -> Dict[str, Any]:
    mask = complete_case_mask(df, PARAMS_LENGTHS)
    sub = df.loc[mask].copy()
    if len(sub) == 0:
        return {"num_elements": 0, "elements": {}}

    exp = sub[[f"rruff_{p}" for p in PARAMS_LENGTHS]].to_numpy(dtype=float)
    pred = sub[[f"pred_{p}" for p in PARAMS_LENGTHS]].to_numpy(dtype=float)
    rel = np.abs(pred - exp) / np.maximum(np.abs(exp), 1e-12)
    sub["mean_relative_length_error"] = rel.mean(axis=1)

    buckets: Dict[str, List[float]] = {}
    for _, row in sub.iterrows():
        raw = row.get("elements")
        if raw is None or (isinstance(raw, float) and math.isnan(raw)):
            continue
        elems = [x.strip() for x in str(raw).split(",") if x.strip()]
        for elem in elems:
            buckets.setdefault(elem, []).append(float(row["mean_relative_length_error"]))

    elements = {
        elem: {
            "n_entries": len(vals),
            "mean_relative_length_error": float(np.mean(vals)),
            "median_relative_length_error": float(np.median(vals)),
        }
        for elem, vals in sorted(buckets.items())
    }
    return {"num_elements": len(elements), "elements": elements}


def recoverable_benchmark_summary(
    df: pd.DataFrame,
    raw_rruff_count: Optional[int],
    unique_valid_chemistry_count: Optional[int],
) -> Dict[str, Any]:
    count_abc_present = int(df[["rruff_a", "rruff_b", "rruff_c"]].notna().all(axis=1).sum())
    count_complete_lattice = int(complete_case_mask(df, PARAMS_ALL).sum())
    crystal_counts = Counter(
        x for x in df["rruff_crystal_system_normalized"].tolist() if x is not None
    )
    return {
        "raw_rruff_entries": raw_rruff_count,
        "unique_valid_chemistry": unique_valid_chemistry_count,
        "total_entries_processed": int(len(df)),
        "entries_with_rruff_abc": count_abc_present,
        "entries_with_complete_lattice": count_complete_lattice,
        "crystal_system_counts": {k: int(crystal_counts.get(k, 0)) for k in CRYSTAL_SYSTEM_ORDER},
    }


def summarize_run(
    run_name: str,
    run_dir: Path,
    df: pd.DataFrame,
    provenance: Dict[str, str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return {
        "run_name": run_name,
        "run_directory": str(run_dir),
        "source": provenance,
        "benchmark_summary": recoverable_benchmark_summary(
            df,
            raw_rruff_count=args.raw_rruff_count,
            unique_valid_chemistry_count=args.unique_valid_chemistry_count,
        ),
        "coverage_metrics": summarize_coverage(df),
        "overall_lattice_accuracy": summarize_overall_lattice(
            df,
            bins=args.jsd_bins,
            alpha=args.laplace_alpha,
        ),
        "method_breakdown": summarize_method_breakdown(df),
        "crystal_system_breakdown": summarize_crystal_system_breakdown(df),
        "distribution_threshold_metrics": summarize_relative_error_thresholds(
            df,
            mode=args.relative_error_mode,
        ),
        "refinement_quality": summarize_refinement_quality(df),
        "runtime_and_failures": summarize_runtime_and_failures(df),
        "similarity_summary": summarize_similarity(df),
        "element_error_summary": summarize_element_errors(df),
    }


def pairwise_distribution_jsd(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    bins: int,
    alpha: float,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for p in PARAMS_ALL:
        arr_a = pd.to_numeric(df_a[f"pred_{p}"], errors="coerce").dropna().to_numpy(dtype=float)
        arr_b = pd.to_numeric(df_b[f"pred_{p}"], errors="coerce").dropna().to_numpy(dtype=float)
        out[p] = {
            "n_run_a": int(len(arr_a)),
            "n_run_b": int(len(arr_b)),
            "jsd": js_divergence(arr_a, arr_b, bins=bins, alpha=alpha),
        }
    return out


def per_run_refinement_comparison(run_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for run_name, summary in run_summaries.items():
        overall = summary["overall_lattice_accuracy"]
        out[run_name] = {
            p: {
                "n": overall[p]["n"],
                "mae": overall[p]["mae"],
                "rmse": overall[p]["rmse"],
                "r2": overall[p]["r2"],
                "jsd": overall[p]["jsd"],
            }
            for p in PARAMS_ALL
        }
    return out


def common_entry_count(dfs: Dict[str, pd.DataFrame]) -> int:
    if not dfs:
        return 0
    sets = [set(df["entry_key"].tolist()) for df in dfs.values()]
    common = set.intersection(*sets) if sets else set()
    return int(len(common))


def build_cross_run_summary(
    run_dataframes: Dict[str, pd.DataFrame],
    run_summaries: Dict[str, Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    pairwise: Dict[str, Any] = {}
    for a, b in combinations(run_dataframes.keys(), 2):
        pairwise[f"{a}__vs__{b}"] = {
            "pair": [a, b],
            "pairwise_predicted_distribution_jsd": pairwise_distribution_jsd(
                run_dataframes[a],
                run_dataframes[b],
                bins=args.jsd_bins,
                alpha=args.laplace_alpha,
            ),
        }

    return {
        "run_order": list(run_dataframes.keys()),
        "n_runs": len(run_dataframes),
        "common_entry_count": common_entry_count(run_dataframes),
        "refinement_comparison": per_run_refinement_comparison(run_summaries),
        "pairwise_run_comparisons": pairwise,
    }


def resolve_run_dir(spec: str, runs_root: Path) -> Path:
    raw = Path(spec)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(Path.cwd() / raw)
        candidates.append(runs_root / raw)
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            return cand.resolve()
    raise FileNotFoundError(f"Could not resolve run directory from '{spec}'")


def main() -> None:
    args = parse_args()
    run_list_path = Path(args.run_list).resolve()
    runs_root = Path(args.runs_root).resolve()
    output_path = Path(args.output).resolve()

    run_specs = read_run_list(run_list_path)

    run_dataframes: Dict[str, pd.DataFrame] = {}
    run_summaries: Dict[str, Dict[str, Any]] = {}

    for spec in run_specs:
        run_dir = resolve_run_dir(spec, runs_root)
        run_name = run_dir.name
        if run_name in run_dataframes:
            raise ValueError(f"Duplicate run name after resolution: {run_name}")
        df, provenance = load_run_dataframe(run_dir)
        run_dataframes[run_name] = df
        run_summaries[run_name] = summarize_run(run_name, run_dir, df, provenance, args)

    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "run_list_file": str(run_list_path),
            "runs_root": str(runs_root),
            "requested_runs": run_specs,
            "resolved_run_names": list(run_dataframes.keys()),
            "jsd_bins": args.jsd_bins,
            "laplace_alpha": args.laplace_alpha,
            "relative_error_mode": args.relative_error_mode,
            "raw_rruff_count": args.raw_rruff_count,
            "unique_valid_chemistry_count": args.unique_valid_chemistry_count,
        },
        "runs": run_summaries,
        "cross_run_summary": build_cross_run_summary(run_dataframes, run_summaries, args),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(jsonable(payload), f, indent=2)

    print(f"Wrote replication summary to {output_path}")


if __name__ == "__main__":
    main()
