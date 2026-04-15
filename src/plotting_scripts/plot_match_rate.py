#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

METHOD_DIRS = {
    "no_refinement": "No refinement",
    "gsas2": "GSAS-II",
    "bmgn": "BMGN",
}


def find_runs_root(start: Path) -> Path:
    if (start / "runs").is_dir():
        return start / "runs"
    if start.name == "runs" and start.is_dir():
        return start
    raise FileNotFoundError(
        "Could not find a runs/ directory. Run this from the project root "
        "or from inside runs/."
    )


def newest_results_csv(method_dir: Path) -> Path:
    candidates = sorted(method_dir.glob("rruff_results_*/results.csv"), key=lambda p: p.parent.name)
    if candidates:
        return candidates[-1]

    fallback = sorted(method_dir.glob("*_results.csv"), key=lambda p: p.name)
    if fallback:
        return fallback[-1]

    raise FileNotFoundError(f"No results.csv found in {method_dir}")


def safe_bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    s = df[col]
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.fillna(False).astype(bool)


def safe_str_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series("", index=df.index, dtype="object")
    return df[col].fillna("").astype(str)


def pct(num: int, den: int) -> float:
    return 100.0 * num / den if den > 0 else float("nan")


def analyze_results(csv_path: Path, method_name: str) -> dict:
    df = pd.read_csv(csv_path)

    pm_success = safe_bool_series(df, "pm_success")
    dg_success = safe_bool_series(df, "dg_success")
    error_mask = safe_str_series(df, "error").str.strip() != ""
    query_used = safe_str_series(df, "query_used")

    total_rows = len(df)
    valid_rows = int((~error_mask).sum())
    error_rows = int(error_mask.sum())

    # Among non-error rows
    pm_success_valid = pm_success & (~error_mask)
    dg_success_valid = dg_success & (~error_mask)
    any_match_valid = (pm_success | dg_success) & (~error_mask)

    # Distinguish formula query vs element fallback using query_used
    element_query_valid = query_used.str.startswith("elements:") & (~error_mask)
    formula_query_valid = (~query_used.str.startswith("elements:")) & (~error_mask)

    # PM successes split by query path
    pm_formula_success = pm_success_valid & formula_query_valid
    pm_element_success = pm_success_valid & element_query_valid

    # Similarity summaries
    pm_similarity = df.loc[pm_success_valid, "pm_similarity"] if "pm_similarity" in df.columns else pd.Series(dtype=float)

    out = {
        "method": method_name,
        "csv_path": str(csv_path),
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "error_rows": error_rows,
        "error_rate_percent": pct(error_rows, total_rows),

        "pm_success_n": int(pm_success_valid.sum()),
        "pm_match_rate_percent": pct(int(pm_success_valid.sum()), valid_rows),

        "formula_query_rows": int(formula_query_valid.sum()),
        "pm_formula_success_n": int(pm_formula_success.sum()),
        "pm_formula_match_rate_percent": pct(int(pm_formula_success.sum()), int(formula_query_valid.sum())),

        "element_query_rows": int(element_query_valid.sum()),
        "pm_element_success_n": int(pm_element_success.sum()),
        "pm_element_match_rate_percent": pct(int(pm_element_success.sum()), int(element_query_valid.sum())),

        "dg_success_n": int(dg_success_valid.sum()),
        "dg_match_rate_percent": pct(int(dg_success_valid.sum()), valid_rows),

        "any_match_n": int(any_match_valid.sum()),
        "any_match_rate_percent": pct(int(any_match_valid.sum()), valid_rows),

        "pm_similarity_mean": float(pm_similarity.mean()) if len(pm_similarity) else float("nan"),
        "pm_similarity_median": float(pm_similarity.median()) if len(pm_similarity) else float("nan"),
        "pm_similarity_min": float(pm_similarity.min()) if len(pm_similarity) else float("nan"),
        "pm_similarity_max": float(pm_similarity.max()) if len(pm_similarity) else float("nan"),
    }
    return out


def main() -> None:
    root = Path.cwd()
    runs = find_runs_root(root)
    print(f"DEBUG: Using runs directory: {runs}", file=sys.stderr)

    rows = []
    for key, pretty in METHOD_DIRS.items():
        method_dir = runs / key
        if not method_dir.is_dir():
            print(f"WARNING: Missing method directory: {method_dir}", file=sys.stderr)
            continue

        csv_path = newest_results_csv(method_dir)
        print(f"DEBUG: {pretty:<13s} -> {csv_path}", file=sys.stderr)
        rows.append(analyze_results(csv_path, pretty))

    if not rows:
        print("ERROR: No results were found.", file=sys.stderr)
        sys.exit(1)

    summary = pd.DataFrame(rows)

    out_csv = runs / "pattern_matching_match_rates.csv"
    summary.to_csv(out_csv, index=False)

    display_cols = [
        "method",
        "valid_rows",
        "error_rows",
        "pm_success_n",
        "pm_match_rate_percent",
        "pm_formula_success_n",
        "pm_formula_match_rate_percent",
        "pm_element_success_n",
        "pm_element_match_rate_percent",
        "dg_success_n",
        "dg_match_rate_percent",
        "any_match_n",
        "any_match_rate_percent",
    ]

    print("\nPattern-matching rate summary:\n")
    print(summary[display_cols].to_string(index=False))
    print(f"\nSaved summary CSV: {out_csv}")


if __name__ == "__main__":
    main()
