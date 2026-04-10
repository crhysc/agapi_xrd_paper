#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams.update({
    "font.family": "serif",
    "axes.linewidth": 0.8,
    "patch.linewidth": 0.0,
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif", "STIX"],
})

import matplotlib.pyplot as plt

from jarvis.core.atoms import Atoms as JarvisAtoms
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


# ── palette / labels ───────────────────────────────────────────────
WVU_BLUE = "#002855"
WVU_GOLD = "#EEAA00"
PLUM = "#8E63CE"

MATCH_GREEN = "#2E8B57"
UNMATCHED_GRAY = "#B0B0B0"

METHODS = [
    ("no_refinement", "No refinement", WVU_BLUE),
    ("gsas2", "GSAS-II", WVU_GOLD),
    ("bmgn", "BMGN", PLUM),
]

CRYSYS_ORDER = [
    "triclinic",
    "monoclinic",
    "orthorhombic",
    "tetragonal",
    "hexagonal",
    "cubic",
    "rhombohedral",
]
CRYSYS_LABELS = [s.capitalize() for s in CRYSYS_ORDER]


# ── CLI ────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot crystal-system distributions for refinement runs with optional max abc filtering."
    )
    parser.add_argument(
        "--max-val",
        type=float,
        default=15.0,
        help="Maximum allowed RRUFF a, b, c value for filtering (default: 15.0).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to runs/refinement_summary_figures.",
    )
    parser.add_argument(
        "--symprec",
        type=float,
        default=0.1,
        help="Symmetry tolerance for determining crystal system from atoms_dict (default: 0.1).",
    )
    return parser.parse_args()


# ── helpers ────────────────────────────────────────────────────────
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _style_axes_like_grid(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", which="major", width=1.4, length=7)
    ax.minorticks_off()


def _find_runs_root(start: Path) -> Path:
    if (start / "runs").is_dir():
        return start / "runs"
    if start.name == "runs" and start.is_dir():
        return start
    raise FileNotFoundError(
        "Could not find a runs/ directory. Run this from the project root or from inside runs/."
    )


def _safe_bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    s = df[col]
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return s.fillna(False).astype(bool)


def _safe_str_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series("", index=df.index, dtype="object")
    return df[col].fillna("").astype(str)


def _pct(num: int, den: int) -> float:
    return 100.0 * num / den if den > 0 else float("nan")


def _normalize_crysys(cs: Optional[str]) -> Optional[str]:
    if cs is None:
        return None
    cs = str(cs).strip().lower()
    if cs == "trigonal":
        return "rhombohedral"
    return cs


def _get_crystal_system_from_atoms_dict(atoms_dict: dict, symprec: float = 0.1) -> Optional[str]:
    try:
        atoms_j = JarvisAtoms.from_dict(atoms_dict)
        pmg = atoms_j.pymatgen_converter()
        sga = SpacegroupAnalyzer(pmg, symprec=symprec)
        conv = sga.get_conventional_standard_structure()
        cs = SpacegroupAnalyzer(conv, symprec=symprec).get_crystal_system()
        cs = _normalize_crysys(cs)
        return cs if cs in CRYSYS_ORDER else None
    except Exception:
        return None


def _crysys_percent_hist(cs_list: List[str]) -> np.ndarray:
    if not cs_list:
        return np.zeros(len(CRYSYS_ORDER))
    counts = pd.Series(cs_list).value_counts()
    total = counts.sum()
    return np.array([(counts.get(cs, 0) / total) * 100.0 for cs in CRYSYS_ORDER], dtype=float)


def _locate_method_files(method_dir: Path) -> Tuple[Path, Optional[Path]]:
    """
    Returns:
      results_csv_path
      all_results_json_path (or None if not found)
    Preference:
      1) newest timestamped rruff_results_*/results.csv + all_results.json
      2) top-level method-specific CSV / results.csv
    """
    # timestamped dirs
    ts_dirs = sorted(method_dir.glob("rruff_results_*"), key=lambda p: p.name)
    if ts_dirs:
        newest = ts_dirs[-1]
        results_csv = newest / "results.csv"
        all_results = newest / "all_results.json"
        if results_csv.is_file():
            return results_csv, all_results if all_results.is_file() else None

    # top-level fallbacks
    top_csv_candidates = [
        method_dir / "results.csv",
        method_dir / f"{method_dir.name}_results.csv",
    ]
    top_json_candidates = [
        method_dir / "all_results.json",
        method_dir / f"{method_dir.name}_all_results.json",
    ]

    results_csv = next((p for p in top_csv_candidates if p.is_file()), None)
    all_results = next((p for p in top_json_candidates if p.is_file()), None)

    if results_csv is None:
        raise FileNotFoundError(f"Could not find results CSV for {method_dir}")

    return results_csv, all_results


def _build_filter_mask(df: pd.DataFrame, max_val: float) -> pd.Series:
    needed = ["rruff_a", "rruff_b", "rruff_c"]
    for col in needed:
        if col not in df.columns:
            raise KeyError(f"Missing required column for max-val filtering: {col}")

    mask = df["rruff_a"].notna() & df["rruff_b"].notna() & df["rruff_c"].notna()
    mask &= (df["rruff_a"] < max_val) & (df["rruff_b"] < max_val) & (df["rruff_c"] < max_val)
    return mask


def _load_method_payload(
    method_key: str,
    max_val: float,
    symprec: float,
) -> Tuple[List[str], Dict[str, float], Path]:
    """
    Returns:
      crystal_systems
      match_rate_info
      runs_dir
    """
    root = Path.cwd().resolve()
    runs_dir = _find_runs_root(root)
    method_dir = runs_dir / method_key
    if not method_dir.is_dir():
        raise FileNotFoundError(f"Missing method directory: {method_dir}")

    results_csv_path, all_results_path = _locate_method_files(method_dir)
    df = pd.read_csv(results_csv_path)

    filter_mask = _build_filter_mask(df, max_val)
    error_mask = _safe_str_series(df, "error").str.strip() != ""

    # Match-rate info computed inside the filtered subset
    pm_success = _safe_bool_series(df, "pm_success")
    dg_success = _safe_bool_series(df, "dg_success")

    filtered_rows = int(filter_mask.sum())
    filtered_valid_rows = int((filter_mask & (~error_mask)).sum())
    filtered_error_rows = int((filter_mask & error_mask).sum())

    pm_success_valid = pm_success & filter_mask & (~error_mask)
    dg_success_valid = dg_success & filter_mask & (~error_mask)
    any_match_valid = (pm_success | dg_success) & filter_mask & (~error_mask)

    match_rate_info = {
        "filtered_rows": filtered_rows,
        "filtered_valid_rows": filtered_valid_rows,
        "filtered_error_rows": filtered_error_rows,
        "pm_success_n": int(pm_success_valid.sum()),
        "pm_match_rate_percent": _pct(int(pm_success_valid.sum()), filtered_valid_rows),
        "dg_success_n": int(dg_success_valid.sum()),
        "dg_match_rate_percent": _pct(int(dg_success_valid.sum()), filtered_valid_rows),
        "any_match_n": int(any_match_valid.sum()),
        "any_match_rate_percent": _pct(int(any_match_valid.sum()), filtered_valid_rows),
        "matched_for_pie": int(pm_success_valid.sum()),
        "unmatched_for_pie": int(filtered_valid_rows - int(pm_success_valid.sum())),
    }

    # Crystal systems
    crystal_systems: List[str] = []

    # Preferred route: use all_results.json if present and row-aligned with results.csv
    if all_results_path is not None:
        try:
            with open(all_results_path, "r") as f:
                payload = json.load(f)

            if len(payload) == len(df):
                for i, rec in enumerate(payload):
                    if not bool(filter_mask.iloc[i]):
                        continue
                    atoms_dict = rec.get("best_match", {}).get("atoms_dict")
                    if not atoms_dict:
                        continue
                    cs = _get_crystal_system_from_atoms_dict(atoms_dict, symprec=symprec)
                    if cs in CRYSYS_ORDER:
                        crystal_systems.append(cs)
            else:
                # Robust fallback if JSON and CSV lengths diverge
                if "rruff_crystal_system" in df.columns:
                    cs_series = df.loc[filter_mask, "rruff_crystal_system"].map(_normalize_crysys)
                    crystal_systems = [cs for cs in cs_series.dropna().tolist() if cs in CRYSYS_ORDER]
        except Exception:
            if "rruff_crystal_system" in df.columns:
                cs_series = df.loc[filter_mask, "rruff_crystal_system"].map(_normalize_crysys)
                crystal_systems = [cs for cs in cs_series.dropna().tolist() if cs in CRYSYS_ORDER]
    else:
        if "rruff_crystal_system" not in df.columns:
            raise KeyError(
                f"Could not determine crystal systems for {method_key}: "
                "no all_results.json and no rruff_crystal_system column in CSV."
            )
        cs_series = df.loc[filter_mask, "rruff_crystal_system"].map(_normalize_crysys)
        crystal_systems = [cs for cs in cs_series.dropna().tolist() if cs in CRYSYS_ORDER]

    return crystal_systems, match_rate_info, runs_dir


# ── plotting ───────────────────────────────────────────────────────
def plot_single_crysys(
    cs_list: List[str],
    method_key: str,
    method_name: str,
    color: str,
    out_dir: Path,
    max_val: float,
) -> None:
    percents = _crysys_percent_hist(cs_list)
    x = np.arange(len(CRYSYS_ORDER))

    fig, ax = plt.subplots(figsize=(11.5, 11.5), constrained_layout=True)

    ax.bar(
        x,
        percents,
        width=0.68,
        alpha=0.82,
        linewidth=0.0,
        edgecolor="none",
        color=color,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(CRYSYS_LABELS, rotation=25)
    ax.set_xlabel("Crystal system", fontsize=30)
    ax.set_ylabel("% of filtered structures", fontsize=30)
    ax.set_title(f"Crystal Systems\n{method_name}  (a,b,c < {max_val:g} Å)", fontsize=36)
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)

    _style_axes_like_grid(ax)

    out_png = out_dir / f"{method_key}_crystal_system_histogram_max_{str(max_val).replace('.', 'p')}.png"
    plt.savefig(out_png, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ wrote {out_png}")


def plot_overlaid_crysys(loaded_data, out_dir: Path, max_val: float) -> None:
    x = np.arange(len(CRYSYS_ORDER))

    fig, ax = plt.subplots(figsize=(13, 9), constrained_layout=True)

    for method_key, method_name, color, cs_list, _ in loaded_data:
        percents = _crysys_percent_hist(cs_list)
        ax.plot(
            x,
            percents,
            marker="o",
            linewidth=2.8,
            markersize=8,
            color=color,
            label=f"{method_name} (n={len(cs_list)})",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(CRYSYS_LABELS, rotation=25)
    ax.set_xlabel("Crystal system", fontsize=24)
    ax.set_ylabel("% of filtered structures", fontsize=24)
    ax.set_title(f"Crystal Systems by Refinement Method  (RRUFF a,b,c < {max_val:g} Å)", fontsize=30)
    ax.legend(fontsize=15, frameon=False)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    _style_axes_like_grid(ax)

    out_png = out_dir / f"crystal_systems_overlaid_max_{str(max_val).replace('.', 'p')}.png"
    plt.savefig(out_png, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ wrote {out_png}")


def plot_match_rate_bar(loaded_data, out_dir: Path, max_val: float) -> None:
    method_names = []
    rates = []
    colors = []
    counts_text = []

    for _, method_name, color, _, match_info in loaded_data:
        method_names.append(method_name)
        rates.append(match_info["pm_match_rate_percent"])
        colors.append(color)
        counts_text.append(f"{match_info['pm_success_n']}/{match_info['filtered_valid_rows']}")

    x = np.arange(len(method_names))

    fig, ax = plt.subplots(figsize=(10.5, 8.5), constrained_layout=True)

    bars = ax.bar(
        x,
        rates,
        width=0.62,
        alpha=0.84,
        linewidth=0.0,
        edgecolor="none",
        color=colors,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=20, ha="right")
    ax.set_ylabel("Pattern-matching match rate (%)", fontsize=24)
    ax.set_title(f"Pattern-Matching Match Rate  (RRUFF a,b,c < {max_val:g} Å)", fontsize=28)
    ax.set_ylim(0, max(100, 1.12 * max(rates) if len(rates) else 100))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    _style_axes_like_grid(ax)

    for bar, rate, label in zip(bars, rates, counts_text):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.0,
            f"{rate:.1f}%\n({label})",
            ha="center",
            va="bottom",
            fontsize=14,
        )

    out_png = out_dir / f"pattern_matching_match_rate_max_{str(max_val).replace('.', 'p')}.png"
    plt.savefig(out_png, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ wrote {out_png}")


def plot_crysys_overlay_plus_match_pie(loaded_data, out_dir: Path, max_val: float) -> None:
    total_matched = 0
    total_unmatched = 0

    for _, _, _, _, match_info in loaded_data:
        total_matched += int(match_info["matched_for_pie"])
        total_unmatched += int(match_info["unmatched_for_pie"])

    x = np.arange(len(CRYSYS_ORDER))

    fig = plt.figure(figsize=(13, 16), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.1])

    ax_top = fig.add_subplot(gs[0, 0])
    ax_pie = fig.add_subplot(gs[1, 0])

    for method_key, method_name, color, cs_list, _ in loaded_data:
        percents = _crysys_percent_hist(cs_list)
        ax_top.plot(
            x,
            percents,
            marker="o",
            linewidth=2.8,
            markersize=8,
            color=color,
            label=f"{method_name} (n={len(cs_list)})",
        )

    ax_top.set_xticks(x)
    ax_top.set_xticklabels(CRYSYS_LABELS, rotation=25)
    ax_top.set_xlabel("Crystal system", fontsize=22)
    ax_top.set_ylabel("% of filtered structures", fontsize=22)
    ax_top.set_title(f"Crystal Systems by Refinement Method  (RRUFF a,b,c < {max_val:g} Å)", fontsize=28)
    ax_top.legend(fontsize=14, frameon=False)
    plt.sca(ax_top)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    _style_axes_like_grid(ax_top)

    pie_vals = [total_matched, total_unmatched]
    pie_labels = [
        f"Matched\n{total_matched}",
        f"Unmatched\n{total_unmatched}",
    ]
    pie_colors = [MATCH_GREEN, UNMATCHED_GRAY]

    ax_pie.pie(
        pie_vals,
        labels=pie_labels,
        colors=pie_colors,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        textprops={"fontsize": 16},
    )
    ax_pie.set_title("Aggregate Pattern-Matching Match Rate", fontsize=24)

    out_png = out_dir / f"crystal_systems_overlaid_plus_match_rate_pie_max_{str(max_val).replace('.', 'p')}.png"
    plt.savefig(out_png, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ wrote {out_png}")


# ── main ───────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    loaded_data = []
    runs_dir = None

    for method_key, method_name, color in METHODS:
        cs_list, match_info, runs_dir = _load_method_payload(
            method_key=method_key,
            max_val=args.max_val,
            symprec=args.symprec,
        )
        loaded_data.append((method_key, method_name, color, cs_list, match_info))

    if runs_dir is None:
        raise RuntimeError("Could not determine runs directory.")

    out_dir = Path(args.output_dir) if args.output_dir is not None else runs_dir / "refinement_summary_figures"
    _ensure_dir(out_dir)

    # individual histograms
    for method_key, method_name, color, cs_list, _ in loaded_data:
        plot_single_crysys(
            cs_list=cs_list,
            method_key=method_key,
            method_name=method_name,
            color=color,
            out_dir=out_dir,
            max_val=args.max_val,
        )

    # overlaid crystal-system plot
    plot_overlaid_crysys(loaded_data, out_dir=out_dir, max_val=args.max_val)

    # combined overlaid plot + aggregate PM pie
    plot_crysys_overlay_plus_match_pie(loaded_data, out_dir=out_dir, max_val=args.max_val)

    # PM match-rate bar plot
    plot_match_rate_bar(loaded_data, out_dir=out_dir, max_val=args.max_val)

    print(f"\n✓ all figures written to: {out_dir}")


if __name__ == "__main__":
    main()
