#!/usr/bin/env python3
from __future__ import annotations

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
    "trigonal",
    "hexagonal",
    "cubic",
]
CRYSYS_LABELS = [s.capitalize() for s in CRYSYS_ORDER]


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


def _newest_results_dir(method_dir: Path) -> Path:
    candidates = sorted(method_dir.glob("rruff_results_*"), key=lambda p: p.name)
    if not candidates:
        raise FileNotFoundError(f"No timestamped rruff_results_* directory found in {method_dir}")
    return candidates[-1]


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


def _get_crystal_system_from_atoms_dict(atoms_dict: dict, symprec: float = 0.1) -> Optional[str]:
    try:
        atoms_j = JarvisAtoms.from_dict(atoms_dict)
        pmg = atoms_j.pymatgen_converter()
        sga = SpacegroupAnalyzer(pmg, symprec=symprec)
        conv = sga.get_conventional_standard_structure()
        cs = SpacegroupAnalyzer(conv, symprec=symprec).get_crystal_system()
        return cs.lower() if isinstance(cs, str) else None
    except Exception:
        return None


def _crysys_percent_hist(cs_list: List[str]) -> np.ndarray:
    if not cs_list:
        return np.zeros(len(CRYSYS_ORDER))
    counts = pd.Series(cs_list).value_counts()
    total = counts.sum()
    return np.array([(counts.get(cs, 0) / total) * 100.0 for cs in CRYSYS_ORDER], dtype=float)


def _load_method_payload(method_key: str) -> Tuple[List[str], Dict[str, float], Path, Path]:
    """
    Returns:
      crystal_systems
      match_rate_info
      runs_dir
      results_dir
    """
    root = Path.cwd().resolve()
    runs_dir = _find_runs_root(root)
    method_dir = runs_dir / method_key
    if not method_dir.is_dir():
        raise FileNotFoundError(f"Missing method directory: {method_dir}")

    results_dir = _newest_results_dir(method_dir)

    all_results_path = results_dir / "all_results.json"
    results_csv_path = results_dir / "results.csv"

    if not all_results_path.is_file():
        raise FileNotFoundError(f"Missing {all_results_path}")
    if not results_csv_path.is_file():
        raise FileNotFoundError(f"Missing {results_csv_path}")

    # Crystal systems from saved best-match structures
    with open(all_results_path, "r") as f:
        payload = json.load(f)

    crystal_systems: List[str] = []
    for rec in payload:
        atoms_dict = rec.get("best_match", {}).get("atoms_dict")
        if not atoms_dict:
            continue
        cs = _get_crystal_system_from_atoms_dict(atoms_dict, symprec=0.1)
        if cs in CRYSYS_ORDER:
            crystal_systems.append(cs)

    # Match rate from saved CSV only (no AGAPI rerun)
    df = pd.read_csv(results_csv_path)
    pm_success = _safe_bool_series(df, "pm_success")
    dg_success = _safe_bool_series(df, "dg_success")
    error_mask = _safe_str_series(df, "error").str.strip() != ""

    total_rows = int(len(df))
    valid_rows = int((~error_mask).sum())
    error_rows = int(error_mask.sum())

    pm_success_valid = pm_success & (~error_mask)
    dg_success_valid = dg_success & (~error_mask)
    any_match_valid = (pm_success | dg_success) & (~error_mask)

    match_rate_info = {
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "error_rows": error_rows,
        "pm_success_n": int(pm_success_valid.sum()),
        "pm_match_rate_percent": _pct(int(pm_success_valid.sum()), valid_rows),
        "dg_success_n": int(dg_success_valid.sum()),
        "dg_match_rate_percent": _pct(int(dg_success_valid.sum()), valid_rows),
        "any_match_n": int(any_match_valid.sum()),
        "any_match_rate_percent": _pct(int(any_match_valid.sum()), valid_rows),
        "matched_for_pie": int(pm_success_valid.sum()),
        "unmatched_for_pie": int(valid_rows - int(pm_success_valid.sum())),
    }

    return crystal_systems, match_rate_info, runs_dir, results_dir


# ── plotting ───────────────────────────────────────────────────────
def plot_single_crysys(
    cs_list: List[str],
    method_key: str,
    method_name: str,
    color: str,
    out_dir: Path,
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
    ax.set_ylabel("% of Total Structures", fontsize=30)
    ax.set_title(f"Crystal Systems\n{method_name}", fontsize=38)
    plt.xticks(fontsize=23)
    plt.yticks(fontsize=23)

    _style_axes_like_grid(ax)

    out_png = out_dir / f"{method_key}_crystal_system_histogram.png"
    plt.savefig(out_png, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ wrote {out_png}")


def plot_stacked_crysys(loaded_data, out_dir: Path) -> None:
    global_max = 0.0
    for _, _, _, cs_list, _ in loaded_data:
        percents = _crysys_percent_hist(cs_list)
        global_max = max(global_max, float(np.max(percents)) if len(percents) else 0.0)

    ymax = 1.12 * global_max if global_max > 0 else 1.0
    x = np.arange(len(CRYSYS_ORDER))

    fig, axes = plt.subplots(3, 1, figsize=(12.5, 22), sharex=True, constrained_layout=True)

    for ax, (method_key, method_name, color, cs_list, _) in zip(axes, loaded_data):
        percents = _crysys_percent_hist(cs_list)

        ax.bar(
            x,
            percents,
            width=0.68,
            alpha=0.82,
            linewidth=0.0,
            edgecolor="none",
            color=color,
        )

        ax.set_ylim(0, ymax)
        ax.set_ylabel("% of Total Structures", fontsize=24)
        ax.set_title(f"{method_name}  (n={len(cs_list)})", fontsize=28)
        plt.sca(ax)
        plt.yticks(fontsize=18)
        _style_axes_like_grid(ax)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(CRYSYS_LABELS, rotation=25)
    axes[-1].set_xlabel("Crystal system", fontsize=26)
    plt.xticks(fontsize=20)

    fig.suptitle("Crystal Systems by Refinement Method", fontsize=34)

    out_png = out_dir / "crystal_system_histograms_stacked.png"
    plt.savefig(out_png, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ wrote {out_png}")


def plot_match_rate_bar(loaded_data, out_dir: Path) -> None:
    method_names = []
    rates = []
    colors = []
    counts_text = []

    for _, method_name, color, _, match_info in loaded_data:
        method_names.append(method_name)
        rates.append(match_info["pm_match_rate_percent"])
        colors.append(color)
        counts_text.append(f"{match_info['pm_success_n']}/{match_info['valid_rows']}")

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
    ax.set_title("Pattern-Matching Match Rate by Refinement Method", fontsize=30)
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

    out_png = out_dir / "pattern_matching_match_rate.png"
    plt.savefig(out_png, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ wrote {out_png}")


def plot_crysys_plus_match_pie(loaded_data, out_dir: Path) -> None:
    """
    4-panel vertical figure:
      top 3 = crystal-system histograms
      bottom = aggregate PM matched vs unmatched pie chart
    """
    global_max = 0.0
    total_matched = 0
    total_unmatched = 0

    for _, _, _, cs_list, match_info in loaded_data:
        percents = _crysys_percent_hist(cs_list)
        global_max = max(global_max, float(np.max(percents)) if len(percents) else 0.0)
        total_matched += int(match_info["matched_for_pie"])
        total_unmatched += int(match_info["unmatched_for_pie"])

    ymax = 1.12 * global_max if global_max > 0 else 1.0
    x = np.arange(len(CRYSYS_ORDER))

    fig = plt.figure(figsize=(13, 28), constrained_layout=True)
    gs = fig.add_gridspec(4, 1)

    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    ax_pie = fig.add_subplot(gs[3, 0])

    for ax, (method_key, method_name, color, cs_list, match_info) in zip(axes, loaded_data):
        percents = _crysys_percent_hist(cs_list)

        ax.bar(
            x,
            percents,
            width=0.68,
            alpha=0.82,
            linewidth=0.0,
            edgecolor="none",
            color=color,
        )

        ax.set_ylim(0, ymax)
        ax.set_ylabel("% of Total Structures", fontsize=22)
        ax.set_title(
            f"{method_name}  (n={len(cs_list)}, PM={match_info['pm_match_rate_percent']:.1f}%)",
            fontsize=26,
        )
        plt.sca(ax)
        plt.yticks(fontsize=17)
        _style_axes_like_grid(ax)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(CRYSYS_LABELS, rotation=25)
    axes[-1].set_xlabel("Crystal system", fontsize=24)
    plt.xticks(fontsize=19)

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
        textprops={"fontsize": 18},
    )
    ax_pie.set_title(
        "Aggregate Pattern-Matching Match Rate\n(across all three refinement runs)",
        fontsize=28,
    )

    fig.suptitle("Crystal Systems and Pattern-Matching Match Rate", fontsize=34)

    out_png = out_dir / "crystal_system_histograms_plus_match_rate_pie.png"
    plt.savefig(out_png, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ wrote {out_png}")


# ── main ───────────────────────────────────────────────────────────
def main() -> None:
    loaded_data = []
    runs_dir = None

    for method_key, method_name, color in METHODS:
        cs_list, match_info, runs_dir, results_dir = _load_method_payload(method_key)
        loaded_data.append((method_key, method_name, color, cs_list, match_info))

    out_dir = runs_dir / "refinement_summary_figures"
    _ensure_dir(out_dir)

    # 3 individual crystal-system histograms
    for method_key, method_name, color, cs_list, match_info in loaded_data:
        plot_single_crysys(
            cs_list=cs_list,
            method_key=method_key,
            method_name=method_name,
            color=color,
            out_dir=out_dir,
        )

    # 1 PNG with the 3 crystal-system histograms stacked
    plot_stacked_crysys(loaded_data, out_dir=out_dir)

    # 1 PNG with the 3 histograms + 1 aggregate PM match-rate pie chart
    plot_crysys_plus_match_pie(loaded_data, out_dir=out_dir)

    # 1 PNG with just the PM match rate
    plot_match_rate_bar(loaded_data, out_dir=out_dir)

    print(f"\n✓ all figures written to: {out_dir}")


if __name__ == "__main__":
    main()
