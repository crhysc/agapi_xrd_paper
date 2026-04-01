#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, List, Tuple

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


def _load_crystal_systems(method_key: str) -> Tuple[List[str], Path, Path]:
    root = Path.cwd().resolve()
    runs = _find_runs_root(root)
    method_dir = runs / method_key
    if not method_dir.is_dir():
        raise FileNotFoundError(f"Missing method directory: {method_dir}")

    results_dir = _newest_results_dir(method_dir)
    all_results_path = results_dir / "all_results.json"
    if not all_results_path.is_file():
        raise FileNotFoundError(f"Missing {all_results_path}")

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

    return crystal_systems, runs, results_dir


# ── plotting ───────────────────────────────────────────────────────
def plot_single_crysys(
    cs_list: List[str],
    method_key: str,
    method_name: str,
    color: str,
    results_dir: Path,
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

    out_png = results_dir / f"{method_key}_crystal_system_histogram.png"
    out_csv = results_dir / f"{method_key}_crystal_system_percentages.csv"

    pd.DataFrame({
        "crystal_system": CRYSYS_LABELS,
        "percent": percents,
        "n_total_structures": [len(cs_list)] * len(CRYSYS_LABELS),
    }).to_csv(out_csv, index=False)

    plt.savefig(out_png, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ wrote {out_png}")
    print(f"✓ wrote {out_csv}")
    print(f"  n = {len(cs_list)}")


def plot_stacked_crysys(loaded_data, runs_dir: Path) -> None:
    global_max = 0.0
    for _, _, _, cs_list, _ in loaded_data:
        percents = _crysys_percent_hist(cs_list)
        global_max = max(global_max, float(np.max(percents)) if len(percents) else 0.0)

    ymax = 1.12 * global_max if global_max > 0 else 1.0
    x = np.arange(len(CRYSYS_ORDER))

    fig, axes = plt.subplots(3, 1, figsize=(12.5, 22), sharex=True, constrained_layout=True)

    summary_rows = []

    for ax, (method_key, method_name, color, cs_list, results_dir) in zip(axes, loaded_data):
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

        for cs_label, pct in zip(CRYSYS_LABELS, percents):
            summary_rows.append({
                "method": method_name,
                "crystal_system": cs_label,
                "percent": pct,
                "n_total_structures": len(cs_list),
            })

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(CRYSYS_LABELS, rotation=25)
    axes[-1].set_xlabel("Crystal system", fontsize=26)
    plt.xticks(fontsize=20)

    fig.suptitle("Crystal Systems by Refinement Method", fontsize=34)

    out_png = runs_dir / "crystal_system_histograms_stacked.png"
    out_csv = runs_dir / "crystal_system_histograms_stacked.csv"

    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)

    plt.savefig(out_png, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ wrote {out_png}")
    print(f"✓ wrote {out_csv}")


# ── main ───────────────────────────────────────────────────────────
def main() -> None:
    loaded_data = []
    runs_dir = None

    for method_key, method_name, color in METHODS:
        cs_list, runs_dir, results_dir = _load_crystal_systems(method_key)
        loaded_data.append((method_key, method_name, color, cs_list, results_dir))

    for method_key, method_name, color, cs_list, results_dir in loaded_data:
        plot_single_crysys(
            cs_list=cs_list,
            method_key=method_key,
            method_name=method_name,
            color=color,
            results_dir=results_dir,
        )

    plot_stacked_crysys(loaded_data, runs_dir)


if __name__ == "__main__":
    main()
