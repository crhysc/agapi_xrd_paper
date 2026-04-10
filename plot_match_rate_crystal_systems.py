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


# ── palette ───────────────────────────────────────────────
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


# ── CLI ───────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-val", type=float, default=15.0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--symprec", type=float, default=0.1)
    return parser.parse_args()


# ── helpers ───────────────────────────────────────────────
def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _style_axes_like_grid(ax):
    ax.tick_params(axis="both", which="major", width=1.4, length=7)
    ax.minorticks_off()


def _find_runs_root(start: Path) -> Path:
    if (start / "runs").is_dir():
        return start / "runs"
    if start.name == "runs":
        return start
    raise RuntimeError("run inside repo root or runs/")


def _safe_bool(df, col):
    return df[col].fillna(False).astype(bool) if col in df else pd.Series(False, index=df.index)


def _safe_str(df, col):
    return df[col].fillna("").astype(str) if col in df else pd.Series("", index=df.index)


def _pct(a, b):
    return 100 * a / b if b > 0 else 0


def _normalize(cs):
    if cs == "trigonal":
        return "rhombohedral"
    return cs


def _crysys_hist(cs_list):
    counts = pd.Series(cs_list).value_counts()
    total = counts.sum()
    return np.array([(counts.get(cs, 0) / total) * 100 for cs in CRYSYS_ORDER])


def _build_filter(df, max_val):
    mask = df["rruff_a"].notna()
    mask &= df["rruff_b"].notna()
    mask &= df["rruff_c"].notna()
    mask &= (df["rruff_a"] <= max_val)
    mask &= (df["rruff_b"] <= max_val)
    mask &= (df["rruff_c"] <= max_val)
    return mask


# ── data loading ───────────────────────────────────────────────
def _load(method_key, max_val, symprec):
    runs = _find_runs_root(Path.cwd())
    method_dir = runs / method_key

    csv = next(method_dir.glob("**/results.csv"))
    df = pd.read_csv(csv)

    mask = _build_filter(df, max_val)
    err = _safe_str(df, "error").str.strip() != ""

    pm = _safe_bool(df, "pm_success")

    valid = mask & (~err)

    match_info = {
        "pm_success_n": int((pm & valid).sum()),
        "filtered_valid_rows": int(valid.sum()),
        "unmatched_for_pie": int(valid.sum() - (pm & valid).sum())
    }

    if "rruff_crystal_system" in df:
        cs = df.loc[mask, "rruff_crystal_system"].map(_normalize)
        cs = [c for c in cs if c in CRYSYS_ORDER]
    else:
        cs = []

    return cs, match_info, runs


def _get_reference(loaded):
    for k, name, c, cs, info in loaded:
        if k == "no_refinement":
            return name, info
    return loaded[0][1], loaded[0][4]


# ── plotting ───────────────────────────────────────────────
def plot_side_by_side(loaded, out_dir, max_val):
    name_ref, info = _get_reference(loaded)

    matched = info["pm_success_n"]
    unmatched = info["unmatched_for_pie"]
    n = info["filtered_valid_rows"]

    x = np.arange(len(CRYSYS_ORDER))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(18, 7),
        gridspec_kw={"width_ratios": [1.8, 1]},
        constrained_layout=True
    )

    # left: overlay
    for k, name, color, cs, _ in loaded:
        ax1.plot(
            x,
            _crysys_hist(cs),
            marker="o",
            linewidth=2.8,
            label=f"{name} (n={len(cs)})",
            color=color,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels(CRYSYS_LABELS, rotation=25)
    ax1.set_ylabel("% of filtered structures")
    ax1.set_title(f"Crystal Systems (a,b,c ≤ {max_val} Å)")
    ax1.legend(frameon=False)
    _style_axes_like_grid(ax1)

    # right: pie
    ax2.pie(
        [matched, unmatched],
        labels=[f"Matched\n{matched}", f"Unmatched\n{unmatched}"],
        colors=[MATCH_GREEN, UNMATCHED_GRAY],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax2.set_title(f"Pattern Matching (n={n})")

    out = out_dir / f"combined_plot_max_{str(max_val).replace('.', 'p')}.png"
    plt.savefig(out, dpi=220)
    plt.close()

    print(f"✓ wrote {out}")


# ── main ───────────────────────────────────────────────
def main():
    args = parse_args()

    loaded = []
    runs = None

    for k, name, color in METHODS:
        cs, info, runs = _load(k, args.max_val, args.symprec)
        loaded.append((k, name, color, cs, info))

    out = Path(args.output_dir) if args.output_dir else runs / "refinement_summary_figures"
    _ensure_dir(out)

    plot_side_by_side(loaded, out, args.max_val)

    print(f"\n✓ done → {out}")


if __name__ == "__main__":
    main()
