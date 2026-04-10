import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Force a non-interactive backend in case you're headless
import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams["font.family"] = "serif"

import matplotlib.pyplot as plt


# ───────────────────────── config / labels ─────────────────────────
REFINEMENT_DIRS = {
    "no_refinement": "No refinement",
    "bmgn": "BMGN",
    "bmgn_alignnff": "BMGN + ALIGNN-FF",
    "gsas2": "GSAS-II",
}

PARAMS = ["a", "b", "c", "alpha", "beta", "gamma"]

ax_label_map = {
    "a": r"$a$",
    "b": r"$b$",
    "c": r"$c$",
    "alpha": r"$\alpha$",
    "beta": r"$\beta$",
    "gamma": r"$\gamma$",
}

length_params = ["a", "b", "c"]
angle_params = ["alpha", "beta", "gamma"]

# Warm colors for lengths; cool colors for angles
param_colors = {
    "a": "#E76F51",
    "b": "#F4A261",
    "c": "#E9C46A",
    "alpha": "#457B9D",
    "beta": "#2A9D8F",
    "gamma": "#577590",
}


def find_runs_root(start: Path) -> Path:
    """Allow running either from project root or from inside runs/."""
    if (start / "runs").is_dir():
        return start / "runs"
    if start.name == "runs" and start.is_dir():
        return start
    raise FileNotFoundError(
        "Could not find a runs/ directory. Run this from the project root "
        "or from inside runs/."
    )


def newest_results_csv(run_dir: Path) -> Path:
    """
    Prefer the newest timestamped rruff_results_*/results.csv.
    Fall back to *_results.csv if needed.
    """
    timestamped = sorted(
        run_dir.glob("rruff_results_*/results.csv"),
        key=lambda p: p.parent.name
    )
    if timestamped:
        return timestamped[-1]

    fallback = sorted(
        run_dir.glob("*_results.csv"),
        key=lambda p: p.name
    )
    if fallback:
        return fallback[-1]

    raise FileNotFoundError(f"No results CSV found in {run_dir}")


def compute_mae(df: pd.DataFrame, param: str):
    """Return (mae, n_valid) for one lattice parameter."""
    exp_col = f"rruff_{param}"
    pred_col = f"pred_{param}"

    if exp_col not in df.columns or pred_col not in df.columns:
        return np.nan, 0

    sub = df[[exp_col, pred_col]].dropna()
    if sub.empty:
        return np.nan, 0

    mae = (sub[pred_col] - sub[exp_col]).abs().mean()
    return float(mae), int(len(sub))


def build_summary(runs_root: Path) -> pd.DataFrame:
    rows = []
    for run_key, pretty_name in REFINEMENT_DIRS.items():
        run_dir = runs_root / run_key
        if not run_dir.is_dir():
            print(f"WARNING: Missing directory {run_dir} — skipped", file=sys.stderr)
            continue

        csv_path = newest_results_csv(run_dir)
        print(f"DEBUG: {pretty_name} -> {csv_path}", file=sys.stderr)

        df = pd.read_csv(csv_path)

        rec = {
            "refinement_key": run_key,
            "refinement_name": pretty_name,
            "csv_path": str(csv_path),
        }

        for p in PARAMS:
            mae, n_valid = compute_mae(df, p)
            rec[f"MAE.{p}"] = mae
            rec[f"N.{p}"] = n_valid

        rows.append(rec)

    if not rows:
        print("ERROR: No refinement result CSVs found.", file=sys.stderr)
        sys.exit(1)

    summary = pd.DataFrame(rows)
    summary["refinement_key"] = pd.Categorical(
        summary["refinement_key"],
        categories=list(REFINEMENT_DIRS.keys()),
        ordered=True,
    )
    summary = summary.sort_values("refinement_key").reset_index(drop=True)
    return summary


def summary_to_mae_df(summary: pd.DataFrame) -> pd.DataFrame:
    mae_cols = [f"MAE.{p}" for p in PARAMS]
    return (
        summary.set_index("refinement_name")[mae_cols]
        .rename(columns=lambda c: ax_label_map[c.split(".")[-1]])
    )


def style_axes(ax, ax2, labels, title):
    ax.set_xlabel("", fontsize=16)
    ax.set_ylabel("Mean Absolute Error — lengths (Å)", fontsize=16)
    ax2.set_ylabel("Mean Absolute Error — angles (°)", fontsize=16)
    ax.set_title(title, fontsize=22)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax2.tick_params(axis="y", labelsize=14)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def plot_mae_figure(
    summary: pd.DataFrame,
    output_path: Path,
    title: str,
    figsize=(13, 8),
):
    mae_df = summary_to_mae_df(summary)

    labels = list(mae_df.index)
    x = np.arange(len(labels))
    width = 0.12

    fig, ax = plt.subplots(figsize=figsize)
    ax2 = ax.twinx()
    ax2.patch.set_alpha(0.0)

    display_len = [ax_label_map[p] for p in length_params]
    display_ang = [ax_label_map[p] for p in angle_params]

    offsets = {
        display_len[0]: -2.5 * width,
        display_len[1]: -1.5 * width,
        display_len[2]: -0.5 * width,
        display_ang[0]:  0.5 * width,
        display_ang[1]:  1.5 * width,
        display_ang[2]:  2.5 * width,
    }

    # Lengths on left y-axis
    for p, col in zip(length_params, display_len):
        ax.bar(
            x + offsets[col],
            mae_df[col].values,
            width=width,
            edgecolor="k",
            color=param_colors[p],
            label=f"{col} (Å)",
            zorder=3,
        )

    # Angles on right y-axis
    for p, col in zip(angle_params, display_ang):
        ax2.bar(
            x + offsets[col],
            mae_df[col].values,
            width=width,
            edgecolor="k",
            color=param_colors[p],
            label=f"{col} (°)",
            zorder=3,
        )

    len_max = (
        np.nanmax(mae_df[display_len].values)
        if not mae_df[display_len].isna().all().all()
        else 1.0
    )
    ang_max = (
        np.nanmax(mae_df[display_ang].values)
        if not mae_df[display_ang].isna().all().all()
        else 1.0
    )

    ax.set_ylim(0, 1.15 * len_max if len_max > 0 else 1.0)
    ax2.set_ylim(0, 1.15 * ang_max if ang_max > 0 else 1.0)

    style_axes(ax, ax2, labels, title=title)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(
        h1 + h2,
        l1 + l2,
        title="Lattice Parameter",
        title_fontsize=14,
        fontsize=13,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )

    plt.tight_layout(rect=[0, 0, 0.84, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ───────────────────────── main workflow ───────────────────────────
ROOT = Path.cwd()
RUNS = find_runs_root(ROOT)
print(f"DEBUG: Using runs directory: {RUNS}", file=sys.stderr)

summary = build_summary(RUNS)

summary_path = RUNS / "refinement_mae_summary.csv"
summary.to_csv(summary_path, index=False)
print(f"DEBUG: Wrote summary -> {summary_path}", file=sys.stderr)

# Full comparison figure
plot_all_path = RUNS / "refinement_mae_all6.png"
plot_mae_figure(
    summary=summary,
    output_path=plot_all_path,
    title="Mean Absolute Error by Refinement Method",
    figsize=(13, 8),
)

# BMGN vs BMGN+ALIGNN-FF figure
bmgn_subset = summary[
    summary["refinement_key"].isin(["bmgn", "bmgn_alignnff"])
].copy()

plot_bmgn_path = RUNS / "refinement_mae_bmgn_vs_bmgn_alignnff.png"
plot_mae_figure(
    summary=bmgn_subset,
    output_path=plot_bmgn_path,
    title="BMGN vs. BMGN + ALIGNN-FF\nMean Absolute Error",
    figsize=(10, 7),
)

# ───────────────────────── console summary ─────────────────────────
display_cols = (
    ["refinement_name"] +
    [f"MAE.{p}" for p in PARAMS] +
    [f"N.{p}" for p in PARAMS] +
    ["csv_path"]
)

print("\nMAE summary:")
print(summary[display_cols].to_string(index=False))
print(f"\nSaved full plot:        {plot_all_path}")
print(f"Saved BMGN comparison:  {plot_bmgn_path}")
print(f"Saved summary:          {summary_path}")
print("DEBUG: All done.", file=sys.stderr)
