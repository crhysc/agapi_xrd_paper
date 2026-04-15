import sys
from pathlib import Path
from string import ascii_lowercase

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib as mpl
mpl.use("Agg")
mpl.rcParams["font.family"] = "serif"

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ───────────────────────── config ─────────────────────────
REFINEMENT_DIRS = {
    "no_refinement": "No refinement",
    "gsas2": "GSAS-II",
    "bmgn": "BMGN",
}

PAIR_ORDER = [
    ("no_refinement", "gsas2"),
    ("gsas2", "bmgn"),
    ("bmgn", "no_refinement"),
]

PARAMS = ["a", "b", "c", "alpha", "beta", "gamma"]

PARAM_LABELS = {
    "a": r"$a$ ($\AA$)",
    "b": r"$b$ ($\AA$)",
    "c": r"$c$ ($\AA$)",
    "alpha": r"$\alpha$ ($^\circ$)",
    "beta": r"$\beta$ ($^\circ$)",
    "gamma": r"$\gamma$ ($^\circ$)",
}

# panel layout: each comparison occupies 2 rows × 3 cols
BLOCK_LAYOUT = [
    ["a", "b", "c"],
    ["alpha", "beta", "gamma"],
]

PANEL_LETTERS = list(ascii_lowercase[:18])


# ───────────────────────── helpers ─────────────────────────
def find_runs_root(start: Path) -> Path:
    """Allow running from project root or from inside runs/."""
    if (start / "runs").is_dir():
        return start / "runs"
    if start.name == "runs" and start.is_dir():
        return start
    raise FileNotFoundError(
        "Could not find a runs/ directory. Run this from the project root "
        "or from inside runs/."
    )


def newest_results_csv(run_dir: Path) -> Path:
    """Prefer newest timestamped rruff_results_*/results.csv; fall back to *_results.csv."""
    timestamped = sorted(
        run_dir.glob("rruff_results_*/results.csv"),
        key=lambda p: p.parent.name
    )
    if timestamped:
        return timestamped[-1]

    fallback = sorted(run_dir.glob("*_results.csv"), key=lambda p: p.name)
    if fallback:
        return fallback[-1]

    raise FileNotFoundError(f"No results CSV found in {run_dir}")


def choose_id_column(df: pd.DataFrame) -> str:
    for col in ["rruff_id", "mineral_name"]:
        if col in df.columns:
            return col
    raise KeyError("Could not find an ID column. Expected 'rruff_id' or 'mineral_name'.")


def load_refinement_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    id_col = choose_id_column(df)

    keep_cols = [id_col] + [f"pred_{p}" for p in PARAMS]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise KeyError(f"{csv_path} is missing columns: {missing}")

    out = df[keep_cols].copy()
    out = out.rename(columns={id_col: "__id__"})
    out = out.dropna(subset=["__id__"]).drop_duplicates(subset="__id__", keep="first")
    return out


def get_common_arrays(df_left: pd.DataFrame, df_right: pd.DataFrame, param: str):
    col = f"pred_{param}"
    merged = df_left[["__id__", col]].merge(
        df_right[["__id__", col]],
        on="__id__",
        how="inner",
        suffixes=("_left", "_right"),
    ).dropna()

    x = merged[f"{col}_left"].to_numpy(dtype=float)
    y = merged[f"{col}_right"].to_numpy(dtype=float)
    return x, y, len(merged)


def make_bins(x: np.ndarray, y: np.ndarray, param: str):
    vals = np.concatenate([x, y]).astype(float)
    lo = float(np.min(vals))
    hi = float(np.max(vals))

    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("Non-finite values encountered while constructing bins.")

    if lo == hi:
        lo -= 1.0
        hi += 1.0

    span = hi - lo
    margin = 0.05 * span if span > 0 else 1.0
    lo -= margin
    hi += margin

    n_bins = 36 if param in {"a", "b", "c"} else 24
    return np.linspace(lo, hi, n_bins + 1)


def jsd_from_histograms(x: np.ndarray, y: np.ndarray, bins, eps: float = 1e-10) -> float:
    hx, _ = np.histogram(x, bins=bins, density=False)
    hy, _ = np.histogram(y, bins=bins, density=False)

    p = hx.astype(np.float64) + eps
    q = hy.astype(np.float64) + eps
    p /= p.sum()
    q /= q.sum()

    m = 0.5 * (p + q)
    jsd = 0.5 * stats.entropy(p, m) + 0.5 * stats.entropy(q, m)
    return float(jsd)


def overlay_hist(ax, x, y, bins, xlabel, title, label_left, label_right, show_legend=False):
    w_x = np.ones_like(x, dtype=float) / max(1, len(x)) * 100.0
    w_y = np.ones_like(y, dtype=float) / max(1, len(y)) * 100.0

    ax.hist(x, bins=bins, weights=w_x, alpha=0.6, color="tab:blue", label=label_left)
    ax.hist(y, bins=bins, weights=w_y, alpha=0.6, color="plum", label=label_right)

    ax.set_xlabel(xlabel)
    ax.set_title(title)

    if show_legend:
        ax.legend(frameon=False, fontsize=13)

    return ax


# ───────────────────────── load results ─────────────────────────
ROOT = Path.cwd()
RUNS = find_runs_root(ROOT)
print(f"DEBUG: Using runs directory: {RUNS}", file=sys.stderr)

method_frames = {}
method_csvs = {}

for key, pretty in REFINEMENT_DIRS.items():
    run_dir = RUNS / key
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Missing run directory: {run_dir}")

    csv_path = newest_results_csv(run_dir)
    print(f"DEBUG: {pretty:<13s} -> {csv_path}", file=sys.stderr)

    method_frames[key] = load_refinement_df(csv_path)
    method_csvs[key] = csv_path


# ───────────────────────── plotting ─────────────────────────
fig = plt.figure(figsize=(16, 34))
the_grid = GridSpec(6, 3, figure=fig)

summary_rows = []
panel_idx = 0

block_titles = [
    "No refinement vs GSAS-II",
    "GSAS-II vs BMGN",
    "BMGN vs No refinement",
]

for block_num, (left_key, right_key) in enumerate(PAIR_ORDER):
    left_name = REFINEMENT_DIRS[left_key]
    right_name = REFINEMENT_DIRS[right_key]

    df_left = method_frames[left_key]
    df_right = method_frames[right_key]

    # add a centered block title above each 2×3 section
    y_text = {0: 0.985, 1: 0.655, 2: 0.325}[block_num]
    fig.text(
        0.5, y_text,
        block_titles[block_num],
        ha="center", va="top",
        fontsize=24
    )

    for local_row, row_params in enumerate(BLOCK_LAYOUT):
        global_row = 2 * block_num + local_row

        for col, param in enumerate(row_params):
            ax = plt.subplot(the_grid[global_row, col])

            x, y, n_common = get_common_arrays(df_left, df_right, param)

            if n_common < 2:
                ax.text(
                    0.5, 0.5,
                    "Insufficient\ncommon data",
                    ha="center", va="center",
                    fontsize=16
                )
                ax.set_title(f"({PANEL_LETTERS[panel_idx]})")
                ax.set_xlabel(PARAM_LABELS[param])
                if col == 0:
                    ax.set_ylabel("Materials dist.")
                panel_idx += 1
                summary_rows.append({
                    "comparison": f"{left_name} vs {right_name}",
                    "left_method": left_name,
                    "right_method": right_name,
                    "parameter": param,
                    "jsd": np.nan,
                    "n_common": n_common,
                    "left_csv": str(method_csvs[left_key]),
                    "right_csv": str(method_csvs[right_key]),
                })
                continue

            bins = make_bins(x, y, param)
            jsd = jsd_from_histograms(x, y, bins=bins)

            title = (
                f"({PANEL_LETTERS[panel_idx]}) "
                f"{param}   JSD={jsd:.4f}"
            )

            overlay_hist(
                ax=ax,
                x=x,
                y=y,
                bins=bins,
                xlabel=PARAM_LABELS[param],
                title=title,
                label_left=left_name,
                label_right=right_name,
                show_legend=(local_row == 0 and col == 0),
            )

            if col == 0:
                ax.set_ylabel("Materials dist.")

            summary_rows.append({
                "comparison": f"{left_name} vs {right_name}",
                "left_method": left_name,
                "right_method": right_name,
                "parameter": param,
                "jsd": jsd,
                "n_common": n_common,
                "left_csv": str(method_csvs[left_key]),
                "right_csv": str(method_csvs[right_key]),
            })

            panel_idx += 1


# ───────────────────────── final layout / save ─────────────────────────
plt.tight_layout(rect=[0.03, 0.03, 0.98, 0.97])
fig.subplots_adjust(hspace=0.75, wspace=0.28)

out_png = RUNS / "refinement_pairwise_jsd_18panel.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close(fig)

summary_df = pd.DataFrame(summary_rows)
summary_csv = RUNS / "refinement_pairwise_jsd_summary.csv"
summary_df.to_csv(summary_csv, index=False)

print(f"✓ saved {out_png}")
print(f"✓ wrote {summary_csv}")
