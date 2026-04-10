#!/usr/bin/env python3
"""
analyse_filtered_rruff.py

Command-line analysis script for AGAPI-XRD / RRUFF results.

Example:
    python analyse_filtered_rruff.py \
        --input runs/no_refinement/results.csv \
        --output-dir runs/no_refinement/analysis

It will:
  - load the CSV
  - filter rows by valid lattice parameters and max RRUFF cell length
  - print summary metrics to stdout
  - save plots into the output directory
  - save a JSON summary into the output directory
  - save a TOML summary into the output directory
"""

import os
import json
import math
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import gaussian_kde, entropy
from scipy.spatial.distance import jensenshannon

import matplotlib
matplotlib.use("Agg")  # headless / bash / cluster-safe
import matplotlib.pyplot as plt


PARAMS = ["a", "b", "c", "alpha", "beta", "gamma"]
ALL_PARAMS = PARAMS + ["volume"]
UNITS = {
    "a": "Å",
    "b": "Å",
    "c": "Å",
    "alpha": "°",
    "beta": "°",
    "gamma": "°",
    "volume": "Å^3",
}

# Explicit plot colors so reruns remain visually stable
EXP_COLOR = "#2F5D8A"      # elegant blue
PRED_COLOR = "#E76FAD"     # elegant pink


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze filtered RRUFF results and save plots."
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to input results.csv file",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory where plots and summary files will be saved",
    )
    parser.add_argument(
        "--max-val",
        type=float,
        default=15.0,
        help="Maximum allowed RRUFF a, b, c value for filtering (default: 15.0)",
    )
    return parser.parse_args()


def ensure_output_dir(path):
    os.makedirs(path, exist_ok=True)


def load_and_filter(csv_path, val):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} entries from {csv_path}")

    mask = df["rruff_a"].notna() & df["pred_a"].notna()
    for p in PARAMS:
        mask &= df[f"rruff_{p}"].notna() & df[f"pred_{p}"].notna()

    mask &= (df["rruff_a"] < val) & (df["rruff_b"] < val) & (df["rruff_c"] < val)

    df_clean = df[mask].copy()
    print(f"Filtered: {len(df_clean)}/{len(df)} entries (RRUFF a,b,c < {val} Å)")

    if "best_method" in df_clean.columns:
        print("\nMethod breakdown:")
        for m, c in df_clean["best_method"].value_counts().items():
            print(f"  {m}: {c}")

    if "rruff_crystal_system" in df_clean.columns:
        print("\nCrystal system breakdown:")
        for cs, c in df_clean["rruff_crystal_system"].value_counts().items():
            print(f"  {cs}: {c}")

    return df, df_clean


def compute_metrics(df_clean):
    print(
        f"\n{'Param':>8s}  {'MAE':>8s}  {'RMSE':>8s}  {'R²':>8s}  "
        f"{'Mean Δ':>9s}  {'Med |Δ|':>8s}  {'N':>5s}"
    )
    print("─" * 60)

    metrics = {}

    for p in ALL_PARAMS:
        exp_col = f"rruff_{p}"
        pred_col = f"pred_{p}"

        if exp_col not in df_clean.columns or pred_col not in df_clean.columns:
            continue

        exp = df_clean[exp_col].values.astype(float)
        pred = df_clean[pred_col].values.astype(float)
        valid = ~(np.isnan(exp) | np.isnan(pred))
        exp, pred = exp[valid], pred[valid]

        if len(exp) > 1:
            delta = pred - exp
            mae = mean_absolute_error(exp, pred)
            rmse = np.sqrt(np.mean(delta ** 2))
            r2 = r2_score(exp, pred) if np.std(exp) > 1e-10 else float("nan")

            metrics[p] = {
                "exp": exp,
                "pred": pred,
                "delta": delta,
                "mae": float(mae),
                "rmse": float(rmse),
                "r2": float(r2) if not np.isnan(r2) else None,
                "mean_delta": float(np.mean(delta)),
                "median_abs": float(np.median(np.abs(delta))),
                "n": int(len(exp)),
            }

            r2_print = f"{r2:8.4f}" if not np.isnan(r2) else f"{'nan':>8s}"
            print(
                f"{p:>8s}  {mae:8.4f}  {rmse:8.4f}  {r2_print}  "
                f"{np.mean(delta):+9.4f}  {np.median(np.abs(delta)):8.4f}  {len(exp):5d}"
            )

    return metrics


def compute_kld_jsd(metrics):
    print(f"\n{'Param':>8s}  {'JSD':>10s}  {'KLD(e||p)':>12s}")
    print("─" * 35)

    kld_results = {}

    for p in ALL_PARAMS:
        if p not in metrics:
            continue

        exp = metrics[p]["exp"]
        pred = metrics[p]["pred"]

        lo = min(exp.min(), pred.min())
        hi = max(exp.max(), pred.max())
        if lo == hi:
            lo -= 1
            hi += 1

        n_bins = min(50, max(10, len(exp) // 5))
        bins = np.linspace(lo, hi, n_bins + 1)

        h_e, _ = np.histogram(exp, bins=bins, density=True)
        h_p, _ = np.histogram(pred, bins=bins, density=True)

        eps = 1e-10
        h_e = h_e + eps
        h_p = h_p + eps
        h_e = h_e / h_e.sum()
        h_p = h_p / h_p.sum()

        kld = entropy(h_e, h_p)
        jsd = jensenshannon(h_e, h_p)

        kld_results[p] = {
            "kld": float(kld),
            "jsd": float(jsd),
        }

        print(f"{p:>8s}  {jsd:10.6f}  {kld:12.6f}")

    return kld_results


def save_parity_plot(metrics, output_dir, n, val):
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for i, p in enumerate(ALL_PARAMS):
        ax = axes[i]
        if p not in metrics:
            ax.set_visible(False)
            continue

        exp = metrics[p]["exp"]
        pred = metrics[p]["pred"]
        mae = metrics[p]["mae"]
        r2 = metrics[p]["r2"]

        ax.scatter(exp, pred, s=30, alpha=0.6, edgecolors="white", linewidths=0.3)

        all_vals = np.concatenate([exp, pred])
        lo, hi = all_vals.min() * 0.95, all_vals.max() * 1.05
        if lo == hi:
            lo -= 1
            hi += 1

        ax.plot([lo, hi], [lo, hi], "--", alpha=0.6, linewidth=1.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel(f"RRUFF {p} ({UNITS[p]})", fontsize=10)
        ax.set_ylabel(f"Predicted {p} ({UNITS[p]})", fontsize=10)
        ax.set_title(
            f"{p}  MAE={mae:.3f}  R²={r2 if r2 is not None else float('nan'):.3f}",
            fontsize=11,
        )
        ax.grid(True, alpha=0.15)

    for j in range(len(ALL_PARAMS), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"RRUFF vs Predicted Lattice Parameters (n={n}, RRUFF a,b,c < {val} Å)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    outpath = os.path.join(output_dir, "parity_filtered.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {outpath}")


def save_distribution_plot(metrics, kld_results, output_dir, n, val):
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for i, p in enumerate(ALL_PARAMS):
        ax = axes[i]
        if p not in metrics:
            ax.set_visible(False)
            continue

        exp = metrics[p]["exp"]
        pred = metrics[p]["pred"]
        jsd = kld_results.get(p, {}).get("jsd", float("nan"))

        lo = min(exp.min(), pred.min())
        hi = max(exp.max(), pred.max())
        margin = 0.05 * (hi - lo + 1)
        bins = np.linspace(lo - margin, hi + margin, min(40, max(10, len(exp) // 5)))

        ax.hist(
            exp,
            bins=bins,
            alpha=0.60,
            density=True,
            label="RRUFF (exp)",
            color=EXP_COLOR,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.hist(
            pred,
            bins=bins,
            alpha=0.60,
            density=True,
            label="Predicted",
            color=PRED_COLOR,
            edgecolor="white",
            linewidth=0.5,
        )

        try:
            x_grid = np.linspace(lo - margin, hi + margin, 300)
            ax.plot(
                x_grid,
                gaussian_kde(exp)(x_grid),
                linewidth=2.2,
                color=EXP_COLOR,
                label="KDE exp",
            )
            ax.plot(
                x_grid,
                gaussian_kde(pred)(x_grid),
                linewidth=2.2,
                color=PRED_COLOR,
                label="KDE pred",
            )
        except Exception:
            pass

        ax.set_xlabel(f"{p} ({UNITS[p]})", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"{p}  JSD={jsd:.4f}", fontsize=11)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.15)

    for j in range(len(ALL_PARAMS), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Distribution: RRUFF (exp) vs Predicted (n={n}, a,b,c < {val} Å)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    outpath = os.path.join(output_dir, "distribution_filtered.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {outpath}")


def save_error_distribution_plot(metrics, output_dir, n, val):
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for i, p in enumerate(ALL_PARAMS):
        ax = axes[i]
        if p not in metrics:
            ax.set_visible(False)
            continue

        delta = metrics[p]["delta"]
        mean_d = metrics[p]["mean_delta"]

        ax.hist(
            delta,
            bins=max(10, len(delta) // 5),
            alpha=0.8,
            density=True,
            edgecolor="white",
        )
        ax.axvline(0, linestyle="--", linewidth=1.5, alpha=0.7)
        ax.axvline(mean_d, linewidth=1.5, alpha=0.7, label=f"mean={mean_d:+.3f}")

        try:
            x_grid = np.linspace(delta.min(), delta.max(), 200)
            ax.plot(x_grid, gaussian_kde(delta)(x_grid), linewidth=2)
        except Exception:
            pass

        std_d = np.std(delta)
        ax.set_xlabel(f"Δ{p} ({UNITS[p]})", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"Δ{p}  μ={mean_d:+.3f} σ={std_d:.3f}", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15)

    for j in range(len(ALL_PARAMS), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Error Distributions: Δ = Predicted − RRUFF (n={n}, a,b,c < {val} Å)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    outpath = os.path.join(output_dir, "error_dist_filtered.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {outpath}")


def save_error_by_method_plot(df_clean, output_dir, n, val):
    if "best_method" not in df_clean.columns:
        return

    methods = df_clean["best_method"].dropna().unique()
    if len(methods) < 2:
        return

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for i, p in enumerate(ALL_PARAMS):
        ax = axes[i]

        for method in sorted(methods):
            subset = df_clean[df_clean["best_method"] == method]
            if len(subset) < 2:
                continue

            exp_col = f"rruff_{p}"
            pred_col = f"pred_{p}"
            if exp_col not in subset.columns or pred_col not in subset.columns:
                continue

            exp_s = subset[exp_col].values.astype(float)
            pred_s = subset[pred_col].values.astype(float)
            valid = ~(np.isnan(exp_s) | np.isnan(pred_s))
            delta = pred_s[valid] - exp_s[valid]

            if len(delta) < 2:
                continue

            label_short = (
                method.replace("pattern_matching (elements)", "PM (elements)")
                .replace("pattern_matching", "PM")
                .replace("diffractgpt", "DG")
            )

            ax.hist(
                delta,
                bins=max(8, len(delta) // 5),
                alpha=0.5,
                density=True,
                label=f"{label_short} (n={len(delta)}, MAE={np.mean(np.abs(delta)):.3f})",
            )

        ax.axvline(0, linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlabel(f"Δ{p} ({UNITS[p]})", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"Δ{p} by method", fontsize=11)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.15)

    for j in range(len(ALL_PARAMS), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Error by Method: PM vs DG (n={n}, a,b,c < {val} Å)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    outpath = os.path.join(output_dir, "error_by_method_filtered.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {outpath}")


def save_error_by_crystal_plot(df_clean, output_dir, n, val):
    if "rruff_crystal_system" not in df_clean.columns:
        return

    cs_counts = df_clean["rruff_crystal_system"].value_counts()
    cs_with_data = cs_counts[cs_counts >= 3].index.tolist()

    if not cs_with_data:
        return

    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for i, p in enumerate(ALL_PARAMS):
        ax = axes[i]
        data_by_cs = []
        labels = []

        for cs in cs_with_data:
            subset = df_clean[df_clean["rruff_crystal_system"] == cs]

            exp_col = f"rruff_{p}"
            pred_col = f"pred_{p}"
            if exp_col not in subset.columns or pred_col not in subset.columns:
                continue

            delta = (subset[pred_col] - subset[exp_col]).dropna().values
            if len(delta) >= 2:
                data_by_cs.append(delta)
                labels.append(f"{cs}\n(n={len(delta)})")

        if data_by_cs:
            ax.boxplot(data_by_cs, labels=labels, patch_artist=True)
            ax.axhline(0, linestyle="--", linewidth=1, alpha=0.5)

        ax.set_ylabel(f"Δ{p} ({UNITS[p]})", fontsize=10)
        ax.set_title(f"Δ{p} by crystal system", fontsize=11)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.grid(True, alpha=0.15, axis="y")

    for j in range(len(ALL_PARAMS), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Error by Crystal System (n={n}, a,b,c < {val} Å)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    outpath = os.path.join(output_dir, "error_by_crystal_filtered.png")
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Saved {outpath}")


def build_summary(df_clean, metrics, kld_results, input_csv, n_total, n_filtered, max_val):
    summary = {
        "input_csv": input_csv,
        "n_total": int(n_total),
        "n_filtered": int(n_filtered),
        "max_val_filter": float(max_val),
        "metrics": {},
    }

    if "best_method" in df_clean.columns:
        summary["method_breakdown"] = {
            str(method): int(count)
            for method, count in df_clean["best_method"].value_counts().items()
        }

    if "rruff_crystal_system" in df_clean.columns:
        summary["crystal_system_breakdown"] = {
            str(cs): int(count)
            for cs, count in df_clean["rruff_crystal_system"].value_counts().items()
        }

    for p in ALL_PARAMS:
        if p not in metrics:
            continue

        r2_val = metrics[p]["r2"]
        summary["metrics"][p] = {
            "unit": UNITS[p],
            "mae": float(metrics[p]["mae"]),
            "rmse": float(metrics[p]["rmse"]),
            "r2": float(r2_val) if r2_val is not None else float("nan"),
            "mean_delta": float(metrics[p]["mean_delta"]),
            "median_abs": float(metrics[p]["median_abs"]),
            "n": int(metrics[p]["n"]),
            "jsd": float(kld_results.get(p, {}).get("jsd", float("nan"))),
            "kld": float(kld_results.get(p, {}).get("kld", float("nan"))),
        }

    return summary


def _toml_escape_string(value):
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )


def _toml_format_scalar(value):
    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)

    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return repr(value)

    if value is None:
        return "nan"

    return f"\"{_toml_escape_string(value)}\""


def _toml_dump_dict(data, parent_key=""):
    lines = []
    scalar_items = []
    dict_items = []

    for key, value in data.items():
        if isinstance(value, dict):
            dict_items.append((key, value))
        else:
            scalar_items.append((key, value))

    if parent_key:
        lines.append(f"[{parent_key}]")

    for key, value in scalar_items:
        lines.append(f"{key} = {_toml_format_scalar(value)}")

    if scalar_items and dict_items:
        lines.append("")

    for i, (key, value) in enumerate(dict_items):
        child_key = f"{parent_key}.{key}" if parent_key else key
        child_lines = _toml_dump_dict(value, child_key)
        lines.extend(child_lines)
        if i != len(dict_items) - 1:
            lines.append("")

    return lines


def save_summary_json(summary, output_dir):
    outpath = os.path.join(output_dir, "analysis_summary.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved {outpath}")


def save_summary_toml(summary, output_dir):
    outpath = os.path.join(output_dir, "analysis_summary.toml")
    toml_text = "\n".join(_toml_dump_dict(summary)) + "\n"

    with open(outpath, "w", encoding="utf-8") as f:
        f.write(toml_text)

    print(f"✓ Saved {outpath}")


def print_summary(metrics, kld_results, n, val, output_dir):
    print(f"\n{'═' * 70}")
    print(f"SUMMARY (n={n}, RRUFF a,b,c < {val} Å)")
    print(f"{'═' * 70}")
    print(
        f"{'Param':>8s}  {'MAE':>8s}  {'RMSE':>8s}  {'R²':>8s}  "
        f"{'JSD':>8s}  {'Mean Δ':>9s}  {'Med |Δ|':>8s}"
    )
    print("─" * 70)

    for p in ALL_PARAMS:
        if p in metrics and p in kld_results:
            m = metrics[p]
            k = kld_results[p]
            r2_val = m["r2"]
            r2_print = f"{r2_val:8.4f}" if r2_val is not None else f"{'nan':>8s}"
            print(
                f"{p:>8s}  {m['mae']:8.4f}  {m['rmse']:8.4f}  {r2_print}  "
                f"{k['jsd']:8.4f}  {m['mean_delta']:+9.4f}  {m['median_abs']:8.4f}"
            )

    print("\nOutput files:")
    print(f"  {os.path.join(output_dir, 'parity_filtered.png')}")
    print(f"  {os.path.join(output_dir, 'distribution_filtered.png')}")
    print(f"  {os.path.join(output_dir, 'error_dist_filtered.png')}")
    print(f"  {os.path.join(output_dir, 'error_by_method_filtered.png')}")
    print(f"  {os.path.join(output_dir, 'error_by_crystal_filtered.png')}")
    print(f"  {os.path.join(output_dir, 'analysis_summary.json')}")
    print(f"  {os.path.join(output_dir, 'analysis_summary.toml')}")


def main():
    args = parse_args()
    ensure_output_dir(args.output_dir)

    df, df_clean = load_and_filter(args.input, args.max_val)
    n = len(df_clean)

    metrics = compute_metrics(df_clean)
    kld_results = compute_kld_jsd(metrics)

    save_parity_plot(metrics, args.output_dir, n, args.max_val)
    save_distribution_plot(metrics, kld_results, args.output_dir, n, args.max_val)
    save_error_distribution_plot(metrics, args.output_dir, n, args.max_val)
    save_error_by_method_plot(df_clean, args.output_dir, n, args.max_val)
    save_error_by_crystal_plot(df_clean, args.output_dir, n, args.max_val)

    summary = build_summary(
        df_clean=df_clean,
        metrics=metrics,
        kld_results=kld_results,
        input_csv=args.input,
        n_total=len(df),
        n_filtered=len(df_clean),
        max_val=args.max_val,
    )

    save_summary_json(summary, args.output_dir)
    save_summary_toml(summary, args.output_dir)

    print_summary(metrics, kld_results, n, args.max_val, args.output_dir)


if __name__ == "__main__":
    main()
