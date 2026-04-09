"""
analyze_filtered_rruff.py — Parity plots + distribution analysis for small cells

Usage (Jupyter):
    %run analyze_filtered_rruff.py

Or paste into a notebook cell.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import gaussian_kde, entropy
from scipy.spatial.distance import jensenshannon
import matplotlib
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════
CSV_PATH = "rruff_results_20260326_024710/results.csv"
VAL = 15  # max RRUFF a, b, c filter (Å)

params = ["a", "b", "c", "alpha", "beta", "gamma"]
all_params = params + ["volume"]
units = {
    "a": "Å", "b": "Å", "c": "Å",
    "alpha": "°", "beta": "°", "gamma": "°",
    "volume": "ų",
}

# ═══════════════════════════════════════════════════════════════════════════
# Load & filter
# ═══════════════════════════════════════════════════════════════════════════
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} entries from {CSV_PATH}")

mask = df["rruff_a"].notna() & df["pred_a"].notna()
for p in params:
    mask &= df[f"rruff_{p}"].notna() & df[f"pred_{p}"].notna()
mask &= (df["rruff_a"] < VAL) & (df["rruff_b"] < VAL) & (df["rruff_c"] < VAL)

df_clean = df[mask].copy()
n = len(df_clean)
print(f"Filtered: {n}/{len(df)} entries (RRUFF a,b,c < {VAL} Å)")

# Method breakdown
if "best_method" in df_clean.columns:
    print(f"\nMethod breakdown:")
    for m, c in df_clean["best_method"].value_counts().items():
        print(f"  {m}: {c}")

# Crystal system breakdown
if "rruff_crystal_system" in df_clean.columns:
    print(f"\nCrystal system breakdown:")
    for cs, c in df_clean["rruff_crystal_system"].value_counts().items():
        print(f"  {cs}: {c}")

# ═══════════════════════════════════════════════════════════════════════════
# MAE / R² table
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'Param':>8s}  {'MAE':>8s}  {'RMSE':>8s}  {'R²':>8s}  "
      f"{'Mean Δ':>9s}  {'Med |Δ|':>8s}  {'N':>5s}")
print(f"{'─' * 60}")

metrics = {}
for p in all_params:
    exp = df_clean[f"rruff_{p}"].values.astype(float)
    pred = df_clean[f"pred_{p}"].values.astype(float)
    valid = ~(np.isnan(exp) | np.isnan(pred))
    exp, pred = exp[valid], pred[valid]
    if len(exp) > 1:
        delta = pred - exp
        mae = mean_absolute_error(exp, pred)
        rmse = np.sqrt(np.mean(delta ** 2))
        r2 = r2_score(exp, pred) if np.std(exp) > 1e-10 else float("nan")
        metrics[p] = {
            "exp": exp, "pred": pred, "delta": delta,
            "mae": mae, "rmse": rmse, "r2": r2,
            "mean_delta": np.mean(delta), "median_abs": np.median(np.abs(delta)),
        }
        print(
            f"{p:>8s}  {mae:8.4f}  {rmse:8.4f}  {r2:8.4f}  "
            f"{np.mean(delta):+9.4f}  {np.median(np.abs(delta)):8.4f}  {len(exp):5d}"
        )

# ═══════════════════════════════════════════════════════════════════════════
# KLD / JSD
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'Param':>8s}  {'JSD':>10s}  {'KLD(e||p)':>12s}")
print(f"{'─' * 35}")
kld_results = {}
for p in all_params:
    if p not in metrics:
        continue
    exp = metrics[p]["exp"]
    pred = metrics[p]["pred"]
    lo = min(exp.min(), pred.min())
    hi = max(exp.max(), pred.max())
    if lo == hi:
        lo -= 1; hi += 1
    n_bins = min(50, max(10, len(exp) // 5))
    bins = np.linspace(lo, hi, n_bins + 1)
    h_e, _ = np.histogram(exp, bins=bins, density=True)
    h_p, _ = np.histogram(pred, bins=bins, density=True)
    eps = 1e-10
    h_e = h_e + eps; h_p = h_p + eps
    h_e = h_e / h_e.sum(); h_p = h_p / h_p.sum()
    kld = entropy(h_e, h_p)
    jsd = jensenshannon(h_e, h_p)
    kld_results[p] = {"kld": kld, "jsd": jsd}
    print(f"{p:>8s}  {jsd:10.6f}  {kld:12.6f}")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 1: Parity plots
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.flatten()

for i, p in enumerate(all_params):
    ax = axes[i]
    if p not in metrics:
        ax.set_visible(False); continue
    exp = metrics[p]["exp"]
    pred = metrics[p]["pred"]
    mae = metrics[p]["mae"]
    r2 = metrics[p]["r2"]

    ax.scatter(exp, pred, c="#3b82f6", s=30, alpha=0.6,
               edgecolors="white", lw=0.3)
    all_vals = np.concatenate([exp, pred])
    lo, hi = all_vals.min() * 0.95, all_vals.max() * 1.05
    if lo == hi: lo -= 1; hi += 1
    ax.plot([lo, hi], [lo, hi], "--", color="#e74c3c", alpha=0.6, lw=1.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(f"RRUFF {p} ({units[p]})", fontsize=10)
    ax.set_ylabel(f"Predicted {p} ({units[p]})", fontsize=10)
    ax.set_title(f"{p}  MAE={mae:.3f}  R²={r2:.3f}", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.15)

for j in range(len(all_params), len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    f"RRUFF vs Predicted Lattice Parameters (n={n}, RRUFF a,b,c < {VAL} Å)",
    fontsize=14, fontweight="bold",
)
plt.tight_layout()
fig.savefig("parity_filtered.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Saved parity_filtered.png")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 2: Distribution comparison (overlaid histograms + KDE)
# ═══════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(2, 4, figsize=(22, 10))
axes2 = axes2.flatten()

for i, p in enumerate(all_params):
    ax = axes2[i]
    if p not in metrics:
        ax.set_visible(False); continue
    exp = metrics[p]["exp"]
    pred = metrics[p]["pred"]
    jsd = kld_results.get(p, {}).get("jsd", float("nan"))

    lo = min(exp.min(), pred.min())
    hi = max(exp.max(), pred.max())
    margin = 0.05 * (hi - lo + 1)
    bins = np.linspace(lo - margin, hi + margin, min(40, max(10, len(exp) // 5)))

    ax.hist(exp, bins=bins, alpha=0.45, color="#3b82f6", label="RRUFF (exp)",
            density=True, edgecolor="white", linewidth=0.5)
    ax.hist(pred, bins=bins, alpha=0.45, color="#ec4899", label="Predicted",
            density=True, edgecolor="white", linewidth=0.5)

    # KDE overlay
    try:
        x_grid = np.linspace(lo - margin, hi + margin, 300)
        ax.plot(x_grid, gaussian_kde(exp)(x_grid), color="#2563eb", lw=2,
                label="KDE exp")
        ax.plot(x_grid, gaussian_kde(pred)(x_grid), color="#db2777", lw=2,
                label="KDE pred")
    except Exception:
        pass

    ax.set_xlabel(f"{p} ({units[p]})", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"{p}  JSD={jsd:.4f}", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.15)

for j in range(len(all_params), len(axes2)):
    axes2[j].set_visible(False)

fig2.suptitle(
    f"Distribution: RRUFF (exp) vs Predicted (n={n}, a,b,c < {VAL} Å)",
    fontsize=14, fontweight="bold",
)
plt.tight_layout()
fig2.savefig("distribution_filtered.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Saved distribution_filtered.png")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 3: Error distributions (Δ = pred - exp)
# ═══════════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(2, 4, figsize=(22, 10))
axes3 = axes3.flatten()

for i, p in enumerate(all_params):
    ax = axes3[i]
    if p not in metrics:
        ax.set_visible(False); continue
    delta = metrics[p]["delta"]
    mean_d = metrics[p]["mean_delta"]

    ax.hist(delta, bins=max(10, len(delta) // 5), color="#8b5cf6",
            edgecolor="white", alpha=0.8, density=True)
    ax.axvline(0, color="#e74c3c", ls="--", lw=1.5, alpha=0.7)
    ax.axvline(mean_d, color="#f59e0b", ls="-", lw=1.5, alpha=0.7,
               label=f"mean={mean_d:+.3f}")

    try:
        x_grid = np.linspace(delta.min(), delta.max(), 200)
        ax.plot(x_grid, gaussian_kde(delta)(x_grid), color="#6d28d9", lw=2)
    except Exception:
        pass

    std_d = np.std(delta)
    ax.set_xlabel(f"Δ{p} ({units[p]})", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(f"Δ{p}  μ={mean_d:+.3f} σ={std_d:.3f}",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)

for j in range(len(all_params), len(axes3)):
    axes3[j].set_visible(False)

fig3.suptitle(
    f"Error Distributions: Δ = Predicted − RRUFF (n={n}, a,b,c < {VAL} Å)",
    fontsize=14, fontweight="bold",
)
plt.tight_layout()
fig3.savefig("error_dist_filtered.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Saved error_dist_filtered.png")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 4: Per-method comparison (PM vs DG)
# ═══════════════════════════════════════════════════════════════════════════
if "best_method" in df_clean.columns:
    methods = df_clean["best_method"].dropna().unique()
    if len(methods) >= 2:
        fig4, axes4 = plt.subplots(2, 4, figsize=(22, 10))
        axes4 = axes4.flatten()

        method_colors = {
            "pattern_matching": "#f59e0b",
            "diffractgpt": "#3b82f6",
            "pattern_matching (elements)": "#22c55e",
        }

        for i, p in enumerate(all_params):
            ax = axes4[i]
            for method in sorted(methods):
                subset = df_clean[df_clean["best_method"] == method]
                if len(subset) < 2:
                    continue
                exp_s = subset[f"rruff_{p}"].values.astype(float)
                pred_s = subset[f"pred_{p}"].values.astype(float)
                valid = ~(np.isnan(exp_s) | np.isnan(pred_s))
                delta = pred_s[valid] - exp_s[valid]
                label_short = (
                    method.replace("pattern_matching (elements)", "PM (elements)")
                    .replace("pattern_matching", "PM")
                    .replace("diffractgpt", "DG")
                )
                color = method_colors.get(method, "#6b7280")
                ax.hist(
                    delta,
                    bins=max(8, len(delta) // 5),
                    alpha=0.5,
                    color=color,
                    label=f"{label_short} (n={len(delta)}, MAE={np.mean(np.abs(delta)):.3f})",
                    density=True,
                )

            ax.axvline(0, color="#e74c3c", ls="--", lw=1, alpha=0.5)
            ax.set_xlabel(f"Δ{p} ({units[p]})", fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.set_title(f"Δ{p} by method", fontsize=11, fontweight="bold")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.15)

        for j in range(len(all_params), len(axes4)):
            axes4[j].set_visible(False)

        fig4.suptitle(
            f"Error by Method: PM vs DG (n={n}, a,b,c < {VAL} Å)",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout()
        fig4.savefig("error_by_method_filtered.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("✓ Saved error_by_method_filtered.png")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 5: Per crystal system boxplots
# ═══════════════════════════════════════════════════════════════════════════
if "rruff_crystal_system" in df_clean.columns:
    cs_counts = df_clean["rruff_crystal_system"].value_counts()
    cs_with_data = cs_counts[cs_counts >= 3].index.tolist()

    if cs_with_data:
        fig5, axes5 = plt.subplots(2, 4, figsize=(22, 10))
        axes5 = axes5.flatten()

        for i, p in enumerate(all_params):
            ax = axes5[i]
            data_by_cs = []
            labels = []
            for cs in cs_with_data:
                subset = df_clean[df_clean["rruff_crystal_system"] == cs]
                delta = (subset[f"pred_{p}"] - subset[f"rruff_{p}"]).dropna().values
                if len(delta) >= 2:
                    data_by_cs.append(delta)
                    labels.append(f"{cs}\n(n={len(delta)})")

            if data_by_cs:
                bp = ax.boxplot(data_by_cs, labels=labels, patch_artist=True)
                colors = plt.cm.Set2(np.linspace(0, 1, len(data_by_cs)))
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                ax.axhline(0, color="#e74c3c", ls="--", lw=1, alpha=0.5)

            ax.set_ylabel(f"Δ{p} ({units[p]})", fontsize=10)
            ax.set_title(f"Δ{p} by crystal system", fontsize=11, fontweight="bold")
            ax.tick_params(axis="x", rotation=45, labelsize=7)
            ax.grid(True, alpha=0.15, axis="y")

        for j in range(len(all_params), len(axes5)):
            axes5[j].set_visible(False)

        fig5.suptitle(
            f"Error by Crystal System (n={n}, a,b,c < {VAL} Å)",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout()
        fig5.savefig("error_by_crystal_filtered.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("✓ Saved error_by_crystal_filtered.png")

# ═══════════════════════════════════════════════════════════════════════════
# Print summary
# ═══════════════════════════════════════════════════════════════════════════
print(f"\n{'═' * 70}")
print(f"SUMMARY (n={n}, RRUFF a,b,c < {VAL} Å)")
print(f"{'═' * 70}")
print(f"{'Param':>8s}  {'MAE':>8s}  {'RMSE':>8s}  {'R²':>8s}  "
      f"{'JSD':>8s}  {'Mean Δ':>9s}  {'Med |Δ|':>8s}")
print(f"{'─' * 70}")
for p in all_params:
    if p in metrics and p in kld_results:
        m = metrics[p]
        k = kld_results[p]
        print(
            f"{p:>8s}  {m['mae']:8.4f}  {m['rmse']:8.4f}  {m['r2']:8.4f}  "
            f"{k['jsd']:8.4f}  {m['mean_delta']:+9.4f}  {m['median_abs']:8.4f}"
        )

print(f"\nOutput files:")
print(f"  parity_filtered.png")
print(f"  distribution_filtered.png")
print(f"  error_dist_filtered.png")
print(f"  error_by_method_filtered.png")
print(f"  error_by_crystal_filtered.png")
