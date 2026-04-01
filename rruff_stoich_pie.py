#!/usr/bin/env python3
from __future__ import annotations

import re
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from jarvis.core.composition import Composition
from jarvis.db.figshare import data as figshare_data

mpl.rcParams["font.family"] = "serif"

_VALID_ELEMENTS = {
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P",
    "S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu",
    "Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo",
    "Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs",
    "Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er",
    "Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl",
    "Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu",
}

def clean_formula(line: str | None) -> str | None:
    if not line or not isinstance(line, str):
        return None
    line = line.strip()
    if not line:
        return None

    line = re.sub(r"\[.*?\]", "", line)
    line = re.sub(r"\^[^^]*\^", "", line)
    line = line.replace("_", "")
    line = re.sub(r"(\d+\.?\d*)-\d+\.?\d*", r"\1", line)
    line = line.replace(" ", "")
    line = re.sub(r"[;·][\dxn]*\.?\d*H2O", "", line)

    return line or None

def formula_to_stoich_dict(formula: str) -> dict[str, float]:
    try:
        comp = Composition.from_string(formula)
        d = comp.to_dict()
        return {el: float(amount) for el, amount in d.items() if el in _VALID_ELEMENTS}
    except Exception:
        raw = re.findall(r"[A-Z][a-z]?", formula)
        elems = [el for el in raw if el in _VALID_ELEMENTS]
        return {el: 1.0 for el in elems}

def collect_rruff_stoichiometry(max_entries: int | None = None) -> pd.Series:
    data = figshare_data("rruff_powder_xrd")
    print(f"Loaded {len(data)} total RRUFF entries")

    totals: dict[str, float] = {}
    n_valid = 0
    n_skipped = 0

    for i, entry in enumerate(data):
        if max_entries is not None and i >= max_entries:
            break

        raw_formula = entry.get("##IDEAL CHEMISTRY")
        formula = clean_formula(raw_formula)

        if not formula:
            n_skipped += 1
            continue

        stoich = formula_to_stoich_dict(formula)
        if not stoich:
            n_skipped += 1
            continue

        for el, amt in stoich.items():
            totals[el] = totals.get(el, 0.0) + amt

        n_valid += 1

    print(f"Usable formulas: {n_valid}")
    print(f"Skipped entries: {n_skipped}")

    if not totals:
        raise RuntimeError("No valid stoichiometric data could be extracted from RRUFF.")

    return pd.Series(totals).sort_values(ascending=False)

def create_stoich_pie_chart(
    element_counts: pd.Series,
    output_dir: Path,
    top_num: int = 23,
    show_top_pcts: int = 11,
    show_other_pct: bool = False,
    label_fontsize: int = 18,
    pct_fontsize: int = 18,
    title_fontsize: int = 30,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    top = element_counts.iloc[:top_num].copy()
    others = element_counts.iloc[top_num:].sum()
    has_other = others > 0
    if has_other:
        top.loc["Other"] = others

    top.to_csv(output_dir / "rruff_element_counts.csv", header=["count"])

    counts = top.values
    labels = top.index.tolist()

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    patches, texts, autotexts = ax.pie(
        counts,
        labels=labels,
        labeldistance=1.05,
        autopct="%1.1f%%",
        pctdistance=0.8,
        radius=1,
        shadow=False,
        startangle=90,
        wedgeprops={"edgecolor": "w", "linewidth": 1},
    )

    for txt in texts:
        txt.set_fontsize(label_fontsize)
        txt.set_fontweight("bold")
        txt.set_color("black")
        txt.set_visible(True)

    other_index = labels.index("Other") if has_other else None

    for i, (pct_txt, wedge) in enumerate(zip(autotexts, patches)):
        is_other = (other_index is not None and i == other_index)
        show_this_pct = (i < show_top_pcts) or (is_other and show_other_pct)

        if show_this_pct:
            ang = 0.5 * (wedge.theta1 + wedge.theta2)

            if 90 < ang < 270:
                ang += 180

            pct_txt.set_rotation(ang)
            pct_txt.set_rotation_mode("anchor")
            pct_txt.set_ha("center")
            pct_txt.set_va("center")
            pct_txt.set_fontsize(pct_fontsize)
            pct_txt.set_visible(True)
        else:
            pct_txt.set_visible(False)

    ax.axis("equal")
    plt.title(
        "Element Proportions in the \n RRUFF Powder XRD Dataset",
        fontsize=title_fontsize,
    )

    plt.savefig(output_dir / "rruff_stoichiometry_pie_chart.png", format="png")
    plt.close(fig)

    print(f"Saved CSV: {output_dir / 'rruff_element_counts.csv'}")
    print(f"Saved PNG: {output_dir / 'rruff_stoichiometry_pie_chart.png'}")
    print(f"Top wedges shown: {top_num}")
    print(f"Top percentage labels shown: {min(show_top_pcts, len(top))}")
    print(f"Show 'Other' percentage: {show_other_pct}")

def build_parser():
    p = argparse.ArgumentParser(
        description="Create a stoichiometry-weighted element pie chart from the RRUFF powder XRD dataset."
    )
    p.add_argument("--output", default="rruff_stoich_output", help="Output directory")
    p.add_argument("--max-entries", type=int, default=None, help="Optional cap on number of RRUFF entries")
    p.add_argument("--top-num", type=int, default=23, help="Number of top elements shown as wedges before grouping the rest into Other")
    p.add_argument("--show-top-pcts", type=int, default=11, help="Number of largest wedges whose percentage labels are shown")
    p.add_argument("--show-other-pct", action="store_true", help="Display a percentage label on the 'Other' wedge")
    p.add_argument("--label-fontsize", type=int, default=18, help="Font size for element labels")
    p.add_argument("--pct-fontsize", type=int, default=18, help="Font size for percentage labels")
    p.add_argument("--title-fontsize", type=int, default=30, help="Font size for title")
    return p

def main():
    args = build_parser().parse_args()
    output_dir = Path(args.output).expanduser().resolve()

    element_counts = collect_rruff_stoichiometry(max_entries=args.max_entries)
    create_stoich_pie_chart(
        element_counts=element_counts,
        output_dir=output_dir,
        top_num=args.top_num,
        show_top_pcts=args.show_top_pcts,
        show_other_pct=args.show_other_pct,
        label_fontsize=args.label_fontsize,
        pct_fontsize=args.pct_fontsize,
        title_fontsize=args.title_fontsize,
    )

if __name__ == "__main__":
    main()
