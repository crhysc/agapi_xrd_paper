def _get_reference_match_info(loaded_data):
    """
    Pattern matching occurs before refinement, so it should not be summed
    across refinement methods. Use a single reference method.
    Prefer no_refinement if present; otherwise use the first entry.
    """
    for method_key, method_name, color, cs_list, match_info in loaded_data:
        if method_key == "no_refinement":
            return method_name, match_info
    return loaded_data[0][1], loaded_data[0][4]


def plot_pattern_matching_pie(loaded_data, out_dir: Path, max_val: float) -> None:
    ref_method_name, match_info = _get_reference_match_info(loaded_data)

    matched = int(match_info["pm_success_n"])
    unmatched = int(match_info["unmatched_for_pie"])
    valid_rows = int(match_info["filtered_valid_rows"])

    fig, ax = plt.subplots(figsize=(8.5, 8.5), constrained_layout=True)

    ax.pie(
        [matched, unmatched],
        labels=[f"Matched\n{matched}", f"Unmatched\n{unmatched}"],
        colors=[MATCH_GREEN, UNMATCHED_GRAY],
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
        textprops={"fontsize": 16},
    )
    ax.set_title(
        f"Pattern-Matching Match Rate\n"
        f"(single pre-refinement result set, n={valid_rows}, a,b,c < {max_val:g} Å)",
        fontsize=22,
    )

    out_png = out_dir / f"pattern_matching_match_rate_pie_max_{str(max_val).replace('.', 'p')}.png"
    plt.savefig(out_png, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ wrote {out_png}")
