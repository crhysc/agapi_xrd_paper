# AGAPI XRD Paper — Scripts Overview

Create a **key.txt** file in the repository root containing your AtomGPT.org API key, as the workflow reads this file at runtime to authenticate all API requests.

**plot_refinement_mae.py**  
    Generates a single MAE chart across all six lattice parameters,  
    comparing the three refinement methods.

**plot_refinement_jsd_18panel.py**  
    Produces a 6×3 panel figure of Jensen–Shannon divergence (JSD)  
    for the three refinement methods across the six lattice parameters.

**plot_match_rate_crystal_systems.py**  
    Outputs six figures:  
        (1–3) Individual normalized crystal system histograms (one per refine method)  
        (4)   3-panel PNG with 3 histograms (one per refine method)  
        (5)   Match rate pie chart  
        (6)   3-panel histogram + match rate pie chart

**stoich.sh**  
    Generates a normalized pie chart histogram showing the most common atomic  
    species in the RRUFF database.
