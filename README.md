# AGAPI XRD Paper — Scripts Overview

**plot_refinement_mae.py**  
    Generates a single MAE chart across all six lattice parameters,  
    comparing the three refinement methods.

**plot_refinement_jsd_18panel.py**  
    Produces a 6×3 panel figure of Jensen–Shannon divergence (JSD)  
    for the three refinement methods across the six lattice parameters.

**plot_match_rate_crystal_systems.py**  
    Outputs six figures:  
        (1–3) Individual normalized crystal system histograms (one per method)  
        (4)   Combined histogram (all three methods)  
        (5)   Match rate pie chart  
        (6)   Combined histogram + match rate pie chart

**stoich.sh**  
    Generates a normalized pie chart showing the most common atomic  
    species in the RRUFF database.
