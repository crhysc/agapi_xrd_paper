echo "plot_refinement_mae.py"
python plot_refinement_mae.py
echo "plot_refinement_jsd_18panel.py"
python plot_refinement_jsd_18panel.py
echo "plot_match_rate_crystal_systems.py"
python plot_match_rate_crystal_systems.py --max-val 10
echo "stoich"
bash stoich.sh
echo "end script"

