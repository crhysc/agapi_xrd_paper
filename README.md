# AGAPI-XRD: Reproducibility Repository for Automated X-ray Diffraction Analysis

This repository contains the benchmark, analysis, and figure-generation scripts used for the AGAPI-XRD study, a hybrid framework for automated crystal-structure identification from powder X-ray diffraction (XRD) data. The workflow combines database-driven pattern matching against JARVIS-DFT and COD, DiffractGPT-based generative structure prediction, optional ALIGNN-FF relaxation, and automated Rietveld refinement. This repository is intended to support reproducibility of the computational experiments and figures reported in the accompanying manuscript.

## Abstract

X-ray diffraction (XRD) remains one of the most powerful experimental techniques for characterizing materials, yet the path from raw diffraction data to a refined atomic structure continues to demand significant domain expertise and manual intervention. We present AGAPI-XRD, a hybrid computational framework that integrates DiffractGPT, a generative pretrained transformer trained on thousands of crystal structures and their simulated XRD patterns, elemental and stoichiometric pattern matching with the JARVIS-DFT and COD materials databases, and classical Rietveld refinement into a unified, accessible API hosted on the AtomGPT.org API (AGAPI) platform. DiffractGPT can rapidly predict candidate atomic structures from experimental XRD patterns. These AI-generated structures then serve as high-quality starting configurations for automated Rietveld refinement, dramatically reducing sensitivity to initial guess quality and accelerating convergence.

The combined pipeline is exposed through the AGAPI interface at `https://atomgpt.org/xrd`, enabling seamless programmatic access for experimentalists, high-throughput screening workflows, and integration with broader agentic AI frameworks for materials discovery. We benchmark the approach on 276 minerals from the RRUFF powder XRD database, achieving 96.7% structure-identification coverage with lattice-parameter mean absolute errors of approximately 1.1 Å for unit-cell lengths and positive skill scores across lattice dimensions and volume. Pattern matching against JARVIS-DFT and COD provides the highest structural accuracy, while DiffractGPT extends coverage to complex minerals absent from existing databases. By bridging generative AI with established crystallographic refinement, AGAPI-XRD lowers the barrier to automated structure determination and advances the vision of fully autonomous materials-characterization pipelines.

## Repository Scope

This repository contains scripts for:

- running AGAPI-XRD benchmark jobs on the RRUFF dataset,
- analyzing returned predictions against experimental lattice parameters,
- compiling replication metrics,
- generating manuscript figures, and
- summarizing the results of different refinement settings.

The benchmark and manuscript focus on a filtered RRUFF subset with `a`, `b`, `c <= 10 Å`, enabling consistent lattice-parameter comparison across the tested structures.

## Installation

This project is part of the JARVIS ecosystem. A minimal conda environment can be created as follows:

```bash
conda create -n jarvis python=3.10 -y
conda activate jarvis
python -m pip install --upgrade pip
python -m pip install jarvis-tools
```

Depending on your local environment and cluster configuration, you may also need additional Python packages used by the analysis and plotting scripts. However, `jarvis-tools` is the essential starting point and should be installed first.

## API Key Setup

Create a `key.txt` file in the repository root containing your AtomGPT.org API key. The workflow reads this file at runtime to authenticate API requests.

```bash
echo "YOUR_ATOMGPT_API_KEY" > key.txt
```

The repository root should then contain a file layout similar to:

```bash
analyse_filtered_rruff.py
analyse_filtered_rruff_runner.sh
compile_agapi_replication_metrics.py
compile_results.sh
generate_figures.sh
key.txt
plot_match_rate_crystal_systems.py
plot_match_rate.py
plot_refinement_jsd_18panel.py
plot_refinement_mae.py
plot_refinement_mae_bmgn_vs_bmgn_alignnff.py
README.md
replication_summary.json
rruff_stoich_pie.py
rruff_xrd_analysis.py
runs/
runs_to_compile.txt
stoich.sh
xrd_pipeline_none.job
xrd_pipeline_gsas2.job
xrd_pipeline_bmgn.job
xrd_pipeline_bmgn_alignnff.job
```

## Reproducing the Benchmark

The main benchmark runs are launched through Slurm job files. From the repository root, submit the desired jobs with:

```bash
sbatch xrd_pipeline_none.job
sbatch xrd_pipeline_gsas2.job
sbatch xrd_pipeline_bmgn.job
sbatch xrd_pipeline_bmgn_alignnff.job
```

These correspond to:

- `xrd_pipeline_none.job`: AGAPI-XRD predictions without Rietveld refinement
- `xrd_pipeline_gsas2.job`: AGAPI-XRD predictions with GSAS-II refinement
- `xrd_pipeline_bmgn.job`: AGAPI-XRD predictions with BGMN-based refinement
- `xrd_pipeline_bmgn_alignnff.job`: AGAPI-XRD predictions with ALIGNN-FF relaxation followed by BGMN-based refinement

After job completion, outputs are organized under the `runs/` directory.

## Re-running the Analysis

Once the Slurm jobs finish, rerun the analysis and figure-generation workflow using the helper scripts included in the repository.

### 1. Analyze returned predictions

```bash
bash analyse_filtered_rruff_runner.sh
```

This step processes the run directories and computes benchmark statistics against the filtered RRUFF ground truth.

### 2. Generate manuscript figures

```bash
bash generate_figures.sh
```

This script regenerates the figure assets from the processed run outputs.

### 3. Compile replication outputs

```bash
bash compile_results.sh
```

This aggregates run-level outputs into a consolidated summary.

### 4. Build the replication summary

```bash
python compile_agapi_replication_metrics.py
```

This produces a consolidated metrics summary, typically written to `replication_summary.json`.

## Expected Outputs

Depending on which jobs and analysis scripts are run, the repository produces:

- processed benchmark outputs under `runs/`,
- figure files for lattice-parameter MAE and JSD comparisons,
- crystal-system and pattern-matching visualizations,
- stoichiometric composition plots,
- compiled result summaries, and
- a replication summary JSON for reviewer-facing inspection.

## Script Overview

### Benchmark and analysis

- `rruff_xrd_analysis.py`  
  Main benchmark driver for querying AGAPI-XRD predictions and storing outputs locally.

- `analyse_filtered_rruff.py`  
  Analyzes prediction outputs against filtered RRUFF ground-truth lattice parameters.

- `analyse_filtered_rruff_runner.sh`  
  Convenience shell wrapper for running the filtered RRUFF analysis workflow.

- `compile_agapi_replication_metrics.py`  
  Compiles benchmark outputs into a reviewer-friendly replication summary JSON.

- `compile_results.sh`  
  Aggregates results from multiple run directories into a consolidated summary.

### Figure generation

- `plot_refinement_mae.py`  
  Generates a single MAE comparison figure across the six lattice parameters for the main refinement settings.

- `plot_refinement_jsd_18panel.py`  
  Produces a multi-panel Jensen-Shannon divergence figure across lattice parameters and refinement settings.

- `plot_match_rate_crystal_systems.py`  
  Generates crystal-system histograms, pattern-matching summaries, and combined presentation figures.

- `plot_refinement_mae_bmgn_vs_bmgn_alignnff.py`  
  Compares lattice-parameter MAE between BMGN and BMGN + ALIGNN-FF workflows.

- `plot_match_rate.py`  
  Generates pattern-matching rate summaries.

### Composition / stoichiometry utilities

- `rruff_stoich_pie.py`  
  Produces stoichiometric or elemental composition visualizations for the RRUFF benchmark set.

- `stoich.sh`  
  Shell helper for generating the stoichiometry figure workflow.

## Notes on Naming

Some repository filenames use the string `bmgn`, while the manuscript discussion refers to the BGMN refinement engine. The filename convention is preserved here to match the code and job files exactly.

## Data and Platform Context

The AGAPI-XRD workflow interfaces with:

- the AtomGPT.org AGAPI platform,
- the JARVIS ecosystem,
- the COD and JARVIS-DFT reference databases, and
- the RRUFF powder XRD benchmark dataset.

This repository is therefore best understood as the reproducibility and analysis companion to the manuscript, rather than as a standalone web service.

## Citation

If you use this repository, please cite the AGAPI-XRD paper and the corresponding JARVIS / AtomGPT ecosystem resources.

## Contact

For scientific questions about the workflow, benchmark, or manuscript, please contact the corresponding authors listed in the paper.
