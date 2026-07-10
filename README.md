# GNN Benchmark Publication Code

This repository contains code for the GNN Benchmark manuscript revision:

Krajnc, M., Comi, T., Miao, S., Hafeez, A., Serviansky, H., Li, P.,
and Stern, T. (2026). *A Controlled in Silico Benchmark for GNN
Prediction of Tissue Dynamics*. Preprint. Online preprint URL/DOI:
pending public posting.

The paper introduces a controlled in silico benchmark for graph neural network
prediction of tissue remodeling. The task is to predict relaxed post-T1
cell-cell interface lengths from simulated epithelial tissues while varying
training-set size, tissue order, mechanics, tissue size, feature availability,
and perturbation structure. This repository provides the code used to generate
the simulated tissue graphs, train and evaluate the GNN models, run the spring
embedding checks, and reproduce the analysis/plotting pipeline.

## Two Ways To Run

The repository supports two main run modes through the MATLAB top-level pipeline
in `pipeline/matlab/`.

### Full Publication Run

This mode is the end-to-end manuscript workflow: tissue generation, dataset
assembly, Bayesian optimization, final training, prediction, embedding, analysis,
and plotting for the full set of paper analyses. It is computationally expensive
and intended for a properly configured compute node or cluster environment.

```matlab
addpath(genpath('/path/to/GNN-Benchmark-Code'))
manifest = GNNBenchmark_run_publication_pipeline( ...
    'mode', 'publication', ...
    'output_root', '/path/to/project/gnn_benchmark_publication_run', ...
    'workers', 20, ...
    'cuda', 0);
```

### Mini Example

This mode uses the same code paths but shrinks the data, epochs, seeds, BO
trials, and simulator durations. It is meant to show that the full pipeline is
installed correctly and can run from start to finish.

```matlab
addpath(genpath('/path/to/GNN-Benchmark-Code'))
manifest = GNNBenchmark_run_publication_pipeline('mode', 'mini', 'cuda', 0);
```

### Integration-Scale Example

This mode is larger than the mini example but far smaller than the full run. It
keeps the manuscript-style dataset and figure layout, uses small generated
datasets, runs all generated condition families, and is meant for an overnight
GPU-node validation.

```matlab
manifest = GNNBenchmark_run_publication_pipeline( ...
    'mode', 'integration', ...
    'output_root', '/path/to/scratch/gnn_benchmark_pipeline_integration', ...
    'workers', 12, ...
    'cuda', 0);
```

See `pipeline/matlab/README.md` for cache policies, integration-profile knobs,
CPU-only examples, and output-folder details.

## Main Pipeline Stages

The chronological workflow is:

1. Initial tissue generation: `data_generation/vertex_model/` compiles and runs
   the vertex-model simulator to create pre-T1 tissue geometries.
2. T1 perturbation and relaxation: `data_generation/vertex_model/` applies one
   or two T1 events, relaxes the tissues, and writes raw `final_*.vt2d` and
   `graph_*.txt` outputs.
3. Dataset assembly and splits: `data_generation/vertex_model/` and its
   `manifests/` convert raw simulator outputs into model-ready weighted and
   unweighted graph datasets with train/validation/test splits.
4. Bayesian optimization: `training/bayesopt/` launches the MPNN/PNA or PPGN
   training code for each trial and records partial and final BO results.
5. Final training: `models/mpnn/` and `models/ppgn/` train the final GraphSAGE,
   GAT, GIN, PNA, and PPGN models. This covers the standard W/UW V1 data and
   the revision analyses: hexagonality, kA, shear, tissue size, two T1 events,
   counterfactual data perturbation with delta = 0.05, and skip/feature
   architecture-ablation analyses.
6. Prediction: `models/mpnn/`, `models/ppgn/`, and `pipeline/matlab/` generate
   and consolidate prediction files for downstream analysis.
7. Spring embedding: `external/spring_embed/` builds the spring-relaxation
   executable; `pipeline/matlab/` and `analysis/matlab/` run and analyze the
   embedding outputs.
8. Post-prediction analysis and plotting: `analysis/matlab/` computes the MAE,
   nMAE, hop-distance, fallback, embedding-error, perturbation, ablation, and
   figure-generation analyses used in the manuscript and revision.

Each run writes outputs under the chosen `output_root`, including generated
data, BO logs, trained checkpoints, predictions, embeddings, analysis tables,
figures, logs, and manifests.

## Repository Layout

Listed roughly in the order used by the end-to-end pipeline:

- `pipeline/matlab/` - MATLAB top-level orchestration for mini and publication-scale end-to-end runs.
- `data_generation/vertex_model/` - vertex-model simulator source and MATLAB wrappers for initial tissue generation, T1 perturbation/relaxation, and model-ready dataset assembly.
- `training/bayesopt/` - MATLAB Bayesian-optimization drivers and final search-space definitions.
- `models/mpnn/` - source snapshot used for GraphSAGE, GAT, GIN, and PNA training and prediction.
- `models/ppgn/` - source snapshots used for PPGN training and prediction, kept split to preserve run provenance.
- `external/spring_embed/` - source for the spring-relaxation executable used for embedding predicted tissues.
- `analysis/matlab/` - MATLAB post-prediction analyses, manuscript diagnostics, and plotting pipeline.
- `manuscript_analyses/` - small manuscript-facing analysis helpers and archived calculation snippets.
- `remote_examples/` - example remote/HPC launch helpers used for long-running jobs.
- `requirements/` and `scripts/` - Python environment requirements, setup helpers, and installation/import checkers.
- `docs/` - installation notes, tested settings, provenance records, and analysis-pipeline documentation.

## Languages

The top-level orchestration, Bayesian optimization drivers, data assembly, and
post-prediction analyses are written in MATLAB. The GNN models are implemented
in Python with PyTorch/PyTorch Geometric. Tissue simulation and spring embedding
use C/C++ source compiled locally. Shell scripts are provided only for setup and
cluster-environment convenience.

## Pipeline Outputs

Generated datasets, predictions, embeddings, model checkpoints, analysis tables,
figures, logs, and manifests are written to the user-chosen `output_root`.

## License

This repository is released under the MIT License; see `LICENSE`. If you use
this code, please cite the accompanying preprint and retain the credits listed in
`CREDITS.md`.

## Credits and Citation

Please cite the accompanying preprint when using this code. Citation metadata is
provided in `CITATION.cff`.

Code credits are listed in `CREDITS.md`. In brief, Siqi Miao and Pan Li provided
the MPNN code used for GraphSAGE, GAT, GIN, and PNA; Troy Comi provided the PPGN
code; Tomer Stern adapted the GNN code for the updated manuscript and ablation
analyses; Matej Krajnc provided the tissue-simulation and spring-embedding code;
and Adnan Hafeez supported code installation and computing-environment setup.

Publication-repository organization, cleanup, and documentation were assisted by
Claude (Anthropic) and Codex (OpenAI), under Tomer Stern's direction and review.

## Installation Instructions

See `docs/INSTALL.md` for environment notes and setup commands, and
`docs/TESTED_SETTINGS.md` for the software/hardware settings that have been
exercised. A typical Lighthouse/shared-HPC setup is:

```bash
git clone https://github.com/Stern-Lab-UM/GNN-Benchmark-Code.git
cd GNN-Benchmark-Code
bash scripts/setup_lh_env.sh --component all --torch default
python scripts/check_install.py --component all
```

For conda or local `venv` setup, see `docs/INSTALL.md`. The checker verifies
package imports and curated MPNN/PPGN source snapshots; it does not require raw
manuscript data or trained checkpoints.

## External Data Package

Manuscript data are archived on Zenodo: [doi:10.5281/zenodo.21286579](https://doi.org/10.5281/zenodo.21286579). See [docs/DATA_PACKAGE.md](docs/DATA_PACKAGE.md) for the archive contents and extraction instructions. After extracting the Zenodo ZIP files into one package root, run `GNNBenchmark_run_from_data_package(package_root)` in MATLAB to rebuild summaries, regenerate analysis figures, and analyze saved embedding outputs without retraining models.

## Contact

For questions or issues, please contact Tomer Stern, University of Michigan,
tomers@umich.edu.
