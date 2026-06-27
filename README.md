# DCG GNN Publication Code

This repository contains code for the DCG/GNN manuscript revision.

The goal is to keep publication-relevant code in one clean repository while
excluding raw data, trained checkpoints, generated figures, large caches, local
screenshots, and cluster-specific scratch logs.

## Repository Layout

- `analysis/matlab/` - MATLAB analysis, manuscript diagnostics, and plotting pipeline.
- `data_generation/vertex_model/` - vertex-model simulator source and MATLAB wrappers for regenerating raw and model-ready tissue-graph datasets.
- `external/spring_embed/` - source for the spring-relaxation executable used by MATLAB embedding example figures.
- `models/mpnn/` - source snapshot used for the GraphSAGE, GAT, GIN, and PNA training and prediction runs.
- `models/ppgn/` - source snapshots used for PPGN, kept split into training, prediction, and GL tail packages to preserve provenance.
- `training/bayesopt/` - MATLAB Bayesian-optimization drivers and final search-space definitions.
- `docs/` - installation notes, provenance records, and analysis-pipeline documentation.

## Bayesian Optimization

The MATLAB BayesOpt drivers are under `training/bayesopt/`. They launch the curated MPNN or PPGN training code once per trial, checkpoint partial `bayesopt` results, and store the final V1 search spaces in `DCG_bayesopt_search_spaces.m`. See `training/bayesopt/README.md`.

## MATLAB Analysis

The MATLAB scripts in `analysis/matlab/` perform the post-prediction analysis, manuscript-specific diagnostics, and plotting. They start from saved prediction files; Bayesian optimization, model training, and prediction generation are implemented in the Python/PyTorch code under `models/`. See `docs/MATLAB_ANALYSIS_PIPELINE.md` for the full MATLAB workflow.

## Vertex-Model Data

The tissue-graph generation pipeline is in `data_generation/vertex_model/`.
It compiles the vertex-model simulator, regenerates raw `final_*.vt2d` and
`graph_*.txt` files, and assembles weighted/unweighted model-ready graph files.
See `data_generation/vertex_model/README.md`.

Embedding example figures additionally require compiling the spring engine under
`external/spring_embed/` and pointing MATLAB to the resulting executable through
`DCG_EMBED_ENGINE` or `DCG_local_config.m`.

## Data Policy

Do not commit raw datasets, prediction dumps, embeddings, model checkpoints,
large `.mat` caches, generated figures, or private cluster credentials. The
publication repository should contain code plus lightweight documentation only.

## Model Code Status

The model source snapshots were copied from provenance-pinned run trees on
2026-06-25. See `models/README.md` and `docs/model_source_hashes_20260625.csv`
for SHA256 hash records. The repository includes environment templates and an
import checker for validating a local setup.

## Installation Check

See `docs/INSTALL.md` for environment notes and setup commands. After installing
dependencies, run:

```bash
python scripts/check_install.py --component all
```

The checker verifies package imports and the curated MPNN/PPGN source snapshots
without requiring manuscript data or trained checkpoints.
