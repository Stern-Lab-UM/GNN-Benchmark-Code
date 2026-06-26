# DCG GNN Publication Code

This repository contains code for the DCG/GNN manuscript revision.

The goal is to keep publication-relevant code in one clean repository while
excluding raw data, trained checkpoints, generated figures, large caches, local
screenshots, and cluster-specific scratch logs.

## Repository Layout

- `analysis/matlab/` - MATLAB analysis and plotting pipeline.
- `models/mpnn/` - source snapshot used for the GraphSAGE, GAT, GIN, and PNA training and prediction runs.
- `models/ppgn/` - source snapshots used for PPGN, kept split into training, prediction, and GL tail packages to preserve provenance.
- `manuscript_analyses/` - focused scripts used to compute manuscript-specific numerical checks, fallback diagnostics, embedding summaries, and tables.
- `remote_examples/` - small example launch scripts for reproducibility.
- `docs/` - installation notes, provenance records, and analysis-pipeline documentation.

## MATLAB Analysis

The MATLAB scripts in `analysis/matlab/` perform the post-prediction analysis and plotting. They start from saved prediction files; Bayesian optimization, model training, and prediction generation are implemented in the Python/PyTorch code under `models/`. See `docs/MATLAB_ANALYSIS_PIPELINE.md` for the full MATLAB workflow.

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
