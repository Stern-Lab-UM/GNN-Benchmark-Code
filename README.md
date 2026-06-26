# DCG GNN Publication Code

This repository is being curated for the DCG/GNN manuscript revision.

The goal is to keep publication-relevant code in one clean, private Git
repository while excluding raw data, trained checkpoints, generated figures,
large caches, local screenshots, and cluster-specific scratch logs.

## Repository Layout

- `analysis/matlab/` - canonical MATLAB analysis and plotting pipeline.
- `models/mpnn/` - source snapshot used for the standard MPNN/PNA
  training and prediction runs.
- `models/ppgn/` - source snapshots used for PPGN, kept split into
  training, prediction, and GL tail packages to preserve provenance.
- `manuscript_analyses/` - focused scripts used to compute manuscript-specific
  numerical checks, fallback diagnostics, embedding summaries, and tables.
- `remote_examples/` - small example launch scripts for reproducibility. These
  should avoid machine-specific credentials, absolute scratch-only paths, and
  logs.
- `docs/` - handoffs, run notes, and curation manifests.

## Curation Status

This repo is not yet publication-ready. Files should be added through
`docs/curation_manifest.csv`, with each row marked as:

- `include` - canonical or required for reproduction.
- `review` - likely useful but needs inspection.
- `exclude` - scratch, cache, figure output, checkpoint, log, or machine-specific
  helper.

## Data Policy

Do not commit raw datasets, prediction dumps, embeddings, model checkpoints,
large `.mat` caches, generated figures, or private cluster credentials. The
publication repository should contain code plus lightweight documentation only.

## Model Code Status

The model source snapshots were copied from the provenance-pinned run trees on
2026-06-25. See `models/README.md` and
`docs/model_source_hashes_20260625.csv` for the exact source paths and SHA256
hash checks. The code is not yet packaged for user installation; an install
helper and cleaned environment files can be added after we finish selecting the
publication scripts.

