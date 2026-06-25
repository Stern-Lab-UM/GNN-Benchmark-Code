# DCG GNN Publication Code

This repository is being curated for the DCG/GNN manuscript revision.

The goal is to keep publication-relevant code in one clean, private Git
repository while excluding raw data, trained checkpoints, generated figures,
large caches, local screenshots, and cluster-specific scratch logs.

## Repository Layout

- `analysis/matlab/` - canonical MATLAB analysis and plotting pipeline.
- `models/mpnn/` - MPNN training/prediction source code and minimal configs.
- `models/ppgn/` - PPGN training/prediction source code and minimal configs.
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
