# Model Source Snapshots

This directory contains the GNN code snapshots curated for the manuscript repository. These are source snapshots, not trained checkpoints.

## Layout

- `mpnn/` contains the standard MPNN/PNA source tree used for GraphSAGE, GAT, GIN, and PNA training and prediction.
- `ppgn/train_gnn_benchmark/` contains the PPGN training package identified in the provenance notes.
- `ppgn/predict_gnn_benchmark/` contains the PPGN prediction package identified in the provenance notes.
- `ppgn/gl_tail_fixed_pkg/` contains the fixed PPGN package used for the late Great Lakes `1_32` PPGN tail run, documented as byte-identical to the Armis2 final runtime.

The split PPGN layout is intentional. It preserves the actual run provenance instead of presenting the PPGN workflow as one artificially unified package.

## Verification

The copied files were hash-checked against their source paths on 2026-06-25. See `../docs/model_source_hashes_20260625.csv`.

## Next Curation Steps

Later passes can add:

- in-file documentation and docstrings;
- small example commands using toy data;
- a cleaned CLI wrapper for publication use.

Do not add raw datasets, predictions, embeddings, trained models, or generated figures to this directory.

Basic installation notes and an import checker now live in `../docs/INSTALL.md` and `../scripts/check_install.py`.
