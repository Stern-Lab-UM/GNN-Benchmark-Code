# Model Code Provenance

This note records the model-code snapshots curated into the publication repository on 2026-06-25.

## MPNN/PNA

Repository destination: `models/mpnn/`

This source tree is the standard training and prediction code for GraphSAGE,
GAT, GIN, and PNA. The copied tree contains `trainer_final.py`,
`predict_final.py`, `dataset.py`, and the local model definitions.

## PPGN

Repository destinations:

- `models/ppgn/train_gnn_benchmark/gnn_benchmark_ppgn/`
- `models/ppgn/predict_gnn_benchmark/gnn_benchmark_ppgn/`
- `models/ppgn/gl_tail_fixed_pkg/gnn_benchmark_ppgn/`

The PPGN split is preserved because the training and prediction code paths were
not a single clean package. The GL tail package is included separately because
it was used for the late `1_32` PPGN tail run and was documented as matching the
runtime used for the other final PPGN predictions.

## Hash Manifest

`docs/model_source_hashes_20260625.csv` records SHA256 hashes for every curated
source file in the repository.
