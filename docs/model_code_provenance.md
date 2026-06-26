# Model Code Provenance

This note records the model-code snapshots curated into the publication repository on 2026-06-25.

## MPNN/PNA

Repository destination: `models/mpnn/`

Source: `Z:\Tomer\gnn_cluster_env_20260529\bundle\src`

This source tree was identified in the provenance handoff as the standard training and prediction code for GraphSAGE, GAT, GIN, and PNA. The copied tree contains `trainer_final.py`, `predict_final.py`, `dataset.py`, and the local model definitions.

## PPGN

Repository destinations:

- `models/ppgn/train_dcg/dcg/`
- `models/ppgn/predict_dcg/dcg/`
- `models/ppgn/gl_tail_fixed_pkg/dcg/`

Sources:

- `Z:\Tomer\venv\dcg_blackwell\dcg\lib\python3.10\site-packages\dcg`
- `Z:\Tomer\PPGN_Tomer\dcg`
- `Z:\Tomer\_gl_finals\dcg_pkg\ppgn_dcg_fixed_pkg\dcg`

The PPGN split is preserved because the training and prediction code paths were not a single clean package. The GL tail package is included separately because it was used for the late Great Lakes `1_32` PPGN tail run and was documented as matching the Armis2 runtime.

## Hash Manifest

`docs/model_source_hashes_20260625.csv` records SHA256 hashes for every copied file and confirms that source and repository hashes matched at copy time.
