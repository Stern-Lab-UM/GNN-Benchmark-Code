# MPNN/PNA Source Snapshot

This is the provenance-pinned source tree used for the standard GraphSAGE, GAT,
GIN, and PNA training/prediction workflow.

Important files:

- `trainer_final.py` - training entry point.
- `predict_final.py` - prediction entry point.
- `dataset.py` - dataset loader and graph construction.
- `models/edge_regressor.py` - regression head used for edge-length prediction.
- `models/pna.py`, `models/gin.py`, `models/sage.py` - local backbones.

GAT is instantiated through PyTorch Geometric in `trainer_final.py`; there is no
separate local `gat.py` file in this snapshot.

The source was copied without Python bytecode caches or backup files. File
hashes are recorded in `../../docs/model_source_hashes_20260625.csv`.
