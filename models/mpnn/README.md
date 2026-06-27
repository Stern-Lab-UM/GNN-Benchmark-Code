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

## Architecture Switches

The default training command is the length-informed revision architecture: the
message-passing backbone receives weighted edge attributes and the final
`EdgeRegressor` head also receives the raw edge attributes.

For the input-to-head ablation used in the revision controls, pass:

```bash
python trainer_final.py --ablate_head_edge_attr ...
```

This removes raw edge attributes from only the final regression head. The
message-passing backbone still receives the same edge features. Checkpoints save
`model_config.ablate_head_edge_attr`, `model_config.backbone_edge_dim`, and the
head `edge_dim`, so `predict_final.py` rebuilds the correct architecture from the
checkpoint without a separate inference flag.