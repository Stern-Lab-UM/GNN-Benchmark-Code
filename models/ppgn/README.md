# PPGN Source Snapshots

PPGN provenance is split across packages, so this directory keeps the split
explicit.

## Directories

- `train_dcg/dcg/` - PPGN training source snapshot.
- `predict_dcg/dcg/` - PPGN prediction source snapshot.
- `gl_tail_fixed_pkg/dcg/` - fixed package used for the late `1_32` PPGN tail run.

The GL tail package is kept because the late `1_32` PPGN tail run used that
fixed package, documented in the run records as byte-identical to the runtime
used for the other final PPGN predictions.

## Notes

The train and predict trees should not be silently merged. They differ in ways
that matter for provenance, especially around prediction-time behavior.

The source was copied without Python bytecode caches or backup files. File
hashes are recorded in `../../docs/model_source_hashes_20260625.csv`.
