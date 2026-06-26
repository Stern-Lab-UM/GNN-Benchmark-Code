# PPGN Source Snapshots

PPGN provenance is split across packages, so this directory keeps the split explicit.

## Directories

- `train_dcg/dcg/` was copied from: `Z:\Tomer\venv\dcg_blackwell\dcg\lib\python3.10\site-packages\dcg`
- `predict_dcg/dcg/` was copied from: `Z:\Tomer\PPGN_Tomer\dcg`
- `gl_tail_fixed_pkg/dcg/` was copied from: `Z:\Tomer\_gl_finals\dcg_pkg\ppgn_dcg_fixed_pkg\dcg`

The GL tail package is kept because the late `1_32` PPGN tail run used that fixed package, documented in the run script as byte-identical to the Armis2 runtime (`dcg` code hash `39490c43`).

## Notes

The train and predict trees should not be silently merged. They differ in ways that matter for provenance, especially around prediction-time behavior.

The source was copied without Python bytecode caches or backup files. File hashes are recorded in `../../docs/model_source_hashes_20260625.csv`.
