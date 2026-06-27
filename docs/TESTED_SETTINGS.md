# Tested Settings

This document records the software, hardware, and command settings used to test
the repository and to run the manuscript-scale code paths. It is intentionally
separate from `INSTALL.md`: installation instructions describe how to set up a
new machine, while this file records what has actually been exercised.

## Public Repository Validation

These checks were run after curating the publication repository.

| Date | Scope | Platform | Python / MATLAB | Command or check | Result |
| --- | --- | --- | --- | --- | --- |
| 2026-06-27 | Source syntax and rename audit | Windows 11 local workstation | Python 3.12.13 bundled runtime | compile all tracked `*.py` files from source with `compile(..., 'exec')`; `git grep` for obsolete project names; `git diff --check` | Passed. No obsolete project-name tokens remained in tracked files. |
| 2026-06-27 | Install checker smoke test | Windows 11 local workstation | Python 3.12.13 bundled runtime | `python scripts/check_install.py --packages-only --component all` | Checker ran, but this bundled Python environment did not contain required ML packages (`torch`, `torch_geometric`, `pyyaml`, `click`, `tqdm`, `hdf5storage`). This is an environment limitation, not a repository syntax failure. |
| 2026-06-27 | MATLAB name-resolution smoke test | Windows 11 local workstation | local MATLAB executable | `matlab -batch ...` minimal path/config check | Not completed: MATLAB failed at startup before loading repository code with a local file-system inconsistency error. |
| 2026-06-27 | Focused PPGN mini pipeline validation after GL handoff fix | Great Lakes `gl1250` A100 MIG slice | MATLAB R2025b; Python 3.11.5 venv; PyTorch 2.12.1+cu130 | `GNNBenchmark_run_publication_pipeline('mode','mini','models',{'PPGN'},'cuda',0,...)` via `_remote_q_gl_test/cmd_018_codex_fix_ppgn_cap11e6` | Passed end-to-end through BO, training, prediction, embedding, analysis, and figure creation. Mini PPGN LR range was capped to `[1e-5, 1.1e-5]`; mini MAE table gave W = 0.0863 and UW = 1.733, below the sanity threshold of 10. |

## Manuscript Run Contexts

The repository contains source snapshots, not trained checkpoints or raw output
trees. The model source snapshots were copied into this repository on
2026-06-25 and hash-recorded in `docs/model_source_hashes_20260625.csv`.

| Code family | Repository path | Runtime context used for manuscript-scale runs | Notes |
| --- | --- | --- | --- |
| GraphSAGE, GAT, GIN, PNA | `models/mpnn/` | Linux HPC GPU jobs; Python/PyTorch/PyG source tree with `trainer_final.py` and `predict_final.py` | The code is written to use PyTorch device selection and can run on CPU for small checks, but manuscript-scale training/prediction was GPU-oriented. |
| PPGN training | `models/ppgn/train_gnn_benchmark/` | Linux HPC GPU jobs; Python package invoked as `python -m gnn_benchmark_ppgn.main train` | Kept separate from prediction code to preserve provenance. |
| PPGN prediction | `models/ppgn/predict_gnn_benchmark/` | Linux HPC GPU jobs; Python package invoked as `python -m gnn_benchmark_ppgn.main predict` | Kept separate from training code to preserve provenance. |
| PPGN late GL tail package | `models/ppgn/gl_tail_fixed_pkg/` | Great Lakes GPU runtime documented as Python 3.10, PyTorch 2.11.0+cu128, CUDA 12.8 | Included because the late PPGN `1_32` tail run used this fixed package. |
| MATLAB analysis/plotting | `analysis/matlab/` | MATLAB on local/HPC-accessible storage reading consolidated prediction files | Starts from saved prediction files and split indices; does not retrain models. |
| Vertex-model data generation | `data_generation/vertex_model/` | Linux/HPC or local machine with MATLAB plus `g++` | Generates raw tissue files and assembles model-ready graph text files. |
| Spring embedding engine | `external/spring_embed/` | Linux/HPC or local machine with a C compiler | Compiled binary is not committed. The committed source uses `kA = 0` and 2000 integration updates. |

## Recommended Reproduction Settings

The following settings are the supported starting point for new users. They are
chosen to match the tested and documented paths above without assuming a
specific cluster.

- Python: 3.10 or 3.11 preferred.
- MATLAB: recent release with `bayesopt` support from Statistics and Machine Learning Toolbox.
- MPNN dependencies: `requirements/mpnn.txt`, plus a PyTorch/PyG wheel set matched to the target CPU/CUDA platform.
- PPGN dependencies: `requirements/ppgn.txt`, plus PyTorch matched to the target CPU/CUDA platform.
- Shared HPC setup: run `bash scripts/setup_lh_env.sh --component all` from a fresh clone; by default it creates an isolated environment under `$SCRATCH` and installs CPU-only PyTorch for safe import checks.
- GPU training: set `CUDA_VISIBLE_DEVICES` through the MATLAB/Python launcher options or the shell environment before launching long jobs. Exact GPU determinism is not guaranteed across PyTorch/CUDA versions.
- Output location: write generated data, checkpoints, prediction files, embeddings, and figures outside the Git clone.

## Commands To Record For A New Test

When validating this repository on a new machine, record the following in a lab
notebook or update this file in a follow-up commit:

```bash
git rev-parse HEAD
python --version
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
python scripts/check_install.py --component all
```

For MATLAB:

```matlab
version
ver
addpath(genpath('/path/to/GNN-Benchmark-Code/analysis/matlab'))
which GNNBenchmark_publication_config
which GNNBenchmark_plot_everything
```

For C/C++ helper tools:

```bash
g++ --version
cmake --version
```

## Known Limits

- The public repository does not commit manuscript-scale raw datasets,
  predictions, embeddings, trained checkpoints, or generated figures.
- Exact numerical equality across independent GPU training runs should not be
  expected. For future exact smoke tests, use fixed inputs, fixed checkpoints,
  and CPU inference with tolerance-based comparisons.
- The Windows local environment used for the 2026-06-27 repository audit did
  not include the ML dependency stack and could not start MATLAB, so it should
  not be treated as a full runtime validation of training or plotting.
