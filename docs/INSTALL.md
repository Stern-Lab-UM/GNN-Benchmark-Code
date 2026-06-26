# Installation And Environment Checks

This repository currently contains source snapshots and analysis code, not a polished Python package. The installation goal at this stage is therefore:

1. create a Python environment with the required dependencies;
2. make sure the curated source snapshots can be imported;
3. later add toy data and checkpoint smoke tests for exact numerical checks.

## Recommended Python Version

Use Python 3.10. The PPGN run packages were Python 3.10-based, and the curated MPNN tree had Python 3.10/3.11 bytecode artifacts in the original run tree.

## Known Run-Time Context

These source snapshots were copied from the code used for the manuscript revision runs:

- MPNN/PNA source: `models/mpnn/`
- PPGN training source: `models/ppgn/train_dcg/`
- PPGN prediction source: `models/ppgn/predict_dcg/`
- PPGN GL tail source: `models/ppgn/gl_tail_fixed_pkg/`

The late PPGN GL tail run was documented as using Python 3.10, PyTorch 2.11.0+cu128, CUDA 12.8, and the fixed `dcg` package hash `39490c43`. External users do not need that exact GPU stack to inspect or import the code, but exact GPU retraining reproducibility may depend on the machine and PyTorch stack.

## Option A: Conda/Mamba Starting Point

```bash
conda env create -f environment.yml
conda activate dcg-gnn
python scripts/check_install.py --component all
```

If PyTorch Geometric installation fails, install PyTorch and PyG using the wheel matrix appropriate for the user's CUDA/CPU platform, then rerun:

```bash
python scripts/check_install.py --component all
```

## Option B: Python venv Starting Point

Linux/macOS:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/mpnn.txt
python scripts/check_install.py --component all
```

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements\mpnn.txt
python scripts\check_install.py --component all
```

For PPGN-only inspection:

```bash
python -m pip install -r requirements/ppgn.txt
python scripts/check_install.py --component ppgn
```

For MPNN/PNA inspection:

```bash
python -m pip install -r requirements/mpnn.txt
python scripts/check_install.py --component mpnn
```

## What The Checker Does

`scripts/check_install.py` verifies:

- Python version and executable path;
- third-party imports such as `torch`, `torch_geometric`, `numpy`, `pandas`, `click`, `pyyaml`, and `hdf5storage`;
- importability of the curated MPNN source tree;
- importability of all three curated PPGN source snapshots.

It does not require raw data, embeddings, trained checkpoints, or generated figures.

## Exact Numerical Smoke Tests

The current checker is an installation/import check. A stronger future smoke test should add:

- a tiny public example graph or sanitized manuscript-style graph;
- a tiny fixed checkpoint, or a deterministic CPU-only toy checkpoint;
- expected prediction and metric files;
- a command that compares produced outputs to expected values within a fixed tolerance.

Training from scratch with a seed is usually not enough for exact equality across machines, especially on GPU. For exact reproducibility, prefer fixed input plus fixed checkpoint plus CPU inference.
