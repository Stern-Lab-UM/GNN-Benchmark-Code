# Installation And Environment Checks

This repository currently contains source snapshots and analysis code, not a polished Python package. The installation goal at this stage is therefore:

1. create a Python environment with the required dependencies;
2. make sure the curated source snapshots can be imported;
3. later add toy data and checkpoint smoke tests for exact numerical checks.

For a record of the platforms, dependency stacks, and validation commands that
have actually been exercised, see `docs/TESTED_SETTINGS.md`.

## Recommended Python Version

Use Python 3.10 or newer; Python 3.10 or 3.11 is preferred. The PPGN run packages were Python 3.10-based, and the curated MPNN tree had Python 3.10/3.11 bytecode artifacts in the original run tree. The LH setup helper auto-detects a usable Python >= 3.10 and refuses older defaults such as Python 3.6.

## Known Run-Time Context

These source snapshots were copied from the code used for the manuscript revision runs:

- MPNN/PNA source: `models/mpnn/`
- PPGN training source: `models/ppgn/train_gnn_benchmark/`
- PPGN prediction source: `models/ppgn/predict_gnn_benchmark/`
- PPGN GL tail source: `models/ppgn/gl_tail_fixed_pkg/`

The late PPGN GL tail run was documented as using Python 3.10, PyTorch 2.11.0+cu128, CUDA 12.8, and the fixed `gnn_benchmark_ppgn` package hash `39490c43`. External users do not need that exact GPU stack to inspect or import the code, but exact GPU retraining reproducibility may depend on the machine and PyTorch stack.

## Option A: Conda/Mamba Starting Point

```bash
conda env create -f environment.yml
conda activate gnn-benchmark
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


## Shared Cluster / Lighthouse Isolation Notes

Multiple copies of this code can coexist on Lighthouse or another shared HPC system if each user keeps both the clone and Python environment isolated.

Recommended pattern for a new user:

```bash
git clone https://github.com/Stern-Lab-UM/GNN-Benchmark-Code.git
cd GNN-Benchmark-Code
bash scripts/setup_lh_env.sh --component all
```

The setup helper creates an isolated virtual environment under `$SCRATCH` by default, installs the relevant requirements, and runs `scripts/check_install.py`. By default it installs CPU-only PyTorch to avoid accidentally downloading a large CUDA stack on no-GPU login or CPU sessions. Advanced users can choose a different environment directory:

```bash
bash scripts/setup_lh_env.sh --component mpnn --env-dir "$SCRATCH/gnn_benchmark_envs/mpnn_publication"
```

If using conda/mamba on a shared system, prefer an explicit environment prefix rather than a generic named environment:

```bash
mamba env create --prefix "$SCRATCH/gnn_benchmark_envs/gnn_benchmark_code_py310" -f environment.yml
conda activate "$SCRATCH/gnn_benchmark_envs/gnn_benchmark_code_py310"
python scripts/check_install.py --component all
```

This avoids collisions with older environments named `gnn-benchmark` or with other versions already present under the same account.

Important collision points:

- Do not permanently export this repository, an older repository, or old run folders in `PYTHONPATH` from `.bashrc` or `.bash_profile`.
- Do not install multiple editable versions of packages named `gnn_benchmark_ppgn` into the same environment. PPGN uses the Python package name `gnn_benchmark_ppgn`, so only one PPGN snapshot should be active on `PYTHONPATH` at a time.
- The MPNN scripts use local top-level module names such as `dataset`, `models`, `trainer_final`, and `predict_final`. Run them from this repository/source tree or use explicit paths so Python does not import an older local file with the same name.
- Activate the intended environment before installing or running anything, and prefer `python -m pip ...` over plain `pip ...`.
- Before long runs, check `which python`, `python -m pip --version`, and `python scripts/check_install.py --component all`.
- Keep outputs, checkpoints, prediction files, embeddings, and figures outside the cloned code repository unless they are tiny documented examples.

The checker prints the Python executable, platform, package versions, and imported repository module paths. If those paths do not point to the user's clone, the environment is not isolated correctly.


## PyTorch Install Modes

`scripts/setup_lh_env.sh` installs CPU-only PyTorch by default:

```bash
bash scripts/setup_lh_env.sh --component all --torch cpu
```

This is the safest choice for no-GPU LH smoke tests and avoids pulling multi-GB CUDA wheels. For a GPU environment where the user intentionally wants the default PyPI CUDA-enabled PyTorch packages, use:

```bash
bash scripts/setup_lh_env.sh --component all --torch default
```

For an already prepared environment, use:

```bash
bash scripts/setup_lh_env.sh --component all --torch skip
```

Exact GPU reproduction may require installing the specific PyTorch/CUDA build appropriate for the cluster before running the checker.


## Bayesian Optimization

The MATLAB BO drivers live under `training/bayesopt/`. Add them to MATLAB's path:

```matlab
addpath(genpath('/path/to/GNN-Benchmark-Code/training/bayesopt'))
spaces = GNNBenchmark_bayesopt_search_spaces();
```

Before launching trials, activate or point to the intended Python environment.
For example:

```bash
export GNN_BENCHMARK_MPNN_MODULE_PREFIX='source /path/to/mpnn_env/bin/activate; '
export GNN_BENCHMARK_PPGN_MODULE_PREFIX='source /path/to/ppgn_env/bin/activate; '
```

`optimize_MPNN.m` defaults to this repository's `models/mpnn/trainer_final.py`;
pass `trainer_py` or set `GNN_BENCHMARK_MPNN_TRAINER` to use another trainer snapshot.
See `training/bayesopt/README.md` for full examples.

## Vertex-Model Generator

The manuscript tissue graphs can be regenerated from the vertex-model simulator
under `data_generation/vertex_model/`. The MATLAB wrapper compiles the C/C++
source with `g++`, runs the simulator, and assembles weighted/unweighted graph
files for the Python training code.

On Linux or Lighthouse:

```matlab
addpath(genpath('/path/to/GNN-Benchmark-Code/data_generation/vertex_model'))
GNNBenchmark_generate_vertex_model_datasets('mode', 'minimal', 'workers', 1)
GNNBenchmark_generate_vertex_model_datasets('mode', 'minimal', 'datasets', {'kA_10'}, 'workers', 1)
```

For the full publication manifests, run from a compute node and write outputs to
scratch or project storage rather than to the git clone:

```matlab
GNNBenchmark_generate_vertex_model_datasets( ...
    'mode', 'publication', ...
    'output_root', '/path/to/generated_data/vertex_model', ...
    'workers', 20)
```

The baseline conditions `kA_100`, `shear_1_0`, and `tissue_256` are aliases of
`standard_16`; the wrapper records this in the output rather than regenerating
three identical datasets. See `data_generation/vertex_model/README.md` for the
condition list, raw file conventions, and output layout.

## Spring Embedding Engine

The MATLAB embedding example figures use a small external spring-relaxation
executable. The source is included under `external/spring_embed/`; compiled
binaries are not committed. The committed engine uses `kA = 0` and 2000
integration updates.

On Linux or a shared cluster with `g++`:

```bash
cd external/spring_embed
make
```

This creates:

```text
external/spring_embed/build/spring_embed
```

Then point MATLAB to it:

```matlab
setenv('GNN_BENCHMARK_EMBED_ENGINE', '/path/to/GNN-Benchmark-Code/external/spring_embed/build/spring_embed')
```

The same directory also contains a `CMakeLists.txt` for users who prefer CMake:

```bash
cmake -S external/spring_embed -B external/spring_embed/build
cmake --build external/spring_embed/build
```

If the engine or `.vt2d` geometry folders are not available, disable embedding
example panels with:

```matlab
GNNBenchmark_CONFIG.embed_examples = false;
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
