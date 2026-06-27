# Vertex-Model Data Generation

This folder contains the vertex-model simulator source and MATLAB wrappers used
to regenerate the tissue-graph datasets for the manuscript revision.

Generated data are intentionally not tracked by git. By default, MATLAB writes
outputs to:

```text
<repo>/generated_data/vertex_model/
```

The output can be moved anywhere by passing `output_root`.

## Conditions

The canonical generated datasets are:

- `standard_16`: 16^2 cells, `kA=100`, shear factor `1.0`, one T1.
- `kA_10`: 16^2 cells, `kA=10`, one T1.
- `kA_1`: 16^2 cells, `kA=1`, one T1.
- `shear_1_2`: 16^2 cells, `kA=100`, shear factor `1.2`, one T1.
- `shear_1_5`: 16^2 cells, `kA=100`, shear factor `1.5`, one T1.
- `tissue_484`: 22^2 cells, `kA=100`, one T1.
- `tissue_784`: 28^2 cells, `kA=100`, one T1.
- `flip_two`: 16^2 cells, `kA=100`, two T1 events.

`kA_100`, `shear_1_0`, and `tissue_256` are aliases of `standard_16`; they are
not separate simulations.

## MATLAB Entry Point

From MATLAB:

```matlab
addpath(genpath('/path/to/GNN-Benchmark-Code/data_generation/vertex_model'))

% One publication-scale graph per generated condition.
DCG_generate_vertex_model_datasets('mode', 'minimal', 'workers', 1)

% Run only selected conditions, useful for validation.
DCG_generate_vertex_model_datasets('mode', 'minimal', 'datasets', {'kA_10'}, 'workers', 1)

% Full manuscript graph manifests. Run this on a compute node.
DCG_generate_vertex_model_datasets( ...
    'mode', 'publication', ...
    'output_root', '/path/to/generated_data/vertex_model', ...
    'workers', 20)
```

The wrapper compiles `src/main.c` with `g++`, runs the initial-tissue and T1
relaxation phases, and assembles model-ready graph files.

## Simulator Command

The compiled simulator takes:

```text
vertex_model_generator Nx kA packageID SigI shearFactor T1EdgeID_1 T1EdgeID_2
```

For example:

```bash
./vertex_model_generator 16 10 10 0 1 -1 -1   # initial tissue
./vertex_model_generator 16 10 10 0 1 1 -1    # one T1 graph
```

The simulator reads and writes only relative to the current working directory,
using an `output/` subfolder.


## Raw Filename Note

The standard 16^2-cell archive used a shorter historical filename such as
`graph_256_10_0_1.txt`. The packaged simulator writes the current explicit form,
for example `graph_256_10_0_1_-1_1.txt`, which includes both T1-edge arguments
and the shear factor. The assembler accepts either form; graph order is governed
by the manifest, not by alphabetical sorting.
## Model-Ready Output

For each dataset key, MATLAB writes:

```text
model_ready/<dataset_key>/2D/<dataset_key>_weighted.txt
model_ready/<dataset_key>/2D/<dataset_key>_unweighted.txt
model_ready/<dataset_key>/2D/train.inds
model_ready/<dataset_key>/2D/val.inds
model_ready/<dataset_key>/2D/test.inds
model_ready/<dataset_key>/splits/<training_set_...>/{train,val,test}.inds
raw/<dataset_key>/output/final_*.vt2d
raw/<dataset_key>/output/graph_*.txt
```

Weighted rows contain:

```text
cell_id_1 cell_id_2 input_preferred_length input_was_flipped output_preferred_length
```

Unweighted rows contain:

```text
cell_id_1 cell_id_2 output_preferred_length
```

For a flipped interface, the removed edge is represented with output length
zero. The newly created interface is added as an extra edge with input length
zero and output equal to the new post-T1 length. The assembler infers that new
interface from the two common post-T1 neighbors of the removed edge and stops
with an error if the inference is ambiguous.

## Manifests

`manifests/*_graph_order.csv` records the graph order used by the manuscript
merged datasets. Split-index files are stored under `manifests/splits/`.
For `standard_16`, the split manifests include the final standard V1 cohort sizes
`standard_2_1`, `standard_2_2`, `standard_2_4`, `standard_2_8`, `standard_2_16`, and `standard_1_32`.


The `kA_1` archive contains 815 graph entries rather than 816; the manifest
records the graph order that exists in the archived merged file.

## Manual Build

On Linux:

```bash
cd /path/to/GNN-Benchmark-Code/data_generation/vertex_model
make
```

The source hashes are recorded in `source_manifest_sha256.csv`.
