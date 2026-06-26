# Spring Embedding Engine

This directory contains the source for the spring-relaxation executable used by
the MATLAB plotting code to generate 2D embedding example panels.

The engine starts from a `.vt2d` vertex-model geometry and a prediction text
file block for one graph. It performs the requested T1 transition, relaxes edge
length springs toward the supplied target lengths, and writes relaxed geometry
and per-edge output tables under `output/` in the engine working directory. The
executable creates that directory when needed.

## Build

The source uses C++ features even though the historical entry file is named
`main.c`, so compile with a C++ compiler.

### Make

```bash
cd external/spring_embed
make
```

This writes:

```text
build/spring_embed
```

To choose a different compiler:

```bash
make CXX=clang++
```

### CMake

```bash
cd external/spring_embed
cmake -S . -B build
cmake --build build
```

## Run

```bash
build/spring_embed <initial.vt2d> <prediction_file> <target_sim_id>
```

The MATLAB analysis pipeline normally runs this executable for you. Point MATLAB
to the compiled binary with either:

```matlab
setenv('DCG_EMBED_ENGINE', '/path/to/GNN-Benchmark-Code/external/spring_embed/build/spring_embed')
```

or by setting `cfg.embed_engine` in an untracked `DCG_local_config.m`.

## Numerical Settings

The publication engine is configured in `src/main.c` with:

- `kA = 0`: length-only embedding; the area term is disabled.
- `kS = 1`: edge spring coefficient.
- `tMAX = 1000`, `h0 = 0.001`: relaxation runs until `Time < tMAX`.

These settings are the ones used to reproduce the saved manuscript embedding
outputs during the publication-code validation run.

## Data Inputs

The executable is not useful by itself without:

- matching `.vt2d` initial geometry files;
- prediction text files containing `Simulation id:` blocks;
- the target simulation id, e.g. `graph_256_17_0.txt`.

The publication repository does not commit raw `.vt2d` geometries, predictions,
or generated embedding outputs.

## Provenance

The source files here started from the validated spring-embedding source tree
used in the MATLAB publication-code validation on 2026-06-26. The repository
version adds only build/packaging hygiene: automatic `output/` directory
creation and removal of unused local variables that produced compiler warnings.
SHA256 hashes for the committed source files are recorded in
`source_manifest_sha256.csv`.
