# Public Data Archive

The manuscript data are archived on Zenodo:

[doi:10.5281/zenodo.21286579](https://doi.org/10.5281/zenodo.21286579)


## Archive Contents

The Zenodo record contains seven logical ZIP archives plus three small metadata
and verification files:

```text
01_readme_manifests.zip
02_predictions_consolidated.zip
03_embeddings_per_graph.zip
04_analysis_tables.zip
05_figures.zip
06_final_models.zip
07_manuscript_analyses.zip
README_ARCHIVES.md
archive_checksums_sha256.csv
archive_listing_test.csv
```

The ZIP archives collectively represent the public data package used for the
manuscript analyses. They include model prediction files, train/validation/test
splits, saved spring-embedding outputs, analysis tables, generated figures,
final trained checkpoints, feature/head ablation outputs, and the corrected
counterfactual-copying analysis.

## Extracting The Data

Download all Zenodo files into one directory, then extract the seven ZIP files
into the same package root. After extraction, the package root should contain:

```text
gnn_benchmark_public_data/
  README_DATA_PACKAGE.md
  predictions/consolidated/
    *.pred.txt
    splits/<split_key>/{train.inds,val.inds,test.inds,_applies_to.txt}
  embeddings/per_graph/
  analysis_tables/analyzer_cache/revision_2026/
  figures/
  final_models/consolidated/
  manuscript_analyses/feature_head_ablation_20260619/
  manuscript_analyses/counterfactual_copying_edgehop14_delta005/
  manifests/
    public_data_manifest.csv
    public_data_summary_by_category.csv
    verification_problems.csv
```

The file-level manifest inside `01_readme_manifests.zip` uses package-relative
paths only and includes SHA-256 checksums for the packaged files. Archive-level
checksums are provided in `archive_checksums_sha256.csv`.

## Reproducing Analyses From The Package

In MATLAB:

```matlab
addpath(genpath('/path/to/GNN-Benchmark-Code'))
package_root = '/path/to/gnn_benchmark_public_data';
report = GNNBenchmark_run_from_data_package(package_root);
```

To choose the destination explicitly:

```matlab
report = GNNBenchmark_run_from_data_package(package_root, ...
    'output_root', '/path/to/reanalysis_outputs');
```

By default the runner reparses the consolidated prediction files rather than
trusting cached summaries. If called with `rebuild_summaries=false`, it uses
existing summaries from `analysis_tables/analyzer_cache/revision_2026/`, with
silent compatibility for older cache-folder names when needed. It also analyzes
`embeddings/per_graph/` when that folder is present.

Embedding example panels inside the main plotting script are off by default
because they require vt2d geometry and a spring executable. The saved per-graph
embedding outputs are sufficient for the manuscript embedding-error bound
analysis.

The corrected counterfactual-copying analysis uses the edge-hop h >= 14,
delta = 0.05 materials and the symmetric-pair PPGN rerun included under
`manuscript_analyses/counterfactual_copying_edgehop14_delta005/`.

The `final_models/consolidated/` folder is included for provenance and optional
reuse. Manuscript figure reproduction should not require retraining if
`predictions/consolidated/`, `splits/`, and the saved embedding outputs are
present.
