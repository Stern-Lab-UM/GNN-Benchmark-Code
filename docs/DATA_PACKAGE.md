# Public Data Package Layout

The manuscript data should be deposited outside the Git repository, but the
recommended archive layout mirrors the output tree created by
`GNNBenchmark_run_publication_pipeline`.

A prepared data package should look like this:

```text
gnn_benchmark_public_data_<date>/
  README_DATA_PACKAGE.md
  predictions/consolidated/
    *.pred.txt
    splits/<split_key>/{train.inds,val.inds,test.inds,_applies_to.txt}
  embeddings/per_graph/
  analysis_tables/analyzer_cache/
  figures/
  final_models/consolidated/
  manuscript_analyses/feature_head_ablation_20260619/
  manifests/
    public_data_manifest.csv
    public_data_summary_by_category.csv
    source_vs_package_counts.csv
    compute_sha256_manifest.ps1
```

For analysis-only reproduction, point the MATLAB analysis code at the
consolidated prediction folder:

```matlab
package_root = '/path/to/gnn_benchmark_public_data_20260627';
data_root = fullfile(package_root, 'predictions', 'consolidated');
setenv('GNN_BENCHMARK_DATA_ROOT', data_root);

GNNBenchmark_rebuild_all_summaries
GNNBenchmark_plot_everything
```

For embedding-bound analyses that start from saved spring-embedding outputs:

```matlab
embedding_root = fullfile(package_root, 'embeddings', 'per_graph');
GNNBenchmark_analyze_embedding_error_bounds('embedding_root', embedding_root)
```

The `final_models/consolidated/` folder is included for provenance and optional
reuse, but manuscript figure reproduction should not require retraining if
`predictions/consolidated/`, `splits/`, and the saved embedding outputs are
present.

The local Stern Lab staging package created on 2026-06-27 is:

```text
Z:\Tomer\gnn_benchmark_public_data_20260627
```

That local staging tree uses hardlinks to avoid duplicating data on `Z:`. A
zipped or tarred deposition archive will contain regular file contents.