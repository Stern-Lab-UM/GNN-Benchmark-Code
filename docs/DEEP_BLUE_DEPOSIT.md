# Deep Blue Data Deposit Preparation

This note collects the materials needed to deposit the manuscript data package
in University of Michigan Deep Blue Data. It is written as a working checklist:
the package should be assembled and checked locally or on a compute filesystem
before Tomer performs the final authenticated Deep Blue submission.

## Recommended Dataset Title

Data for: A Controlled in Silico Benchmark for GNN Prediction of Tissue Dynamics

## Recommended Creators

- Matej Krajnc
- Troy Comi
- Siqi Miao
- Adnan Hafeez
- Hadar Serviansky
- Pan Li
- Tomer Stern

Use the final manuscript author order if it changes.

## Contact

Tomer Stern, University of Michigan, tomers@umich.edu

## Short Description

This dataset supports a controlled in silico benchmark for graph neural network
prediction of epithelial tissue remodeling after T1 transitions. It contains the
model prediction files, train/validation/test split manifests, trained model
checkpoints, spring-embedding outputs, cached analysis tables, and manuscript
analysis outputs needed to reproduce the post-prediction analyses and figures.
The paired code repository is:

https://github.com/Stern-Lab-UM/GNN-Benchmark-Code

## Suggested Abstract

This data package accompanies the manuscript "A Controlled in Silico Benchmark
for GNN Prediction of Tissue Dynamics." The benchmark asks graph neural networks
to predict relaxed post-T1 cell-cell interface lengths from simulated epithelial
tissues. The package includes weighted and unweighted prediction outputs for
GraphSAGE, GAT, GIN, PNA, and PPGN across training-set sizes and revision
conditions; train/validation/test split files; final trained model checkpoints;
per-graph spring-embedding outputs; analysis caches; generated manuscript
figures; feature/head ablation outputs; and the corrected counterfactual
copying analysis in which distal pre-T1 input lengths were perturbed by
delta = 0.05 at edge-hop h >= 14. These files are intended to be used with the
paired GitHub repository to regenerate summaries, diagnostics, and manuscript
figures without retraining all models.

## Keywords

graph neural networks; tissue dynamics; epithelial tissue; T1 transition;
vertex model; prediction benchmark; message passing neural network; PPGN;
simulation; spring embedding

## Recommended License

Use an open attribution license for the data, such as CC BY 4.0, if Deep Blue
offers it for this deposit. The code repository itself uses the MIT License.

## Package Layout

The prepared package should follow `docs/DATA_PACKAGE.md`:

```text
gnn_benchmark_public_data_<date>/
  README_DATA_PACKAGE.md
  predictions/consolidated/
  embeddings/per_graph/
  analysis_tables/analyzer_cache/revision_2026/
  figures/
  final_models/consolidated/
  manuscript_analyses/
  manifests/
```

The key manuscript-specific analysis folders under `manuscript_analyses/` are:

- `feature_head_ablation_20260619/`
- `counterfactual_copying_edgehop14_delta005/`

The counterfactual-copying folder must include the symmetric-pair PPGN rerun and
must not use the obsolete raw-directed PPGN outputs for manuscript conclusions.

## Deep Blue Practical Notes

As of 2026-07-08, Deep Blue Data states that deposits are free for eligible
University of Michigan depositors, DOI assignment is provided when the dataset
is published, browser upload is limited to 5 GB per deposit, and deposits above
1 TB require consultation. This package is expected to be far below 1 TB but
above the browser-upload threshold, so contact Deep Blue staff before the final
upload and ask whether they prefer a mediated transfer, Globus, or multiple
archive parts.

Useful Deep Blue pages:

- FAQ: https://deepblue.lib.umich.edu/data/help
- About: https://deepblue.lib.umich.edu/data/about
- Policies and terms: https://deepblue.lib.umich.edu/data/agreement

## Curator Contact Draft

Subject: Deep Blue Data deposit for GNN tissue-dynamics benchmark dataset

Hello Deep Blue Data team,

I am preparing a University of Michigan research dataset for the manuscript
"A Controlled in Silico Benchmark for GNN Prediction of Tissue Dynamics." The
dataset is approximately 50 GB before final compression and contains many small
prediction, embedding, analysis-table, and checkpoint files. The data are
simulation/model outputs and are intended for public reuse with an accompanying
GitHub code repository.

Because the dataset is above the browser-upload threshold, could you advise on
the preferred upload route? I can provide the package as one archive, several
smaller archive parts, or by another mechanism such as Globus if preferred.

Contact: Tomer Stern, University of Michigan, tomers@umich.edu

Thank you.

## User-Only Submission Steps

These steps require Tomer's U-M account or final depositor decisions:

1. Log into Deep Blue Data with U-M credentials.
2. Confirm final author order, ORCID values, funding fields, and related
   manuscript/preprint citation.
3. Choose the final data license, preferably CC BY 4.0 if appropriate.
4. Decide whether the dataset should be public immediately or embargoed until
   manuscript/preprint posting.
5. Coordinate the large-file upload route with Deep Blue staff.
6. Approve and submit the final deposit.
