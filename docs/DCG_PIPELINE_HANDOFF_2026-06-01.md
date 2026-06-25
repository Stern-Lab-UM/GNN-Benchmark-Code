# DCG GNN-Benchmark Analysis Pipeline â€” Handoff & Math Reference
**Date:** 2026-06-01  Â·  **Scope:** from *prediction files on disk* â†’ *all numeric analyses + manuscript figures*
**Pipeline home (CANONICAL):** `Z:\Tomer\DCG 2024\analysis_pipeline_2026\` â€” the 13 pipeline `.m` files + this
handoff; runs standalone (scripts cross-reference via relative `code_dir`; `readmatrix`/`turbo` resolve to
R2024a built-ins). **Add this folder to your MATLAB path.**  Â·  **Canonical data:**
`Z:\Tomer\gnn_benchmark_consolidated_20260530\`
**Pre-move archive:** the identical files remain in `Z:\Tomer\DCG 2024\Code\` with a `_20260601` suffix
(frozen 2026-06-01 snapshot; not loaded under their normal names).

> Provenance of this document: the analyzer (`DCG_analyze_results_2026_codex.m`), `load_dataset.m`, and
> `DCG_consolidated_paths_2026_codex.m` were read line-for-line. The plotter's normalization / SD /
> baseline / aggregation code was verified directly against source (lines cited below). The remaining
> plotter helpers and the Flip_two / composites / calibrate / verify modules are documented from a
> detailed line-cited code audit. All file:line citations below were taken from the live files in the
> code root on 2026-06-01.
>
> **Line-number caveat:** the in-script documentation pass (later on 2026-06-01) inserted comment blocks
> (+~1710 lines in the plotter, +~800 in Flip_two), so citations into *those two files* now read ~that many
> lines LOW. Anchor on the **function name / quoted code snippet** (stable), not the raw line number.

---

## 0. TL;DR of the math decisions (the things you asked about)

| Decision | What we do | Where |
|---|---|---|
| **Removed vs. new interface** | **DROP the removed (eliminated) edge** (flag==1, post-T1 length 0); **KEEP the newly-formed edge** (pre-T1 length 0). Analysis is on the new interface + all surviving edges. | analyzer 1163-1168, 1180-1195, 1350-1357; `is_new_interface_extra` |
| **Ground truth (per edge)** | true post-T1 length = column `end-1` of the W matrix | analyzer 870, 880 |
| **Prediction (per edge)** | model output = column `end` | analyzer 879 |
| **Error (per edge)** | `abs(pred âˆ’ truth)` (absolute; never squared) | analyzer 881 |
| **Baseline ("no_learning")** | identity: "nothing moves" â†’ prediction = pre-T1 length (column 3); baseline error = `abs(pre-T1 âˆ’ truth)` | analyzer 614-667; plotter 1443-1445 |
| **Normalization (nMAE)** | `nMAE = model_MAE / baseline_MAE`; plotted as `log2(nMAE)` | plotter 585-586, 613-614, 1462; assumptions item 7 |
| **Averaging order** | edges â†’ **mean within a graph** â†’ **mean across graphs (graph-weighted)** â†’ **mean across the 5 seeds**; log2 placement = **MAE arithmetic** (log after graph-mean) / **nMAE geometric** (log per graph), Â§7.11 | plotter 2453-2455, 2488-2525, 1449-1463 |
| **Uncertainty reported** | **SD across the 5 seeds** (shaded band) | plotter 2455, 2525, 217 |
| **Split** | **test only**, hard-enforced everywhere | analyzer 267-270; plotter 322-329 |
| **Reference model** (GT/baseline/edges source) | priority `PNA â†’ GraphSAGE â†’ GAT â†’ GIN â†’ PPGN`; first one with real data | analyzer 636-654 |
| **Hop distance** | row-preserving cell-share BFS from the T1 root edge; one hop = two interfaces share a cell; multi-T1 â†’ distance to **nearest** T1 | analyzer 836-851 |

Â§7 lists the now-**resolved** issues (SD estimator, Flip_two nMAE, stale header, hex PPGN naming, and the
size-vs-distance aggregation rule â€” MAE arithmetic / nMAE geometric, unified Â§7.11) and three **dead
functions** â€” read before trusting edge cases.

---

## 1. What the pipeline does (end to end)

Each GNN model (4 message-passing nets â€” GraphSAGE, GAT, GIN, PNA â€” plus PPGN) was trained to predict,
for every cellâ€“cell interface (edge) in a tissue graph, the interface's **length after a T1 transition**.
For each (dataset, model, seed) there is one prediction text file. This pipeline:

1. **Parses** every prediction file into a uniform in-memory struct `I` (per model, per cohort, per seed).
2. **Cleans** each graph (dedup edges, drop the eliminated-edge row), builds the row-preserving **cell-share interface graph**, computes
   each edge's **hop distance** from the T1 site for single-root datasets, and assembles per-edge
   **prediction / ground-truth / error** matrices (one column per model). Flip_two gets an additional
   two-root analysis that recomputes distances to both concurrent T1 interfaces.
3. **Summarizes** into `S` (a `[n_seeds Ã— n_size_bins]` cell structure) and saves `results_summary.mat`.
4. **Plots**: MAE-vs-dataset-size, generic single-root MAE-vs-distance-from-T1, scatter examples,
   manuscript hexagonality panels, a PPGN "fallback" diagnostic, 2-D embedding example figures, the
   Flip_two two-T1 interaction/single-vs-two comparison analysis, and multi-panel composites; plus
   per-figure-type y-range calibration.

The science target is **how prediction error grows with distance from the T1 site**, **how models compare
to the identity baseline (normalized MAE)**, and **how error depends on local tissue regularity
(hexagonality)**.

---

## 2. Run order & file map

### 2.1 What you actually invoke (two steps)

**Step 1 â€” build numeric analyses** (prediction files â†’ `results_summary.mat`, one per dataset):
```matlab
% All 10 datasets. hex and v1_UW are NOT in the driver's default list, so pass them explicitly.
datasets = {'v1_2_16_W','v1_UW','hex','Shear_1_2','Shear_1_5', ...
            'kA_1','kA_10','Flip_two','Tissue_484','Tissue_784'};
rebuild_summaries  = true;     % force fresh parse from the .pred.txt files
plot_after_summary = false;    % summaries only; no figures yet
run('Z:\Tomer\DCG 2024\Code\DCG_run_revision_analyses_2026_codex.m');
```
(Equivalent helper used 2026-06-01: `%TEMP%\dcg_rebuild_all10.m`. A permanent "rebuild all 10" wrapper
is *not* yet committed to the code root â€” see Â§7.)

**Step 2 â€” make all figures + reported numbers:**
```matlab
run('Z:\Tomer\DCG 2024\Code\DCG_plot_everything_2026_codex.m');
```

### 2.2 The 11 in-scope files

| Role | File | Functions |
|---|---|---|
| **Orchestrator (master plot)** | `DCG_plot_everything_2026_codex.m` | script |
| **Orchestrator (build + plot driver)** | `DCG_run_revision_analyses_2026_codex.m` | + `first_existing` |
| **Orchestrator (revision plots only)** | `DCG_plot_all_revision_figures_2026_codex.m` | script |
| **Rebuild wrapper (all 10 summaries)** | `DCG_rebuild_all_summaries_2026_codex.m` | script (added 2026-06-01) |
| **Analyzer (pred â†’ summary)** | `DCG_analyze_results_2026_codex.m` | + `extract_MP_results`, `extract_PPGN_results`, `read_dataset_inds` |
| **Pred-file parser** | `load_dataset.m` | 1 |
| **Snapshot path/inds resolver** | `DCG_consolidated_paths_2026_codex.m` | + `i_prefix_parts`, `i_inds_dir` |
| **Plotter (summary â†’ figs + numbers)** | `DCG_plot_results_2026_codex.m` | 69 local fns |
| **Y-range calibration** | `DCG_calibrate_saved_fig_y_ranges_2026_codex.m` | 5 |
| **Composite figures** | `DCG_make_revision_transverse_dist_composites_2026_codex.m` | 7 |
| **Flip_two two-T1 interaction + single-vs-two comparison** | `DCG_plot_Flip_two_interaction_2026_codex.m` | 37 |
| **Flip_two figure QA** | `DCG_verify_Flip_two_figures_2026_codex.m` | 1 |

**Out of scope** (these run *before* prediction files exist â€” model training / Bayesian HPO):
`run_PPGN_*_BO_*_2026_codex.m` (8 files).

### 2.3 Call graph
```
DCG_plot_everything            DCG_run_revision_analyses (rebuild_summaries=true)
  â”œâ”€ DCG_plot_results (v1_W)     â””â”€ DCG_analyze_results            â”€â”€ per dataset
  â”œâ”€ DCG_plot_results (v1_UW)         â”œâ”€ extract_MP_results  â”€â”
  â”œâ”€ DCG_plot_results (hex)           â”œâ”€ extract_PPGN_results â”œâ”€ load_dataset
  â””â”€ DCG_run_revision_analyses        â”œâ”€ read_dataset_inds   â”€â”˜
       (plot_after_summary=true)      â””â”€ DCG_consolidated_paths   (pred_file / inds_dir)
        â”œâ”€ DCG_plot_results  (per dataset, all the science figures)
        â”œâ”€ DCG_calibrate_saved_fig_y_ranges
        â”œâ”€ DCG_make_revision_transverse_dist_composites
        â”œâ”€ DCG_plot_Flip_two_interaction
        â””â”€ DCG_verify_Flip_two_figures
```

---

## 3. Data model

### 3.1 Consolidated snapshot layout (`DCG_consolidated_paths_2026_codex.m`)
```
<root>\<task>_<model>_<W|UW>_<size>_s<seed>.pred.txt      (one file per modelÃ—seedÃ—cohort)
<root>\splits\<key>\{train,val,test}.inds                (0-based; md5-deduped)
<root>\splits\<key>\_applies_to.txt                      (lists the flat prefixes this split serves)
```
`<task>` tokens: `standard-flip` (v1), `Hexagonality` (hex), `Shear`, `kA`, `Tissue`, `Flip-two`.
A flat prefix (`v1_2_16_W`, `hex_2_8_W`, `rev_kA_1`, â€¦) â†’ (task, weighting, size) is decoded in
`i_prefix_parts` (paths 54-85). Split folders are deduped: a prefix â†’ folder lookup is driven by each
folder's `_applies_to.txt` (`i_inds_dir`, paths 88-112). **`.inds` are 0-based; +1 on read** to MATLAB
indexing (`read_dataset_inds`, analyzer 1441).

> **Size-token convention (data hygiene).** Every model file of a given dataset must share the **same**
> `<size>` token â€” e.g. all hex files are `â€¦_W_2_8_â€¦`, all Flip_two files are `â€¦_W_na_â€¦`. A 2026-06-01 bug
> had hex **PPGN** as `â€¦_W_na_â€¦` while hex's MP models were `â€¦_W_2_8_â€¦`, so the analyzer silently missed
> PPGN (Â§7.9). If you add a model and it shows as absent despite its files existing, check its `<size>` token.

### 3.2 Prediction-file columns (after `load_dataset` + cleanup)
| Task | Raw cols | Meaning |
|---|---|---|
| **W** (`lengths_to_lengths`) | 6 | `[u, v, pre-T1 length, flag, true post-T1 (end-1), predicted post-T1 (end)]` |
| **UW** (`none_to_lengths`) | 4 | `[u, v, true post-T1 (end-1), predicted (end)]` |
| **PPGN W / UW** | 6 / 5 (+ interleaved 38-col node rows) | node rows truncated by `consider_nodes`; UW extra zero col-3 stripped (analyzer Â§8) |

Per-graph row hygiene (analyzer 1147-1168, 1338-1357): drop `u>v` rows (keep one direction), dedup exact
`(u,v)` keeping first occurrence (stable), sort by `(u,v)`, then **drop_flag1** (below). A 256-cell tissue
â†’ 769 raw W rows â†’ **768** after drop_flag1.

> **W is the master.** `lengths_to_lengths` always loads the `_W` file even for a `_UW` run; UW files are
> only used for the `none_to_lengths` task (analyzer 1109-1115, 304-310).

---

## 4. The math, in detail (with code citations)

### 4.1 The T1 transition â€” removed vs. new interface (`drop_flag1`)  â† you asked about this
A T1 transition **removes** one cellâ€“cell interface and **creates** a new one perpendicular to it. In the
raw W file each graph therefore has two special rows:
- the **eliminated** edge: `flag == 1`, has a positive pre-T1 length but **post-T1 length 0**;
- the **newly-formed** edge: pre-T1 length **0**, positive post-T1 length.

**Decision (BASELINE-FIX 2026-05-14, `drop_flag1`):** we **delete the eliminated-edge row entirely**
(analyzer 1165-1168, 1354-1357) and **keep the newly-formed edge** as the T1 hop-0 row.
Rationale (analyzer 1180-1195): the eliminated edge's post-T1 length is **known a priori to be 0**, so
evaluating either model error or the identity baseline on it is meaningless and, in the *old* merged-row
scheme, it inflated the apparent improvement-over-baseline by ~0.07 nMAE aggregate (~0.5 at hop 0). Gated
by `is_new_interface_extra` (=1 for all current data; analyzer 240-241) so legacy files without a separate
eliminated row pass through unchanged.

The **hop-0 root edge** is then located as the newly-formed edge `pre-T1 length == 0`
(`find(curr_G(:,3)==0)`, analyzer 839). For graphs with several concurrent T1s (Flip_two) there are several
roots (Â§4.13).

### 4.2 Per-edge quantities
For each surviving edge of each graph:
- **ground truth** = true post-T1 length = column `end-1` (analyzer 870, 880);
- **prediction** = model output = column `end` (analyzer 879);
- **error** = `abs(pred âˆ’ ground_truth)` (analyzer 881; UW 890). **Absolute error, never squared.**
Ground truth and the edge list are physical labels (identical across all models) â€” confirmed byte-identical
12240/12240 columns (analyzer 626-628).

### 4.3 The identity baseline (`no_learning`)
A synthetic 6th "model" whose prediction for every edge is the **pre-T1 length** (column 3): i.e. "assume
nothing changes across the T1." Built by cloning the reference model and overwriting the prediction column
with col 3 (analyzer 657-667). Its per-edge error is therefore `abs(pre-T1 âˆ’ truth)`. In the summary it is
appended as the **last** model column and later renamed `Baseline` for plotting (plotter 370).

### 4.4 Reference-model priority (GENERALIZATION 2026-05-31)
Ground truth, the edge list, and the `no_learning` baseline are all read from one **reference model**.
Historically hard-coded to PNA; now chosen by priority **`{PNA, GraphSAGE, GAT, GIN, PPGN}`** â€” the first
model that has *real* (non-NaN) data (analyzer 636-654). This is why, when PNA pred files were still being
generated, Shear/Flip_two/Tissue_484 fell through to **GraphSAGE** as reference (seen in the 2026-06-01
rebuild log) instead of silently writing an empty summary. It errors loudly if **no** model has real data
(analyzer 649-653). These columns are model-independent, so any present model is a valid reference.

### 4.5 Hop distance from the T1 site
Build the row-preserving **cell-share interface graph** of the tissue graph (vertices = interfaces/edges;
two interfaces adjacent iff they share a cell): `dcg_line_graph_preserve_rows(curr_G(:,1:2))` (analyzer 836).
Hop distance = unweighted BFS `distances(line_G, root_edge)` from the hop-0 root. For periodic tissues,
periodic wrap hops are included through the wrap-around interfaces present in the source graph matrix.
**Multi-T1 graphs** (Flip_two) have a
root *vector*; distance is collapsed to the **nearest** T1: `min(dist_mat,[],1)` (analyzer 848-850).
2026-06-06 audit hardening: the analyzer now errors if the root count is not exactly one for non-Flip_two
datasets or exactly two for Flip_two, verifies that root rows still match line-graph node rows, and refuses
to save square-periodic summaries with impossible old-style non-periodic hop ranges (256-cell: >12; larger
square tissues: loose `ceil(sqrt(n_cells))` guard).
Edges are then sorted by hop and bucketed (`mat2cell` by per-hop counts `h`, analyzer 908-935), so each
`S.prediction_errors.<task>{seed,size}.test{graph}` is a cell of `[n_edges_at_hop Ã— n_models]` matrices.

### 4.6 Aggregation hierarchy (graph-weighted)
1. **within a graph**: MAE = `mean` of that graph's per-edge errors (`omitnan`) â€” plotter 2453, 2492.
2. **across graphs (per seed)**: mean of per-graph MAEs â€” **each graph equal weight** (plotter 2453-2454,
   2524). *This is a deliberate change from the old edge-weighted distance plot* (comment plotter 2482-2484).
3. **across the 5 seeds**: mean of per-seed values (plotter 2454, 2524).
`seeds = 1:5` (plotter 217). Empty size bins emit a NaN row of model-width so concatenation is clean
(plotter 2448-2451, 2477).

### 4.7 Normalization â€” `nMAE` (verified against source)
`calculate_normalization_factors` (plotter 1383-1467) computes the **baseline (identity) MAE** using the
*same* graph-weighting:
- per edge at hop h: `abs(pre_vals âˆ’ post_vals)` where `pre_vals = pred_cells{h}(:,end)` (the `no_learning`
  column = pre-T1) and `post_vals` = ground truth (plotter 1443-1445);
- â†’ `graph_mae_all(g)` = within-graph mean over all hops (1449);
- â†’ `per_size_seed(r)` = mean across graphs within a seed (1452);
- â†’ `per_size(siz)` = mean across seeds (1462). A per-hop vector `per_dist{siz}(h)` (1453, 1463) and a
  per-hexagonality-bin version (1455-1458, 1464) are built the same way.

The normalized pass (plotter 573-625, **verified**): divide every per-edge error by the scalar
`per_size(ss)` (586), re-run `extract_MAEs` with `use_log=1, normalized=1` (geometric: `log2` per graph,
then averaged), restore `S`; then divide by the per-hop vector `per_dist{ss}(h)` (614) and re-extract. Net
plotted quantity:
> **`log2(nMAE) = log2( model_MAE / baseline_MAE )`** (assumptions file item 7).

**Denominator safety** (`sanitize_normalization_scalar/vector`, plotter 1470-1486): any denominator that is
empty / non-finite / **â‰¤ `eps`** (~2.2e-16) becomes **NaN** (not clamped to a floor), so near-zero baselines
drop the point instead of exploding it.

**Baseline forced to exactly 0** (`force_baseline_reference_to_zero`, plotter 727-745, called 600-601,
624-625): the Baseline column of the normalized arrays is hard-set to 0 (since `log2(baseline/baseline)=0`),
giving a clean reference line with zero band. Baseline column resolved by name `Baseline|no_learning`, else
last column (plotter 596-599).

### 4.8 Uncertainty â€” SD across seeds
The shaded band on MAE-vs-size and MAE-vs-distance plots is the **standard deviation across the 5 seeds**
(not SEM, not CI). SEM was considered and rejected (commented `/sqrt(length(seeds))`, plotter 2558).
**All per-seed SDs are sample SD (Ã·Nâˆ’1)** as of 2026-06-01 â€” MAE-vs-size, MAE-vs-distance, and the
hexagonality path all use `std(...,0,...)`. (Before the fix the size path used `std(...,1)` = Ã·N, a
population-vs-sample mismatch; now resolved â€” see Â§7.1.) For the manuscript hexagonality panels and
`summarize_by_x`, `uncertainty_from_values` (plotter 2201-2213) supports `'sd'` (default,
`hex_paper_uncertainty='sd'`) or `'sem'`=SD/âˆšn.

### 4.9 Binning
- **Dataset size**: cohorts at 1,2,4,8,16,32 cells â†’ size-bin `log2(size)+1` = 1..6 (analyzer 761, 793).
  x-axis plotted as `2.^(0:n-1)` (plotter 2663). Revision datasets are single-size (one populated bin).
- **Hop distance**: integer hops 0,1,2,â€¦; `max_cell_dist` seeded per dataset (24 for 256-cell, 40/50 for
  484/784) and **auto-raised** from observed data if deeper (plotter 387-400).
- **Hexagonality (quality) bins**: `h_bins_for_quality_analysis = 0.4:0.1:1` (plotter 220), half-open
  `(bin(b), bin(b+1)]` (plotter 1456). Manuscript panels use finer `0.4:0.025:1` (plotter 222).
- **Active noise** (disorder) bins: `0:0.01:0.5`, used only when >80 unique noise levels else one point per
  level (plotter 223, 2026).

### 4.10 Hexagonality definitions (analyzer Â§14)
- **cell hexagonality** = the cell's degree in the dual graph = number of edges touching it (analyzer 941).
- **whole-graph hexagonality** = **fraction of cells with exactly 6 neighbors** (analyzer 943; plotter
  1746-1747). A perfect hex tissue â†’ 1.0.
- **edge ("neighborhood") hexagonality** = for each edge, the mean of `abs(degree âˆ’ 6)` over its **4-cell
  neighborhood** (the two cells it separates + their two shared neighbors); vectorized as
  `B * |cell_hex âˆ’ 6| / 4` (analyzer 960-980). A warning (not a halt) fires if a neighborhood â‰  4 cells,
  which can legitimately happen on revision tissues (analyzer 973-978).
- **disorder / active noise** = the generator's noise scale `graph_id.disorder`, with a `>1 â†’ /100` rescale
  in the plotter (plotter 1751-1753).

### 4.11 W vs UW
Two tasks: `lengths_to_lengths` (W, weighted) and `none_to_lengths` (UW). `lengths_to_lengths` always reads
the `_W` file (prefix swap, analyzer 1109-1110). A `v1_UW` run loads **both** tasks (analyzer 148). PPGN UW
files carry an always-zero extra column 3 that is stripped (analyzer Â§8, 534-542). W/UW per-graph row order
is aligned (analyzer Â§9); post-`drop_flag1` the `(u,v)` sets match exactly so the alignment is a near no-op.

### 4.12 PPGN specifics & the "fallback" diagnostic
- **PPGN is never run on the large tissues** (`Tissue_484`, `Tissue_784`) due to VRAM â€” `has_ppgn=false`
  there (analyzer 198, 205). PPGN NaN for Tissue is **by design**, not missing data.
- **hex PPGN was a *naming* gap, now fixed (2026-06-01).** The hex PPGN predictions existed on disk but
  were mis-tokened `Hexagonality_PPGN_W_na_s*` instead of `â€¦_W_2_8_s*` (the token its four MP models use),
  so the `hex_2_8_W â†’ â€¦_W_2_8` lookup missed them and hex was built without PPGN (analyzer logged
  `PPGN prefixes matched: 0/1`). Fixed by renaming the 5 pred + 5 model files `na`â†’`2_8` and rebuilding hex;
  validated independently (PPGN test-MAE 0.00284, nMAE 0.271, S-vs-pred Î”=3e-18). hex now carries all 5
  models. Root cause + guard in Â§7.9/Â§7.10.
- PPGN availability is auto-detected from files; partial seed coverage is tolerated and NaN-filled
  (analyzer 295-329, Â§6).
- **Fallback analysis** (`plot_PPGN_fallback`, plotter 1030-1344): per graph/model/hop,
  `FB(h) = log2( MAE(pred, true post-T1) / MAE(pred, pre-T1 baseline) )`. Negative â‡’ the model tracks the
  true post-T1 length; positive â‡’ it "fell back" to the do-nothing baseline. A `findchangepts` split
  (1 changepoint, mean statistic) gives before/after segment means; a graph is flagged "fallback" iff
  `mean_before â‰¤ âˆ’log2(2)` **and** `mean_after â‰¥ +log2(2)` â€” i.e. a factor-of-2 crossover
  (`ratio_threshold_identity_function = log2(2) = 1`, plotter 241; rule 1134-1135). Needs â‰¥4 valid non-hop-0
  hops; **hop 0 is excluded** (NaN). Reports `fallback_analysis_table.csv` + per-(model,size) fraction of
  graphs flagged. Focus model = PPGN if present else PNA.

### 4.13 Flip_two - two concurrent T1s (`DCG_plot_Flip_two_interaction_2026_codex.m`)
- **2026-06-03 update:** this function now implements the reviewer-facing minimal two-T1 package: overall
  single-T1 vs two-T1 graph error, nearest-T1 distance profiles, and graph error vs inter-flip distance,
  in addition to the previous zones/heatmaps/close-middle-far diagnostics.
- PPGN is **included**. PNA remains the canonical graph/GT/topology source; other models provide their
  aligned prediction column only.
- A Flip_two graph must have **exactly two** roots (`col 3 == 0`); otherwise it is counted in
  `skipped.not_two_roots`. The weighted single-T1 reference (`v1_W - analyses data.mat`) must have
  **exactly one** root; otherwise it is counted in `skipped_single.not_one_root`.
- Per Flip_two edge: `d1`,`d2` = cell-share hops to each root; `d_near=min(d1,d2)`,
  `d_far=max(d1,d2)`. **Inter-flip distance** = minimal periodic cell-share hop distance between the two roots. The single-T1
  reference stores its single-root distance as `d_near` so it can be overlaid against Flip_two
  `h_min(e)=min(h1,h2)`.
- The single-vs-two comparison is **unpaired**. `graph_idx` is retained for traceability, but the code
  does not assume that the v1_W and Flip_two graphs are the same tissues or matched perturbations.
- **Zones** (`near_radius = 3` hops): "near exactly one flip" (`xor`), "near both" (`&`), and "far from
  both"; roots are treated as interchangeable.
- **Close/middle/far** split = tertiles of the per-graph inter-flip distance.
- Error per edge = `abs(model_pred - GT)`; baseline = `abs(col3 - GT)`. Aggregation
  (`summarize_wide_records`): mean within graph/bin -> mean within seed -> mean across seeds; **SD is
  across seed means**.
- nMAE (**geometric**, per the unified rule Section 7.11): `mean(log2(model/baseline))` per seed (log per
  record, then averaged); MAE shown as log2 is **arithmetic** `log2(mean(model))`. Zero/negative errors are
  dropped from the log metrics.
- Outputs are written directly into the canonical `_figures\revision_codex_2026\Flip_two` dataset folder:
  original Flip_two record/summary tables, added `single_T1_*` and `single_vs_two_T1_*` CSVs, a pair-record
  `.mat`, an assumptions txt with both root-skip counters, and 11 `.fig` outputs: 3 single-vs-two
  comparison figures plus 8 two-source interaction figures.
### 4.14 2-D embedding example figures (plotter 2902-3493, opt-in)
Only when `DCG_CONFIG.embed_examples=true`. For a few example graphs it relaxes the GT, predicted, and
pre-T1 interface-length sets to 2-D cell polygons using an **external C++ engine** `spring_embed.exe`
(default `%TEMP%\springs_embed\spring_embed.exe`, plotter 2904; overridable via `DCG_CONFIG.embed_engine`),
recovers periodic topology by vertex snapping (tol 1e-9; neighbor iff â‰¥2 shared vertices), unwraps across
the periodic box by BFS, and **rigidly aligns** pred/base onto GT via a Kabsch/SVD fit with a reflection
guard (`emb_align_to`, 3432-3442). A new-T1 interface is identified from W column 4 > **0.5** (3304). A
**safety guard** recomputes `mean|col6âˆ’col5|` from the raw file and **skips** the panel if it disagrees with
the S-derived MAE by `> max(5e-3, 0.1Â·|sMAE|)` (3010-3014). This is the C++ vertex-model embedder
(`springs` project).

### 4.15 Figure post-processing
- **Y-range calibration** (`DCG_calibrate_saved_fig_y_ranges_2026_codex.m`): groups saved `.fig` by exact
  filename across datasets and unifies their y-limits to a common padded range, then **overwrites the
  `.fig`**. Padding = symmetric **5%** of the data span; for `raw MAE` figures the lower bound is clamped to
  0 (`prefer_zero_for_positive_raw`, calibrate 51, 135-137). âš  The extent uses object `YData`, which for
  errorbars is the **center only** (whisker deltas not included) â€” see Â§7.
- **Composites** (`DCG_make_revision_transverse_dist_composites_2026_codex.m`): builds 3Ã—3 panels from the
  per-dataset "MAE vs traverse dist" figures (column 1 = v1 reference; columns 2-3 = perturbed variants);
  unifies y across all 9 panels (same 5% padding / raw-MAE-zero rule), per-panel x padded Â±0.5.

---

## 5. Outputs

### 5.1 Numeric summaries (Step 1)
`<root>\_analyzer_cache\revision_codex_2026\<dataset> - results_summary.mat` holding `S`, `all_models`,
`tasks`, `data_sets`; plus a large `<dataset> - analyses data.mat` cache of raw `I`.
**`S` layout:** every field is a `[n_seeds Ã— n_size_bins]` cell; each populated cell is a struct with a
`.test` field = `{n_graphs Ã— 1}`; per-graph entries are cells of per-hop `[n_edges Ã— n_models]` matrices.
**Model-column order** = `all_models` = `[PPGN?, GraphSAGE, GAT, GIN, PNA, no_learning]` (PPGN only when
present; `no_learning` always last).

### 5.2 Figures (Step 2) â€” per dataset folder
`MAE vs dataset size (raw MAE | log2 MAE | log2 nMAE).fig`; `MAE vs traverse dist (â€¦ Ã—3).fig`;
`Scatter plot examples (<task>).fig`; `Manuscript hexagonality panels.fig`;
`<focus> fallback hop distribution / example change point / example scatter plot.fig`;
`Embedding examples overlay/per-edge (<task>).fig/.png`; plus the composite `Revision transverse dist
comparison (â€¦).fig`.

### 5.3 CSVs + "assumptions" text files (the in-tree methodology record)
`fallback_analysis_table.csv`; `manuscript_hexagonality_vs_active_noise_data.csv`;
`manuscript_identity_normalized_error_vs_hexagonality_data.csv`; the Flip_two record/summary CSVs; and
self-documenting `*_assumptions.txt` files (`normalization_assumptions.txt`,
`fallback_analysis_assumptions.txt`, `manuscript_hexagonality_assumptions.txt`, the Flip_two assumptions) â€”
these are written *by the code* and are the authoritative per-figure methodology statements.

---

## 6. Validation performed (why we trust the numbers)

1. **Split integrity** â€” for all 10 datasets, `test âˆ© train = 0` and `test âˆ© val = 0`.
2. **Test-set provenance** â€” an *independent* recompute of test-MAE directly from the raw `.pred.txt` files
   (re-applying the same dedup + drop_flag1 on `test.inds`) matches `S` to **~1e-18** (machine precision) for
   every in-sync model. This proves `S` holds the **test** split and applies the documented cleanups.
3. **Raw-units sanity** â€” test/val MAE ratio = **0.89â€“1.16** across datasets (no overfitting; the test set
   behaves like validation). (BO `min_objective` is a *normalized validation loss* and is **not** comparable
   to raw test distance â€” do not compare them.)
4. **Seed consistency** â€” 5 distinct seeds; per-model CV typically 2â€“20% (a few high-variance cases:
   GIN on kA_1, PNA on hex).
5. **Cross-dataset ranking** â€” model difficulty ordering is consistent (GraphSAGEâ†”GIN Spearman â‰ˆ +0.89,
   GraphSAGEâ†”GAT â‰ˆ +0.77); GAT is consistently the weakest learner (nMAE â‰ˆ 0.83â€“1.12).
6. **Clean rebuild (2026-06-01)** â€” all 10 summaries rebuilt from the now-complete prediction set; the
   re-validation run confirms every in-sync model still matches to ~1e-18. After the hex-PPGN naming fix
   (Â§4.12), hex PPGN also matches to **Î”=3e-18**, so **8/10 datasets now carry all 5 models** (Tissue Ã—2 are
   PPGN-less by design).

The validation harness lives at `%TEMP%\dcg_validate_all.m` (PARTs Aâ€“E).

---

## 7. Known issues, caveats & decisions to be aware of

1. **SD estimator inconsistency â€” RESOLVED 2026-06-01.** All per-seed SDs now use the sample estimator
   `std(â€¦,0,â€¦)` (Ã·(Nâˆ’1)): MAE-vs-size, MAE-vs-distance, and hexagonality. (The size & hexagonality paths
   previously used Ã·N.)
2. **Flip_two nMAE/MAE aggregation â€” RESOLVED 2026-06-01.** Per the unified rule (Â§7.11), Flip_two's
   `nmae_seed` is **geometric** â€” `mean(log2(model/baseline))` per seed â€” and its `log_seed` (log2(MAE)) is
   **arithmetic** â€” `log2(mean(model))`. (An interim 2026-06-01 edit had briefly made `nmae_seed` the
   arithmetic `log2(mean/mean)`; reverted so nMAE stays geometric as the paper defines it.)
3. **Dead code, unlabeled**: `perform_MAE_normalization` (plotter 2567) â€” stale normalization path with a
   **hard-coded 24-hop** assumption (wrong for 484/784 tissues); not called. `plot_hexagonality_distribution`
   (1528) and `plot_MAE_vs_hexagonality` (2810) are intentionally not invoked. Harmless but should be marked
   deprecated.
4. **Stale file header**: `DCG_analyze_results_2026_codex.m` lines 1-7 claim it is a non-editable
   "documented copy" of `/home/tomers/DCG_analyze_results_2024.m` with "identical code". **False** â€” this is
   the live, diverged file (carries the 2026-05-31 `ref_priority` fix and ran the rebuild). **Corrected 2026-06-01** â€” the header now
   accurately describes the live, diverged file.
5. **Calibrate/composite y-extent ignores error-bar whiskers**: extent comes from object `YData`, which for
   `errorbar` is the center; `YNegativeDelta`/`YPositiveDelta` are not included, so a unified y-range can
   clip visible whiskers (calibrate `axis_data_extent`).
6. **Flip_two QA is structural only** (`DCG_verify_Flip_two_figures_2026_codex.m`): it checks axes/lines/
   labels/colors/`limits_cover_data` and a `ppgn_expected` text scan (color tol 1e-4, y-coverage tol ~1e-8) but
   **recomputes no scientific values** and emits **no overall PASS/FAIL** â€” a figure with wrong data but
   correct structure passes. Only the Y axis is coverage-checked.
6b. **PPGN absent on Tissue is by design** (VRAM), not a data gap (analyzer 198/205).
7. **Driver default list excludes `hex` and `v1_UW`** (`DCG_run_revision_analyses` line 23). They must be
   passed explicitly to rebuild them (done in Step 1 above). A committed "rebuild all 10" wrapper now exists:
   **`DCG_rebuild_all_summaries_2026_codex.m`** (added 2026-06-01) â€” it passes the full 10-set explicitly.
8. **Still-generating data** (as of 2026-06-01): if any (model,seed) pred files arrive later, re-run Step 1
   for those datasets â€” a summary built before its files exist silently lacks them (now caught by the
   validation harness, PART C).
9. **hex PPGN naming mismatch â€” RESOLVED (2026-06-01).** hex PPGN files were mis-tokened `â€¦_W_na_â€¦`
   instead of `â€¦_W_2_8_â€¦`, so the analyzer silently built hex without PPGN. Renamed `na`â†’`2_8`, rebuilt hex,
   validated (Î”=3e-18). Details in Â§4.12.
10. **Upstream consolidation gap.** The snapshot's MP consolidation (`Z:\Tomer\_consolidate_copy.py` /
   `_consolidate_audit.py`) correctly maps `hexâ†’2_8` but only handles `GAT/GIN/GraphSAGE`; **PPGN/PNA were
   consolidated by a separate, unlocated (likely cluster-side) step** that emitted hex PPGN as `na`, and the
   audit's `MODELS` list excludes PPGN/PNA so it never checked. **Guard ADDED 2026-06-01:**
   `_consolidate_audit.py` now runs a cross-model size-token consistency check (all models of a
   (task,weighting) must share one `<size>` token) â€” it would have caught the hex case.
11. **Graph-aggregation rule unified â€” RESOLVED 2026-06-01.** Final rule (per Tomer): **MAE is always
   arithmetic** (even displayed as log2 it is `log2(arithmetic-mean MAE)` â€” graphs averaged linearly, `log2`
   taken *after* the graph-mean); **nMAE is always geometric** (`log2` *per graph*, then averaged).
   `extract_MAEs` now takes a `normalized` flag that selects the graph-level `log2` placement, applied
   **identically in the size and distance paths**: `normalized=0` â†’ arithmetic (raw + log2(MAE) passes);
   `normalized=1` â†’ geometric (log2(nMAE) passes; callers at plotter ~649/677 pass `1`). Previously the
   **size** path was geometric and the **distance** path arithmetic for *both* MAE and nMAE â€” now they agree:
   log2(MAE) is arithmetic everywhere, log2(nMAE) geometric everywhere. This changes the v1 size-sweep
   log2(MAE) numbers (geometricâ†’arithmetic) and the distance log2(nMAE) numbers (arithmeticâ†’geometric);
   raw MAE is unchanged. Flip_two follows the same rule (Â§4.13, Â§7.2).
12. **PPGN now included in plots â€” 2026-06-01.** The master `DCG_plot_everything` previously hard-excluded
   PPGN (`models_to_exclude={'PPGN'}`) "until data is ready"; the data is validated (Â§4.12) so the exclusion
   is cleared (`{}`). `DCG_run_revision_analyses`' default also changed from `{'PPGN'}` to `{}` and now keys
   on `~exist` only (so a caller passing `{}` is honored, not silently overridden). TissueÃ—2 still show no
   PPGN (no data exists).
13. **STALE-CACHE TRAP â€” found 2026-06-01.** The plotter's default `data_root` is
   `C:\Users\tomers\Desktop\GNN benchmark results`, so it loads `â€¦\_analyzer_cache\<dataset> -
   results_summary.mat`. This session's rebuilds were written to a NESTED dir
   `â€¦\_analyzer_cache\revision_codex_2026\` (the revision driver's `codex_cache_root`), so a *default* plot
   SILENTLY loaded the **23-May** summaries (pre-PPGN-fix) instead of the fresh ones. Symptom: PPGN looked
   terrible (the old anomaly â€” e.g. PPGN MAE 0.034 vs the correct 0.003) **and** its error bars were absurd
   (between-seed SD ~0.02 vs the correct ~0.0004) â€” both are the *same* artifact. FIX: make sure the cache
   the plotter actually reads holds the current summaries; check a summary's mtime before trusting a figure.
   (2026-06-01: the correct summaries were promoted into `â€¦\_analyzer_cache\`, stale ones moved to
   `_stale_may23_backup_20260601\`.)
14. **v1 MP size-sweep gap + PPGN merge â€” 2026-06-01.** The consolidated snapshot only consolidated the v1
   MP models at the `2_16` cohort (the `_consolidate` MAP maps only `v1_2_16`), so a consolidated rebuild of
   v1_W/v1_UW had correct PPGN at every size but **MP at one size only**. The full v1 MP size-sweep exists
   only in the legacy Desktop data (`pred_v1_<size>_<W|UW>__<MP>_s*.txt`, all 6 cohorts), whose PPGN was the
   old anomaly. FIX: the corrected PPGN (consolidated `standard-flip_PPGN_<W|UW>_<size>_s*.pred.txt`) was
   copied into the legacy layout as `pred_v1_<size>_<W|UW>__PPGN_s*.txt` (anomalous originals â†’ 
   `_v1_PPGN_anomalous_backup_20260601\`) and v1_W/v1_UW rebuilt from the merged legacy data. **Alignment was
   proven before merging**: the test-split indices are byte-identical between `inds\v1_2_16_W\test.inds` and
   consolidated `splits\standard_2_16\test.inds`. Verified result: v1_W PPGN 0.0027â€“0.0056 (identical to the
   consolidated rebuild), all 4 MP present at all 6 sizes, PPGN wins (dominates the small cohorts). NOTE:
   `v1_2_16_W` is not a real analyzer dataset â€” the revision driver aliases it to `v1_W` (line 127-129).
15. **Flip_two interaction read a stale cache â€” FIXED 2026-06-01** (Codex catch). The two-source Flip_two
   diagnostics (`DCG_plot_Flip_two_interaction_2026_codex`) were fed `Flip_two - analyses data.mat` from
   `source_cache_root` (the parent `_analyzer_cache`), but the driver rebuilds the fresh copy into
   `codex_cache_root` (`revision_codex_2026`). So it silently loaded an older Desktop copy (the assumptions
   file confirmed it). FIX (`DCG_run_revision_analyses` ~244): look in `codex_cache_root` FIRST, fall back to
   `source_cache_root` only for legacy layouts. Same stale-cache class as Â§7.13.
16. **Hex manuscript panel used 24 bins, not the paper's 20 â€” FIXED 2026-06-01** (Codex catch). The
   identity-normalized-error-vs-hexagonality panel binned hexagonality with `hex_paper_bins = 0.4:0.025:1`
   (24 bins). Hexagonality is continuous (range â‰ˆ [0.406, 0.969], ~31 distinct values, n=200), so the bin
   count is a plotting choice â€” changed to `0.4:0.03:1` = **20 even bins over [0.4,1]** (plotter line 277) to
   match the manuscript. Overridable via `DCG_CONFIG.hex_paper_bins`; if the paper uses different exact edges,
   set them there.
17. **Cache-staleness guard â€” ADDED 2026-06-01** (the durable fix for the silent-stale-cache class, Â§7.13/Â§7.15).
   The pipeline is parse-once / plot-many: the analyzer builds `S` and writes `results_summary.mat`; the
   plotter is a PURE CONSUMER that loads that summary and never re-parses. So a summary that drifts from its
   source folder silently yields wrong figures (exactly what bit us). GUARD: the analyzer now stamps each
   summary with `source_manifest` â€” a fingerprint (name+size+mtime of every `.pred.txt`) of the source folder
   via `dcg_source_manifest_2026_codex.m`; the plotter recomputes it from the current `data_root` and
   **ERRORS** (`DCG:staleCache`) on mismatch, **WARNS** (`DCG:noCacheStamp`) for pre-guard summaries. Override:
   `DCG_CONFIG.skip_cache_guard=true`. Verified end-to-end (stamp saved into the `.mat`; matchâ†’pass,
   mismatchâ†’error). Residual gap: an in-place overwrite preserving BOTH size and mtime isn't seen by the
   metadata fingerprint â€” covered by the force-rebuild below. **CANONICAL WORKFLOW:** rebuild from ONE folder
   with `DCG_rebuild_all_summaries` (`rebuild_summaries=true`), then plot; the guard makes skipping the rebuild
   a hard error rather than a silent wrong plot.
   2026-06-06 extension: summaries also carry `analysis_algorithm_version =
   '2026-06-06_cellshare_hops_v1'`; the plotter errors (`DCG:staleAnalysisAlgorithm`) if the stamp is missing
   or mismatched, so old pre-cell-share summaries cannot silently drive hop-distance/fallback plots.

---

## 8. Appendix â€” function index (per file)

**`DCG_analyze_results_2026_codex.m`** â€” main script Â§Â§1-16 (dataset dispatch 135-213; grid NaN-fill Â§6;
test-split filter Â§7; `no_learning` Â§10; summary build Â§14; persist Â§16). Helpers: `extract_MP_results`
(1044), `extract_PPGN_results` (1250), `read_dataset_inds` (1419).

**`load_dataset.m`** â€” `load_dataset` (1): regex-parse a pred file â†’ graph names, raw blocks, header,
per-graph metadata, per-graph numeric matrices; v2 handles 6/7/legacy name tokens and trailing-junk
truncation.

**`DCG_consolidated_paths_2026_codex.m`** â€” `DCG_consolidated_paths_2026_codex` (1, dispatch:
`is_consolidated|pred_file|pred_glob|inds_dir`), `i_prefix_parts` (54), `i_inds_dir` (88).

**`DCG_run_revision_analyses_2026_codex.m`** â€” batch driver (per-dataset try/catch; build via analyzer,
plot via plotter, then calibrate + composites + Flip_two). Helper: `first_existing` (260).

**`DCG_plot_results_2026_codex.m`** â€” 69 functions. Core math: `calculate_normalization_factors` (1383),
`sanitize_normalization_scalar/vector` (1470/1481), `force_baseline_reference_to_zero` (727),
`extract_MAEs` (2394), `uncertainty_from_values` (2201). Plot builders: `plot_MAE_vs_dataset_size` (2635),
`plot_MAE_vs_dist` (2696), `plot_scatter_plot_examples` (2222), `plot_PPGN_fallback` (1030),
`fallback_graph_values` (1347), the manuscript-hex family (1545-2176), `plot_embedding_examples` (2902) +
`emb_*` (3186-3493). Model styling: `paper_model_colors` (889), `paper_model_plot_order` (914),
`order_models_for_paper` (936), `drop_models_from_summary` (979). Housekeeping: `keep_*` (774-808),
`remove_stale_*` (823-888), `write_normalization_assumptions` (959). *(Full THOROUGH/THIN audit per function
all 69 now documented to a uniform standard (2026-06-01); 3 dead â€” see Â§7.3.)*

**`DCG_calibrate_saved_fig_y_ranges_2026_codex.m`** â€” main (1) + `dcg_savefig_visible` (82),
`axis_data_extent` (96), `padded_axis_limits` (118), `data_axes` (142).

**`DCG_make_revision_transverse_dist_composites_2026_codex.m`** â€” main (1) + `best_data_axes` (135),
`copy_axis_contents` (163), `dcg_savefig_visible` (181), `axis_data_extent` (192), `padded_axis_limits`
(214), `write_composite_assumptions` (238).

**`DCG_plot_Flip_two_interaction_2026_codex.m`** â€” main (1) + `extract_flip_two_records` (73, analytical
core), `summarize_wide_records` (220, aggregation), tertile/zone helpers (271-306), 9 plot builders
(352-540), styling/IO (561-650).

**`DCG_verify_Flip_two_figures_2026_codex.m`** â€” `DCG_verify_Flip_two_figures_2026_codex` (1): structural
QA of saved Flip_two figures.

**`DCG_plot_everything_2026_codex.m` / `DCG_plot_all_revision_figures_2026_codex.m`** â€” top-level plotting
scripts (no local functions).

---
*End of handoff. Companion (done 2026-06-01): the in-script per-function documentation pass brought every function
above to a uniform doc standard (~109 functions across 11 files, comment-only, diff-verified,
checkcode-clean); this file is the cross-cutting math/architecture reference.*

