%==========================================================================
%  DCG_rebuild_all_summaries  --  Step-1 rebuild wrapper (all 10)
%
%  PURPOSE
%    Rebuild EVERY results_summary.mat from the consolidated prediction files
%    (the numeric-analysis half of the pipeline; no figures). This is the
%    permanent, committed equivalent of the ad-hoc %TEMP%\dcg_rebuild_all10.m
%    used on 2026-06-01.
%
%  WHY THIS EXISTS
%    DCG_run_revision_analyses's DEFAULT dataset list omits `hex`
%    and `v1_UW` (its comment: "manuscript hexagonality / unweighted analyses
%    belong to their own runs"). Passing the FULL 10-dataset list explicitly
%    makes the driver build them too (the omission is only in the default).
%
%  WHAT IT DOES
%    Sets datasets = the 10 canonical sets, rebuild_summaries = true,
%    plot_after_summary = false, then runs the driver. Each summary is written
%    to <data_root>\_analyzer_cache\revision_2026\<dataset> - results_summary.mat.
%
%  OUTPUT
%    None returned; side effect is the 10 rebuilt .mat summaries (+ analyses
%    caches). Per-dataset OK/FAILED status is printed by the driver.
%
%  USAGE
%    >> run('DCG_rebuild_all_summaries.m')
%    Then run DCG_plot_everything.m for the figures.
%
%  NOTE
%    To rebuild a SINGLE dataset (e.g. after new pred files land for it), set
%    `datasets = {'hex'};` (or whichever) before the run() -- far cheaper than
%    rebuilding all 10.
%==========================================================================

close all;
clc;

% The 10 canonical datasets (hex and v1_UW must be listed explicitly; see header).
datasets = {'v1_2_16_W', 'v1_UW', 'hex', 'Shear_1_2', 'Shear_1_5', ...
            'kA_1', 'kA_10', 'Flip_two', 'Tissue_484', 'Tissue_784'};

rebuild_summaries  = true;     % force a fresh parse of the .pred.txt files
plot_after_summary = false;    % summaries only; figures come from DCG_plot_everything

code_dir = fileparts(mfilename('fullpath'));
fprintf('[DCG_rebuild_all_summaries] rebuilding %d datasets (no plots)...\n', numel(datasets));
run(fullfile(code_dir, 'DCG_run_revision_analyses.m'));
fprintf('[DCG_rebuild_all_summaries] done. Run DCG_plot_everything.m for figures.\n');
