%==========================================================================
% GNNBenchmark_plot_all_revision_figures
%
% Convenience wrapper for regenerating the revision figure set from existing
% result summaries. This is the script to run when you want all current
% revision figures without rebuilding analysis summaries.
%
% Defaults:
%   - revision datasets only:
%       Shear_1_2, Shear_1_5, kA_1, kA_10, Flip_two, Tissue_484, Tissue_784
%   - include all models (PPGN validated/ready 2026-06-01; previously excluded);
%   - skip v1-only hop demo scatter examples;
%   - skip dataset-size plots for single-size revision datasets;
%   - save MATLAB .fig files;
%   - match y-ranges across corresponding saved figure types.
%==========================================================================

close all;
clc;

datasets = {'Shear_1_2', 'Shear_1_5', 'kA_1', 'kA_10', ...
            'Flip_two', 'Tissue_484', 'Tissue_784'};

rebuild_summaries = false;
plot_after_summary = true;
models_to_exclude = {};   % 2026-06-01: PPGN validated/ready -> include all models (was {'PPGN'})
calibrate_y_ranges = true;
save_png = false;
figure_panel_size = [340, 300];

code_dir = fileparts(mfilename('fullpath'));
run(fullfile(code_dir, 'GNNBenchmark_run_revision_analyses.m'));
