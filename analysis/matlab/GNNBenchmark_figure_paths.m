function out = GNNBenchmark_figure_paths(action, figures_root, key)
%GNNBENCHMARK_FIGURE_PATHS  Canonical publication figure output layout.
%
%   OUT = GNNBENCHMARK_FIGURE_PATHS(ACTION, FIGURES_ROOT, KEY) returns the
%   canonical output folder for manuscript/revision figures. The helper keeps
%   all figure-writing scripts on the same human-readable folder tree instead
%   of mixing historical roots such as figures/main, figures/revision_2026,
%   or dataset-key folders directly under figures/.
%
%   Supported ACTION values
%   -----------------------
%     'dataset_dir'     KEY is a dataset key such as 'v1_W', 'Shear_1_5',
%                       'Flip_two', etc. OUT is the directory for that
%                       dataset's figures and sidecar files.
%     'composite_dir'   KEY is a composite group such as
%                       'revision_transverse_dist_composites' or
%                       'revision_extreme_task_composites'.
%     'diagnostic_dir'  KEY is a manuscript diagnostic such as
%                       'embedding_error_bounds' or
%                       'counterfactual_copying'.
%     'mini_dir'        OUT is the mini-pipeline smoke-test figure directory.
%
%   Canonical tree
%   --------------
%     figures/00_mini_smoke/
%     figures/01_standard_v1/weighted_all_cohorts/
%     figures/01_standard_v1/unweighted_all_cohorts/
%     figures/02_hexagonality/
%     figures/03_condition_comparisons/standard_16_cohorts/
%     figures/03_condition_comparisons/kA/kA_10/
%     figures/03_condition_comparisons/kA/kA_1/
%     figures/03_condition_comparisons/shear/shear_1_2/
%     figures/03_condition_comparisons/shear/shear_1_5/
%     figures/03_condition_comparisons/tissue_size/22x22_484_cells/
%     figures/03_condition_comparisons/tissue_size/28x28_784_cells/
%     figures/04_two_T1_events/
%     figures/05_summary_panels/
%     figures/06_embedding_error_bounds/
%     figures/07_counterfactual_copying/

if nargin < 1 || isempty(action)
    error('GNNBenchmark:figurePathActionMissing', 'Pass a figure path action.');
end
if nargin < 2 || isempty(figures_root)
    error('GNNBenchmark:figureRootMissing', 'Pass the root figures directory.');
end
if nargin < 3
    key = '';
end

action = char(action);
figures_root = char(figures_root);
key = char(key);

switch lower(action)
    case 'dataset_dir'
        rel = dataset_relative_dir(key);
    case 'composite_dir'
        rel = composite_relative_dir(key);
    case 'diagnostic_dir'
        rel = diagnostic_relative_dir(key);
    case 'mini_dir'
        rel = fullfile('00_mini_smoke');
    otherwise
        error('GNNBenchmark:unknownFigurePathAction', 'Unknown figure path action: %s', action);
end

out = fullfile(figures_root, rel);
end

function rel = dataset_relative_dir(dataset)
switch char(dataset)
    case 'v1_W'
        rel = fullfile('01_standard_v1', 'weighted_all_cohorts');
    case 'v1_UW'
        rel = fullfile('01_standard_v1', 'unweighted_all_cohorts');
    case 'hex'
        rel = fullfile('02_hexagonality');
    case 'v1_2_16_W'
        rel = fullfile('03_condition_comparisons', 'standard_16_cohorts');
    case 'kA_10'
        rel = fullfile('03_condition_comparisons', 'kA', 'kA_10');
    case 'kA_1'
        rel = fullfile('03_condition_comparisons', 'kA', 'kA_1');
    case 'Shear_1_2'
        rel = fullfile('03_condition_comparisons', 'shear', 'shear_1_2');
    case 'Shear_1_5'
        rel = fullfile('03_condition_comparisons', 'shear', 'shear_1_5');
    case 'Tissue_484'
        rel = fullfile('03_condition_comparisons', 'tissue_size', '22x22_484_cells');
    case 'Tissue_784'
        rel = fullfile('03_condition_comparisons', 'tissue_size', '28x28_784_cells');
    case 'Flip_two'
        rel = fullfile('04_two_T1_events');
    otherwise
        rel = matlab.lang.makeValidName(char(dataset));
end
end

function rel = composite_relative_dir(key)
switch char(key)
    case 'revision_transverse_dist_composites'
        rel = fullfile('05_summary_panels', 'condition_distance_profiles');
    case 'revision_extreme_task_composites'
        rel = fullfile('05_summary_panels', 'condition_task_composites');
    otherwise
        rel = fullfile('05_summary_panels', matlab.lang.makeValidName(char(key)));
end
end

function rel = diagnostic_relative_dir(key)
switch char(key)
    case 'embedding_error_bounds'
        rel = fullfile('06_embedding_error_bounds');
    case 'counterfactual_copying'
        rel = fullfile('07_counterfactual_copying');
    otherwise
        rel = fullfile('08_other_diagnostics', matlab.lang.makeValidName(char(key)));
end
end