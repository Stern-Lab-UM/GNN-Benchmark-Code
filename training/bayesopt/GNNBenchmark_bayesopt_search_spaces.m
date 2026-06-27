function spaces = GNNBenchmark_bayesopt_search_spaces()
% GNNBenchmark_bayesopt_search_spaces  Implement GNN Benchmark bayesopt search spaces for this MATLAB workflow.
% Inputs: none.
% Outputs: spaces
%GNNBenchmark_BAYESOPT_SEARCH_SPACES  Final hyperparameter spaces used for BO.
%
%   SPACES = GNNBenchmark_BAYESOPT_SEARCH_SPACES() returns structs that can be passed
%   to OPTIMIZE_MPNN and OPTIMIZE_PPGN. The spaces record the final V1
%   revision optimization setup described in the manuscript Methods: MPNN
%   depth was fixed at 16 layers, and PPGN architecture size was fixed while
%   optimizer/scheduler knobs were swept.
%
%   Example:
%       spaces = GNNBenchmark_bayesopt_search_spaces();
%       results = optimize_MPNN(dataset_file, split_dir, 'GIN', ...
%           spaces.mpnn_v1_l16.hp_ranges, 20, ...
%           'max_epochs', spaces.mpnn_v1_l16.max_epochs, ...
%           'patience', spaces.mpnn_v1_l16.patience, ...
%           'early_stop_patience', spaces.mpnn_v1_l16.early_stop_patience, ...
%           'early_stop_min_delta', spaces.mpnn_v1_l16.early_stop_min_delta, ...
%           'trainer_py', '/path/to/models/mpnn/trainer_final.py');

spaces = struct();

mpnn = struct();
mpnn.description = 'Final MPNN V1 search: 16 layers fixed; GraphSAGE/GAT/GIN/PNA.';
mpnn.hp_ranges = struct();
mpnn.hp_ranges.hidden_channels = {'64', '128'};
mpnn.hp_ranges.dropout = {'0', '0.1', '0.2'};
mpnn.hp_ranges.batch_size = {'1', '2', '4'};
mpnn.hp_ranges.lr = [1e-4, 1e-2];
mpnn.fixed = struct('num_layers', 16, 'weight_decay', 0, 'factor', 0.75);
mpnn.max_epochs = 120;
mpnn.patience = 20;
mpnn.early_stop_patience = 40;
mpnn.early_stop_min_delta = 1e-4;
mpnn.num_seed_points = 6;
mpnn.n_trials = 20;
spaces.mpnn_v1_l16 = mpnn;

ppgn = struct();
ppgn.description = 'Final PPGN V1 search: architecture fixed; optimizer/scheduler knobs swept.';
ppgn.hp_ranges = struct();
ppgn.hp_ranges.learning_rate = [1e-5, 1e-2];
ppgn.hp_ranges.batch_size = {'2', '4', '8', '16', '32'};
ppgn.hp_ranges.gradient_clipping = {'0.001', '0.01', '0.1', '1'};
ppgn.hp_ranges.factor = {'0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8'};
ppgn.fixed = struct('block_features', '[400,400,400]', 'depth_of_mlp', 2);
ppgn.max_epochs = 120;
ppgn.patience = 20;
ppgn.early_stop = 40;
ppgn.threshold = 1e-4;
ppgn.num_seed_points = 6;
ppgn.n_trials = 20;
spaces.ppgn_v1 = ppgn;
end
