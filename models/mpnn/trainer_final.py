"""trainer_final.py -- MPNN trainer with a SELECTABLE prediction head.

Head-selectable copy of nano-main/src/trainer_final.py. Identical to the
published trainer except the prediction head is now chosen on the command line.

Head options (CLI flag --head):
  regressor  (DEFAULT)  EdgeRegressor: y = MLP([h_u || h_v || edge_attr]),
                        linear output, NO sigmoid. Trains directly on physical
                        edge lengths. Removes the sigmoid-floor collapse at the
                        T1 edge. Gradient-clipped (the linear head is unbounded).
  sigmoid               LinkPredictor: y = sigmoid(MLP(h_i * h_j)), the ORIGINAL
                        published head. Trains on y scaled to [0, 1];
                        inverse_scale_y restores physical units. Pass
                        --head sigmoid to reproduce the preprint.

The head choice is coupled to several other things, all handled here:
  * y-scaling : sigmoid trains on dataset.scale_y(y) in [0,1]; regressor on raw y.
  * edge_attr : passed to the head only for the regressor. The optional
                --ablate_head_edge_attr flag removes raw edge attributes from
                the final regressor head while leaving the message-passing
                backbone unchanged.
  * grad clip : applied only for the regressor (the linear head is unbounded).
  * checkpoint: model_config['head'] and model_config['edge_dim'] are recorded so
                predict_final.py can rebuild the matching head; head weights are
                saved under 'head_state_dict' for BOTH heads.
  * file name : gnn_<MODEL>_weighted_<W>_<head>_<norm>_seed_<S>.pth
                (ablation checkpoints append _ablateHeadEdgeAttr_True).

Everything else (data pipeline, hyperparameters, GPU monitoring, TB logging) is
unchanged from the published trainer_final.py.
"""
import os
import os.path as osp
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import PNAConv
from torch_geometric.nn.models import GAT, GCN
from torch.utils.tensorboard import SummaryWriter
import time, json  # NEW
from gpu_monitor import GPUMonitor, bytes_to_gb  # NEW

from dataset import Nano
from models import LinkPredictor, EdgeRegressor, GraphSAGE, GIN, PNA

import random          # NEW TOMER
import numpy as np      # NEW TOMER

from torch_geometric.nn import InstanceNorm

# --- per-graph InstanceNorm (audit issue I3) ---------------------------------
# PyG 2.3.1's BasicGNN/MLP call their norm layers as norm(x) -- they do NOT
# thread a per-graph `batch` vector -- so InstanceNorm pools the whole
# minibatch, making predictions batch-size dependent (I3). We register a
# forward-pre-hook on every InstanceNorm that rewrites norm(x) -> norm(x, batch)
# with the current per-graph batch vector, so each graph is normalized on its
# own. norm_mode 'pooled' makes the hook a no-op -> the original behavior.
# This works on any PyG version and needs no change to the model files.
_CURRENT_BATCH = {'node': None, 'edge': None}


def _install_pergraph_norm_hooks(module, kind):
    """Attach per-graph InstanceNorm hooks to a model component.

    PyTorch Geometric calls these normalization layers as ``norm(x)``. The
    hook rewrites that call to ``norm(x, batch)`` when per-graph normalization
    is requested, so normalization statistics are not pooled across graphs in a
    minibatch.

    Args:
        module: Backbone or prediction-head module to scan recursively.
        kind: Batch-vector namespace, either ``'node'`` for backbone features or
            ``'edge'`` for head features.

    Returns:
        Number of ``InstanceNorm`` modules that received a hook.
    """
    def _pre_hook(_mod, args):
        """
        Implement the pre hook step for models / mpnn / trainer_final.py.

        Args:
            _mod: Caller-supplied value used by this routine.
            args: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        b = _CURRENT_BATCH[kind]
        if b is None:
            return None                 # pooled: leave norm(x) untouched
        return (args[0], b)             # -> norm(x, batch) -> per-graph
    n = 0
    for m in module.modules():
        if isinstance(m, InstanceNorm):
            m.register_forward_pre_hook(_pre_hook)
            n += 1
    return n


def _set_current_batch(data, norm_mode):
    """
    Stash the per-graph batch vectors that the InstanceNorm hooks read.

    Args:
        data: Caller-supplied value used by this routine.
        norm_mode: Caller-supplied value used by this routine.

    Returns:
        None; the function updates object state, files, logs, or external process state.
    """
    nb = getattr(data, 'batch', None)
    if norm_mode == 'per_graph' and nb is not None:
        _CURRENT_BATCH['node'] = nb
        _CURRENT_BATCH['edge'] = nb[data.edge_index[0]]
    else:
        _CURRENT_BATCH['node'] = None
        _CURRENT_BATCH['edge'] = None


@torch.no_grad()
def eval_one_batch(gnn, head, data, criterion, optimizer, dataset, head_type, norm_mode):
    """
    Evaluate one mini-batch without gradient updates and return loss plus predictions.

    Args:
        gnn: Caller-supplied value used by this routine.
        head: Caller-supplied value used by this routine.
        data: Caller-supplied value used by this routine.
        criterion: Caller-supplied value used by this routine.
        optimizer: Caller-supplied value used by this routine.
        dataset: Caller-supplied value used by this routine.
        head_type: Caller-supplied value used by this routine.
        norm_mode: Caller-supplied value used by this routine.

    Returns:
        Computed value used by the caller.
    """
    gnn.eval()
    head.eval()
    _set_current_batch(data, norm_mode)

    edge_attr = data.edge_attr if data.edge_attr.shape[0] > 0 else None
    node_emb = gnn(data.x, data.edge_index, edge_attr=edge_attr,
                   edge_weight=data.edge_attr[:, 0] if edge_attr is not None else None)

    if head_type == 'sigmoid':
        scaled_y_pred = head(node_emb, data.edge_index)          # in [0, 1] (sigmoid)
        y_pred = dataset.inverse_scale_y(scaled_y_pred)          # -> physical units
    else:
        y_pred = head(node_emb, data.edge_index, edge_attr=edge_attr)  # physical units

    loss = criterion(y_pred, data.y)
    return loss.data.cpu(), y_pred.data.cpu()


def train_one_batch(gnn, head, data, criterion, optimizer, dataset, head_type, norm_mode):
    """
    Train on one mini-batch and return the physical-unit loss plus predictions.

    Args:
        gnn: Caller-supplied value used by this routine.
        head: Caller-supplied value used by this routine.
        data: Caller-supplied value used by this routine.
        criterion: Caller-supplied value used by this routine.
        optimizer: Caller-supplied value used by this routine.
        dataset: Caller-supplied value used by this routine.
        head_type: Caller-supplied value used by this routine.
        norm_mode: Caller-supplied value used by this routine.

    Returns:
        Computed value used by the caller.
    """
    gnn.train()
    head.train()
    _set_current_batch(data, norm_mode)

    edge_attr = data.edge_attr if data.edge_attr.shape[0] > 0 else None
    node_emb = gnn(data.x, data.edge_index, edge_attr=edge_attr,
                   edge_weight=data.edge_attr[:, 0] if edge_attr is not None else None)  # GCN uses edge_weight

    if head_type == 'sigmoid':
        # Original protocol: the head outputs [0, 1], so back-prop against y
        # scaled to [0, 1]; report the loss in physical units after inverse-scaling.
        scaled_y_pred = head(node_emb, data.edge_index)          # in [0, 1] as we use sigmoid
        scaled_y_true = dataset.scale_y(data.y)                  # scale y_true to [0, 1]
        loss_for_backward = criterion(scaled_y_pred, scaled_y_true)
        y_pred = dataset.inverse_scale_y(scaled_y_pred)          # inverse scale to original values
    else:
        # Regressor: the head outputs physical edge lengths directly, no y-scaling.
        y_pred = head(node_emb, data.edge_index, edge_attr=edge_attr)
        loss_for_backward = criterion(y_pred, data.y)

    optimizer.zero_grad()
    loss_for_backward.backward()
    if head_type == 'regressor':
        # the linear EdgeRegressor head is unbounded above -> clip for stability
        torch.nn.utils.clip_grad_norm_(
            list(gnn.parameters()) + list(head.parameters()), max_norm=1.0)
    optimizer.step()

    loss = criterion(y_pred, data.y)                            # report loss in physical units
    return loss.data.cpu(), y_pred.data.cpu()


def run_one_epoch(gnn, head, data_loader, criterion, optimizer, device, seed, epoch, phase, dataset, writer, head_type, norm_mode):
    """
    Run one train/validation/test epoch over a data loader.

    Args:
        gnn: Caller-supplied value used by this routine.
        head: Caller-supplied value used by this routine.
        data_loader: Caller-supplied value used by this routine.
        criterion: Caller-supplied value used by this routine.
        optimizer: Caller-supplied value used by this routine.
        device: Caller-supplied value used by this routine.
        seed: Caller-supplied value used by this routine.
        epoch: Caller-supplied value used by this routine.
        phase: Caller-supplied value used by this routine.
        dataset: Caller-supplied value used by this routine.
        writer: Caller-supplied value used by this routine.
        head_type: Caller-supplied value used by this routine.
        norm_mode: Caller-supplied value used by this routine.

    Returns:
        Computed value used by the caller.
    """
    loader_len = len(data_loader)
    run_one_batch = train_one_batch if phase == 'train' else eval_one_batch

    pbar = tqdm(data_loader)
    all_y_pred, all_y_true = [], []
    for idx, data in enumerate(pbar):
        data = data.to(device)
        if data.x.size(1) >= 5:
            data.x[:, :5] = data.x[:, :5] / dataset.deg_max  # Normalize degree (LDP only)

        batch_loss, batch_y_pred = run_one_batch(gnn, head, data.to(device), criterion, optimizer, dataset, head_type, norm_mode)

        batch_y_true = data.y.data.cpu()
        all_y_pred.append(batch_y_pred), all_y_true.append(batch_y_true)
        desc = log_batch(seed, epoch, phase, batch_loss)

        if idx == loader_len - 1:
            desc, loss = log_epoch(seed, epoch, phase, all_y_pred, all_y_true, criterion, writer)
        pbar.set_description(desc)
    return loss, [torch.cat(all_y_pred), torch.cat(all_y_true)]


def log_batch(seed, epoch, phase, batch_loss):
    """
    Format one mini-batch progress message.

    Args:
        seed: Caller-supplied value used by this routine.
        epoch: Caller-supplied value used by this routine.
        phase: Caller-supplied value used by this routine.
        batch_loss: Caller-supplied value used by this routine.

    Returns:
        Computed value used by the caller.
    """
    desc = f'[Seed {seed}, Epoch: {epoch}]: {phase}....., '
    desc += f'loss: {batch_loss:.4f}, '
    return desc

def log_epoch(seed, epoch, phase, y_pred, y_true, criterion, writer):
    """
    Aggregate one epoch loss, write TensorBoard output, and format a status message.

    Args:
        seed: Caller-supplied value used by this routine.
        epoch: Caller-supplied value used by this routine.
        phase: Caller-supplied value used by this routine.
        y_pred: Caller-supplied value used by this routine.
        y_true: Caller-supplied value used by this routine.
        criterion: Caller-supplied value used by this routine.
        writer: Caller-supplied value used by this routine.

    Returns:
        Computed value used by the caller.
    """
    loss = criterion(torch.cat(y_pred), torch.cat(y_true))
    desc = f'[Seed {seed}, Epoch: {epoch}]: {phase} done, '
    desc += f'loss: {loss:.4f}, '

    if writer is not None:
        writer.add_scalar(f'{phase.strip()}/loss', loss, epoch)
    return desc, loss.data.cpu()


def get_model(model_name, train_loader):
    """
    Resolve a model name into the corresponding architecture class and optional degree histogram.

    Args:
        model_name: Caller-supplied value used by this routine.
        train_loader: Caller-supplied value used by this routine.

    Returns:
        Computed value used by the caller.
    """
    deg = None
    if model_name == 'GraphSAGE':
        Model = GraphSAGE
    elif model_name == 'GAT':
        Model = GAT
    elif model_name == 'GIN':
        Model = GIN
    elif model_name == 'GCN':
        Model = GCN
    elif model_name == 'PNA':
        Model = PNA
        deg = PNAConv.get_degree_histogram(train_loader)
    else:
        print(f'Unknown model: {model_name}, please choose from [GraphSAGE, GAT, GIN, GCN, PNA].')
        raise NotImplementedError
    return Model, deg


def build_head(head_type, hidden_channels, out_channels, dropout, edge_dim):
    """Construct the prediction head. Both heads use a 2-layer MLP.

    head_type == 'sigmoid'   -> LinkPredictor  (original; ignores edge_attr)
    head_type == 'regressor' -> EdgeRegressor  (default; concatenates edge_attr)

    Args:
        head_type: Caller-supplied value used by this routine.
        hidden_channels: Caller-supplied value used by this routine.
        out_channels: Caller-supplied value used by this routine.
        dropout: Caller-supplied value used by this routine.
        edge_dim: Caller-supplied value used by this routine.

    Returns:
        Computed value used by the caller.
    """
    if head_type == 'sigmoid':
        return LinkPredictor(hidden_channels, hidden_channels, out_channels, 2, dropout)
    elif head_type == 'regressor':
        return EdgeRegressor(hidden_channels, hidden_channels, out_channels,
                             num_layers=2, dropout=dropout, edge_dim=edge_dim)
    else:
        raise ValueError(f"Unknown head '{head_type}', expected 'regressor' or 'sigmoid'.")


def main():
    """
    Parse command-line arguments and run this script entry point.

    Returns:
        None; the function updates object state, files, logs, or external process state.
    """
    parser = argparse.ArgumentParser(description='Train SAT')
    parser.add_argument('-d', '--dim', type=str, default='2D')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the folder that directly contains the input .txt file(s) '
                             '(e.g. *_weighted.txt / *_unweighted.txt) and train.inds / val.inds / test.inds.')
    parser.add_argument('-m', '--model', type=str, default='GraphSAGE')
    parser.add_argument('--head', type=str, choices=['regressor', 'sigmoid'], default='regressor',
                        help="Prediction head. 'regressor' (DEFAULT): EdgeRegressor -- linear, no "
                             "sigmoid; fixes the T1 collapse. 'sigmoid': original LinkPredictor -- "
                             "reproduces the preprint.")
    parser.add_argument('--norm_mode', type=str, choices=['per_graph', 'pooled'], default='per_graph',
                        help="InstanceNorm scope. 'per_graph' (DEFAULT): normalize each graph on "
                             "its own -- fixes the batch-size dependence (I3). 'pooled': normalize "
                             "over the whole minibatch -- the original published behavior.")
    parser.add_argument('--ablate_head_edge_attr', action='store_true',
                        help='Ablation mode for length-informed MPNNs: remove raw edge attributes from the final EdgeRegressor head only. The message-passing backbone still receives edge features.')
    parser.add_argument('--weighted', type=str, default = True)
    parser.add_argument('--use_node_feats', type=str, default="True",
                    help="Use 35 node features (LDP+Laplacian). Set to 'False' to disable.")
    parser.add_argument('--cuda', type=int, help='cuda device id, -1 for cpu', default=0)
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    parser.add_argument('--hidden_channels', type=int, help='hidden channels', default=256)
    parser.add_argument('--out_channels', type=int, help='out_channels', default=1)
    parser.add_argument('--num_layers', type=int, help='number of layers', default=16)
    parser.add_argument('--dropout', type=float, help='dropout', default=0.5)
    parser.add_argument('--lr', type=float, help='learning rate', default=1.0e-4)
    parser.add_argument('--wd', type=float, help='wd', default=0)
    parser.add_argument('--epochs', type=int, help='number of epochs', default=5000)
    parser.add_argument('--batch_size', type=int, help='batch size', default=8)
    parser.add_argument('--factor', type=float, help='factor', default=0.75)
    parser.add_argument('--patience', type=int, help='patience', default=20)
    parser.add_argument('--early_stop_patience', type=int, default=60,
                        help='epochs without a meaningful validation improvement before stopping')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0,
                        help='minimum validation-loss decrease that resets early-stop patience')
    args = parser.parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cuda_id = args.cuda
    dim = args.dim
    head_type = args.head
    norm_mode = args.norm_mode
    ablate_head_edge_attr = bool(args.ablate_head_edge_attr)
    if (args.weighted == "False"):
        is_weighted = False
    elif (args.weighted == "True"):
        is_weighted = True
    else:
        is_weighted = None  # set to None if dim == '3D'
    use_node_feats = True if args.use_node_feats == "True" else False

    model_name = args.model

    root = args.data_dir
    print(f'Training {osp.basename(osp.normpath(root))} with the "{head_type}" head...')
    dataset = Nano(root=root, is_weighted=is_weighted, use_node_feats=use_node_feats, dim=dim)

    timestamp = datetime.now().strftime("%m_%d_%Y-%H_%M_%S.%f")[:-3]
    # Build the log-subdir name ONCE so the SummaryWriter dir and the
    # tensorboard2csv path at the end cannot drift apart.
    log_subdir = (
        f'{timestamp}_dim_{args.dim}_weighted_{args.weighted}_nodefeats_{args.use_node_feats}'
        f'_model_{args.model}_head_{head_type}_norm_{norm_mode}'
        f'_ablateHeadEdgeAttr_{ablate_head_edge_attr}_epochs_{args.epochs}'
        f'_hiddenChannels_{args.hidden_channels}_numLayers_{args.num_layers}'
        f'_dropout_{args.dropout}_ls_{args.lr}_wd_{args.wd}'
    )
    writer = SummaryWriter(osp.join(root, 'logs', log_subdir))
    device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')
    in_channels = dataset[0].x.size(1)
    edge_dim = dataset[0].edge_attr.size(1) if dataset[0].edge_attr.shape[0] > 0 else None

    head_edge_dim = 0 if ablate_head_edge_attr else edge_dim
    ### Hyperparameters ###
    hidden_channels = args.hidden_channels
    out_channels = args.out_channels
    num_layers = args.num_layers
    dropout = args.dropout
    lr = args.lr
    wd = args.wd
    epochs = args.epochs
    batch_size = args.batch_size
    factor = args.factor
    patience = args.patience
    early_termination = args.early_stop_patience
    early_stop_min_delta = args.early_stop_min_delta
    criterion = torch.nn.L1Loss()
    #######################

    train_loader = DataLoader(dataset[dataset.idx_split['train']], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset[dataset.idx_split['val']], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset[dataset.idx_split['test']], batch_size=batch_size, shuffle=False)

    Model, deg = get_model(model_name, train_loader)
    model_kwargs = dict(dropout=dropout, norm='instance_norm', jk='cat', edge_dim=edge_dim)
    if deg is not None:
        model_kwargs['deg'] = deg
    gnn = Model(in_channels, hidden_channels, num_layers, hidden_channels, **model_kwargs).to(device)
    head = build_head(head_type, hidden_channels, out_channels, dropout, head_edge_dim).to(device)
    print(f'[INFO] Prediction head: {head_type} ({type(head).__name__})')
    print(f'[INFO] ablate_head_edge_attr: {ablate_head_edge_attr}; backbone edge_dim={edge_dim}; head edge_dim={head_edge_dim}')
    # I3 fix: make every InstanceNorm normalize per-graph (see top of file).
    n_node_norms = _install_pergraph_norm_hooks(gnn, 'node')
    n_edge_norms = _install_pergraph_norm_hooks(head, 'edge')
    print(f'[INFO] norm_mode: {norm_mode}  '
          f'(per-graph InstanceNorm hooks: {n_node_norms} backbone + {n_edge_norms} head)')
    # -------------------------------------------------------------
    # NEW: report parameter count
    num_params = sum(p.numel() for p in gnn.parameters() if p.requires_grad) \
            + sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"[INFO] Model parameters (gnn + head): {num_params:,}")
    # ===================== GPU MONITOR - START OF TRAIN =====================
    is_cuda = (cuda_id >= 0 and torch.cuda.is_available())
    gpu_monitor = None
    gpu_global_peak_alloc = 0
    gpu_global_peak_reserved = 0
    train_t0 = time.perf_counter()

    if is_cuda:
        # Start background NVML sampling (utilization, power, mem used)
        gpu_monitor = GPUMonitor(device_index=cuda_id, interval_sec=0.25)
        gpu_monitor.start()

        torch.cuda.set_device(cuda_id)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        base_alloc = torch.cuda.memory_allocated(device)
        base_resv  = torch.cuda.memory_reserved(device)

        # Log and print baseline memory after model+optimizer are on the device
        writer.add_scalar('gpu/baseline_alloc_bytes', base_alloc, 0)
        writer.add_scalar('gpu/baseline_reserved_bytes', base_resv, 0)
        writer.add_scalar('model/num_params', num_params, 0)
        writer.add_text('gpu/device', f'CUDA:{cuda_id}', 0)
        print(f"[GPU] Baseline (after model+opt) on device {cuda_id}: "
            f"allocated={bytes_to_gb(base_alloc)} GB, reserved={bytes_to_gb(base_resv)} GB")
    # =======================================================================

    # -------------------------------------------------------------
    optimizer = torch.optim.Adam(list(gnn.parameters()) + list(head.parameters()), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience)

    best_train_loss, best_val_loss, best_test_loss = float('inf'), float('inf'), float('inf')
    best_epoch = 0
    early_stop_best_val_loss = float('inf')
    early_stop_best_epoch = 0
    for epoch in range(epochs):
        # -------- TRAIN phase --------
        if is_cuda: torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        train_loss, _ = run_one_epoch(
            gnn, head, train_loader, criterion, optimizer,
            device, seed, epoch, 'train', dataset, writer, head_type, norm_mode
        )
        if is_cuda: torch.cuda.synchronize(device)
        train_sec = time.perf_counter() - t0
        writer.add_scalar('time/train_sec', train_sec, epoch)
        if is_cuda:
            peak_alloc = torch.cuda.max_memory_allocated(device)
            peak_resv  = torch.cuda.max_memory_reserved(device)
            writer.add_scalar('gpu/train_peak_alloc_bytes', peak_alloc, epoch)
            writer.add_scalar('gpu/train_peak_reserved_bytes', peak_resv, epoch)
            gpu_global_peak_alloc = max(gpu_global_peak_alloc, peak_alloc)
            gpu_global_peak_reserved = max(gpu_global_peak_reserved, peak_resv)

        # -------- VAL phase --------
        if is_cuda: torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        val_loss, _ = run_one_epoch(
            gnn, head, val_loader, criterion, None,
            device, seed, epoch, 'val  ', dataset, writer, head_type, norm_mode
        )
        if is_cuda: torch.cuda.synchronize(device)
        val_sec = time.perf_counter() - t0
        writer.add_scalar('time/val_sec', val_sec, epoch)
        if is_cuda:
            peak_alloc = torch.cuda.max_memory_allocated(device)
            peak_resv  = torch.cuda.max_memory_reserved(device)
            writer.add_scalar('gpu/val_peak_alloc_bytes', peak_alloc, epoch)
            writer.add_scalar('gpu/val_peak_reserved_bytes', peak_resv, epoch)
            gpu_global_peak_alloc = max(gpu_global_peak_alloc, peak_alloc)
            gpu_global_peak_reserved = max(gpu_global_peak_reserved, peak_resv)

        # -------- TEST phase --------
        if is_cuda: torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        test_loss, _ = run_one_epoch(
            gnn, head, test_loader, criterion, None,
            device, seed, epoch, 'test ', dataset, writer, head_type, norm_mode
        )
        if is_cuda: torch.cuda.synchronize(device)
        test_sec = time.perf_counter() - t0
        writer.add_scalar('time/test_sec', test_sec, epoch)
        if is_cuda:
            peak_alloc = torch.cuda.max_memory_allocated(device)
            peak_resv  = torch.cuda.max_memory_reserved(device)
            writer.add_scalar('gpu/test_peak_alloc_bytes', peak_alloc, epoch)
            writer.add_scalar('gpu/test_peak_reserved_bytes', peak_resv, epoch)
            gpu_global_peak_alloc = max(gpu_global_peak_alloc, peak_alloc)
            gpu_global_peak_reserved = max(gpu_global_peak_reserved, peak_resv)

        # keep your LR scheduler, best-loss tracking, checkpointing, etc.
        scheduler.step(val_loss)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        val_loss_float = float(val_loss)

        if val_loss < best_val_loss:
            best_train_loss, best_val_loss, best_test_loss = train_loss, val_loss, test_loss
            best_epoch = epoch
            checkpoint_tag = '_ablateHeadEdgeAttr_True' if ablate_head_edge_attr else ''
            checkpoint_name = f'gnn_{model_name}_weighted_{is_weighted}_{head_type}_{norm_mode}{checkpoint_tag}_seed_{seed}.pth'
            torch.save(
                {
                    'gnn_state_dict': gnn.state_dict(),
                    'head_state_dict': head.state_dict(),
                    'model_config': {
                        'head': head_type,
                        'norm_mode': norm_mode,
                        'hidden_channels': hidden_channels,
                        'out_channels': out_channels,
                        'num_layers': num_layers,
                        'dropout': dropout,
                        'lr': lr,
                        'wd': wd,
                        'epochs': epochs,
                        'batch_size': batch_size,
                        'factor': factor,
                        'patience': patience,
                        'early_stop_patience': early_termination,
                        'early_stop_min_delta': early_stop_min_delta,
                        'criterion': criterion,
                        'use_node_feats': use_node_feats,
                        'cuda': cuda_id,
                        'edge_dim': head_edge_dim,
                        'backbone_edge_dim': edge_dim,
                        'ablate_head_edge_attr': ablate_head_edge_attr,
                        'y_min': float(dataset.y_min),
                        'y_max': float(dataset.y_max),
                        'deg_max': float(dataset.deg_max),
                    }
                },
                osp.join(root, checkpoint_name)
            )

        if early_stop_min_delta > 0:
            improved_for_early_stop = val_loss_float <= early_stop_best_val_loss - early_stop_min_delta
        else:
            improved_for_early_stop = val_loss_float < early_stop_best_val_loss
        if improved_for_early_stop:
            early_stop_best_val_loss = val_loss_float
            early_stop_best_epoch = epoch

        writer.add_scalar('train/best_train_loss', best_train_loss, epoch)
        writer.add_scalar('val/best_val_loss', best_val_loss, epoch)
        writer.add_scalar('test/best_test_loss', best_test_loss, epoch)

        print('-'*100)
        print('-'*100)

        if (epoch - early_stop_best_epoch) >= early_termination:
            print('Early termination')
            break
    # ===================== GPU MONITOR - END OF TRAIN =====================
    train_sec = time.perf_counter() - train_t0
    if is_cuda:
        if gpu_monitor:
            gpu_monitor.stop()
        nvml_summary = gpu_monitor.summary() if gpu_monitor else {'nvml_available': False}

        gpu_json = {
            'model': model_name,
            'head': head_type,
            'num_params': int(num_params),
            'epochs_run': int(best_epoch + 1 if 'best_epoch' in locals() else epoch + 1),
            'training_time_sec': float(train_sec),
            'pytorch_global_peak_alloc_bytes': int(gpu_global_peak_alloc),
            'pytorch_global_peak_reserved_bytes': int(gpu_global_peak_reserved),
            'nvml': nvml_summary
        }
        gpu_json_path = osp.join(writer.log_dir, 'gpu_report.json')
        with open(gpu_json_path, 'w') as f:
            json.dump(gpu_json, f, indent=2)
        print(f"[GPU] Wrote training GPU report -> {gpu_json_path}")

        # Also pin key NVML stats to TB (step=0) so tensorboard2csv can aggregate them:
        if nvml_summary.get('nvml_available', False):
            writer.add_scalar('gpu/nvml_avg_util_percent', nvml_summary.get('avg_util_percent', 0.0), 0)
            writer.add_scalar('gpu/nvml_avg_power_w', nvml_summary.get('avg_power_w', 0.0), 0)
            writer.add_scalar('gpu/nvml_peak_mem_used_bytes', nvml_summary.get('peak_mem_used_bytes_nvml', 0), 0)
    # ======================================================================

    print("\nConverting tensorboard to cvs... ", end="")
    tb2csv = osp.join(osp.dirname(osp.abspath(__file__)), 'tensorboard2csv.py')
    command = f'python "{tb2csv}" --path "{osp.join(root, "logs", log_subdir)}"'
    os.system(command)
    print("done.\n")

if __name__ == '__main__':
    torch.set_num_threads(5)
    main()
