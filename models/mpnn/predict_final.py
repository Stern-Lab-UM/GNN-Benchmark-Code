#!/usr/bin/env python3
"""
predict_final.py -- Inference on **exactly one** Nano `.txt` file with an exact
checkpoint. Head-selectable copy of nano-main/src/predict_final.py.

### Key features
1. **Exact checkpoint** -- uses only the `.pth` you specify.
2. **Exact data file** -- processes exactly the `.txt` you specify by staging it
   into a temporary folder that satisfies Nano's `assert '2D' in root` rule.
3. **Exact output filename** -- writes predictions to the path you pass with
   `--out` (or `basename_pred.txt` beside the data file if omitted).
4. **Selectable head** -- auto-detects the prediction head from the checkpoint
   (model_config['head'], or the saved state-dict key for legacy checkpoints)
   and rebuilds the matching head. Override with --head if needed.
5. **Ablation-aware head width** -- if the checkpoint was trained with
   ablate_head_edge_attr, the final EdgeRegressor is rebuilt without raw
   edge attributes while the backbone still receives the graph edge features.

NOTE: this copy imports get_model/run_one_epoch from `trainer_final` -- the
published predict_final.py imported a stale `trainer` module (audit issue I4).
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import re
import shutil
import tempfile
from typing import Optional

import torch
from torch_geometric.loader import DataLoader

from dataset import Nano
from trainer_final import get_model, run_one_epoch, _install_pergraph_norm_hooks
from models import LinkPredictor, EdgeRegressor

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

_CKPT_RE = re.compile(r"gnn_(.+?)_weighted_", re.IGNORECASE)


def _parse_model_name(ckpt_path: str) -> str:
    """
    Implement the parse model name step for models / mpnn / predict_final.py.

    Args:
        ckpt_path: Caller-supplied value used by this routine.

    Returns:
        Computed value used by the caller.
    """
    m = _CKPT_RE.search(osp.basename(ckpt_path))
    if not m:
        raise ValueError(f"Cannot parse model name from '{osp.basename(ckpt_path)}'.")
    return m.group(1)


def _require_checkpoint(path: str) -> str:
    """
    Implement the require checkpoint step for models / mpnn / predict_final.py.

    Args:
        path: Caller-supplied value used by this routine.

    Returns:
        Computed value used by the caller.
    """
    if osp.isfile(path) and path.endswith(".pth"):
        return osp.abspath(path)
    raise FileNotFoundError(f"Checkpoint must be an existing .pth file - got {path}")


def _detect_weighted(data_file: str) -> Optional[bool]:
    """
    Implement the detect weighted step for models / mpnn / predict_final.py.

    Args:
        data_file: Caller-supplied value used by this routine.

    Returns:
        Computed value used by the caller.
    """
    if data_file.endswith("_weighted.txt"):
        return True
    if data_file.endswith("_unweighted.txt"):
        return False
    return None


def _load_single_file_dataset(
    data_file: str,
    is_weighted: Optional[bool],
    use_node_feats: bool,
) -> Nano:
    """Stage one graph file into an isolated temporary Nano dataset root.

    The temporary folder name contains ``2D`` so the Nano loader infers the
    dimensionality correctly, and only ``data_file`` is copied into it so no
    neighboring datasets are accidentally processed during prediction.

    Args:
        data_file: Path to one Nano-format graph text file to predict.
        is_weighted: Whether to parse weighted edge attributes; ``None`` keeps
            the dataset loader's automatic handling.
        use_node_feats: Whether to construct the full node-feature transform.

    Returns:
        Loaded ``Nano`` dataset whose temporary directory is kept alive by the
        returned object.
    """
    if not osp.isfile(data_file):
        raise FileNotFoundError(f"Data file '{data_file}' not found.")

    tmp_dir = tempfile.TemporaryDirectory(prefix="singlefile_2D_")
    staged_root = tmp_dir.name  # contains '2D' to satisfy dataset assertion

    shutil.copy2(data_file, osp.join(staged_root, osp.basename(data_file)))

    dataset = Nano(root=staged_root, is_weighted=is_weighted, use_node_feats=use_node_feats)
    dataset._tmp_dir = tmp_dir  # keep tempfile alive with dataset
    return dataset

# -----------------------------------------------------------------------------
# Prediction pipeline
# -----------------------------------------------------------------------------

def predict(
    model_path: str,
    data_file: str,
    output_path: Optional[str] = None,
    batch_size: int = 8,
    cuda_id: Optional[int] = None,
    use_node_feats: Optional[bool] = None,
    head: Optional[str] = None,
    norm_mode: Optional[str] = None,
) -> str:
    """
    Run model inference and return or save predictions.

    Args:
        model_path: Caller-supplied value used by this routine.
        data_file: Caller-supplied value used by this routine.
        output_path: Caller-supplied value used by this routine.
        batch_size: Caller-supplied value used by this routine.
        cuda_id: Caller-supplied value used by this routine.
        use_node_feats: Caller-supplied value used by this routine.
        head: Caller-supplied value used by this routine.
        norm_mode: Caller-supplied value used by this routine.

    Returns:
        Computed value used by the caller.
    """
    ckpt_path = _require_checkpoint(model_path)

    print(f"[INFO] Using checkpoint: {ckpt_path}")
    # Load to CPU first so we can read model_config before choosing the device.
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_name = _parse_model_name(ckpt_path)
    cfg = checkpoint["model_config"]

    # ---- Which prediction head did this checkpoint train with? ----
    # Priority: explicit --head override -> model_config['head'] -> infer from
    # the saved state-dict key (new checkpoints save 'head_state_dict'; the
    # original published checkpoints saved 'link_predictor_state_dict').
    head_type = head or cfg.get("head")
    if head_type is None:
        head_type = "sigmoid" if "link_predictor_state_dict" in checkpoint else "regressor"
    if head_type not in ("regressor", "sigmoid"):
        raise ValueError(f"Unknown head '{head_type}', expected 'regressor' or 'sigmoid'.")
    print(f"[INFO] Prediction head: {head_type}")

    ablate_head_edge_attr = bool(cfg.get("ablate_head_edge_attr", False))
    print(f"[INFO] ablate_head_edge_attr: {ablate_head_edge_attr}")
    # Which InstanceNorm scope did this checkpoint train with?
    norm_mode_resolved = norm_mode or cfg.get("norm_mode")
    if norm_mode_resolved is None:
        norm_mode_resolved = "pooled"   # legacy checkpoints predate the I3 fix
    if norm_mode_resolved not in ("per_graph", "pooled"):
        raise ValueError(f"Unknown norm_mode '{norm_mode_resolved}'.")
    print(f"[INFO] norm_mode: {norm_mode_resolved}")

    # Device: CLI overrides, otherwise use the GPU the checkpoint was trained on.
    if cuda_id is None:
        cuda_id = int(cfg.get("cuda", -1))
    if cuda_id >= 0 and not torch.cuda.is_available():
        print(f"[WARN] Checkpoint trained on cuda:{cuda_id} but no CUDA is available; falling back to CPU.")
        cuda_id = -1
    device = torch.device(f"cuda:{cuda_id}" if cuda_id >= 0 else "cpu")
    print(f"[INFO] Running inference on {device}")

    # Determine feature setting: CLI overrides, otherwise use what was saved in training.
    if use_node_feats is None:
        use_node_feats = bool(cfg.get("use_node_feats", True))

    is_weighted = _detect_weighted(data_file)
    dataset = _load_single_file_dataset(data_file, is_weighted, use_node_feats)

    # Override the single-file dataset's y range with the training-set range
    # saved in the checkpoint. Only the sigmoid head uses it (inverse_scale_y);
    # it is harmless (unused) for the regressor head.
    if 'y_min' in cfg and 'y_max' in cfg:
        dataset.set_y_range(cfg['y_min'], cfg['y_max'])
        print(f"[INFO] Using training-set y range from checkpoint: "
              f"[{cfg['y_min']:.6f}, {cfg['y_max']:.6f}]")
    else:
        print("[WARN] Checkpoint lacks y_min/y_max; falling back to single-file "
              "y range. Sigmoid-head predictions in physical units may be incorrect.")

    # Same for deg_max (LDP column scale used in run_one_epoch).
    if 'deg_max' in cfg:
        dataset.set_deg_max(cfg['deg_max'])
        print(f"[INFO] Using training-set deg_max from checkpoint: {cfg['deg_max']:.6f}")
    else:
        print("[WARN] Checkpoint lacks deg_max; falling back to single-file value.")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    in_channels = dataset[0].x.size(1)
    edge_dim = dataset[0].edge_attr.size(1) if dataset[0].edge_attr.numel() else None
    Model, deg = get_model(model_name, dataset)

    model_kwargs = dict(dropout=0.0, norm="instance_norm", jk="cat", edge_dim=edge_dim)
    if deg is not None:
        model_kwargs["deg"] = deg
    gnn = Model(
        in_channels,
        cfg["hidden_channels"],
        cfg["num_layers"],
        cfg["hidden_channels"],
        **model_kwargs,
    ).to(device)

    # Build the head that matches the checkpoint.
    if head_type == "sigmoid":
        head_module = LinkPredictor(
            cfg["hidden_channels"],
            cfg["hidden_channels"],
            cfg["out_channels"],
            2,
            0.0,
        ).to(device)
    else:
        head_edge_dim = cfg.get("edge_dim", edge_dim)
        print(f"[INFO] Backbone edge_dim={edge_dim}; head edge_dim={head_edge_dim}")
        head_module = EdgeRegressor(
            cfg["hidden_channels"],
            cfg["hidden_channels"],
            cfg["out_channels"],
            num_layers=2,
            dropout=0.0,
            edge_dim=head_edge_dim,
        ).to(device)

    gnn.load_state_dict(checkpoint["gnn_state_dict"])
    head_state = checkpoint.get("head_state_dict", checkpoint.get("link_predictor_state_dict"))
    if head_state is None:
        raise KeyError("Checkpoint has neither 'head_state_dict' nor 'link_predictor_state_dict'.")
    head_module.load_state_dict(head_state)
    criterion = cfg["criterion"]

    # I3 fix: per-graph InstanceNorm via forward-hooks (see trainer_final.py).
    _install_pergraph_norm_hooks(gnn, 'node')
    _install_pergraph_norm_hooks(head_module, 'edge')

    _, (all_y_pred, all_y_true) = run_one_epoch(
        gnn,
        head_module,
        loader,
        criterion,
        None,
        device,
        0,
        999,
        "test ",
        dataset,
        None,
        head_type,
        norm_mode_resolved,
    )

    # --------------------------- write predictions ---------------------------
    if output_path is None:
        base = osp.splitext(osp.basename(data_file))[0] + "_pred.txt"
        output_path = osp.join(osp.dirname(data_file), base)

    out_dir = osp.dirname(osp.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)
    # Concurrency fix (audit): save_preds() writes a fixed name
    # <input>_pred.txt that is NOT model/seed-unique, so parallel predicts in
    # the same directory clobber each other's intermediate file. Stage into a
    # private per-process temp dir on the SAME filesystem, then atomically
    # rename into place (os.replace is atomic; cross-process safe).
    stage_dir = tempfile.mkdtemp(dir=out_dir, prefix=".predstage_")
    try:
        dataset.save_preds(all_y_pred, all_y_true, stage_dir)
        staged = osp.join(stage_dir,
                          osp.splitext(osp.basename(data_file))[0] + "_pred.txt")
        os.replace(staged, output_path)
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)

    print(f"[INFO] Inference complete on {len(dataset)} graphs; results -> {output_path}")
    return osp.abspath(output_path)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _cli_parser() -> argparse.ArgumentParser:
    """
    Implement the cli parser step for models / mpnn / predict_final.py.

    Returns:
        Computed value used by the caller.
    """
    p = argparse.ArgumentParser(description="Single-file Nano inference script")
    p.add_argument("-m", "--model-path", required=True,
                   help="Path to the exact .pth checkpoint to load")
    p.add_argument("-d", "--data-file", required=True,
                   help="Path to the *single* Nano .txt file to process")
    p.add_argument("-o", "--out", help="Exact output filename for predictions")
    p.add_argument("-b", "--batch-size", type=int, default=8,
                   help="Mini-batch size")
    p.add_argument("-c", "--cuda", type=int, default=None,
                   help="CUDA device ID; -1 for CPU. Default: use the device saved in the checkpoint.")
    p.add_argument("--use-node-feats", choices=["True", "False"], default=None,
               help="Use the 35 node features. Default: auto-detect from checkpoint.")
    p.add_argument("--head", choices=["regressor", "sigmoid"], default=None,
               help="Prediction head. Default: auto-detect from the checkpoint "
                    "(model_config['head'], else the saved state-dict key).")
    p.add_argument("--norm_mode", choices=["per_graph", "pooled"], default=None,
               help="InstanceNorm scope. Default: auto-detect from the checkpoint "
                    "(model_config['norm_mode'], else 'pooled' for legacy checkpoints).")
    return p


def main() -> None:
    """
    Parse command-line arguments and run this script entry point.

    Returns:
        None; the function updates object state, files, logs, or external process state.
    """
    args = _cli_parser().parse_args()
    predict(
        model_path=args.model_path,
        data_file=args.data_file,
        output_path=args.out,
        batch_size=args.batch_size,
        cuda_id=args.cuda,
        use_node_feats=(None if args.use_node_feats is None else args.use_node_feats == "True"),
        head=args.head,
        norm_mode=args.norm_mode,
    )


if __name__ == "__main__":
    main()
