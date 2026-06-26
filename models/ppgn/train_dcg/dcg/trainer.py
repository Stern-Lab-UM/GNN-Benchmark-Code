from typing import Dict
import copy
import torch
import torch.nn as nn
import torch.optim
import pandas as pd
import numpy as np
from datetime import datetime
import click


class Trainer(object):
    def __init__(self, model, dataloader, config, device):
        self.best_epoch = -1
        self.cur_epoch = 0
        self.device = device
        self.verbose = True
        self.early_stop = config['training_info']['early_stop']
        self.penalties = None
        self.clipping = config['training_info']['gradient_clipping']

        self.model = model
        self.data = dataloader

        self.norm_factors = dataloader.norm_factors
        self.target_features = dataloader.target_features
        self.num_targets = model.num_targets

        params = config['hyperparameters']

        loss = params['loss']
        loss_fns = {
            'L1': nn.L1Loss,
            'L2': nn.MSELoss,
            'BCE': nn.BCEWithLogitsLoss,
            'MSLE': RMSLELoss,
        }
        if loss in loss_fns:
            self.loss_fn = loss_fns[loss]()
        else:
            raise ValueError(f'Unsupported loss function "{loss}"\n'
                             f'Options are [{", ".join(loss_fns.keys())}]')

        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=params['learning_rate'])

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=params['patience'],
            factor=params['factor'],
            threshold=params['threshold'],
        )

        # ---- candidate-mask + pos_weight switches ----
        # Both default to True. Either can be turned off via --args
        # "candidate_mask=false" / "pos_weight=false" to recover original behavior.
        self.candidate_mask_on = bool(
            config['training_info'].get('candidate_mask', True))
        self.pos_weight_on = bool(
            config['training_info'].get('pos_weight', True))
        self.loss_name = loss

        # Identify the candidate-edge column (accept legacy name).
        self._cand_col = None
        if hasattr(self.data, 'input_features'):
            for name in ('is_candidate', 'is_candidate_edge'):
                if name in self.data.input_features:
                    self._cand_col = name
                    break
        if self.candidate_mask_on and self._cand_col is None and self.verbose:
            click.echo("  [trainer] candidate_mask=True but neither "
                       "'is_candidate' nor 'is_candidate_edge' is in "
                       "input_features; falling back to full output mask.")

        # Per-target loss functions (default: shared self.loss_fn).
        # When pos_weight is on AND loss is BCE, override with per-target
        # BCEWithLogitsLoss(pos_weight=N_neg/N_pos) computed from training data.
        self.loss_fns = [self.loss_fn] * len(self.target_features)
        if self.pos_weight_on:
            if loss != 'BCE':
                if self.verbose:
                    click.echo(f"  [trainer] pos_weight=True ignored: "
                               f"only applies to BCE loss (got '{loss}').")
            else:
                self.loss_fns = self._build_pos_weighted_losses()

    def _build_pos_weighted_losses(self):
        """Iterate the training set once to count pos/neg per target over the
        same edges that loss_and_results will mask to (candidate edges if
        candidate_mask_on, otherwise the model's full output_mask), then
        return one BCEWithLogitsLoss(pos_weight=N_neg/N_pos) per target."""
        n_t = len(self.target_features)
        pos = torch.zeros(n_t, dtype=torch.float64)
        neg = torch.zeros(n_t, dtype=torch.float64)
        cand_idx = None
        if self.candidate_mask_on and self._cand_col is not None:
            cand_idx = 1 + self.data.input_features.index(self._cand_col)
        for graphs, targets in self.data.train:
            if cand_idx is not None:
                m = (graphs[:, cand_idx] == 1)            # (B, N, N)
            else:
                # full output mask (use first target slice; same shape per task)
                m = self.model.output_mask(graphs.to(self.device))[:, 0].cpu()
            for i in range(n_t):
                t = targets[:, i][m]
                pos[i] += (t == 1).sum().item()
                neg[i] += (t == 0).sum().item()
        # Avoid divide-by-zero: if pos[i]==0, fall back to weight 1 (no balancing)
        pw = torch.where(pos > 0, neg / pos.clamp(min=1.0), torch.ones_like(pos))
        if self.verbose:
            click.echo("  [trainer] pos_weight per target (= N_neg / N_pos):")
            for i, n in enumerate(self.target_features):
                click.echo(f"    {n}: pos={int(pos[i].item())} "
                           f"neg={int(neg[i].item())} "
                           f"-> pos_weight={pw[i].item():.4f}")
        return [
            nn.BCEWithLogitsLoss(
                pos_weight=pw[i].to(self.device, dtype=torch.float32))
            for i in range(n_t)
        ]

    def train(self, epochs):
        """
        Trains for the num of epochs.
        """
        best_loss = float('inf')
        best_dists = None
        metrics = []

        try:
            for self.cur_epoch in range(0, epochs):

                train_dists, train_loss = self.train_epoch(self.cur_epoch)
                val_dists, val_loss = self.validate(self.cur_epoch, best_loss)

                # step scheduler once per epoch
                self.scheduler.step(val_loss)

                entry = {
                    'train_loss': train_loss,
                    'val_loss': val_loss}
                entry.update({f'train_dists_{n}': v
                              for n, v in
                              zip(self.target_features, train_dists)})
                entry.update({f'val_dists_{n}': v
                              for n, v in
                              zip(self.target_features, val_dists)})
                metrics.append(entry)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_dists = val_dists
                    self.best_epoch = self.cur_epoch
                    self.best_model = copy.deepcopy(self.model)

                if (self.early_stop is not None and
                        self.best_epoch <= self.cur_epoch - self.early_stop):
                    if self.verbose:
                        click.echo('Stopping training due to no '
                                   'improvement over the last '
                                   f'{self.early_stop} epochs...')
                    break

        except (KeyboardInterrupt, RuntimeError) as e:
            # sometimes, keyboard interrupts during backprop raise runtime
            # errors.  In order to not mask out of memory, check the message
            if 'memory' in str(e):
                raise e
            if self.verbose:
                click.echo('\nStopping training due to keyboard interrupt...')

        best_epoch = self.best_epoch
        if self.verbose:
            click.echo(f'Best valuation results: {best_epoch}) '
                       f'loss: {best_loss:.4f} dists: {best_dists}')

        metrics = pd.DataFrame(metrics)
        metrics.index.name = 'epochs'

        return metrics

    def train_epoch(self, epoch):
        """
        implement the logic of epoch:
        -train all batches

        Train one epoch
        :param epoch: cur epoch number
        :return dist and loss on train set
        """
        result = self.evaluate_batches(self.data.train, 'Training', epoch,
                                       optimize=True)

        return result

    def validate(self, epoch, best_loss):
        """
        Perform forward pass on the model with the validation set
        :param epoch: Epoch number
        :return: (val_dists, val_loss)
        """
        with torch.no_grad():
            return self.evaluate_batches(self.data.val, 'Val',
                                         epoch, best_loss=best_loss)

    def load_best_model(self):
        self.model = self.best_model
        self.cur_epoch = self.best_epoch

    def test(self):
        """
        Perform forward pass on the model for the test set
        :param load_best_model: True to load the lowest validation loss model
        :return: (test_dists, test_loss)
        """
        with torch.no_grad():
            return self.evaluate_batches(self.data.test, 'Test',
                                         self.cur_epoch)

    def evaluate_batches(self, dataset, name, epoch,
                         optimize=False, best_loss=None):
        start_time = datetime.now()

        if optimize:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0
        total_dists = np.zeros(self.num_targets)

        # iterate over batches
        samples = 0
        for graphs, targets in dataset:
            graphs, targets = graphs.to(self.device), targets.to(self.device)
            if optimize:
                graphs.requires_grad_()
            scores = self.model(graphs)
            loss, dists = self.loss_and_results(
                scores, targets, graphs)

            if optimize:
                self.optimizer.zero_grad()
                loss.backward()

                if self.clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.clipping)

                self.optimizer.step()

            total_loss += loss.cpu().item()
            total_dists += dists  # np.array
            samples += graphs.shape[0]

        total_loss /= samples
        total_dists /= samples

        run_time = str(datetime.now() - start_time).split(".")[0]
        dists_s = ', '.join([f'{d:.4f}' for d in total_dists])

        if best_loss is not None:
            if total_loss < best_loss:
                name = '-> ' + name
            else:
                name = '   ' + name

        if self.verbose:
            click.echo(f'\t{name:<10}- {epoch:4} loss: '
                       f'{total_loss:.6f} '
                       f'-- dists: {dists_s} -- runtime: {run_time}')

        return total_dists, total_loss

    def loss_and_results(self, scores, targets, graphs):
        """
        :param scores: shape (B,X,N,N)
        :param targets: shape (B,X,N,N)
        :param graphs: Original graphs (B,Y,N,N), [:, 0] is adjacency
        :return: tuple of (loss tensor, array of distance per targets)

        If self.candidate_mask_on and an is_candidate column is available,
        intersect the model's output mask with (is_candidate == 1) so loss
        is computed only on candidate edges. Otherwise use the full mask
        (original behavior).
        """
        total_loss = 0
        dists = np.zeros(scores.shape[1])
        mask = self.model.output_mask(graphs)

        if self.candidate_mask_on and self._cand_col is not None:
            ci = 1 + self.data.input_features.index(self._cand_col)
            cand_mask = (graphs[:, ci] == 1).unsqueeze(1)   # (B, 1, N, N)
            mask = mask & cand_mask

        for i, n in enumerate(self.target_features):
            if not mask[:, i].any():
                continue                                    # no positions to score
            loss_fn_i = self.loss_fns[i]
            loss = loss_fn_i(scores[:, i][mask[:, i]],
                             targets[:, i][mask[:, i]])
            total_loss += loss
            if (self.penalties is not None
                    and n in self.penalties
                    and self.penalties[n] != 0):
                penalty = self.penalties[n]
                total_loss += penalty * loss_fn_i(
                    scores[:, i][mask[:, i]].sum(),
                    targets[:, i][mask[:, i]].sum()
                )
            dists[i] = loss.cpu().item() * self.norm_factors.loc[n, 'std']

        return total_loss, dists

    def set_penalties(self, penalties: Dict[str, float]) -> None:
        '''
        Set the penalty attribute of the trainer
        Used to scale the loss of features differently
        '''
        self.penalties = penalties


class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))
