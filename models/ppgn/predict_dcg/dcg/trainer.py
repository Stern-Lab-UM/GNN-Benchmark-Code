"""Utilities for models / ppgn / predict_dcg / dcg / trainer.py in the DCG benchmark codebase."""

from typing import Dict
import copy
import torch
import torch.nn as nn
import torch.optim
import pandas as pd
import numpy as np
from datetime import datetime
import click
import logging
import torch.nn.functional as F

class Trainer(object):
    """
    train and evaluate PPGN models


    Role:
        Trainer groups state and methods for this repository component.
    """
    def __init__(self, model, dataloader, config, device):
        """
        Initialize the Trainer instance and store constructor configuration.

        Args:
            model: Caller-supplied value used by this routine.
            dataloader: Caller-supplied value used by this routine.
            config: Caller-supplied value used by this routine.
            device: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
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

        self.is_bce = (loss == 'BCE')

        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=params['learning_rate'])

        # NOTE: `verbose=True` was removed from ReduceLROnPlateau in PyTorch
        # 2.x. PPGN_Tomer was written against older PyTorch where it was
        # still accepted. Both cluster (torch 2.7.x) and runpod images
        # (torch 2.8.x) trip on it -- drop it; the scheduler still applies
        # the LR reduction silently.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            patience=params['patience'],
            factor=params['factor'],
            threshold=params['threshold'],
        )


        # THIS IS FOR LOSS FUNCTION << ADDED BY RIYA SAMANTA.
        ratio = params.get('bce_ratio', None)
        fp_w  = params.get('bce_fp_weight', None)
        fn_w  = params.get('bce_fn_weight', None)


        if ratio is not None:                        # CASE B
            self.pos_weight_val = float(ratio)       # (= fn / fp)
        elif fp_w is not None and fn_w is not None:  # CASE A
            self.pos_weight_val = float(fn_w) / max(float(fp_w), 1e-12)
        """
        else:

            pos_cnt = neg_cnt = 0                               # CASE C – compute from data
            with torch.no_grad():                              # no gradients needed
                for g, t in dataloader.train:                  # loop over train set
                    g, t = g.to(device), t.to(device)

                    # ---- LOCAL mask, used only here ----
                    local_mask = self.model.output_mask(g)     # structural

                    # candidate filter (present or not)
#                    if 'is_candidate' in dataloader.input_features:
#                        idx = 1 + dataloader.input_features.index('is_candidate') #first channel is reserved for the adjacency matrix
#                        local_mask = local_mask & (g[:, idx] > 0).unsqueeze(1)
                    # ------------------------------------
                    # assuming “is_mother” is target channel 0
                    mother_idx = dataloader.target_features.index('is_mother') # earlier assumption was that the first channel is "is_mother" - which is true in our case, but definitely should be generalized.
                    is_pos = (t[:, mother_idx] == 1) & local_mask[:, mother_idx]

                    #is_pos = (t[:, 0] == 1) & local_mask[:, 0]
                    pos_cnt += is_pos.sum().item()
#                    neg_cnt += (local_mask[:, 0].sum() - is_pos.sum()).item()
                    neg_cnt += (local_mask[:, mother_idx].sum() - is_pos.sum()).item()

            #print (pos_cnt,neg_cnt)
            self.pos_weight_val = neg_cnt / max(pos_cnt, 1)    # fallback to 1 if no positives

        """
        self.pos_weight_val = 1.0; # for now, we don't need it
        # Keep a tensor ready on the correct device
        self.pos_weight = torch.tensor(self.pos_weight_val,
                               dtype=torch.float32, device=device)
        #print(self.pos_weight)
        #print(self.pos_weight_val)

        # create the masker (enabled by default, can be turned off via config)
#        self.candidate_masker = CandidateMask(enabled=True)

        #<< EDITED BY RIYA - END


    def train(self, epochs):
        """
        Trains for the num of epochs.

        Args:
            epochs: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
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
            import traceback, sys
            traceback.print_exc()
            sys.exit(1)
            #if 'memory' in str(e):
            #    raise e
            #if self.verbose:
            #    click.echo('\nStopping training due to keyboard interrupt...')

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
        :return dist and loss on train set

        Args:
            epoch: Caller-supplied value used by this routine.
        """
        return self.evaluate_batches(self.data.train, 'Training', epoch, optimize=True)

#        def validate(self, epoch, best_loss):
#            """
#            Perform forward pass on the model with the validation set
#            :param epoch: Epoch number
#            :return: (val_dists, val_loss)
#            """
#            with torch.no_grad():
#                return self.evaluate_batches(self.data.val, 'Val',
#                                            epoch, best_loss=best_loss)

    def validate(self, epoch, best_loss):
        """
        Forward pass on the validation set (no gradient).

        Args:
            epoch: Caller-supplied value used by this routine.
            best_loss: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        with torch.no_grad():
            return self.evaluate_batches(self.data.val, 'Val', epoch, best_loss=best_loss)

    def load_best_model(self):
        """
        Restore the best checkpointed model state.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        self.model = self.best_model
        self.cur_epoch = self.best_epoch

#    def test(self):
#        """
#        Perform forward pass on the model for the test set
#        :param load_best_model: True to load the lowest validation loss model
#        :return: (test_dists, test_loss)
#        """
#        with torch.no_grad():
#            return self.evaluate_batches(self.data.test, 'Test',
#                                         self.cur_epoch)

    def test(self):
        """
        Forward pass on the test set (no gradient).

        Returns:
            Computed value used by the caller.
        """
        with torch.no_grad():
            return self.evaluate_batches(self.data.test, 'Test', self.cur_epoch)

    def evaluate_batches(self, dataset, name, epoch,
                         optimize=False, best_loss=None):
        """
        Evaluate all batches in a split and aggregate metrics.

        Args:
            dataset: Caller-supplied value used by this routine.
            name: Caller-supplied value used by this routine.
            epoch: Caller-supplied value used by this routine.
            optimize: Caller-supplied value used by this routine.
            best_loss: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
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
        """
        total_loss = 0
        dists = np.zeros(scores.shape[1])
        mask = self.model.output_mask(graphs)

        # ADDED BY RIYA SAMANTA
#        cand_mask = self.candidate_masker(graphs, self.data.input_features)                     # (B, 1, N, N)
        #mask2 = mask # real mask
#        mask = mask & cand_mask                                   # broadcast over X

        #breakpoint()
        # Original resumes
        for i, n in enumerate(self.target_features):
            if not mask[:, i].any():
                continue                                      # ADDED BY RIYA SAMANTA

            logits = scores[:, i][mask[:, i]]
            truth = targets[:, i][mask[:, i]]

            #logits2 = scores[:,i][mask2[:,i]] # for debugging
            #truth2 = targets[:,i][mask2[:,i]] # for debugging

            loss = self.loss_fn(logits,truth)

            #if self.is_bce:                                   # ADDED BY RIYA

             #   loss = F.binary_cross_entropy_with_logits(
             #           logits, truth, pos_weight=self.pos_weight)

                #loss2 = F.binary_cross_entropy_with_logits(logits2, truth2)
                #loss3 = F.binary_cross_entropy_with_logits(logits2,truth2,pos_weight=self.pos_weight)
                #loss = F.binary_cross_entropy_with_logits(
                #    logits, truth, weight=sample_weight, pos_weight=pos_w)
                #print(logits,truth,logits2,truth2,loss,loss2,loss3)
                #breakpoint()

              #  loss_fn = torch.nn.BCEWithLogitsLoss(
              #          weight=sample_weight,
              #          pos_weight=pos_w,
              #          reduction='none'
              #          )
              #  loss = loss_fn(logits,truth)

            #else:
            #    loss = self.loss_fn(logits,truth)

            #breakpoint()
            total_loss += loss
            if (self.penalties is not None
                    and n in self.penalties
                    and self.penalties[n] != 0):
                penalty = self.penalties[n]
                total_loss += penalty * self.loss_fn(
                    scores[:, i][mask[:, i]].sum(),
                    targets[:, i][mask[:, i]].sum()
                )
            dists[i] = loss.cpu().item() * self.norm_factors.loc[n, 'std']
            #breakpoint()

        return total_loss, dists

    def set_penalties(self, penalties: Dict[str, float]) -> None:
        '''
        Set the penalty attribute of the trainer
        Used to scale the loss of features differently

        Args:
            penalties: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        '''
        self.penalties = penalties







class RMSLELoss(nn.Module):
    """
    Provide the rmsleloss component used by models / ppgn / predict_dcg / dcg / trainer.py.


    Role:
        RMSLELoss groups state and methods for this repository component.
    """
    def __init__(self):
        """
        Initialize the RMSLELoss instance and store constructor configuration.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        """
        Run the neural-network forward pass for this module.

        Args:
            pred: Caller-supplied value used by this routine.
            actual: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))


#class CandidateMask:
#    """
#    Optional filter that ignores samples whose `is_candidate` == 0.
#
#    enabled         : set to False to disable the filter at runtime
#    feature_name    : channel name that carries the flag (default 'is_candidate')
#    """
#    def __init__(self, feature_name: str = 'is_candidate', enabled: bool = True):
#        self.feature_name = feature_name
#        self.enabled = enabled
#        self._channel_idx = None          # discovered on first call
#
#    def __call__(self, graphs: torch.Tensor, input_features: list) -> torch.Tensor:
#        """
#        Returns a boolean tensor of shape (B, 1, N, N).
#        All-True if the filter is disabled *or* the feature is absent.
#        """
#        if not self.enabled:
#            # disabled → keep everything
#            return torch.ones(
#                graphs.shape[0], 1, graphs.shape[-2], graphs.shape[-1],
#                dtype=torch.bool, device=graphs.device)
#
#        # discover the channel index once
#        if self._channel_idx is None:
#            if self.feature_name not in input_features:
#                # feature not present → fall back to all-True mask
#                return torch.ones(
#                    graphs.shape[0], 1, graphs.shape[-2], graphs.shape[-1],
#                    dtype=torch.bool, device=graphs.device)
#            self._channel_idx = input_features.index(self.feature_name) + 1  # +1 for adjacency
#
#        return (graphs[:, self._channel_idx] == 1).unsqueeze(1)
