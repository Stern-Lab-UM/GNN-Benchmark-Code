"""Utilities for models / ppgn / predict_dcg / dcg / recorder.py in the DCG benchmark codebase."""

import os
import yaml
from contextlib import contextmanager
import torch
import gzip
import pandas as pd
import numpy as np

from dcg.configuration import Configuration, DEVICE


class Recorder():
    '''
    Responsible for writing models, supporting information, and performance
    measurements

    Role:
        Recorder groups state and methods for this repository component.
    '''
    def __init__(self, out_dir):
        """
        Initialize the Recorder instance and store constructor configuration.

        Args:
            out_dir: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        self.out_dir = out_dir
        if os.path.isfile(out_dir):
            raise ValueError(f'Output directory "{out_dir}" is a file')

        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = os.path.abspath(out_dir)
        self.config_file = 'config.yaml'
        self.metrics_file = 'metrics.csv'
        self.model_file = 'model.tar'
        self.ind_template = '{key}.inds'

    @contextmanager
    def get_out_file(self, basename, mode='w', zipped=False):
        '''
        get the basename joined with out_dir and yield a file
        Mostly used for testing purposes

        Args:
            basename: Caller-supplied value used by this routine.
            mode: Caller-supplied value used by this routine.
            zipped: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        '''
        name = os.path.join(self.out_dir, basename)
        if zipped:
            handle = gzip.open(name, mode)
        else:
            handle = open(name, mode)

        try:
            yield handle
        finally:
            handle.close()

    def write_config(self, config: Configuration):
        '''
        dump the config object as a yaml file in the out_dir

        Args:
            config: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        '''
        with self.get_out_file(self.config_file) as outfile:
            yaml.safe_dump(config.config, outfile)

    def write_metrics(self, metrics):
        """
        Append metric values to the run metrics table.

        Args:
            metrics: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        with self.get_out_file(self.metrics_file) as outfile:
            metrics.to_csv(outfile)

    def write_model(self, trainer, norm_factors):
        """
        Persist the current model checkpoint and normalization state.

        Args:
            trainer: Caller-supplied value used by this routine.
            norm_factors: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        with self.get_out_file(self.model_file, zipped=True) as outfile:
            torch.save({'epoch': trainer.cur_epoch,
                        'model_state_dict': trainer.model.state_dict(),
                        'optimizer_state_dict': trainer.optimizer.state_dict(),
                        'norm_factors': norm_factors,
                        },
                       outfile)

    def write_indices(self, indices):
        """
        Persist train/validation/test split indices.

        Args:
            indices: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        for key, array in indices.items():
            with self.get_out_file(self.ind_template.format(key=key)) \
                    as outfile:
                np.savetxt(outfile, array, fmt='%d')

    def load_config(self):
        '''
        Return configuration object with the yaml values from training

        Returns:
            Computed value used by the caller.
        '''
        with self.get_out_file(self.config_file, mode='r') as outfile:
            return Configuration(outfile)

    def load_metrics(self):
        """
        Load a saved metrics table.

        Returns:
            Computed value used by the caller.
        """
        with self.get_out_file(self.metrics_file, mode='r') as outfile:
            return pd.read_csv(outfile)

    def load_model_state(self, config: Configuration):
        """
        Load a saved model checkpoint state.

        Args:
            config: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        model = config.build_model()
        with self.get_out_file(self.model_file, zipped=True, mode='r') as file:
            saved_states = torch.load(file, map_location=DEVICE, weights_only=False)
        model.load_state_dict(saved_states['model_state_dict'])
        model.set_norm_factors(saved_states['norm_factors'],
                               config, DEVICE)
        return model, saved_states['epoch'], saved_states['norm_factors']

    def load_indices(self):
        """
        Load saved train/validation/test split indices.

        Returns:
            Computed value used by the caller.
        """
        result = {}
        for key in ('train', 'val', 'test'):
            with self.get_out_file(self.ind_template.format(key=key),
                                   mode='r') as infile:
                result[key] = np.loadtxt(infile, dtype=int)

        return result


class SavedModel():
    """
    Provide the saved model component used by models / ppgn / predict_dcg / dcg / recorder.py.


    Role:
        SavedModel groups state and methods for this repository component.
    """
    def __init__(self, base_dir, need_metrics=False):
        """
        Initialize the SavedModel instance and store constructor configuration.

        Args:
            base_dir: Caller-supplied value used by this routine.
            need_metrics: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        recorder = Recorder(base_dir)
        self.config = recorder.load_config()

        self.metrics = None
        if need_metrics:
            self.metrics = recorder.load_metrics()

        (self.model, self.epoch, self.norm_factors) = \
            recorder.load_model_state(self.config)
