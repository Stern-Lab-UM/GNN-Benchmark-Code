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
    '''
    def __init__(self, out_dir):
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
        '''
        with self.get_out_file(self.config_file) as outfile:
            yaml.safe_dump(config.config, outfile)

    def write_metrics(self, metrics):
        with self.get_out_file(self.metrics_file) as outfile:
            metrics.to_csv(outfile)

    def write_model(self, trainer, norm_factors):
        with self.get_out_file(self.model_file, zipped=True) as outfile:
            torch.save({'epoch': trainer.cur_epoch,
                        'model_state_dict': trainer.model.state_dict(),
                        'optimizer_state_dict': trainer.optimizer.state_dict(),
                        'norm_factors': norm_factors,
                        },
                       outfile)

    def write_indices(self, indices):
        for key, array in indices.items():
            with self.get_out_file(self.ind_template.format(key=key)) \
                    as outfile:
                np.savetxt(outfile, array, fmt='%d')

    def load_config(self):
        '''
        Return configuration object with the yaml values from training
        '''
        with self.get_out_file(self.config_file, mode='r') as outfile:
            return Configuration(outfile)

    def load_metrics(self):
        with self.get_out_file(self.metrics_file, mode='r') as outfile:
            return pd.read_csv(outfile)

    def load_model_state(self, config: Configuration):
        model = config.build_model()
        with self.get_out_file(self.model_file, zipped=True, mode='r') as file:
            saved_states = torch.load(file, map_location=DEVICE, weights_only=False)
        model.load_state_dict(saved_states['model_state_dict'])
        model.set_norm_factors(saved_states['norm_factors'],
                               config, DEVICE)
        return model, saved_states['epoch'], saved_states['norm_factors']

    def load_indices(self):
        result = {}
        for key in ('train', 'val', 'test'):
            with self.get_out_file(self.ind_template.format(key=key),
                                   mode='r') as infile:
                result[key] = np.loadtxt(infile, dtype=int)

        return result


class SavedModel():
    def __init__(self, base_dir, need_metrics=False):
        recorder = Recorder(base_dir)
        self.config = recorder.load_config()

        self.metrics = None
        if need_metrics:
            self.metrics = recorder.load_metrics()

        (self.model, self.epoch, self.norm_factors) = \
            recorder.load_model_state(self.config)
