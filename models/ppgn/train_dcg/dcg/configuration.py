"""Utilities for models / ppgn / train_dcg / dcg / configuration.py in the DCG benchmark codebase."""

import yaml
import copy
import torch

from dcg.cell_loader import CellLoader
from dcg.base_model import BaseModel
from dcg.trainer import Trainer


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DEFAULT_OPTIONS = {
    'hyperparameters': {
        'epochs': 500,
        'block_features': [400, 400, 400],
        'depth_of_mlp': 2,
        'learning_rate': 0.0001,
        'loss': 'L1',
        'factor': 0.1,
        'patience': 20,
        'threshold': 1e-4,
        'disable_first_skip': False,
    },
    'training_info': {
        'target_features': None,
        'input_features': None,
        'non_negative': None,
        'edge_only': 'all',
        'total_penalties': None,
        'batch_size': 64,
        'subset': None,
        'training_fraction': 0.8,
        'early_stop': 30,
        'normalize': 'all',
        'gradient_clipping': None,
        'instance_normalization': False,
        'candidate_mask': True,
        'pos_weight': True,
    },
    'execution': {
        'gpu': 0,
    }
}

HELP_OPTIONS = {
    'hyperparameters': {
        'epochs': '# INT maximum number of epocs to run',
        'block_features': ('# LIST[INT] number of features in each mlp block\n'
                           '# and number of blocks in model'),
        'depth_of_mlp': '# INT number of mlp layers for each block',
        'learning_rate': '# DOUBLE lr parameter to Adam optimizer',
        'loss': ('# Loss function, L1 -> L1loss, L2 -> MSELoss, '
                 'BCE -> BCEWithLogitsLoss, ' 'MSLE -> Mean Squared Log Error'),
        'factor': '# DOUBLE factor parameter to ReduceLROnPlateau scheduler',
        'patience': '# INT patience parameter to ReduceLROnPlateau scheduler',
        'threshold': ('# DOUBLE threshold parameter to '
                      'ReduceLROnPlateau scheduler'),
        'disable_first_skip': ('# BOOL ablation switch. False is the normal PPGN; '
                               '# True removes only the first RegularBlock skip path.'),
    },
    'training_info': {
        'target_features': ('# LIST[STR] which features of'
                            ' training data to target.\n'
                            '# null means target all'),
        'input_features': ('# LIST[STR] which features of'
                           ' training data to use as input.\n'
                           '# null means input only adjacency'),
        'non_negative': ('# LIST[STR] which targets to constrain '
                         'as non-negative.\n# null means none, '
                         'all means all targets'),
        'edge_only': ('# LIST[STR] targets to constrain '
                      'as zero along unconnected edges.\n# null means none, '
                      'all means all targets'),
        'total_penalties': ('# LIST[STR: DOUBLE] Target features (keys)'
                            ' to penalize for \n'
                            '# differences in total values. '
                            ' The value provided is a scale \n'
                            '# factor.  null sets all scales to 0,'
                            ' missing values are set as 0'),
        'batch_size': '# INT number of graphs to include in training batch',
        'subset': ('# INT number of graphs to load from training data.\n'
                   '# null means utilize all'),
        'training_fraction': ('# DOUBLE fraction of data to include in '
                              'training set.\n# Remaining data is split '
                              'between validation and test set'),
        'early_stop': ('# INT how many epochs to run without improvement '
                       'before stopping.\n# null means do not stop early'),
        'normalize': ('# LIST[STR] features to standardize (z-score).\n'
                      '# null means no normalization, '
                      'all means standardize all'),
        'gradient_clipping': ('# DOUBLE gradient clipping threshold.\n'
                              '# null means no clipping'
                              ),
        'instance_normalization': ('# BOOL if True, z-score each input feature\n'
                                   '# per-graph (using that graph\'s own mean/std)\n'
                                   '# instead of using global training-set stats.\n'
                                   '# Only affects input features listed in "normalize";\n'
                                   '# target features remain globally normalized.\n'
                                   '# Default: False (legacy dataset-level behavior)'),
        'candidate_mask': ('# BOOL if True, mask BCE loss to is_candidate=1 only.\n'
                           '# Concentrates gradient on the candidate edges, which\n'
                           '# are typically <2% of all edges; otherwise loss is\n'
                           '# dominated by trivial 0-targets on non-candidate\n'
                           '# edges and the prediction signal is drowned out.\n'
                           '# Requires "is_candidate" (or legacy "is_candidate_edge")\n'
                           '# in input_features; falls back to full mask + warning.\n'
                           '# Default: True. Set False to recover original behavior.'),
        'pos_weight': ('# BOOL if True, balance positive/negative class\n'
                       '# frequencies in BCE loss using pos_weight = N_neg/N_pos\n'
                       '# computed once per target from the training data\n'
                       '# (over candidate edges if available). No effect for\n'
                       '# non-BCE losses. Default: True.'),
    },
    'execution': {
        'gpu': '# INT index of GPU to use; ignored if gpu not available',
    }
}


def parse_args(*parsers):
    """
    Return a parser/converter for structured configuration values.

    Args:
        *parsers: Caller-supplied value used by this routine.

    Returns:
        Computed value used by the caller.
    """
    def parse(string):
        """
        Parse the supplied object or text into the structure expected by this module.

        Args:
            string: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        for parser in parsers:
            try:
                value = parser(string)
                return value
            except:
                continue
            raise ValueError(f'Unable to parse {string}')
    return parse


def list_or(other):
    """
    Implement the list or step for models / ppgn / train_dcg / dcg / configuration.py.

    Args:
        other: Caller-supplied value used by this routine.

    Returns:
        Computed value used by the caller.
    """
    def parser(string):
        """
        Implement the parser step for models / ppgn / train_dcg / dcg / configuration.py.

        Args:
            string: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        if ',' in string:
            return [other(s.strip()) for s in string.strip('[]').split(',')]
        else:
            return [other(string.strip('[]'))]
    return parser


def check_none(string):
    """
    Check none and report whether it is valid.

    Args:
        string: Caller-supplied value used by this routine.

    Returns:
        Computed value used by the caller.
    """
    string = string.lower()
    if string == 'none' or string == 'null':
        return None
    raise ValueError


def parse_bool(string):
    """
    Parse bool into structured values.

    Args:
        string: Caller-supplied value used by this routine.

    Returns:
        Computed value used by the caller.
    """
    s = string.strip().lower()
    if s in ('true', '1', 'yes'):
        return True
    if s in ('false', '0', 'no'):
        return False
    raise ValueError(f'Unable to parse "{string}" as bool')


def is_all(string: str) -> str:
    '''
    Normalizes the string to be 'all' or raises a value error

    Args:
        string: Caller-supplied value used by this routine.

    Returns:
        Computed value used by the caller.
    '''
    if string.strip() == 'all':
        return 'all'
    raise ValueError


def dict_of(other):
    '''
    Handles dicts of the form [a:10, b:20]
    other is the type of the values
    returns dict {'a': 10, 'b': 20}

    Args:
        other: Caller-supplied value used by this routine.

    Returns:
        Computed value used by the caller.
    '''
    def parser(string):
        """
        Implement the parser step for models / ppgn / train_dcg / dcg / configuration.py.

        Args:
            string: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        string = string.strip('[]')
        result = {}
        for token in string.split(','):
            key, value = token.split(':')
            key = key.strip()
            value = other(value.strip())
            result[key] = value
        return result

    return parser


TYPE_PARSERS = {
    'hyperparameters': {
        'epochs': int,
        'block_features': parse_args(list_or(int)),
        'depth_of_mlp': int,
        'learning_rate': float,
        'loss': str,
        'factor': float,
        'patience': int,
        'threshold': float,
        'disable_first_skip': parse_bool,
    },
    'training_info': {
        'target_features': parse_args(check_none, is_all, list_or(str)),
        'input_features': parse_args(check_none, is_all, list_or(str)),
        'non_negative': parse_args(check_none, is_all, list_or(str)),
        'edge_only': parse_args(check_none, is_all, list_or(str)),
        'total_penalties': parse_args(check_none, dict_of(float)),
        'batch_size': int,
        'subset': parse_args(check_none, int),
        'training_fraction': float,
        'early_stop': parse_args(check_none, int),
        'normalize': parse_args(check_none, is_all, list_or(str)),
        'gradient_clipping': parse_args(check_none, float),
        'instance_normalization': parse_bool,
        'candidate_mask': parse_bool,
        'pos_weight': parse_bool,
    },
    'execution': {
        'gpu': int,
    }
}


class Configuration():
    """
    build model/training configuration


    Role:
        Configuration groups state and methods for this repository component.
    """
    def __init__(self, yaml_file=None):
        """
        Initialize the Configuration instance and store constructor configuration.

        Args:
            yaml_file: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        self.config = copy.deepcopy(DEFAULT_OPTIONS)
        self.update_with_yaml(yaml_file)

    @staticmethod
    def print_defaults():
        '''
        Returns a string representation of a valid yaml file with defaults
        and help

        Returns:
            Computed value used by the caller.
        '''
        result = '---'
        for base_key, values in DEFAULT_OPTIONS.items():
            result += f'\n{base_key}:\n'
            for key, value in values.items():
                # indent help string properly
                help_str = '\n  '.join(HELP_OPTIONS[base_key][key].split('\n'))
                result += f'  {help_str}\n'
                if isinstance(value, str):
                    value = f'"{value}"'  # surround with quotes
                if value is None:
                    value = 'null'
                result += f'  {key}: {value}\n'
        return result

    @property
    def gpu(self):
        """
        Implement the gpu step for models / ppgn / train_dcg / dcg / configuration.py.

        Returns:
            Computed value used by the caller.
        """
        return self.config['execution']['gpu']

    @property
    def epochs(self):
        """
        Implement the epochs step for models / ppgn / train_dcg / dcg / configuration.py.

        Returns:
            Computed value used by the caller.
        """
        return self.config['hyperparameters']['epochs']

    @property
    def target_features(self):
        """
        Implement the target features step for models / ppgn / train_dcg / dcg / configuration.py.

        Returns:
            Computed value used by the caller.
        """
        return self.config['training_info']['target_features']

    def update_with_yaml(self, yaml_file):
        """
        Merge YAML configuration values into the current configuration.

        Args:
            yaml_file: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        if yaml_file is None:
            return

        config = yaml.safe_load(yaml_file)
        for base_key, values in self.config.items():
            for key, value in values.items():
                if base_key in config and key in config[base_key]:
                    self.config[base_key][key] = config[base_key][key]
                elif key in config:
                    self.config[base_key][key] = config[key]

    def update_with_args(self, arguments: str):
        """
        Merge command-line configuration overrides into the current configuration.

        Args:
            arguments: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        if arguments is None:
            return
        for arg in arguments.split(';'):
            try:
                k, v = arg.split('=')
            except ValueError:
                raise ValueError(f'Unable to parse token "{arg}"')
            for base_key, values in self.config.items():
                if k in values:
                    values[k] = TYPE_PARSERS[base_key][k](v)
                    break
            else:  # not found
                raise ValueError(f'Unable to match config key "{k}"')

    def build_loader(self):
        """
        Create the data loader requested by the configuration.

        Returns:
            Computed value used by the caller.
        """
        loader = CellLoader()
        loader.apply_config(self.config)
        return loader

    def build_model(self):
        """
        Create the model requested by the configuration.

        Returns:
            Computed value used by the caller.
        """
        model = BaseModel(
            len(self.config['training_info']['input_features']) + 1,
            len(self.config['training_info']['target_features']),
            self.config['hyperparameters']['block_features'],
            self.config['hyperparameters']['depth_of_mlp'],
            self.config['hyperparameters'].get('disable_first_skip', False))
        model.to(DEVICE)
        model.set_indices(self)
        model.set_non_negative(self)
        return model

    def build_trainer(self, model, loader):
        """
        Create the trainer object requested by the configuration.

        Args:
            model: Caller-supplied value used by this routine.
            loader: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        penalties = self.config['training_info']['total_penalties']
        targets = self.config['training_info']['target_features']
        if penalties is not None:
            for key in penalties:
                if key not in targets:
                    raise ValueError(f'Unable to match penalty for "{key}" '
                                     'to a target feature in '
                                     f'"[{", ".join(targets)}]"')
        trainer = Trainer(model, loader, self.config, DEVICE)
        trainer.set_penalties(penalties)
        return trainer

    def set_feature_names(self, inputs, outputs):
        """
        Resolve configured input and target feature names.

        Args:
            inputs: Caller-supplied value used by this routine.
            outputs: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        self.config['training_info']['input_features'] = inputs
        self.config['training_info']['target_features'] = outputs

    def set_node_features(self, input_index, output_index):
        """
        Resolve configured node-feature names.

        Args:
            input_index: Caller-supplied value used by this routine.
            output_index: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        self.config['training_info']['input_node'] = input_index
        self.config['training_info']['target_node'] = output_index
