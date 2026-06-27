"""Utilities for models / mpnn / tensorboard2csv.py in the DCG benchmark codebase."""

import os
import numpy as np
import pandas as pd
import argparse

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

parser = argparse.ArgumentParser(description='Inference')
parser.add_argument('--path', type=str, default='')
args = parser.parse_args()

def _event_entries(dpath):
    """
    Implement the event entries step for models / mpnn / tensorboard2csv.py.

    Args:
        dpath: Caller-supplied value used by this routine.

    Returns:
        Computed value used by the caller.
    """
    return [d for d in os.listdir(dpath)
            if os.path.isdir(os.path.join(dpath, d)) or d.startswith('events.out.tfevents')]


def tabulate_events(dpath):
    """
    Implement the tabulate events step for models / mpnn / tensorboard2csv.py.

    Args:
        dpath: Caller-supplied value used by this routine.

    Returns:
        Computed value used by the caller.
    """
    summary_iterators = [EventAccumulator(os.path.join(dpath, dname)).Reload() for dname in _event_entries(dpath)]

    tags = summary_iterators[0].Tags()['scalars']

    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags

    out = defaultdict(list)
    steps_by_tag = {}

    for tag in tags:
        steps_by_tag[tag] = [e.step for e in summary_iterators[0].Scalars(tag)]

        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            assert len(set(e.step for e in events)) == 1

            out[tag].append([e.value for e in events])

    return out, steps_by_tag


def to_csv(dpath):
    """
    Implement the to csv step for models / mpnn / tensorboard2csv.py.

    Args:
        dpath: Caller-supplied value used by this routine.

    Returns:
        None; the function updates object state, files, logs, or external process state.
    """
    dirs = _event_entries(dpath)

    d, steps_by_tag = tabulate_events(dpath)

    for tag, values in d.items():
        df = pd.DataFrame(values, index=steps_by_tag[tag], columns=dirs)
        df.to_csv(get_file_path(dpath, tag))


def get_file_path(dpath, tag):
    """
    Return file path for the current object or workflow.

    Args:
        dpath: Caller-supplied value used by this routine.
        tag: Caller-supplied value used by this routine.

    Returns:
        Computed value used by the caller.
    """
    file_name = tag.replace("/", "_") + '.csv'
    folder_path = os.path.join(dpath, 'csv')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return os.path.join(folder_path, file_name)


if __name__ == '__main__':
    path = args.path
    to_csv(path)