"""Utilities for models / ppgn / gl_tail_fixed_pkg / gnn_benchmark_ppgn / main.py in the GNN Benchmark codebase."""

import click
import torch
import numpy as np
import os
import hdf5storage

from gnn_benchmark_ppgn.recorder import Recorder, SavedModel
from gnn_benchmark_ppgn.configuration import Configuration, DEVICE
from gnn_benchmark_ppgn.notebook_renderer import Renderer
from gnn_benchmark_ppgn.simulator import Simulator
from gnn_benchmark_ppgn.file_reader import NodeMap


# override the alphabetical order of help text
# https://github.com/pallets/click/issues/513
class NaturalOrderGroup(click.Group):
    """
    Provide the natural order group component used by models / ppgn / gl_tail_fixed_pkg / gnn_benchmark_ppgn / main.py.


    Role:
        NaturalOrderGroup groups state and methods for this repository component.
    """
    def list_commands(self, ctx):
        """
        Implement the list commands step for models / ppgn / gl_tail_fixed_pkg / gnn_benchmark_ppgn / main.py.

        Args:
            ctx: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        return self.commands.keys()


@click.group(cls=NaturalOrderGroup, invoke_without_command=True)
@click.option('--print-config', is_flag=True,
              help='Print default config yaml to standard out and exit')
@click.pass_context
def cli(ctx, print_config):
    '''
    GNN Benchmark

    Simulating cellular interactions in a network of neighbors

    Args:
        ctx: Caller-supplied value used by this routine.
        print_config: Caller-supplied value used by this routine.

    Returns:
        None; the function updates object state, files, logs, or external process state.
    '''
    if ctx.invoked_subcommand is None and not print_config:
        click.echo(ctx.get_help())
        ctx.exit()

    if print_config:
        click.echo(Configuration.print_defaults())
        ctx.exit()

    np.random.seed(int(os.environ.get('GNN_BENCHMARK_SEED','100')))
    torch.manual_seed(int(os.environ.get('GNN_BENCHMARK_SEED','100')))

    if torch.cuda.is_available():
        click.echo('GPU found')
        torch.cuda.manual_seed(int(os.environ.get('GNN_BENCHMARK_SEED','100')))
        torch.cuda.manual_seed_all(int(os.environ.get('GNN_BENCHMARK_SEED','100')))


@cli.command(short_help='Train a new PPGN')
@click.option('--training-data', required=True, type=str,
              help='Ground-truth data to train and evaluate against')
@click.option('--out-dir', required=True,
              type=click.Path(file_okay=False, writable=True),
              help='Directory to store resulting model.')
@click.option('--inds-dir',
              type=click.Path(file_okay=False),
              help='Directory with indices for splitting graphs into '
              'train, test and val sets.', default=None)
@click.option('-c', '--config-file', type=click.File('r'),
              help='yaml file with configuration for model creation')
@click.option('--transfer-from', default=None,
              type=click.Path(file_okay=False, exists=True),
              help='Trained gnn_benchmark_ppgn model to initialize inner layers')
@click.option('--args', type=str, default=None,
              help='Other configuration settings to override config.  Use '
              'the format \'key=value;key2=value2\'.  Note the single quotes!')
@click.option('--evaluation/--no-evaluation', default=True,
              help='Switch to control if evaulation notebook is created')
def train(config_file, args, out_dir, training_data,
          evaluation, inds_dir, transfer_from):
    '''
    Given a set of ground-truth networks with associated physical progperties,
    train a provably powerful graph network to relate adjacencies to output
    properties.  By default, the performance is evaluated and reported in the
    output directory along with the model parameters.

    Args:
        config_file: Caller-supplied value used by this routine.
        args: Caller-supplied value used by this routine.
        out_dir: Caller-supplied value used by this routine.
        training_data: Caller-supplied value used by this routine.
        evaluation: Caller-supplied value used by this routine.
        inds_dir: Caller-supplied value used by this routine.
        transfer_from: Caller-supplied value used by this routine.

    Returns:
        None; the function updates object state, files, logs, or external process state.
    '''
    click.echo('Training')
    recorder = Recorder(out_dir)

    config = Configuration()

    config.update_with_yaml(config_file)
    config.update_with_args(args)

    if torch.cuda.is_available():
        torch.cuda.set_device(int(config.gpu))

    # load data
    loader = config.build_loader()
    if inds_dir is not None:
        inds = Recorder(inds_dir).load_indices()
        loader.indices = inds
        click.echo(f'Using indices in {inds_dir} to split graphs')
    num_graphs = loader.load_cells(training_data)
    click.echo(f'Loaded {num_graphs} graphs')
    if inds_dir is None:
        recorder.write_indices(loader.indices)

    # save info for loading model
    click.echo(f'Input features: [{", ".join(loader.input_features)}]')
    click.echo(f'Target features: [{", ".join(loader.target_features)}]')
    click.echo()
    click.echo(f'Normalize: {loader.normalize}')
    click.echo('Normalization factors:')
    click.echo(loader.norm_factors)
    config.set_feature_names(loader.input_features, loader.target_features)
    config.set_node_features(*loader.node_inds)
    recorder.write_config(config)

    model = config.build_model()

    if transfer_from is not None:
        old_model = SavedModel(transfer_from)
        model.initialize_from(old_model.model)
        del old_model
        click.echo('Transferring from trained model')

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    click.echo(f'The model has {num_params} trainable parameters')

    trainer = config.build_trainer(model, loader)
    trainer.recorder = recorder  # PATCH: enable periodic model saves
    metrics = trainer.train(config.epochs)

    if trainer.best_epoch == -1:  # not trained
        click.echo('No epochs completed, exiting')
        return

    recorder.write_metrics(metrics)

    click.echo('Evaluating over test set')
    trainer.load_best_model()
    test_dists, test_loss = trainer.test()
    recorder.write_model(trainer, loader.norm_factors)

    if evaluation:
        click.echo('Performing detailed evaluation')
        out_dir = recorder.out_dir
        notebook = os.path.join(out_dir, 'evaluation.ipynb')
        html = os.path.join(out_dir, 'evaluation.html')
        with open(notebook, 'w') as notebook, open(html, 'w') as html:
            renderer = Renderer()
            renderer.set_preamble(
                [out_dir], training_data, loader.indices)
            renderer.render(notebook, html)


@cli.command(short_help='Check PPGN performance')
@click.option('--test-data', required=True, type=str,
              help='File with only testing data.')
@click.option('--notebook', type=click.File('w'), default=None,
              help='Where to save jupyter notebook')
@click.option('--html', type=click.File('w'), default=None,
              help='Where to save exported html notebook')
@click.argument('models', required=True, nargs=-1,
                type=click.Path(file_okay=False, exists=True))
@click.pass_context
def evaluate(ctx, models, test_data, notebook, html):
    '''
    With one or more trained MODELS directories and testing data,
    report the model performance in a jupyter notebook or html.

    Args:
        ctx: Caller-supplied value used by this routine.
        models: Caller-supplied value used by this routine.
        test_data: Caller-supplied value used by this routine.
        notebook: Caller-supplied value used by this routine.
        html: Caller-supplied value used by this routine.

    Returns:
        None; the function updates object state, files, logs, or external process state.
    '''
    if notebook is None and html is None:
        click.echo("Must specify at least one output, "
                   "'--html' or '--notebook'",
                   err=True)
        ctx.exit(1)
    models = list(models)
    click.echo(f'Evaluating {len(models)} models')
    renderer = Renderer()
    renderer.set_preamble(models, test_data, indices=None)
    renderer.render(notebook, html)


@cli.command(short_help='Predict a set of graphs')
@click.option('--model', required=True,
              type=click.Path(file_okay=False, exists=True),
              help='Trained gnn_benchmark_ppgn model to use')
@click.option('--input', required=True, type=str,
              help='Input file, may contain targets')
@click.option('--output', required=True, type=str,
              help='Output file, matches input with predicted values added')
@click.pass_context
def predict(ctx, model, input: str, output):
    """
    Run model inference and return or save predictions.

    Args:
        ctx: Caller-supplied value used by this routine.
        model: Caller-supplied value used by this routine.
        input: Caller-supplied value used by this routine.
        output: Caller-supplied value used by this routine.

    Returns:
        None; the function updates object state, files, logs, or external process state.
    """
    saved = SavedModel(model)
    loader = saved.config.build_loader()
    inputs = loader.input_features

    # override values to limit loading
    loader.target_features = None
    if output.endswith('.txt'):
        loader.batch_size = 1

    loaded = loader.load_cells(input, apply_norm=False, only_test=True)
    if inputs != loader.input_features:
        click.echo(f'Expected input features of {inputs}')
        click.echo(f'Found: {loader.input_features}')
        raise ValueError('Unable to match input features')

    click.echo(f'Input features: [{", ".join(loader.input_features)}]')
    click.echo(f'Target features: [{", ".join(saved.config.target_features)}]')

    click.echo(f'Running prediction on {loaded} graphs')

    # txt, mat, or np input to mat output
    if output.endswith('.mat'):
        predictions = np.concatenate(
            [saved.model.predict(graph.to(DEVICE)).cpu().numpy()
             for graph, _ in loader.test])

        data = {
            'data': predictions,
            'Format': saved.config.target_features
        }
        hdf5storage.savemat(output, data)

    # txt, mat, or np input to np output
    elif output.endswith('.npz'):
        predictions = np.concatenate(
            [saved.model.predict(graph.to(DEVICE)).cpu().numpy()
             for graph, _ in loader.test])

        np.savez(
            output,
            data=predictions,
            Format=saved.config.target_features
        )

    elif input.endswith('.txt'):
        input = open(input, 'r')
        output = open(output, 'w')

        for line in input:
            if line.startswith('Format'):
                line = (line.strip() + ' ' +
                        ' '.join(saved.config.target_features) + '\n')
            output.write(line)
            if line.startswith('Simulation id:'):
                break

        # now on first graph
        node_map = NodeMap()
        for graph, _ in loader.test:
            prediction = saved.model.predict(graph.to(DEVICE)).cpu().numpy()

            for line in input:
                if line.startswith('Simulation id:'):
                    output.write(line)
                    node_map = NodeMap()
                    break

                if line == '\n' or line.startswith('#cells'):
                    output.write(line)
                    continue

                i, j, *rest = line.split()
                i = node_map[i]
                j = node_map[j]
                line = line.strip() + ' '
                line += ' '.join(str(d) for d in prediction[0, :, i, j]) + '\n'
                output.write(line)

    elif input.endswith('.mat') or input.endswith('.npz'):
        raise ValueError('Unable to produce text output from binary input')

    else:
        raise ValueError('Unsupported input or output type, '
                         'must be txt, mat, or npz')


@cli.command(short_help='Simulate cell network')
@click.option('--model', required=True,
              type=click.Path(file_okay=False, exists=True),
              help='Path to training output')
@click.option('--polygonality', required=True, type=click.File('r'),
              help='Initial polygonality of each node, one value per line')
@click.option('--topology', required=True, type=click.File('r'),
              help='Adjacency list with space-separated columns of edges '
              '"(node1, node2), (node3, node4)"')
@click.option('--output', required=True, type=click.File('w'),
              help='CSV file with order parameters per step')
@click.option('--kbT', default=60, type=click.IntRange(0, 120),
              help='Boltzmann energy, '
              'unitless between 0 and 120. Default = 60')
@click.option('--steps', default=100,
              help='Number of transitions to simulate. Default = 100')
@click.option('--seed', default=100,
              help='Random number seed. Default = 100')
def simulate(model, polygonality, topology, output, kbt, steps, seed):
    '''
    Simulate the dynamics of a cell network using a trained model.

    Args:
        model: Caller-supplied value used by this routine.
        polygonality: Caller-supplied value used by this routine.
        topology: Caller-supplied value used by this routine.
        output: Caller-supplied value used by this routine.
        kbt: Caller-supplied value used by this routine.
        steps: Caller-supplied value used by this routine.
        seed: Caller-supplied value used by this routine.

    Returns:
        None; the function updates object state, files, logs, or external process state.
    '''
    click.echo('Simulating')
    np.random.seed(seed)
    torch.manual_seed(seed)

    simulator = Simulator(topology, polygonality, kbt)
    saved = SavedModel(model)
    simulator.model = saved.model

    results = simulator.simulate(steps)
    results.to_csv(output)


if __name__ == '__main__':
    cli()
