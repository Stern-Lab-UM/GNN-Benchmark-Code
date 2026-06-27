"""Utilities for models / ppgn / predict_gnn_benchmark / gnn_benchmark_ppgn / notebook_renderer.py in the GNN Benchmark codebase."""

import nbformat
import nbconvert
import pickle
import textwrap
from typing import List
import pkgutil
import os


class Renderer():
    """
    Provide the renderer component used by models / ppgn / predict_gnn_benchmark / gnn_benchmark_ppgn / notebook_renderer.py.


    Role:
        Renderer groups state and methods for this repository component.
    """
    def __init__(self):
        """
        Initialize the Renderer instance and store constructor configuration.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        self.notebook = nbformat.reads(
            pkgutil.get_data(__name__, 'templates/evaluation_template.ipynb'),
            as_version=4)

    def set_preamble(self,
                     model_directories: List,
                     test_data: str,
                     indices=None):  # dict of np arrays from loader
        """
        Set notebook or report preamble content before rendering.

        Args:
            model_directories: Caller-supplied value used by this routine.
            test_data: Caller-supplied value used by this routine.
            indices: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        model_directories = [os.path.normpath(m) for m in model_directories]
        GNNBenchmark_data = {
            'model_directories': model_directories,
            'test_data': test_data,
            'indices': indices,
        }
        preamble = textwrap.dedent(
            '''
            ### AUTO GENERATED PREAMBLE DO NOT EDIT
            import pickle
            GNNBenchmark_data = pickle.loads({GNNBenchmark_data})
            ###END PREAMBLE
            ''').format(GNNBenchmark_data=pickle.dumps(GNNBenchmark_data))
        self.notebook['cells'][0]['source'] = preamble

    def render(self, notebook, html):
        """
        Render the configured notebook/report artifact.

        Args:
            notebook: Caller-supplied value used by this routine.
            html: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        nb = nbconvert.preprocessors.ExecutePreprocessor(kernel_name='python3')
        nb.preprocess(self.notebook)
        if notebook is not None:
            nbformat.write(self.notebook, notebook)

        if html is not None:
            ht = nbconvert.exporters.HTMLExporter()
            (body, resources) = ht.from_notebook_node(self.notebook)
            html.write(body)
