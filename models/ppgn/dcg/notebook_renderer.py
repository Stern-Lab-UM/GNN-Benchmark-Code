import nbformat
import nbconvert
import pickle
import textwrap
from typing import List
import pkgutil
import os


class Renderer():
    def __init__(self):
        self.notebook = nbformat.reads(
            pkgutil.get_data(__name__, 'templates/evaluation_template.ipynb'),
            as_version=4)

    def set_preamble(self,
                     model_directories: List,
                     test_data: str,
                     indices=None):  # dict of np arrays from loader
        model_directories = [os.path.normpath(m) for m in model_directories]
        dcg_data = {
            'model_directories': model_directories,
            'test_data': test_data,
            'indices': indices,
        }
        preamble = textwrap.dedent(
            '''
            ### AUTO GENERATED PREAMBLE DO NOT EDIT
            import pickle
            dcg_data = pickle.loads({dcg_data})
            ###END PREAMBLE
            ''').format(dcg_data=pickle.dumps(dcg_data))
        self.notebook['cells'][0]['source'] = preamble

    def render(self, notebook, html):
        nb = nbconvert.preprocessors.ExecutePreprocessor(kernel_name='python3')
        nb.preprocess(self.notebook)
        if notebook is not None:
            nbformat.write(self.notebook, notebook)

        if html is not None:
            ht = nbconvert.exporters.HTMLExporter()
            (body, resources) = ht.from_notebook_node(self.notebook)
            html.write(body)
