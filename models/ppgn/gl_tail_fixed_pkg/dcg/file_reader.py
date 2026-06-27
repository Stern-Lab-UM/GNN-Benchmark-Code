"""Utilities for models / ppgn / gl_tail_fixed_pkg / dcg / file_reader.py in the DCG benchmark codebase."""

from typing import List, Union
import re
import numpy as np
import hdf5storage


class FileReader():
    """
    Coordinate file reader responsibilities for the DCG PPGN workflow.


    Role:
        FileReader groups state and methods for this repository component.
    """
    def __init__(self, subset, targets, inputs):
        """
        Initialize the FileReader instance and store constructor configuration.

        Args:
            subset: Caller-supplied value used by this routine.
            targets: Caller-supplied value used by this routine.
            inputs: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        self.test_inds = None
        self.subset = subset
        self.num_graphs = None

        # when subset is a list, those are the targets and want to read
        # to the max index
        if isinstance(subset, (np.ndarray, list)):
            self.test_inds = subset
            self.subset = max(subset) + 1

        self.targets = targets
        self.inputs = inputs
        self.num_cells = None
        self.num_edges = None
        self.node_inds = None

    def read_cells(self, file):
        """
        Read cell graph tensors and target arrays from the configured file.

        Args:
            file: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        if isinstance(file, str):
            if file.endswith('.txt'):
                with open(file, 'r') as infile:
                    return self._read_cells_txt(infile)
            elif file.endswith('.mat'):
                return self._read_cells_mat(file)
            elif file.endswith('.npz'):
                return self._read_cells_npz(file)
            else:
                raise ValueError(f'Unsupported input file type: "{file}"')

        # should be an open file
        return self._read_cells_txt(file)

    def _read_cells_txt(self, file):
        """
        Read cell graph tensors and target arrays from the configured file.

        Args:
            file: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        current_graph = None  # name of graph (str)
        current_index = -1  # index to insert into
        graph_num = -1  # currently on the ith graph of the file
        edges = 0
        nodes = 0

        graphs = None
        targets = None
        parser = CellParser(self.inputs, self.targets)

        for line in file:
            if line.startswith('Total graphs:'):
                self.parse_total_graphs(line)

            elif line.startswith('#cells'):
                self.parse_cells(line)

            elif line.startswith('Format'):
                parser.update_format(line)

            elif line.startswith('Simulation id'):
                if current_graph is not None:  # not first graph
                    if edges != self.num_edges * 2:  # each pair specified
                        raise ValueError(f'{current_graph} has {edges} edges,'
                                         f' expected {self.num_edges * 2}')
                    if nodes != 0 and nodes != self.num_cells:
                        raise ValueError(f'{current_graph} has {nodes} nodes,'
                                         f' expected {self.num_cells}')

                if not parser.initialized():
                    raise ValueError('Missing "Format" header')

                if graphs is None:  # first graph
                    if self.num_graphs is None:
                        raise ValueError(
                            'Missing total graphs information, '
                            'check input format')

                    if self.num_cells is None:
                        # when there are multiple sizes in file
                        graphs = np.empty(self.num_graphs, dtype=object)
                        targets = np.empty(self.num_graphs, dtype=object)
                    else:
                        # one size in file
                        graphs = np.zeros((self.num_graphs,
                                           parser.input_len,
                                           self.num_cells, self.num_cells))
                        targets = np.zeros((self.num_graphs, parser.output_len,
                                            self.num_cells, self.num_cells))

                if self.subset is not None and graph_num >= self.subset-1:
                    break

                graph_num += 1

                # this graph is in the test set or no test set
                if self.test_inds is None or graph_num in self.test_inds:
                    current_index += 1
                    edges = 0
                    nodes = 0
                    current_graph = line.split(': ')[-1].strip()
                    # reset parser nodemap
                    parser.node_map = NodeMap()

            elif line != '\n':
                if (self.test_inds is not None \
                        and graph_num not in self.test_inds):
                    continue
                if graphs[current_index] is None:
                    # for object array on first insert
                    if self.num_cells is None:
                        raise ValueError(
                            'Unable to parse graph metadata. '
                            'Expected:\n'
                            '#cells: {}, [#vertices: {}, ]#edges: {}')
                    graphs[current_index] = np.zeros(
                        (parser.input_len, self.num_cells, self.num_cells))
                    targets[current_index] = np.zeros(
                        (parser.output_len, self.num_cells, self.num_cells))

                result = parser.parse(line)

                inval, outval = result['input'], result['output']
                graphs[current_index][inval['index']] = inval['values']
                targets[current_index][outval['index']] = outval['values']

                if result['type'] == 'edge':
                    edges += 1
                else:
                    nodes += 1

        # check number of graphs
        graph_num += 1  # convert to 1-based
        if graph_num != self.num_graphs:
            # this tests for when the test_inds are provided
            if self.test_inds is None or graph_num != max(self.test_inds) + 1:
                raise ValueError(f'Expected {self.num_graphs} graphs, '
                                 f'found {graph_num}')

        # check number of edges and nodes for last graph
        if edges != self.num_edges * 2:  # each pair specified
            raise ValueError(f'{current_graph} has {edges} edges, '
                             f'expected {self.num_edges * 2}')

        if nodes != 0 and nodes != self.num_cells:
            raise ValueError(f'{current_graph} has {nodes} nodes,'
                             f' expected {self.num_cells}')

        self.check_symmetry(targets)

        self.inputs = parser.input_names
        self.targets = parser.output_names
        self.node_inds = parser.node_indices

        return graphs, targets

    def _read_cells_mat(self, file):
        """
        Read cell graph tensors and target arrays from the configured file.

        Args:
            file: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        data = hdf5storage.loadmat(file)
        if 'Format' not in data or 'data' not in data:
            raise ValueError('Unable to parse matlab file')

        return self._parse_binary(data)

    def _read_cells_npz(self, file):
        """
        Read cell graph tensors and target arrays from the configured file.

        Args:
            file: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        with np.load(file) as data:
            if 'Format' not in data or 'data' not in data:
                raise ValueError('Unable to parse npz file')

            return self._parse_binary(data)

    def _parse_binary(self, data):
        """
        Implement the parse binary step for models / ppgn / gl_tail_fixed_pkg / dcg / file_reader.py.

        Args:
            data: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        if isinstance(data['Format'], list):
            names = data['Format']
        elif isinstance(data['Format'], np.ndarray):
            names = data['Format'].tolist()
        else:
            # matlab saves string lists as 2d arrays of strings
            names = [name[0][0] for name in data['Format'][0]]

        # since the nodes/edges can be in any order, need to find the nodes
        # sum absolute diagonal values of all features, leaving axis 1
        nodes = np.abs(np.diagonal(data['data'],
                                   axis1=-1, axis2=-2)).sum(axis=(0, -1))
        # nodes are where there is a non-zero diagonal value
        nodes = nodes != 0

        if len(nodes) != len(names):
            raise ValueError(
                f'Found {len(nodes)} features in data, '
                f'but Format has {len(names)}')

        if nodes[0] == True:
            raise ValueError(
                'Adjacency matrix had self loop, ensure diagonal is all 0')

        # use parser logic for finding features and inputs/outputs
        parser = CellParser(self.inputs, self.targets)
        # the 1: skips the adjacency
        parser.update_tokens([name
                              for name, node in zip(names[1:], nodes[1:])
                              if node], edges=False)
        parser.update_tokens([name
                              for name, node in zip(names[1:], nodes[1:])
                              if not node], edges=True)

        # get indices of inputs
        node_inds = np.where(nodes)[0]
        edge_inds = np.where(~nodes)[0][1:]  # skip adj

        graphs = [0]  # adjacency
        graphs += [edge_inds[i] for i in parser.input_edges.inds]
        graphs += [node_inds[i] for i in parser.input_nodes.inds]

        targets = [edge_inds[i] for i in parser.output_edges.inds]
        targets += [node_inds[i] for i in parser.output_nodes.inds]

        self.inputs = parser.input_names
        self.targets = parser.output_names
        self.node_inds = parser.node_indices

        data = data['data']
        if self.subset is not None and self.subset < data.shape[0]:
            data = data[:self.subset]

        self.num_graphs = data.shape[0]
        self.num_cells = data.shape[-1]

        return data[:, graphs, ...], data[:, targets, ...]

    def parse_total_graphs(self, line):
        """
        Parse the total graph count from a benchmark file header.

        Args:
            line: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        match = re.search(r'^Total graphs: (?P<num_graphs>\d+)', line)

        if not match:
            raise ValueError('Unable to parse "Total graphs"')

        num_graphs = int(match.group('num_graphs'))

        if self.test_inds is not None:
            self.test_inds = {i for i in self.test_inds
                              if i < num_graphs}
            self.num_graphs = len(self.test_inds)
            self.subset = max(self.test_inds) + 1

        elif self.subset is not None:
            self.num_graphs = min(num_graphs, self.subset)

        else:
            self.num_graphs = num_graphs

    def parse_cells(self, line):
        """
        Parse cell graph records from a benchmark file payload.

        Args:
            line: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        match = re.search(r'#cells: (?P<num_cells>\d+)', line)
        if not match:
            raise ValueError('Unable to parse graph metadata. Expected:\n'
                             '#cells: {}, [#vertices: {}, ]#edges: {}')

        self.num_cells = int(match.group('num_cells'))

        match = re.search(r'#edges: (?P<num_edges>\d+)', line)
        if not match:
            raise ValueError('Unable to parse graph metadata. Expected:\n'
                             '#cells: {}, [#vertices: {}, ]#edges: {}')

        self.num_edges = int(match.group('num_edges'))

    def check_symmetry(self, targets):
        """
        Validate that paired adjacency-style tensors are symmetric where required.

        Args:
            targets: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        for graph_i in range(targets.shape[0]):
            for target_i in range(targets[graph_i].shape[0]):
                if np.any(targets[graph_i][target_i, :, :].T !=
                          targets[graph_i][target_i, :, :]):
                    raise ValueError(f'Graph {graph_i+1} is not symmetric')

    def __eq__(self, obj):
        '''
        test equality for testing equivalence of reading file types

        Args:
            obj: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        '''
        return (isinstance(obj, FileReader) and
                self.test_inds == obj.test_inds and
                self.subset == obj.subset and
                self.num_graphs == obj.num_graphs and
                self.targets == obj.targets and
                self.inputs == obj.inputs and
                self.num_cells == obj.num_cells and
                self.node_inds == obj.node_inds)


class CellParser():
    """
    Coordinate cell parser responsibilities for the DCG PPGN workflow.


    Role:
        CellParser groups state and methods for this repository component.
    """
    def __init__(self,
                 requested_inputs: Union[None, List[str]],
                 requested_targets: Union[None, List[str]]):
        """
        Initialize the CellParser instance and store constructor configuration.

        Args:
            requested_inputs: Caller-supplied value used by this routine.
            requested_targets: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        self.requested_inputs = requested_inputs
        self.requested_outputs = requested_targets
        self.input_edges = None
        self.output_edges = None
        self.input_nodes = None
        self.output_nodes = None
        self.node_map = NodeMap()

    def update_format(self, format_str: str) -> None:
        '''
        Given the format string, update state to handle parsing from string

        Args:
            format_str: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        '''
        if format_str.startswith('Format:'):  # edges
            tokens = remove_prefix(format_str, 'Format: ').split()

            if (len(tokens) < 2 or
                    tokens[0] != 'cell_id_1' or
                    tokens[1] != 'cell_id_2'):
                raise ValueError('Unexpected edge format, required:'
                                 ' "Format: cell_id_1 cell_id_2 '
                                 '[input_{name} ...] [[output_]{name}]"')

        elif format_str.startswith('Format nodes:'):
            tokens = remove_prefix(format_str, 'Format nodes: ').split()

            if (len(tokens) < 2 or
                    tokens[0] != 'cell_id_1' or
                    tokens[1] != 'cell_id_1'):
                raise ValueError('Unexpected node format, required:'
                                 ' "Format: cell_id_1 cell_id_1 '
                                 '[input_{name} ...] [[output_]{name}]"')

        else:
            raise ValueError(f'Unexpected format: {format_str}')

        self.update_tokens(tokens[2:], edges=format_str.startswith('Format:'))

    def update_tokens(self, tokens, edges):
        """
        Update parser state from the current tokenized line.

        Args:
            tokens: Caller-supplied value used by this routine.
            edges: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        tokens_no_prefix = [
            remove_prefix(remove_prefix(t, 'input_'), 'output_')
            for t in tokens]

        # check if tokens contain duplicates
        if len(tokens_no_prefix) != len(set(tokens_no_prefix)):
            token_set = set()
            for token in tokens_no_prefix:
                if token in token_set:
                    raise ValueError("Format line contained "
                                     f"duplicate target '{token}'")
                token_set.add(token)

        # check against existing tokens
        for named_token in (self.input_edges, self.output_edges,
                            self.input_nodes, self.output_nodes):
            if named_token is None:
                continue
            for token in named_token.names:
                if token in tokens_no_prefix:
                    raise ValueError("Format line contained "
                                     f"duplicate target '{token}'")

        if self.requested_inputs is None:
            inputs = [no_prefix
                      for no_prefix, t in zip(tokens_no_prefix, tokens)
                      if t.startswith('input_') and
                      (self.requested_outputs is None or
                       no_prefix not in self.requested_outputs)]
        else:
            inputs = self.requested_inputs

        if self.requested_outputs is None:
            outputs = [no_prefix
                       for no_prefix, t in zip(tokens_no_prefix,
                                               tokens)
                       if not t.startswith('input_') and
                       (self.requested_inputs is None or
                        no_prefix not in self.requested_inputs)]
        else:
            outputs = self.requested_outputs

        if edges:
            self.input_edges = NamedTokens(inputs,
                                           enumerate(tokens_no_prefix))

            self.output_edges = NamedTokens(outputs,
                                            enumerate(tokens_no_prefix))
        else:
            self.input_nodes = NamedTokens(inputs,
                                           enumerate(tokens_no_prefix))

            self.output_nodes = NamedTokens(outputs,
                                            enumerate(tokens_no_prefix))

    def parse(self, line):
        '''
        Given a string, returns a dict with indices and values for the input
        and target matrices.  Indices are (property, node1, node2)

        Args:
            line: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        '''
        i, j, *values = line.split()
        i = self.node_map[i]
        j = self.node_map[j]
        in_edges, out_edges = self.node_indices

        if i == j:
            return {
                'type': 'node',
                'input': {
                    'index': (
                        (np.arange(self.input_nodes.toks) +
                         in_edges),
                        np.full(self.input_nodes.toks, i),
                        np.full(self.input_nodes.toks, j),
                    ),
                    'values': [float(values[x])
                               for x in self.input_nodes.inds]
                },
                'output': {
                    'index': (
                        (np.arange(self.output_nodes.toks) +
                         out_edges),
                        np.full(self.output_nodes.toks, i),
                        np.full(self.output_nodes.toks, j),
                    ),
                    'values': [float(values[x])
                               for x in self.output_nodes.inds]
                }}

        else:
            return {
                'type': 'edge',
                'input': {
                    'index': (
                        np.arange(in_edges),
                        np.full(in_edges, i),
                        np.full(in_edges, j),
                    ),
                    'values': [1.] + [float(values[x])
                                      for x in self.input_edges.inds]
                },
                'output': {
                    'index': (
                        np.arange(out_edges),
                        np.full(out_edges, i),
                        np.full(out_edges, j),
                    ),
                    'values': [float(values[x])
                               for x in self.output_edges.inds]
                }}

    def initialized(self):
        """
        Report whether the parser has enough format information to parse rows.

        Returns:
            Computed value used by the caller.
        """
        # returns true if at least one format spec is provided
        return self.input_edges is not None or self.input_nodes is not None

    @property
    def input_names(self):
        """
        Return names of configured input features.

        Returns:
            Computed value used by the caller.
        """
        ed = self.input_edges.names if self.input_edges is not None else []
        no = self.input_nodes.names if self.input_nodes is not None else []
        return ed + no

    @property
    def output_names(self):
        """
        Return names of configured output features.

        Returns:
            Computed value used by the caller.
        """
        ed = self.output_edges.names if self.output_edges is not None else []
        no = self.output_nodes.names if self.output_nodes is not None else []
        return ed + no

    @property
    def input_len(self):
        """
        Return the number of configured input features.

        Returns:
            Computed value used by the caller.
        """
        return len(self.input_names) + 1  # for adjacency

    @property
    def output_len(self):
        """
        Return the number of configured output features.

        Returns:
            Computed value used by the caller.
        """
        return len(self.output_names)

    @property
    def node_indices(self):
        """
        Return node-index columns parsed from the current graph format.

        Returns:
            Computed value used by the caller.
        """
        # nodes are always the last indices of the matrices
        # report the first node index, can be last index in matrix
        in_edges = self.input_edges.toks if self.input_edges else 0
        out_edges = self.output_edges.toks if self.output_edges else 0
        return (in_edges + 1, out_edges)


class NamedTokens():
    '''
    A collection of indices and names for tokens

    Role:
        NamedTokens groups state and methods for this repository component.
    '''
    def __init__(self, requested, present):
        '''
        Match the requested to present tokens

        Args:
            requested: Caller-supplied value used by this routine.
            present: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        '''
        self.tokens = [(i, n) for i, n in present
                       if requested is None or n in requested]

    def __repr__(self):
        """
        Return a compact printable representation for debugging.

        Returns:
            Computed value used by the caller.
        """
        return self.tokens.__repr__()

    @property
    def inds(self):
        """
        Implement the inds step for models / ppgn / gl_tail_fixed_pkg / dcg / file_reader.py.

        Returns:
            Computed value used by the caller.
        """
        return [t[0] for t in self.tokens]

    @property
    def names(self):
        """
        Implement the names step for models / ppgn / gl_tail_fixed_pkg / dcg / file_reader.py.

        Returns:
            Computed value used by the caller.
        """
        return [t[1] for t in self.tokens]

    @property
    def toks(self):
        """
        Implement the toks step for models / ppgn / gl_tail_fixed_pkg / dcg / file_reader.py.

        Returns:
            Computed value used by the caller.
        """
        return len(self.tokens)


def remove_prefix(text, prefix):
    """
    Return tokens or names with the configured prefix removed.

    Args:
        text: Caller-supplied value used by this routine.
        prefix: Caller-supplied value used by this routine.

    Returns:
        Computed value used by the caller.
    """
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


class NodeMap(dict):
    """
    Provide the node map component used by models / ppgn / gl_tail_fixed_pkg / dcg / file_reader.py.


    Role:
        NodeMap groups state and methods for this repository component.
    """
    # map strings to range in order of appearance
    def __missing__(self, key):
        """
        Create and store a missing mapping entry on demand.

        Args:
            key: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        self[key] = len(self)
        return self[key]
