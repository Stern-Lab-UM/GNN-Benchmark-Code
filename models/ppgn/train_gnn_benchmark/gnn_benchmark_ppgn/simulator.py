"""Utilities for models / ppgn / train_gnn_benchmark / gnn_benchmark_ppgn / simulator.py in the GNN Benchmark codebase."""

import numpy as np
import pandas as pd
import torch


class Simulator:
    """
    simulate graph dynamics


    Role:
        Simulator groups state and methods for this repository component.
    """

    def __init__(self, adjacency, polygonality, kbT):
        '''
        Initialize a gnn_benchmark_ppgn simulator
        Adjacency and polygonality should be readable files
        Adjacency is expected to have 4 space separated columns of
        (node1, node2), (node3, node4)
        where the first pair is a connected edge and the second are
        connected along the first pair
        Each node in the edge list should be 1-indexed!
        Polygonality is the polygonality value for each node,
            one value per line
        kbT is unitless boltzmann temperature

        Args:
            adjacency: Caller-supplied value used by this routine.
            polygonality: Caller-supplied value used by this routine.
            kbT: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        '''
        adj = np.loadtxt(adjacency, dtype=int) - 1  # 1 to 0 based indexing

        self.num_edges = adj.shape[0]

        self.edge_list = adj[:, 0:2]
        self.other_edges = adj[:, 2:]

        self.polygonality = np.loadtxt(polygonality, dtype=int)

        self.num_cells = len(self.polygonality)

        self.edges = np.zeros((self.num_cells, self.num_cells), dtype=np.int)

        self.edges[(self.edge_list[:, 0],
                    self.edge_list[:, 1])] = np.arange(len(self.edge_list))
        self.edges[(self.edge_list[:, 1],
                    self.edge_list[:, 0])] = np.arange(len(self.edge_list))

        self.energies = None
        self.lengths = None

        self.pick_edge = self.kramers
        self.kbT = kbT
        self.model = None

    def simulate(self, steps):
        """
        Run the simulator for the configured model and graph state.

        Args:
            steps: Caller-supplied value used by this routine.

        Returns:
            Computed value used by the caller.
        """
        step_data = []

        try:
            for _ in range(steps):

                step_data.append({
                    'order_parameter': self.order_parameter(),
                })

                self.predict_energy_barriers()

                eID = self.pick_edge()
                self.perform_T1(eID)

        except KeyboardInterrupt:
            pass

        step_data = pd.DataFrame(step_data)
        step_data.index.name = 'step'

        return step_data

    def order_parameter(self):
        """
        Compute the simulator order parameter for the current state.

        Returns:
            Computed value used by the caller.
        """
        # fraction of cells with polygonality of 6
        return (self.polygonality == 6).mean()

    def predict_energy_barriers(self):
        """
        Predict energy barriers for candidate rearrangements.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        if self.model is None:
            raise ValueError('Unable to predict energy, set model first')

        adjacency_matrix = np.zeros(
            shape=(1, 1, self.num_cells, self.num_cells),
            dtype=np.float32)

        adjacency_matrix[0, 0, self.edge_list[:, 0], self.edge_list[:, 1]] = 1
        adjacency_matrix[0, 0, self.edge_list[:, 1], self.edge_list[:, 0]] = 1

        result = self.model.predict(torch.from_numpy(adjacency_matrix))

        self.energies = result[0, 0,
                               self.edge_list[:, 0],
                               self.edge_list[:, 1]].numpy()
        self.lengths = result[0, 1,
                              self.edge_list[:, 0],
                              self.edge_list[:, 1]].numpy()

    def set_edge_method(self, method='kramers'):
        """
        Configure how simulator edges are selected.

        Args:
            method: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        if method == 'kramers':
            self.pick_edge = self.kramers
        if method == 'random':
            self.pick_edge = self.random_edge

    def kramers(self):
        """
        Convert energy barriers into Kramers-style transition rates.

        Returns:
            Computed value used by the caller.
        """
        # converts predicted energy barriers to probabilities
        # for t1 and return the chosen edge

        # kramers relation
        e_P = np.exp(-self.energies / self.kbT)

        # zero-out energies with low polygonality (< 4)
        e_P[(self.edge_list[:, :, None] ==
             np.where(self.polygonality < 4)[0][None, None, :]
             ).sum(axis=(1, 2)) != 0] = 0

        # normalize
        e_P /= e_P.sum()

        # choose an edge from the distribution
        return np.random.choice(self.num_edges, p=e_P)

    def random_edge(self):
        """
        Sample a candidate edge according to the configured transition rates.

        Returns:
            Computed value used by the caller.
        """
        p = np.ones(self.edge_list.shape[0])
        p[(self.edge_list[:, :, None] ==
           np.where(self.polygonality < 4)[0][None, None, :]
           ).sum(axis=(1, 2)) != 0] = 0
        p /= p.sum()
        return np.random.choice(self.num_edges, p=p)

    def perform_T1(self, eID):
        """
        Apply one T1 transition to the simulator state.

        Args:
            eID: Caller-supplied value used by this routine.

        Returns:
            None; the function updates object state, files, logs, or external process state.
        """
        # performs t1 transition (change topology of graphs):
        #      \ c3/             \     /
        #      e1 e2             e1   e2
        #       \ /               \ c3/
        #     c1 | c2   -->     c1 ---  c2
        #       / \               / c4\
        #      e3 e4             e3   e4
        #      / c4\             /     \

        # check polygonality is high enough to avoid triangles
        if (self.polygonality[self.edge_list[eID]] <= 3).any():
            return

        # identify cells
        c1, c2 = self.edge_list[eID, :]
        c3, c4 = self.other_edges[eID, :]

        # identify edges
        e1, e2 = self.edges[[c1, c2], c3]
        e3, e4 = self.edges[[c1, c2], c4]

        # update topology
        self.edge_list[eID, :] = c3, c4
        self.other_edges[eID, :] = c1, c2

        self.edges[c1, c2] = 0
        self.edges[c2, c1] = 0
        self.edges[c3, c4] = eID
        self.edges[c4, c3] = eID

        # update other edges
        # e1: c2 -> c4
        self.other_edges[e1, self.other_edges[e1, :] == c2] = c4
        self.other_edges[e2, self.other_edges[e2, :] == c1] = c4
        self.other_edges[e3, self.other_edges[e3, :] == c2] = c3
        self.other_edges[e4, self.other_edges[e4, :] == c1] = c3

        # before
        self.polygonality[[c1, c2]] -= 1
        self.polygonality[[c3, c4]] += 1
