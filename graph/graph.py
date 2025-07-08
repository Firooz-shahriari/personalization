import networkx as nx
import numpy as np

from utils.utils import *


class RandomGraph:
    def __init__(self):
        self.num_nodes = config.graph.num_nodes
        self.prob = config.graph.edge_prob
        self.laplacian_factor = config.graph.laplacian_factor
        self.zero_sum = None
        self.zero_row_sum = None
        self.zero_col_sum = None
        self.stochastic = None
        self.row_stochastic = None
        self.col_stochastic = None

        if config.graph.directed:
            self.generate_directed()
        else:
            self.generate_undirected()

    def generate_directed(self):
        graph = nx.gnp_random_graph(n=self.num_nodes, p=self.prob, directed=True)
        graph.remove_edges_from(nx.selfloop_edges(graph))

        if not nx.is_strongly_connected(graph):
            print("Error: Generated graph is not strongly connected.")
            sys.exit(1)

        adjacency = nx.adjacency_matrix(graph).toarray()

        row_sum = adjacency.sum(axis=1)
        col_sum = adjacency.sum(axis=0)

        row_zero_sum_matrix = np.diag(row_sum) - adjacency
        col_zero_sum_matrix = np.diag(col_sum) - adjacency

        self.zero_row_sum = row_zero_sum_matrix / (
            self.laplacian_factor * np.max(row_sum)
        )
        self.zero_col_sum = col_zero_sum_matrix / (
            self.laplacian_factor * np.max(col_sum)
        )

        self.row_stochastic = np.eye(len(graph)) - self.zero_row_sum
        self.col_stochastic = np.eye(len(graph)) - self.zero_col_sum

    def generate_undirected(self):
        graph = nx.gnp_random_graph(n=self.num_nodes, p=self.prob, directed=False)
        graph.remove_edges_from(nx.selfloop_edges(graph))

        if not nx.is_connected(graph):
            print("Error: Generated graph is not connected.")
            sys.exit(1)

        adjacency = nx.adjacency_matrix(graph).toarray()

        degree_sum = adjacency.sum(axis=1)

        zero_sum_matrix = np.diag(degree_sum) - adjacency
        zero_row_and_col_sum = zero_sum_matrix / (
            self.laplacian_factor * np.max(degree_sum)
        )

        stochastic_matrix = np.eye(len(graph)) - zero_row_and_col_sum

        self.zero_row_sum = zero_row_and_col_sum
        self.zero_col_sum = zero_row_and_col_sum

        self.row_stochastic = stochastic_matrix
        self.col_stochastic = stochastic_matrix
