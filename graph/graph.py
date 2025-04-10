import networkx as nx
import numpy as np

from utils.utils import *

class RandomGraph:
    def __init__(self):
        self.num_nodes = config.graph.num_nodes
        self.prob = config.graph.edge_prob
        self.laplacian_factor = config.graph.laplacian_factor

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

        zero_row_sum = row_zero_sum_matrix / (self.laplacian_factor * np.max(row_sum))
        zero_col_sum = col_zero_sum_matrix / (self.laplacian_factor * np.max(col_sum))

        row_stochastic = np.eye(len(graph)) - zero_row_sum
        col_stochastic = np.eye(len(graph)) - zero_col_sum

        return zero_row_sum, zero_col_sum, row_stochastic, col_stochastic

    def generate_undirected(self):
        graph = nx.gnp_random_graph(n=self.num_nodes, p=self.prob, directed=False)
        graph.remove_edges_from(nx.selfloop_edges(graph))

        if not nx.is_connected(graph):
            print("Error: Generated graph is not connected.")
            sys.exit(1)  

        adjacency = nx.adjacency_matrix(graph).toarray()

        degree_sum = adjacency.sum(axis=1)

        zero_sum_matrix = np.diag(degree_sum) - adjacency
        zero_row_and_col_sum = zero_sum_matrix / (self.laplacian_factor * np.max(degree_sum))

        stochastic_matrix = np.eye(len(graph)) - zero_row_and_col_sum

        return zero_row_and_col_sum, stochastic_matrix


