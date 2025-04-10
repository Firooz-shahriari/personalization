import os
import sys

import numpy as np

from utils.utils import *

# This class minimizes the function \sum_v \log(\cosh(A_v^T x^v - b_v))
# subject to the constraint \|x^v - \bTx\|_2 \leq \epsilon
# with respect to [{x^v}_v, x].


class synthetic:
    def __init__(self):

        self.num_nodes = config.graph.num_nodes
        self.local_dim = config.pr.local_dimension
        self.global_dim = config.pr.global_dim

        self.A = np.random.randn(self.num_nodes, self.local_dim)
        self.epsilon = config.pr.epsilon
        self.map = np.random.randn(self.local_dim, self.global_dim)
        # self.map = np.eye(self.local_dim)

        self.theta_global_opt = np.random.randn(self.global_dim)
        self.theta_locals_opt = sample_random_vectors_near_y(
            self.map, self.theta_global_opt, self.epsilon, self.num_nodes
        )

        self.b = np.sum(self.A * self.theta_locals_opt, axis=1)
        self.optimal_value = 0.0

        if config.pr.global_objective_exists:
            self.C = np.random.randn(self.num_nodes, self.global_dim)
            self.d = np.matmul(self.C, self.theta_global_opt)

    def F_val(self, theta_local, theta_global=None):
        if config.pr.global_objective_exists:
            return np.sum(
                np.log10(np.cosh(np.sum(self.A * theta_locals, axis=1) - self.b))
            ) + np.sum(np.log10(np.cosh(np.matmul(self.C, theta_global) - self.d)))
        return np.sum(np.log10(np.cosh(np.sum(self.A * theta_local, axis=1) - self.b)))

    def localgrad(self, theta_locals, idx):
        grad_local = (
            (1 / np.cosh(np.inner(self.A[idx], theta_locals[idx]) - self.b[idx]))
            * np.sinh(np.inner(self.A[idx], theta_locals[idx]) - self.b[idx])
            * self.A[idx]
        )

        grad = np.zeros((self.num_nodes, self.local_dim))
        grad[idx] = grad_local
        return grad_local, grad

    def globalgrad(self, theta_global):
        grad_global = (
            (1 / np.cosh(np.matmul(self.C, theta_global) - self.d))
            * np.sinh(np.matmul(self.C, theta_global) - self.d)
            * self.C
        )
        return grad_global

    def networkgrad(self, theta_locals):
        grad = np.zeros((self.num_nodes, self.local_dim))
        for i in range(self.num_nodes):
            grad_local, _ = self.localgrad(theta_locals, i)
            grad[i] = grad_local
        return grad

    def grad(self, theta_locals):
        return self.networkgrad(theta_locals)
