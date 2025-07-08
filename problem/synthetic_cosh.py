import os
import sys

import numpy as np

from utils.utils import *


class synthetic:
    def __init__(self):

        self.local_dim = config.pr.local_dim
        self.global_dim = config.pr.global_dim
        self.epsilon = config.pr.epsilon

        self.A = np.random.randn(config.graph.num_nodes, self.local_dim)
        self.map_mat = np.random.randn(self.local_dim, self.global_dim)
        # self.map_mat = np.eye(self.local_dim)

        self.theta_global_opt = np.random.randn(self.global_dim)
        self.theta_locals_opt = sample_random_vectors_near_y(
            self.map_mat, self.theta_global_opt, self.epsilon, config.graph.num_nodes
        )

        self.b = np.sum(self.A * self.theta_locals_opt, axis=1)
        self.optimal_value = 0.0

        if config.pr.global_objective_exists:
            self.C = np.random.randn(config.graph.num_nodes, self.global_dim)
            self.d = np.matmul(self.C, self.theta_global_opt)

    def F_val(self, theta_locals, theta_global=None):
        if config.pr.global_objective_exists:
            return np.sum(
                np.log10(np.cosh(np.sum(self.A * theta_locals, axis=1) - self.b))
            ) + np.sum(np.log10(np.cosh(np.sum(self.C * theta_global, axis=1) - self.d)))
        else:
            return np.sum(
                np.log10(np.cosh(np.sum(self.A * theta_locals, axis=1) - self.b))
            )

    def localgrad(self, theta_locals, idx):
        grad_local = (
            (1 / np.cosh(np.inner(self.A[idx], theta_locals[idx]) - self.b[idx]))
            * np.sinh(np.inner(self.A[idx], theta_locals[idx]) - self.b[idx])
            * self.A[idx]
        )

        grad = np.zeros((config.graph.num_nodes, self.local_dim))
        grad[idx] = grad_local
        return grad_local, grad

    def globalgrad_local(self, theta_global, idx):
        grad_global = (
            (1 / np.cosh(np.inner(self.C[idx], theta_global[idx]) - self.d[idx]))
            * np.sinh(np.inner(self.C[idx], theta_global[idx]) - self.d[idx])
            * self.C[idx]
        )
        return grad_global

    def globalgrad(self, theta_global):
        grad_global = np.zeros(self.global_dim)
        for i in range(config.graph.num_nodes):
            grad_global += self.globalgrad_local(theta_global, i)
        return grad_global

    def network_globalgrad(self, theta_global):
        grad_global = np.zeros((config.graph.num_nodes, self.global_dim))
        for i in range(config.graph.num_nodes):
            grad_global[i] = self.globalgrad_local(theta_global, i)
        return grad_global

    def networkgrad(self, theta_locals):
        grad = np.zeros((config.graph.num_nodes, self.local_dim))
        for i in range(config.graph.num_nodes):
            grad_local, _ = self.localgrad(theta_locals, i)
            grad[i] = grad_local
        return grad

    def grad(self, theta_locals):
        return self.networkgrad(theta_locals)
