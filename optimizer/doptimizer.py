# Decneralized Optimization Algorithms

import copy as cp

from numpy import linalg as LA
import numpy as np
from tqdm import tqdm

from utils.utils import *


def PushPull(pr, R, C, learning_rate, K, theta_0):
    theta = [cp.deepcopy(theta_0)]
    g = pr.networkgrad(theta[-1])
    last_g = np.copy(g)
    y = np.copy(g)
    for k in range(K):
        theta.append(np.matmul(R, theta[-1] - learning_rate * y))
        last_g = g
        g = pr.networkgrad(theta[-1])
        y = np.matmul(C, y) + g - last_g
        ut.monitor("PushPull", k, K)
    return theta


def DAGP(pr, W, Q, learning_rate, K, x0, rho, alpha, cons=True):
    x = [cp.deepcopy(x0)]
    z = [cp.deepcopy(x0)]

    f_grad = pr.networkgrad(x[-1])
    g = np.zeros(f_grad.shape)
    h = np.zeros(g.shape)

    h_iterates = [cp.deepcopy(h)]
    g_iterates = [cp.deepcopy(g)]

    for k in range(K):
        z.append(x[-1] - np.matmul(W, x[-1]) + learning_rate * (g - f_grad))
        if cons:
            x.append(pr.network_projection(z[-1]))
        else:
            x.append(z[-1])
        local_grad = pr.networkgrad(x[-1])
        new_h = h - np.matmul(Q, h - g)
        g = g + rho * (f_grad - g + (z[-1] - x[-1]) / learning_rate) + alpha * (h - g)
        f_grad = local_grad
        h = new_h
        h_iterates.append(h)
        g_iterates.append(g)
        ut.monitor("DAGP", k, K)
    return x, z, h_iterates, g_iterates


def dagp_modified_for_personalized(
    pr, W, Q, learning_rate, K, x0, rho, alpha, cons=True
):
    pass


def p_dagp(pr, graph):
    x0_locals = np.random.randn(config.graph.num_nodes, pr.local_dim)
    x0_globals = np.random.randn(graph.num_nodes, pr.global_dim)

    x_locals = [cp.deepcopy(x0_locals)]
    x_globals = [cp.deepcopy(x0_globals)]

    f_grad_locals = pr.networkgrad(x_locals[-1])
    f_grad_global = pr.network_globalgrad(x_globals[-1])

    g_globals = np.zeros(f_grad_global.shape)
    h_globals = np.zeros(g_globals.shape)

    h_global_iters = [cp.deepcopy(h_globals)]
    g_global_iters = [cp.deepcopy(g_globals)]
    f_values = [pr.F_val(x_locals[-1], x_globals[-1])]

    bar = tqdm(total=config.p_dagp.max_iter, leave=False)

    for k in range(config.p_dagp.max_iter):
        z_locals, z_globals = update_z(
            x_locals[-1],
            x_globals[-1],
            g_global_iters[-1],
            graph,
            config.p_dagp.lr,
            f_grad_locals,
            f_grad_global,
        )

        x_locals_tmp, x_globals_tmp = project_z(
            z_locals, z_globals, pr.map_mat, pr.epsilon
        )
        x_locals.append(cp.deepcopy(x_locals_tmp))
        x_globals.append(cp.deepcopy(x_globals_tmp))

        h_globals_new = h_globals - np.matmul(graph.zero_col_sum, h_globals - g_globals)
        g_globals = (
            g_globals
            + config.p_dagp.rho
            * (
                f_grad_global
                - g_globals
                + (z_globals - x_globals_tmp) / config.p_dagp.lr
            )
            + config.p_dagp.alpha * (h_globals - g_globals)
        )

        f_grad_locals = pr.networkgrad(x_locals[-1])
        f_grad_global = pr.network_globalgrad(x_globals[-1])
        h_globals = h_globals_new

        bar.set_postfix({"objective_val": pr.F_val(x_locals[-1], x_globals[-1])})
        bar.update()
        h_global_iters.append(cp.deepcopy(h_globals))
        g_global_iters.append(cp.deepcopy(g_globals))
        f_values.append(pr.F_val(x_locals[-1], x_globals[-1]))

    return (
        x_locals,
        x_globals,
        h_global_iters,
        g_global_iters,
    )


def update_z(
    x_locals, x_globals, g_globals, graph, learning_rate, f_grad_locals, f_grad_globals
):
    z_locals = x_locals - config.p_dagp.lr * f_grad_locals

    if config.pr.global_objective_exists:
        z_globals = (
            x_globals
            - np.matmul(graph.zero_row_sum, x_globals)
            + config.p_dagp.lr * (g_globals - f_grad_globals)
        )
    else:
        z_globals = (
            x_globals
            - np.matmul(graph.zero_row_sum, x_globals)
            + config.p_dagp.lr * g_globals
        )
    return z_locals, z_globals


def project_z(x_locals, x_globals, map_mat, epsilon):
    glob_tmp = cp.deepcopy(x_globals)
    local_tmp = cp.deepcopy(x_locals)
    for node in range(config.graph.num_nodes):
        local_tmp[node], glob_tmp[node] = project_onto_quadratic_set(
            x_locals[node], x_globals[node], map_mat, epsilon
        )
    return local_tmp, glob_tmp
