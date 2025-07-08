## This file contains the implementation of the centralized optimization algorithms

import copy as cp

import numpy as np
from tqdm import tqdm

from utils.utils import *


## Centralized gradient descent
def pcgd(pr):
    """
    Centralized Projected Gradient Descent (CGD) algorithm.
    Args:
        pr: Problem instance.
        config.pcgd.lr: Learning rate for the optimization.
        config.pcgd.max_iter: Maximum number of iterations.
        theta_0: Initial parameter vector.
    Returns:
        theta: List of parameter vectors at each iteration.
        theta_opt: Optimal parameter vector.
        F_opt: Optimal function value.
    """

    theta_locals = [np.random.randn(pr.num_nodes, pr.local_dim)]
    theta_global = [np.random.randn(pr.global_dim)]

    if config.pr.global_objective_exists:
        f_values = [pr.F_val(theta_locals[-1], theta_global[-1])]
    else:
        f_values = [pr.F_val(theta_locals[-1])]

    bar = tqdm(total=config.pcgd.max_iter, leave=False)

    for itr in range(config.pcgd.max_iter):
        theta_locals_tmp = theta_locals[-1] - config.pcgd.lr * pr.grad(theta_locals[-1])
        if config.pr.global_objective_exists:
            theta_global_tmp = theta_global[-1] - config.pcgd.lr * pr.globalgrad(
                theta_global[-1]
            )
        else:
            theta_global_tmp = theta_global[-1]
        for _ in range(config.pcgd.projection_iters):
            for idx in range(pr.num_nodes):
                theta_locals_tmp[idx], theta_global_tmp = project_onto_quadratic_set(
                    theta_locals_tmp[idx], theta_global_tmp, pr.map, pr.epsilon
                )

        theta_locals.append(cp.copy(theta_locals_tmp))
        theta_global.append(cp.copy(theta_global_tmp))

        if config.pr.global_objective_exists:
            f_values.append(pr.F_val(theta_locals[-1], theta_global[-1]))
        else:
            f_values.append(pr.F_val(theta_locals[-1]))

        results = {"objective_val": f_values[-1]}
        bar.set_postfix(results)
        bar.update()

    return theta_locals, theta_global, f_values
