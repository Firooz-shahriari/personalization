import os

import matplotlib.pyplot as plt
import numpy as np

from analysis.analysis import error
from graph.graph import RandomGraph
from optimizer.coptimizer import pcgd
from problem.logistic_regression import lr_l2
from problem.synthetic_cosh import synthetic
from utils.utils import *

if __name__ == "__main__":

    z_row_sum, z_col_sum, row_stoch, col_stoch = RandomGraph().generate_directed()
    pr = synthetic()
    error_pr = error(pr, pr.theta_locals_opt, pr.optimal_value)

    (
        theta_locals,
        theta_global,
        f_values,
        theta_locals_pcgd,
        theta_global_pcgd,
        F_cgd,
    ) = pcgd(pr)

    os.makedirs(save_path, exist_ok=True)
    plt.figure()
    plt.plot(f_values)
    plt.yscale("log")
    plt.title("Objective Value vs Iterations")
    plt.hlines(0, 0, len(f_values), colors="r", linestyles="--")
    plt.savefig(f"{save_path}f_values.png")
    plt.close()

    distance_to_optimal_local = np.zeros((config.graph.num_nodes, config.pcgd.max_iter))
    distance_to_optimal_global = np.zeros(config.pcgd.max_iter)
    distance_between_locals = np.zeros(
        (config.graph.num_nodes, config.graph.num_nodes, config.pcgd.max_iter)
    )
    distance_between_locals_and_global = np.zeros(
        (config.graph.num_nodes, config.pcgd.max_iter)
    )

    constraint_satisfaction = np.zeros((config.graph.num_nodes, config.pcgd.max_iter))
    for i in range(config.pcgd.max_iter):
        for n in range(pr.num_nodes):
            constraint_satisfaction[n, i] = (
                np.linalg.norm(theta_locals[i][n] - np.matmul(pr.map, theta_global[i]))
                - pr.epsilon
            )
            distance_between_locals_and_global[n, i] = np.linalg.norm(
                theta_locals[i][n] - theta_global[i]
            )

    for i in range(config.pcgd.max_iter):
        for n in range(pr.num_nodes):
            for m in range(pr.num_nodes):
                distance_between_locals[n, m, i] = np.linalg.norm(
                    theta_locals[i][n] - theta_locals[i][m]
                )

    for n in range(pr.num_nodes):
        for i in range(config.pcgd.max_iter):
            distance_to_optimal_local[n, i] = np.linalg.norm(
                theta_locals[i][n] - pr.theta_locals_opt[n]
            )
            if n == 0:
                distance_to_optimal_global[i] = np.linalg.norm(
                    theta_global[i] - pr.theta_global_opt
                )

    plt.figure()
    for n in range(pr.num_nodes):
        plt.plot(
            distance_between_locals_and_global[n], label=f"node {n + 1}", linestyle="--"
        )
    plt.title("Distance between Local and Global Variables vs Iterations")
    plt.yscale("log")
    # plt.legend()
    plt.savefig(f"{save_path}distance_between_locals_and_global.png")
    plt.close()

    plt.figure()
    for n in range(pr.num_nodes):
        plt.plot(constraint_satisfaction[n], label=f"node {n + 1}", linestyle="--")
    plt.title("Constraint Satisfaction vs Iterations")
    plt.yscale("log")
    # plt.hlines(pr.epsilon, 0, len(f_values), colors="r", linestyles="--")
    # plt.legend()
    plt.savefig(f"{save_path}constraint_satisfaction.png")
    plt.close()

    plt.figure()
    node = 2
    for n in range(pr.num_nodes):
        plt.plot(
            distance_between_locals[n, node], label=f"nodes {n + 1} and {node + 1}"
        )
    plt.yscale("log")
    plt.title("Distance between Local Variables and local variable of one node")
    # plt.legend()
    plt.savefig(f"{save_path}distance_between_locals.png")
    plt.close()

    plt.figure()
    for n in range(pr.num_nodes):
        plt.plot(distance_to_optimal_local[n], label=f"Local {n + 1}")
    plt.plot(distance_to_optimal_global, label="Global")
    plt.yscale("log")
    plt.title("Distance to Optimal_variables vs Iterations")
    plt.legend()
    plt.savefig(f"{save_path}distance_to_optimal.png")
    plt.close()

    LOGGER = log_results(config, save_path)

    # if config.graph.directed:
    #     z_row_sum, z_col_sum, row_stoch, col_stoch = RandomGraph().generate_directed()
    # else:
    #     z_row_and_col_sum, stoch_matrix = RandomGraph().generate_undirected()

    # if config.pr.type == "synthetic":
    #     prd = synthetic()
    #     error_prd = error()
    # elif config.pr.type == "LogisticRegression":
    #     prd = lr_l2()
    #     error_prd = error()

    # theta_DAGP, _, h_itrs, g_itrs = dopt.DAGP(
    #     prd,
    #     zero_row_sum,
    #     zero_column_sum,
    #     step_size_DAGP,
    #     int(depoch),
    #     theta_0,
    #     rho_DAGP,
    #     alpha_DAGP,
    #     cons=True,
    # )

    # res_F_DAGP = error_prd.cost_path(np.sum(theta_DAGP, axis=1) / num_nodes)
    # fesgp_DAGP = error_prd.feasibility_gap_syn(np.sum(theta_DAGP, axis=1) / num_nodes)


# if __name__ == "__main__":
#     n = 10
#     zl = np.random.randn(n)
#     zg = np.random.randn(n)
#     # zl = np.array([1, 2, 3, 4])
#     # zg = np.array([1, 2, 3, 4])
#     # A = np.eye(n)
#     A = 10 * np.random.randn(n, n)

#     xl, xg = project_onto_quadratic_set(zl, zg, A, 0.0001)
#     print("xl:", xl)
#     print("xg:", xg)
#     print(np.linalg.norm(xl - np.matmul(A, xg)))


# if __name__ == "__main__":
#     prd = synthetic()
