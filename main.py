import os

import matplotlib.pyplot as plt
import numpy as np

from analysis.analysis import error
from graph.graph import RandomGraph
from optimizer.coptimizer import pcgd
from optimizer.doptimizer import p_dagp
from problem.logistic_regression import lr_l2
from problem.synthetic_cosh import synthetic
from utils.utils import *

if __name__ == "__main__":
    gr = RandomGraph()
    pr = synthetic()
    er = error(pr, pr.theta_locals_opt, pr.optimal_value)

    x_locals, x_global, h_global_iters, g_global_iters = p_dagp(pr, gr)

    # for i in range(config.p_dagp.max_iter):
    #     print(np.linalg.norm(np.sum(h_global_iters[i], axis=0)))

    # check if the constraints are satisfied
    constraint_satisfaction = np.zeros((gr.num_nodes, config.p_dagp.max_iter))
    for i in range(config.p_dagp.max_iter):
        x_global_avg = np.mean(x_global[i], axis=0)
        for n in range(gr.num_nodes):
            constraint_satisfaction[n][i] = (
                np.linalg.norm(x_locals[i][n] - np.matmul(pr.map_mat, x_global_avg))
                - pr.epsilon
            )
    os.makedirs(save_path, exist_ok=True)
    plt.figure()
    for n in range(gr.num_nodes):
        plt.plot(constraint_satisfaction[n], label=f"node {n + 1}", linestyle="--")
    plt.title("Constraint Satisfaction (taking average of node_globals) vs Iterations")
    plt.yscale("log")
    plt.savefig(f"{save_path}constraint_satisfaction.png")
    plt.close()

    # check if the global variables are equal (globals reach consensus)
    node = 2
    distance_between_globals = np.zeros((gr.num_nodes, config.p_dagp.max_iter))
    for i in range(config.p_dagp.max_iter):
        for n in range(gr.num_nodes):
            distance_between_globals[n][i] = np.linalg.norm(
                x_global[i][n] - x_global[i][node]
            )

    plt.figure()
    for n in range(gr.num_nodes):
        plt.plot(distance_between_globals[n], label=f"nodes {n + 1} and {node + 1}")
    plt.yscale("log")
    plt.title("Distance between Global Variables and global variable of one node")
    plt.savefig(f"{save_path}distance_between_globals.png")
    plt.close()

    # check if the stopping point is optimal: both locals and globals
    f_values = np.zeros(config.p_dagp.max_iter)
    for i in range(config.p_dagp.max_iter):
          f_values[i] = pr.F_val(
                x_locals[i], x_global[i]
            )
      
    plt.figure()
    plt.plot(f_values)
    plt.yscale("log")
    plt.title("Objective Value vs Iterations")
    plt.savefig(f"{save_path}f_values.png")
    plt.close()
            


#### this code chekcs the centralized algorithm
if __name__ == "__main__":
    gr = RandomGraph().generate_directed()
    pr = synthetic()
    er = error(pr, pr.theta_locals_opt, pr.optimal_value)
    theta_locals, theta_global, f_values = pcgd(pr)

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
    plt.savefig(f"{save_path}distance_between_locals_and_global.png")
    plt.close()

    plt.figure()
    for n in range(pr.num_nodes):
        plt.plot(constraint_satisfaction[n], label=f"node {n + 1}", linestyle="--")
    plt.title("Constraint Satisfaction vs Iterations")
    plt.yscale("log")
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
    plt.savefig(f"{save_path}distance_between_locals.png")
    plt.close()

    plt.figure()
    for n in range(pr.num_nodes):
        plt.plot(distance_to_optimal_local[n])
    plt.plot(distance_to_optimal_global, label="Global")
    plt.yscale("log")
    plt.title("Distance to Optimal_variables vs Iterations")
    plt.legend()
    plt.savefig(f"{save_path}distance_to_optimal.png")
    plt.close()

    LOGGER = log_results(config, save_path)
