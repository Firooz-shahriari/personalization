import copy as cp
from datetime import datetime
import os
import random
import sys

from dotenv import load_dotenv
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from utils.config_parser import Config
from utils.logger import getLOGGER

load_dotenv()
config_path = os.environ.get("CONFIGPATH")
config = Config(config_path)

if config.reproducibility:
    random.seed(config.seed)
    np.random.seed(config.seed)

plot = os.environ.get("PLOT") == "True"

plt.rcParams["figure.figsize"] = [9, 6]
plt.rcParams["figure.dpi"] = 250  # 200 e.g. is really fine, but slower
plt.rcParams.update({"figure.max_open_warning": 0})
plt.rcParams["font.size"] = 10


now = datetime.now().strftime("%Y%m%d_%H%M%S")
result_path = os.environ.get("RESULTPATH")
save_path = f"{result_path}/{now}/"

# Define custom colors for 0 and 1 (e.g., 0 -> blue, 1 -> red)
colors = [
    (0, "white"),
    (1, "blue"),
]  # Replace 'blue' and 'red' with your desired colors

# Create a custom colormap using the colors
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)


def log_results(config, save_path, terminal=False):
    os.makedirs(save_path, exist_ok=True)
    LOGGER = getLOGGER(
        name=f"{now}",
        log_on_file=True,
        save_path=save_path,
        terminal=terminal,
    )

    LOGGER.info(f"Reproducibility: {config.reproducibility}")
    LOGGER.info(f"Seed: {config.seed}")
    LOGGER.info(f"Graph: {config.graph.num_nodes} nodes")
    LOGGER.info(f"Graph: {config.graph.edge_prob} edge probability")
    LOGGER.info(f"Graph: is it directed: {config.graph.directed}")
    LOGGER.info(f"Graph: {config.graph.laplacian_factor} laplacian factor")
    LOGGER.info("\n")
    LOGGER.info(f"Problem: {config.pr.local_dimension} local dimension")
    LOGGER.info(f"Problem: {config.pr.global_dim} global dimension")
    LOGGER.info(f"Problem: {config.pr.epsilon} epsilon")
    LOGGER.info("\n")
    LOGGER.info(f"Optimizer: {config.pcgd.max_iter} max iter")
    LOGGER.info(f"Optimizer: {config.pcgd.lr} learning rate")
    LOGGER.info(f"Optimizer: {config.pcgd.min} min learning rate")
    LOGGER.info(f"Optimizer: {config.pcgd.decay_rate} decay rate")
    LOGGER.info(f"Optimizer: {config.pcgd.decay_steps} decay steps")
    LOGGER.info(f"Optimizer: {config.pcgd.projection_iters} projection iterations")
    LOGGER.info("\n")
    LOGGER.info(f"Result path: {save_path}")

    return LOGGER


def project_onto_quadratic_set(z1, z2, A, epsilon, tol=1e-6):
    """
    Projects the point (z1, z2) onto the quadratic set defined by:
        { (x1, x2) | ||x1 - A x2||_2^2 <= epsilon }

    Parameters:
        z1 (ndarray): Vector in R^n.
        z2 (ndarray): Vector in R^m.
        A (ndarray): Matrix of size (n x m).
        epsilon (float): Scalar defining the radius of the quadratic set.
        tol (float, optional): Tolerance for convergence. Default is 1e-6.

    Returns:
        x1_proj (ndarray): Projected vector corresponding to z1.
        x2_proj (ndarray): Projected vector corresponding to z2.
    """

    # Check feasibility: If the point is already in the set, return it directly
    residual = np.linalg.norm(z1 - A @ z2)
    if residual <= epsilon:
        return z1, z2

    # Otherwise, solve numerically using bisection method
    λ_min, λ_max = 0.0, 1.0

    # Find an upper bound λ_max where the residual is less than epsilon
    while True:
        x2_trial = compute_x2(z1, z2, A, λ_max)
        x1_trial = compute_x1(z1, A, x2_trial, λ_max)
        if np.linalg.norm(x1_trial - A @ x2_trial) <= epsilon:
            break
        λ_max *= 2

    iteration = 0
    # Perform bisection search
    while True:
        iteration += 1
        λ_mid = (λ_min + λ_max) / 2

        # Compute intermediate projection for given λ_mid
        x2_mid = compute_x2(z1, z2, A, λ_mid)
        x1_mid = compute_x1(z1, A, x2_mid, λ_mid)
        residual_mid = np.linalg.norm(x1_mid - A @ x2_mid)

        # Update λ bounds based on residual
        if residual_mid > epsilon:
            λ_min = λ_mid
        else:
            λ_max = λ_mid

        # Check convergence
        if abs(residual_mid - epsilon) < tol:
            break

    return x1_mid, x2_mid


def compute_x2(z1, z2, A, λ):
    """
    Computes x2(λ) based on the optimality conditions.
    """

    m = z2.shape[0]
    identity_m = np.eye(m)

    matrix = identity_m + (λ / (1 + λ)) * (A.T @ A)
    rhs = z2 + (λ / (1 + λ)) * (A.T @ z1)
    x2 = np.linalg.solve(matrix, rhs)

    return x2


def compute_x1(z1, A, x2, λ):
    """
    Computes x1(λ) based on the optimality conditions.
    """

    return (z1 + λ * (A @ x2)) / (1 + λ)


def equidistant_point(X):
    # X: n x d matrix, each row is a point x_i
    x1 = X[0]
    A = 2 * (X[1:] - x1)  # shape: (n-1) x d
    b = np.sum(X[1:] ** 2, axis=1) - np.sum(x1**2)

    # Solve least squares
    y, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return y


def sample_random_vectors_near_y(A, x_star, epsilon, n):
    m = A.shape[0]
    y = A @ x_star

    samples = np.zeros((n, m))

    for i in range(n):
        # random direction
        u = np.random.randn(m)
        u_unit = u / np.linalg.norm(u)

        # random radius for uniform sampling in ball
        radius = epsilon * np.random.rand() ** (1 / m)

        # create random sample
        samples[i] = y + radius * u_unit

    return samples
