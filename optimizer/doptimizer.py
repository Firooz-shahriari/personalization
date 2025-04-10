# Decneralized Optimization Algorithms

import copy as cp

from numpy import linalg as LA
import numpy as np
import tensorly
from tqdm import tqdm

from utils.utils import *


def GP(prd, B, learning_rate, K, theta_0):
    theta = [cp.deepcopy(theta_0)]
    grad = prd.networkgrad(theta[-1])
    Y = np.ones(B.shape[1])
    for k in range(K):
        theta.append(np.matmul(B, theta[-1]) - learning_rate * grad)
        Y = np.matmul(B, Y)
        YY = np.diag(Y)
        z = np.matmul(LA.inv(YY), theta[-1])
        grad = prd.networkgrad(z)
        ut.monitor("GP", k, K)
    return theta


def ADDOPT(prd, B1, B2, learning_rate, K, theta_0):
    theta = [cp.deepcopy(theta_0)]
    grad = prd.networkgrad(theta[-1])
    tracker = cp.deepcopy(grad)
    Y = np.ones(B1.shape[1])
    for k in range(K):
        theta.append(np.matmul(B1, theta[-1]) - learning_rate * tracker)
        grad_last = cp.deepcopy(grad)
        Y = np.matmul(B1, Y)
        YY = np.diag(Y)
        z = np.matmul(LA.inv(YY), theta[-1])
        grad = prd.networkgrad(z)
        tracker = np.matmul(B2, tracker) + grad - grad_last
        ut.monitor("ADDOPT", k, K)
    return theta


def SGP(prd, B, learning_rate, K, theta_0):
    theta = cp.deepcopy(theta_0)
    theta_epoch = [cp.deepcopy(theta)]
    sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.n)])
    grad = prd.networkgrad(theta, sample_vec)
    Y = np.ones(B.shape[1])
    for k in range(K):
        theta = np.matmul(B, theta) - learning_rate * grad
        Y = np.matmul(B, Y)
        YY = np.diag(Y)
        z = np.matmul(LA.inv(YY), theta)
        sample_vec = np.array(
            [np.random.choice(prd.data_distr[i]) for i in range(prd.n)]
        )
        grad = prd.networkgrad(z, sample_vec)
        ut.monitor("SGP", k, K)
        if (k + 1) % prd.b == 0:
            theta_epoch.append(cp.deepcopy(theta))
    return theta_epoch


def SADDOPT(prd, B1, B2, learning_rate, K, theta_0):
    theta = cp.deepcopy(theta_0)
    theta_epoch = [cp.deepcopy(theta)]
    sample_vec = np.array([np.random.choice(prd.data_distr[i]) for i in range(prd.n)])
    grad = prd.networkgrad(theta, sample_vec)
    tracker = cp.deepcopy(grad)
    Y = np.ones(B1.shape[1])
    for k in range(K):
        theta = np.matmul(B1, theta) - learning_rate * tracker
        grad_last = cp.deepcopy(grad)
        Y = np.matmul(B1, Y)
        YY = np.diag(Y)
        z = np.matmul(LA.inv(YY), theta)
        sample_vec = np.array(
            [np.random.choice(prd.data_distr[i]) for i in range(prd.n)]
        )
        grad = prd.networkgrad(z, sample_vec)
        tracker = np.matmul(B2, tracker) + grad - grad_last
        ut.monitor("SADDOPT", k, K)
        if (k + 1) % prd.b == 0:
            theta_epoch.append(cp.deepcopy(theta))
    return theta_epoch


def PushPull(prd, R, C, learning_rate, K, theta_0):
    theta = [cp.deepcopy(theta_0)]
    g = prd.networkgrad(theta[-1])
    last_g = np.copy(g)
    y = np.copy(g)
    for k in range(K):
        theta.append(np.matmul(R, theta[-1] - learning_rate * y))
        last_g = g
        g = prd.networkgrad(theta[-1])
        y = np.matmul(C, y) + g - last_g
        ut.monitor("PushPull", k, K)
    return theta


def DAGP(prd, W, Q, learning_rate, K, x0, rho, alpha, cons=True):
    x = [cp.deepcopy(x0)]
    z = [cp.deepcopy(x0)]

    f_grad = prd.networkgrad(x[-1])
    g = np.zeros(f_grad.shape)
    h = np.zeros(g.shape)

    h_iterates = [cp.deepcopy(h)]
    g_iterates = [cp.deepcopy(g)]

    for k in range(K):
        z.append(x[-1] - np.matmul(W, x[-1]) + learning_rate * (g - f_grad))
        if cons:
            x.append(prd.network_projection(z[-1]))
        else:
            x.append(z[-1])
        local_grad = prd.networkgrad(x[-1])
        new_h = h - np.matmul(Q, h - g)
        g = g + rho * (f_grad - g + (z[-1] - x[-1]) / learning_rate) + alpha * (h - g)
        f_grad = local_grad
        h = new_h
        h_iterates.append(h)
        g_iterates.append(g)
        ut.monitor("DAGP", k, K)
    return x, z, h_iterates, g_iterates


def DDPS(prd, R, C, p, K, x0, eps):  # DDPS p: decaying power, eps: parameter
    x = [cp.deepcopy(x0)]
    z = [cp.deepcopy(x0)]
    last_x = cp.deepcopy(x0)

    f_grad = prd.networkgrad(x[-1])
    y = [cp.deepcopy(f_grad)]

    for k in range(K):
        alpha = (k + 1) ** (-p)
        z.append(np.matmul(R, x[-1]) + eps * y[-1] - alpha * f_grad)
        last_x = x[-1]
        x.append(prd.network_projection(z[-1]))
        y.append(last_x - np.matmul(R, last_x) + np.matmul(C, y[-1]) - eps * y[-1])
        f_grad = prd.networkgrad(x[-1])
        ut.monitor("DDPS", k, K)
    return x
