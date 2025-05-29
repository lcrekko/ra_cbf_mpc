"""
This is a debugging test script for adaptive linaer control
"""

import numpy as np
from rls.rls_main import RLS_constant
from rls.rls_utils import interleave_vec, interleave_diag
from nmpc.diverse_functions import linear_dynamics, linear_f, linear_kernel
from nmpc.controller import MPCController

# -------------- Test disturbance matrix generation ---------------

# state and input dimensions
dt = 1 # trivial sampling time
x_dim = 2
u_dim = 1

u_lim = 1

# disturbance information
w_lim = 0.1

# disturbance matrix
H_w = interleave_diag(-w_lim * np.ones(x_dim), w_lim * np.ones(x_dim))

# print("H_w:", H_w)

# --------------- Test parameter matrix generation ---------------
num_para = 3
para_0 = np.array([0.0, 0.0, 0.0])
para_star = np.array([0.8, 0.2, -0.5])

# parameter bound 
LB_para = [-1, -1, -1]
UB_para = [1, 1, 1]

# parameter matrix
H_para = interleave_diag(-np.ones(num_para), np.ones(num_para))
h_para = interleave_vec(LB_para, UB_para)

# print("H_para:", H_para)
# print("h_para:", h_para)

# --------------- Test dynamics and kernel function ---------------
# initial state
x_0 = np.array([2, 3])  # specify the initial state

# --------------- Test MPC solver ----------------
# prediction horizon
N = 10

# weighting matrices
Q = np.array([[1, 0], [0, 1]]) # the distance D is not heavily penalized
R = np.array([[1]]) # the input weakly penalized
P = np.array([[1.467, 0.207], [0.207, 1.731]])

# reference state and input (set-point tracking)
x_ref = np.array([0, 0])
u_ref = np.array([0])

# ------------- One-step MPC Simulation --------------

# initialize the MPC controller
my_mpc = MPCController(N, dt,
                       x_dim, u_dim,
                       x_ref, u_ref,
                       -u_lim, u_lim,
                       Q, R, P,
                       linear_dynamics, num_para)

u_0 = my_mpc.solve_closed(x_0, para_0)

print("Initial control input:", u_0)