"""
This is a debugging test script for adaptive cruise control
"""

import numpy as np
from rls.rls_main import RLS_constant
from rls.rls_utils import interleave_vec, interleave_diag
from nmpc.diverse_functions import acc_dynamics, acc_f, acc_kernel

# -------------- Test disturbance matrix generation ---------------
# sampling time
dt = 0.1

# state and input dimensions
x_dim = 2
u_dim = 1

# disturbance information
w_lim = 1

# disturbance matrix
# H_w = interleave_diag(-w_lim * dt * np.ones(x_dim), w_lim * dt * np.ones(x_dim))

# print("H_w:", H_w)

# --------------- Test parameter matrix generation ---------------
num_para = 3
para_0 = np.array([0.1, 0.5, 0.25])

# parameter bound 
LB_para = [0, 0, 0]
UB_para = [1, 5, 2.5]

# parameter matrix
H_para = interleave_diag(-np.ones(num_para), np.ones(num_para))
h_para = interleave_vec(LB_para, UB_para)

# print("H_para:", H_para)
# print("h_para:", h_para)

# --------------- Test dynamics and kernel function ---------------
# initial state
v_0 = 20 # specify the initial velocity
x_0 = np.array([v_0, 60 + 1.8 * v_0])

# --------------- Test MPC solver ----------------
# prediction horizon
N = 20

# weighting matrices
Q = np.array([[1, 0], [0, 0.5]]) # the distance D is not heavily penalized
R = np.array([[5 * 1e-6]]) # the input weakly penalized
P = 2 * Q

# reference state and input (set-point tracking)
x_ref = np.array([24, 24 * 2])
u_ref = np.array([0])

# state and input dimensions
x_dim = 2
u_dim = 1




