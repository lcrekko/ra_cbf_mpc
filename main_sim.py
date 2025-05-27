"""
This script is the main script for simulation and obtain the regret curve

The below example is for adaptive cruise control, the initial state 
is estimated from the Figure 1 in the paper **Adaptive Safety with 
Control Barrier Functions** by A.J. Taylor and A.D. Ames
"""
import numpy as np
import matplotlib.pyplot as plt
from nmpc.controller import MPCController
from nmpc.diverse_functions import acc_dynamics, acc_kernel, acc_f, acc_g
# from optimization_utils.metric import max_l1_deviation_value
from rls.rls_main import RLS_constant
from rls.rls_utils import interleave_vec, interleave_diag
# from plotters.simulation import simulate_regret
from plotters.utils import RegretPlotter

# ------------- Controller Parameters --------------

# prediction horizon
N = 5

# weighting matrices
Q = np.array([[1, 0], [0, 1e-3]]) # the distance D is not heavily penalized
R = 1e-3 # the input is not heavily penalized, so tracking u_ref = 0 is not prioritized
R = np.atleast_2d(R) # for cost compuations, keep the data type consistent for MIMO systems
P = 2 * Q

# ------------- System parameters & Control Objective -------------

# reference state and input (set-point tracking)
X_REF = np.array([24, 24 * 2])
U_REF = np.array([0])

# state and input dimensions
X_DIM = 2
U_DIM = 1

# initial state
v_initial = 20 # specify the initial velocity
x_initial = np.array([v_initial, 60 + 1.8 * v_initial])

# input hard limits
U_LIM = 10

# sampling time
dt = 0.01

# simulation time & simulation steps
T_sim = 1
T_step = int(T_sim / dt)

time_plot = np.arange(0, T_step)

# -------------- Uncertainty & adaptation ---------------

# initial parameter estimate
NUM_PARA = 3
para_star = np.array([1.0, -2.0, 3.0])
para_0 = 5 * para_star

# parameter bound
LB_PARA = [-4, -10, -9]
UB_PARA = [6, 6, 15]

# parameter matrix
H_para = interleave_diag(-np.ones(NUM_PARA), np.ones(NUM_PARA))
h_para = interleave_vec(LB_PARA, UB_PARA)

# disturbance information
W_LIM = 10 # disturbance limit, the disturbance is uniformly distributed in [-W_LIM, W_LIM]
NUM_SIM = 2 # number of simulations to be performed

# disturbance matrix
H_w = interleave_diag(-W_LIM * dt * np.ones(X_DIM), W_LIM * dt * np.ones(X_DIM))


# generate disturbance
np.random.seed(22) # fix the seed for reproducibility
w_sim = np.random.uniform(-W_LIM, W_LIM, size=(X_DIM, T_step, NUM_SIM))

# ------------- Initialize the MPC and RLS instances --------------

# initialize the MPC controller
ampc_controller = MPCController(N, dt,
                                X_DIM, U_DIM,
                                X_REF, U_REF,
                                -U_LIM, U_LIM,
                                Q, R, P,
                                acc_dynamics, NUM_PARA)

# set a default learning rate
MU_0 = 0.05

# initialize the RLS estimator
my_rls = RLS_constant(NUM_PARA, MU_0, acc_kernel, acc_f, acc_g, H_w)

# ------------- Run the simulation to obtain regret data -------------
