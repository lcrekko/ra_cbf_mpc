"""
This script is a test script for the MPC submodule

The below example is for adaptive cruise control, the initial state 
is estimated from the Figure 1 in the paper **Adaptive Safety with 
Control Barrier Functions** by A.J. Taylor and A.D. Ames
"""
import numpy as np
from nmpc.controller import MPCController
from nmpc.diverse_functions import acc_dynamics, acc_kernel, acc_f, acc_g
# from optimization_utils.metric import max_l1_deviation_value
from rls.rls_main import RLS_constant
from rls.rls_utils import interleave_vec, interleave_diag

# ------------- Controller Parameters --------------

# prediction horizon
N = 5

# weighting matrices
Q = np.array([[1, 0], [0, 1e-3]]) # the distance D is not heavily penalized
R = 1e-3 # the input is not heavily penalized, so tracking u_ref = 0 is not prioritized
P = 2 * Q 

# ------------- System parameters & Control Objective -------------

# reference state and input (set-point tracking)
X_REF = np.array([24, 24 * 2])
U_REF = np.array([0])

# state and input dimensions
X_DIM = 2
U_DIM = 1

# initial state
V_INITIAL = 20 # specify the initial velocity
X_INITIAL = np.array([V_INITIAL, 60 + 1.8 * V_INITIAL])

# input hard limits
U_LIM = 10

# sampling time
DT = 0.01

# simulation time & simulation steps
T_SIM = 20
T_STEP = int(20 / DT)

# -------------- Uncertainty & adaptation ---------------

# initial parameter estimate
NUM_PARA = 3
PARA_STAR = np.array([1.0, -2.0, 3.0])
PARA_0 = 5 * PARA_STAR

# parameter bound 
LB_PARA = [-4, -10, -9]
UB_PARA = [6, 6, 15]

# parameter matrix
H_para = interleave_diag(-np.ones(NUM_PARA), np.ones(NUM_PARA))
h_para = interleave_vec(LB_PARA, UB_PARA)

# disturbance information
W_LIM = 10

# disturbance matrix
H_w = interleave_diag(-W_LIM * DT * np.ones(X_DIM), W_LIM * DT * np.ones(X_DIM))


# generate disturbance
np.random.seed(22) # fix the seed for reproducibility
w_sim = np.random.uniform(-W_LIM, W_LIM, size=(X_DIM, T_STEP))

# ------------- One-step MPC Simulation --------------

# initialize the MPC controller
ampc_controller = MPCController(N, DT,
                                X_DIM, U_DIM,
                                X_REF, U_REF,
                                -U_LIM, U_LIM,
                                Q, R, P,
                                acc_dynamics)

# solve the mpc to get the first input
u_0 = ampc_controller.solve_closed(X_INITIAL, PARA_0)

# simulate the system with the first input to get the next state
x_plus = acc_dynamics(X_INITIAL, u_0, DT, PARA_STAR) + DT * w_sim[:, 0] # type: ignore

# -------------- RLS Identification Simulation ---------------

# set a default learning rate
MU_0 = 0.5

# initialize the RLS instance
my_rls = RLS_constant(NUM_PARA, MU_0, acc_kernel, acc_f, acc_g, H_w)

# update the parameter estimate and parameter set estiamte
f_prior, f_add = my_rls.update_para(x_plus, X_INITIAL, u_0, PARA_0, 1)
H_f, h_f = my_rls.update_paraset(x_plus, X_INITIAL, u_0, H_para, h_para, 1)
f_posterior = my_rls.projection(H_f, h_f, f_prior)

print("Control input is:", u_0)

print("The posterior estimate", f_posterior)

