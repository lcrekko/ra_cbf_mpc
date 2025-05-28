"""
This script is a test script for the submodules

The below example is for adaptive cruise control, the initial state 
is estimated from the Figure 1 in the paper **Adaptive Safety with 
Control Barrier Functions** by A.J. Taylor and A.D. Ames
"""
import numpy as np
import matplotlib.pyplot as plt
from nmpc.controller import MPCController
from nmpc.diverse_functions import acc_dynamics, acc_f, acc_g, acc_kernel
# from optimization_utils.metric import max_l1_deviation_value
from rls.rls_main import RLS_constant
from rls.rls_utils import interleave_vec, interleave_diag
from plotters.simulation import SimulateRegret

# ------------- Controller Parameters --------------

# prediction horizon
N = 20

# weighting matrices
Q = np.array([[1, 0], [0, 0.5]]) # the distance D is not heavily penalized
R = np.array([[5 * 1e-6]]) # the input weakly penalized
P = 2 * Q

# ------------- System parameters & Control Objective -------------

# reference state and input (set-point tracking)
x_ref = np.array([24, 24 * 2])
u_ref = np.array([0])

# state and input dimensions
x_dim = 2
u_dim = 1

# initial state
v_0 = 20 # specify the initial velocity
x_0 = np.array([v_0, 60 + 1.8 * v_0])

# input hard limits
u_lim = 1e3

# sampling time
dt = 0.1

# simulation time & simulation steps
T_sim = 100
T_step = int(T_sim / dt)

# generate the x-axis for plotting
time_state = dt * np.arange(0, T_step + 1)
time_input = dt * np.arange(0, T_step)

# -------------- Uncertainty & adaptation ---------------

# initial parameter estimate
num_para = 3
para_star = np.array([0.1, 0.5, 0.25])
para_0 = np.zeros(num_para)

# parameter bound 
LB_para = [0, 0, 0]
UB_para = [1, 5, 2.5]

# parameter matrix
H_para = interleave_diag(-np.ones(num_para), np.ones(num_para))
h_para = interleave_vec(LB_para, UB_para)

# disturbance information
w_lim = 1

# disturbance matrix
H_w = interleave_diag(-w_lim * dt * np.ones(x_dim), w_lim * dt * np.ones(x_dim))


# generate disturbance
np.random.seed(22) # fix the seed for reproducibility
w_sim = np.random.uniform(-w_lim, w_lim, size=(x_dim, T_step))

# ------------- One-step MPC Simulation --------------

# initialize the MPC controller
my_mpc = MPCController(N, dt,
                       x_dim, u_dim,
                       x_ref, u_ref,
                       -u_lim, u_lim,
                       Q, R, P,
                       acc_dynamics, num_para)

# set a default learning rate
MU_0 = 0.5

# initialize the RLS instance
my_rls = RLS_constant(num_para, MU_0, acc_kernel, acc_f, acc_g, H_w)

u_0 = my_mpc.solve_closed(x_0, para_0)

u_0 = np.asarray(u_0)  # ensure u_0 is a numpy array
u_0 = u_0.reshape((u_dim,))  # ensure u_0 is a column vector

print("Initial input:", u_0)

x_next = acc_dynamics(x_0, u_0, dt, para_star, mode="SIM") + dt * w_sim[:, 0] # type: ignore

print("Next state:", x_next)

# update the parameter estimate and parameter set estiamte
para_prior, _ = my_rls.update_para(x_next, x_0, u_0, para_0, 1)

print("Prior parameter estimate:", para_prior)

H_f, h_f = my_rls.update_paraset(x_next, x_0, u_0, H_para, h_para, 1)

print("Updated H matrix:", H_f)
print("Updated h vector:", h_f)

para_next = my_rls.projection(H_f, h_f, para_prior)

print("Posterior parameter estimate:", para_next)