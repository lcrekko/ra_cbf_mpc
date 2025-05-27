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
PARA_0 = 5 * para_star

# parameter bound 
LB_para = [0, 0, 0]
UB_para = [1, 5, 2.5]

# parameter matrix
H_para = interleave_diag(-np.ones(num_para), np.ones(num_para))
h_para = interleave_vec(LB_para, UB_para)

# disturbance information
w_lim = 2

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

my_sim = SimulateRegret(my_mpc, my_rls,
                        x_0, para_star, u_dim,
                        Q, R,
                        dt, T_step)

out_opt = my_sim.nominal_mpc_sim(w_sim)

v_traj = out_opt["x_traj"][0, :]
D_traj = out_opt["x_traj"][1, :]
u_traj = out_opt["u_traj"][0, :]

# Create subplots (3 rows, 1 column)
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

# Plot 1
axes[0].plot(time_state, v_traj, label='velocity', color='b')
axes[0].set_title("Adaptive Cruise Control using MPC")
axes[0].legend()
axes[0].set_facecolor((0.95, 0.95, 0.95))
axes[0].grid(True, linestyle='--', color='white', linewidth=1)

# Plot 2
axes[1].plot(time_state, D_traj, label='distance', color='g')
axes[1].legend()
axes[1].set_facecolor((0.95, 0.95, 0.95))
axes[1].grid(True, linestyle='--', color='white', linewidth=1)

# Plot 3
axes[2].plot(time_input, u_traj, label='acceleration', color='r')
axes[2].legend()
axes[2].set_facecolor((0.95, 0.95, 0.95))
axes[2].grid(True, linestyle='--', color='white', linewidth=1)

# Set x-axis label only on the last plot
axes[2].set_xlabel('Time')

# Improve spacing
plt.tight_layout()

# Show plot
plt.show()

