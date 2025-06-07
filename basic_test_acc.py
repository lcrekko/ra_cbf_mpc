"""
This is a debugging test script for adaptive linaer control
"""

import numpy as np
import matplotlib.pyplot as plt
from rls.rls_main import RLSProjection
from rls.rls_utils import interleave_vec, interleave_diag
from nmpc.diverse_functions import acc_dynamics, acc_f, acc_kernel
from nmpc.controller import MPCController
from plotters.simulation import SimulateRegret

# -------------- Test disturbance matrix generation ---------------

# state and input dimensions
dt = 0.1 # trivial sampling time
x_dim = 2
u_dim = 1

u_lim = 1e3

# disturbance information
w_lim = 1

# disturbance matrix
H_w = interleave_diag(-w_lim * np.ones(x_dim), w_lim * np.ones(x_dim))

# print("H_w:", H_w)

# --------------- Test parameter matrix generation ---------------
num_para = 3
para_0 = np.array([0.0, 0.0, 0.0])
para_star = np.array([0.1, 0.5, 0.25])

# parameter bound 
LB_para = [0.0, 0.0, 0.0]
UB_para = [1.0, 5, 2.5]

# parameter matrix
H_para = interleave_diag(-np.ones(num_para), np.ones(num_para))
h_para = interleave_vec(LB_para, UB_para)

# print("H_para:", H_para)
# print("h_para:", h_para)

# --------------- Test dynamics and kernel function ---------------
# initial state
x_0 = np.array([20, 96])  # specify the initial state

# --------------- Test MPC solver ----------------
# prediction horizon
N = 10

# weighting matrices
Q = np.array([[1, 0], [0, 0.1]]) # the distance D is not heavily penalized
R = np.array([[1e-3]]) # the input weakly penalized
P = 2 * Q

# reference state and input (set-point tracking)
x_ref = np.array([24, 48])
u_ref = np.array([0])

# ------------- One-step MPC Simulation --------------

# initialize the MPC controller
my_mpc = MPCController(N, dt,
                       x_dim, u_dim,
                       x_ref, u_ref,
                       -u_lim, u_lim,
                       Q, R, P,
                       acc_dynamics, num_para)

# ------------- Basic test RLS --------------
my_mu = 1e2
my_rls = RLSProjection(num_para, x_dim, acc_f, acc_kernel, dt, H_w)

T_sim = 200
time_state = dt * np.arange(0, T_sim + 1)
time_input = dt * np.arange(0, T_sim)
w_sim = np.random.uniform(-w_lim, w_lim, size=(x_dim, T_sim))

my_sim = SimulateRegret(my_mpc, my_rls, acc_dynamics,
                        x_0, para_star, u_dim,
                        Q, R,
                        dt, T_sim)

# out_sim_1 = my_sim.nominal_mpc_sim(w_sim)
out_sim_2 = my_sim.learning_mpc_sim(w_sim, para_0, H_para, h_para, my_mu)
# regret = my_sim.regret_final(out_sim_1["cost"], out_sim_2["cost"])

# ------------- Plotting the results -----------------
# x_1_traj = out_sim_2["x_traj"][0, :]
# x_2_traj = out_sim_2["x_traj"][1, :]
# u_traj = out_sim_2["u_traj"][0, :]

para_1_traj = out_sim_2["para_est"][0, :]
para_2_traj = out_sim_2["para_est"][1, :]
para_3_traj = out_sim_2["para_est"][2, :]

# Create subplots (3 rows, 1 column)
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

# Plot 1
# axes[0].plot(time_state, v_traj_opt, label='velocity (OPT)')
axes[0].plot(time_state, para_1_traj, label='parameter 1')
axes[0].axhline(y=para_star[0], color='r', linestyle='--', linewidth=2, label='true value')
axes[0].set_title("RLS Performance")
axes[0].legend()
axes[0].set_facecolor((0.95, 0.95, 0.95))
axes[0].grid(True, linestyle='--', color='white', linewidth=1)

# Plot 2
# axes[1].plot(time_state, D_traj_opt, label='distance (OPT)')
axes[1].plot(time_state, para_2_traj, label='parameter 2')
axes[1].axhline(y=para_star[1], color='r', linestyle='--', linewidth=2, label='true value')
axes[1].legend()
axes[1].set_facecolor((0.95, 0.95, 0.95))
axes[1].grid(True, linestyle='--', color='white', linewidth=1)

# Plot 3
# axes[2].plot(time_state, u_traj_opt, label='force (OPT)')
axes[2].plot(time_state, para_3_traj, label='parameter 3')
axes[2].axhline(y=para_star[2], color='r', linestyle='--', linewidth=2, label='true value')
axes[2].legend()
axes[2].set_facecolor((0.95, 0.95, 0.95))
axes[2].grid(True, linestyle='--', color='white', linewidth=1)

# Set x-axis label only on the last plot
axes[2].set_xlabel('Time Step')

# Improve spacing
plt.tight_layout()


# ------------ State plot ------------
# Create subplots (3 rows, 1 column)
# fig_2, axes_2 = plt.subplots(3, 1, figsize=(10, 8))

# # Plot 1
# # axes[0].plot(time_state, v_traj_opt, label='velocity (OPT)')
# axes_2[0].plot(time_state, x_1_traj, label='state 2')
# axes_2[0].set_title("RLS MPC State and Input")
# axes_2[0].legend()
# axes_2[0].set_facecolor((0.95, 0.95, 0.95))
# axes_2[0].grid(True, linestyle='--', color='white', linewidth=1)

# # Plot 2
# # axes[1].plot(time_state, D_traj_opt, label='distance (OPT)')
# axes_2[1].plot(time_state, x_2_traj, label='state 1')
# axes_2[1].legend()
# axes_2[1].set_facecolor((0.95, 0.95, 0.95))
# axes_2[1].grid(True, linestyle='--', color='white', linewidth=1)

# # Plot 3
# # axes[2].plot(time_state, u_traj_opt, label='force (OPT)')
# axes_2[2].plot(time_input, u_traj, label='input')
# axes_2[2].legend()
# axes_2[2].set_facecolor((0.95, 0.95, 0.95))
# axes_2[2].grid(True, linestyle='--', color='white', linewidth=1)

# # Set x-axis label only on the last plot
# axes_2[2].set_xlabel('Time')

# plt.tight_layout()

plt.savefig("rls_acc.pdf", format="pdf", dpi=300, bbox_inches='tight')
plt.show()
