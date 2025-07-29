"""
This is test script for the speed dynamics of adaptive cruise control
"""

import numpy as np
import matplotlib.pyplot as plt
from rls.rls_main import RLSProjection
from rls.rls_utils import interleave_vec, interleave_diag
from nmpc.diverse_functions import vacc_dynamics, vacc_fg, vacc_kernel

# state and input dimensions
dt = 0.1 # trivial sampling time
x_dim = 1
u_dim = 1

# large limit, no special constraints
u_lim = 1e3

#  --------------- disturbance information -----------------
w_lim = 0.01

# disturbance matrix
H_w = interleave_diag(-w_lim * np.ones(x_dim), w_lim * np.ones(x_dim))

# print("H_w:", H_w)

# ---------------- parameter information -----------------
num_para = 3
para_0 = np.array([0.4, 4, 0.3])
para_star = np.array([0.5, 5, 0.25])

# parameter bound 
LB_para = [0.0, 0.0, 0.0]
UB_para = [1.0, 10, 2.5]

# parameter matrix
H_para = interleave_diag(-np.ones(num_para), np.ones(num_para))
h_para = interleave_vec(LB_para, UB_para)

# print("H_para:", H_para)
# print("h_para:", h_para)

# ----------- test dynamics and one step estimation test --------------
x_0 = np.array([50])
u_0 = 100 * np.random.randn(1)

# sample a random disturbance
w = dt * np.random.uniform(-w_lim, w_lim)

x_p = vacc_dynamics(x_0, u_0, para_star) + w
# x_p2 = vacc_fg(x_0, u_0) - vacc_kernel(x_0).T @ para_star

# print("next state integrated:", x_p)
# print("next state separated:", x_p2)

# initialize our RLS module
cov = 100 * np.eye(num_para)
my_rls = RLSProjection(num_para, x_dim, vacc_fg, vacc_kernel, dt, H_w)

# para_new = my_rls.update_para(x_p, x_0, u_0, para_0, cov, 1)

# print("new parameter is:", para_new["para"])
# print("the increment is:", para_new["diff"])


# ------------ A simple naive exploration (no control) ------------------

# simulation time
T = 100
time = dt * np.arange(0, T + 1)
time_s = dt * np.arange(0, T)
# lr = 0.1

para_traj = np.zeros((T+1, num_para))
x_traj = np.zeros((T+1, x_dim))
innovation = np.zeros(T)

para_traj[0, :] = para_0
x_traj[0, :] = x_0

mag_u = 1e3

# main simulation loop
for t in range(T):
    # a random exploration input
    u_temp = 2000 + mag_u * np.random.randn(1)

    # sample a noise
    w = dt * np.random.uniform(-w_lim, w_lim)

    # obtain the measured next state using the true parameter
    x_traj[t+1, :] = vacc_dynamics(x_traj[t, :], u_temp, para_star) + w

    para_info = my_rls.update_para(x_traj[t+1, :], x_traj[t, :], u_temp, para_traj[t, :], cov, t+1)

    para_traj[t+1, :] = para_info["para"]
    cov = para_info["cov"]
    innovation[t] = para_info["inc_state"]

para_1_traj = para_traj[:, 0]
para_2_traj = para_traj[:, 1]
para_3_traj = para_traj[:, 2]

# Create subplots (3 rows, 1 column)
fig, axes = plt.subplots(3, 1, figsize=(10, 6))

# Plot 1
# axes[0].plot(time_state, v_traj_opt, label='velocity (OPT)')
axes[0].plot(time, para_1_traj, label='parameter 1')
#axes[0].axhline(y=para_star[0], color='r', linestyle='--', linewidth=2)
axes[0].set_title("RLS Performance with Random Exploration")
axes[0].legend()
axes[0].set_facecolor((0.95, 0.95, 0.95))
axes[0].grid(True, linestyle='--', color='white', linewidth=1)

# Plot 2
# axes[1].plot(time_state, D_traj_opt, label='distance (OPT)')
axes[1].plot(time, para_2_traj, label='parameter 2')
#axes[1].axhline(y=para_star[1], color='r', linestyle='--', linewidth=2)
axes[1].legend()
axes[1].set_facecolor((0.95, 0.95, 0.95))
axes[1].grid(True, linestyle='--', color='white', linewidth=1)

# Plot 3
# axes[2].plot(time_state, u_traj_opt, label='force (OPT)')
axes[2].plot(time, para_3_traj, label='parameter 3')
#axes[2].axhline(y=para_star[2], color='r', linestyle='--', linewidth=2)
axes[2].legend()
axes[2].set_facecolor((0.95, 0.95, 0.95))
axes[2].grid(True, linestyle='--', color='white', linewidth=1)

# Set x-axis label only on the last plot
axes[2].set_xlabel('Time Step')

# Improve spacing
plt.tight_layout()


fig_2, axes_2 = plt.subplots(2, 1, figsize=(10, 4))

# Plot 1
# axes[0].plot(time_state, v_traj_opt, label='velocity (OPT)')
axes_2[0].plot(time, x_traj[:, 0], label='vehicle velocity')
#axes[0].axhline(y=para_star[0], color='r', linestyle='--', linewidth=2)
axes_2[0].set_title("State and Innovation Inspection")
axes_2[0].legend()
axes_2[0].set_facecolor((0.95, 0.95, 0.95))
axes_2[0].grid(True, linestyle='--', color='white', linewidth=1)

# axes[2].plot(time_state, u_traj_opt, label='force (OPT)')
axes_2[1].plot(time_s, innovation, label='innovation')
#axes[2].axhline(y=para_star[2], color='r', linestyle='--', linewidth=2)
axes_2[1].legend()
axes_2[1].set_facecolor((0.95, 0.95, 0.95))
axes_2[1].grid(True, linestyle='--', color='white', linewidth=1)


plt.tight_layout()
plt.show()
