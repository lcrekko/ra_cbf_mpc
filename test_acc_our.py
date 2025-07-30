"""
This is test script for ACC model in our case
"""

import numpy as np
import matplotlib.pyplot as plt
from rls.rls_main import RLSProjection
from rls.rls_utils import interleave_vec, interleave_diag
from nmpc.diverse_functions import sacc_dynamics, sacc_fg, sacc_kernel

# state and input dimensions
dt = 0.1 # trivial sampling time
x_dim = 2
u_dim = 1

# large limit, no special constraints
u_lim = 1e3

#  --------------- disturbance information -----------------
w_lim = np.array([0.1, 1])

# disturbance matrix
H_w = interleave_diag(-w_lim, w_lim)

# print("H_w:", H_w)

# ---------------- parameter information -----------------
num_para = 2
para_0 = np.array([0.55, 30])
para_star = np.array([0.45, 24])

# parameter bound
LB_para = [0.3, 20]
UB_para = [0.6, 35]

# parameter matrix
H_para = interleave_diag(-np.ones(num_para), np.ones(num_para))
h_para = interleave_vec(LB_para, UB_para)

# print("H_para:", H_para)
# print("h_para:", h_para)

# ----------- test dynamics and one step estimation test --------------
x_0 = np.array([20, 96])
u_0 = 100 * np.random.randn(1)

# sample a random disturbance
w = dt * np.random.uniform(-w_lim, w_lim, size=x_dim)

x_p = sacc_dynamics(x_0, u_0, para_star) + w
# x_p2 = vacc_fg(x_0, u_0) - vacc_kernel(x_0).T @ para_star

# print("next state integrated:", x_p)
# print("next state separated:", x_p2)

# initialize our RLS module
var_para = np.array([100, 50])
cov = np.diag(var_para)
my_rls = RLSProjection(num_para, x_dim, sacc_fg, sacc_kernel, dt, H_w)

# para_new = my_rls.update_para(x_p, x_0, u_0, para_0, cov, 1)

# print("new parameter is:", para_new["para"])
# print("the increment is:", para_new["diff"])


# ------------ A simple naive exploration (no control) ------------------

# simulation time
T = 200
T_gap = 1
T_rls = int(T / T_gap)
time_rls = np.arange(0, T_rls + 1)
time_rls_s = np.arange(0, T_rls)
time = dt * np.arange(0, T + 1)
time_s = dt * np.arange(0, T)
# lr = 0.1

para_traj = np.zeros((T_rls + 1, num_para))
x_traj = np.zeros((T + 1, x_dim))
# innovation = np.zeros(T_rls)

para_traj[0, :] = para_0
x_traj[0, :] = x_0

bias_u = 2000
mag_u = 0 * bias_u
t_rls = 0

# main simulation loop
for t in range(T):
    # a random exploration input
    u_temp = bias_u + mag_u * np.random.randn(1)

    # sample a noise
    w = dt * np.random.uniform(-w_lim, w_lim, size=x_dim)

    # obtain the measured next state using the true parameter
    x_traj[t+1, :] = sacc_dynamics(x_traj[t, :], u_temp, para_star) + w

    if (t+1) % T_gap == 0:
        para_info = my_rls.update_para(x_traj[t+1, :], x_traj[t, :], u_temp, para_traj[t_rls, :], cov, t+1)

        # para_traj[t_rls + 1, :] = para_info["para"]
        H_para_next, h_para_next = my_rls.update_paraset(x_traj[t+1, :], x_traj[t, :], u_temp,
                                                         H_para, h_para, t+1)
        para_traj[t_rls + 1, :] = my_rls.projection(H_para_next, h_para_next, para_info["para"])
        cov = para_info["cov"]
        # innovation[t_rls] = para_info["inc_state"]
        t_rls += 1

para_1_traj = para_traj[:, 0]
para_2_traj = para_traj[:, 1]

# Create subplots (3 rows, 1 column)
fig, axes = plt.subplots(2, 1, figsize=(6, 4))

# Plot 1
# axes[0].plot(time_state, v_traj_opt, label='velocity (OPT)')
axes[0].plot(dt * time_rls, para_1_traj, label='estimated drag coefficient')
axes[0].axhline(y=para_star[0], label = 'true drag coefficient', color='r', linestyle='--', linewidth=2)
axes[0].set_title("RLS Performance with Random Exploration")
axes[0].legend()
axes[0].set_facecolor((0.95, 0.95, 0.95))
axes[0].grid(True, linestyle='--', color='white', linewidth=1)

# Plot 2
# axes[1].plot(time_state, D_traj_opt, label='distance (OPT)')
axes[1].plot(dt * time_rls, para_2_traj, label='estimated velocity')
axes[1].axhline(y=para_star[1], label = 'true velocity', color='r', linestyle='--', linewidth=2)
axes[1].legend()
axes[1].set_facecolor((0.95, 0.95, 0.95))
axes[1].grid(True, linestyle='--', color='white', linewidth=1)

# Set x-axis label only on the last plot
axes[1].set_xlabel('Time[s]')

# Improve spacing
plt.tight_layout()
plt.show()
