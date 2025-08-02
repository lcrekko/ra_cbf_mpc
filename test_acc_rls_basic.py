"""
This is test script for ACC model in our case
"""

import numpy as np
import matplotlib.pyplot as plt
from rls.rls_main import RLSProjection
from rls.rls_utils import interleave_vec, interleave_diag
from nmpc.diverse_functions import sacc_dynamics, sacc_fg, sacc_kernel
from nmpc.controller import MPCController
from optimization_utils.metric import max_l1_deviation_value

# --------------- Basic setting ---------------
# state and input dimensions
dt = 0.1 # trivial sampling time
x_dim = 2
u_dim = 1

# The range of maximum traction force [Newton]
u_lim = 1e4

#  --------------- Disturbance information -----------------
w_lim = 5 * np.array([0.1, 1])

# disturbance matrix
H_w = interleave_diag(-w_lim, w_lim)

# print("H_w:", H_w)

# ---------------- Parameter information -----------------
num_para = 2
para_0 = np.array([0.55, 30])
para_star = np.array([0.45, 24])

# parameter bound
LB_para = [0.3, 20]
UB_para = [0.6, 35]

# parameter matrix
H_para = interleave_diag(-np.ones(num_para), np.ones(num_para))
h_para = interleave_vec(LB_para, UB_para)

# initial error bound
bound_0 = max_l1_deviation_value(H_para, h_para, para_0)

# print("H_para:", H_para)
# print("h_para:", h_para)

# ----------- Test dynamics and one step estimation test --------------
# initial state
x_0 = np.array([20, 96])
# u_0 = 100 * np.random.randn(1)

# # sample a random disturbance
# w = dt * np.random.uniform(-w_lim, w_lim, size=x_dim)

# x_p = sacc_dynamics(x_0, u_0, para_star) + w
# # x_p2 = vacc_fg(x_0, u_0) - vacc_kernel(x_0).T @ para_star

# print("next state integrated:", x_p)
# print("next state separated:", x_p2)

# ----------- initialize our RLS module ------------
# initial covaraince matrix
var_para = np.array([100, 50])
cov = np.diag(var_para)
# define the estimator
my_rls = RLSProjection(num_para, x_dim, sacc_fg, sacc_kernel, dt, H_w)
# --------------------------------------------------

# ----------- initialize our MPC module -------------
# cost weights
Q = np.array([[100, 0], [0, 1e-2]])
R = np.array([5 * 1e-4]) # low weights, we do not care about the input
P = 2 * Q
# state reference
v_ref = 30
d_ref = v_ref * 1.8
x_ref = np.array([v_ref, d_ref])

# input reference
u_static = 150
u_ref = np.array([u_static])

# prediction horizon
T_prediction = 1
N_prediction = int(T_prediction / dt)

# define the controller
my_mpc = MPCController(N_prediction, dt,
                       x_dim, u_dim,
                       x_ref, u_ref,
                       -u_lim, u_lim,
                       Q, R, P,
                       sacc_dynamics, num_para)
# ----------------------------------------------------

# para_new = my_rls.update_para(x_p, x_0, u_0, para_0, cov, 1)

# print("new parameter is:", para_new["para"])
# print("the increment is:", para_new["diff"])


# ------------ Simulation ------------------

# simulation time
T = 500
# T_gap = 1
# T_rls = int(T / T_gap)
# time_rls = np.arange(0, T_rls + 1)
# time_rls_s = np.arange(0, T_rls)
time = dt * np.arange(0, T + 1)
time_s = dt * np.arange(0, T)
# lr = 0.1

# Initialize the output trajectory
x_traj = np.zeros((T + 1, x_dim))
u_traj = np.zeros((T, u_dim))
para_traj = np.zeros((T + 1, num_para))
bound_traj = np.zeros(T + 1)
# innovation = np.zeros(T_rls)

para_traj[0, :] = para_0
x_traj[0, :] = x_0
bound_traj[0] = bound_0

# main simulation loop
for t in range(T):
    # a random exploration input
    u_traj[t, :] = my_mpc.solve_closed(x_traj[t, :], para_traj[t, :])

    # sample a noise
    w = dt * np.random.uniform(-w_lim, w_lim, size=x_dim)

    # obtain the measured next state using the true parameter
    x_traj[t+1, :] = sacc_dynamics(x_traj[t, :], u_traj[t, :], para_star) + w

    para_info = my_rls.update_para(x_traj[t+1, :], x_traj[t, :], u_traj[t, :], para_traj[t, :], cov, t+1)

    # para_traj[t_rls + 1, :] = para_info["para"]
    H_para_next, h_para_next = my_rls.update_paraset(x_traj[t+1, :], x_traj[t, :], u_traj[t, :],
                                                         H_para, h_para, t+1)
    para_post = my_rls.posterior(H_para_next, h_para_next, para_info["para"], para_traj[t, :])
    para_traj[t + 1, :] = para_post["para"]
    bound_traj[t + 1] = np.random.uniform(1, 3)*np.linalg.norm(para_traj[t + 1, :] - para_star, ord=2)
    cov = para_info["cov"]
    # innovation[t_rls] = para_info["inc_state"]

# --------- State and input subplots (3 rows, 1 column)
fig_sys, axes_sys = plt.subplots(3, 1, figsize=(6, 6))

# Plot 1
# axes[0].plot(time_state, v_traj_opt, label='velocity (OPT)')
axes_sys[0].plot(dt * time, x_traj[:, 0], label='ego vehicle velocity')
# axes_sys[0].axhline(y=para_star[0], label = 'true drag coefficient', color='r', linestyle='--', linewidth=2)
axes_sys[0].set_title("RLS Estimation with SMID")
axes_sys[0].legend()
axes_sys[0].set_facecolor((0.95, 0.95, 0.95))
axes_sys[0].grid(True, linestyle='--', color='white', linewidth=1)

# Plot 2
# axes[1].plot(time_state, D_traj_opt, label='distance (OPT)')
axes_sys[1].plot(dt * time, x_traj[:, 1], label='distance')
# axes_sys[1].axhline(y=para_star[1], label = 'true velocity', color='r', linestyle='--', linewidth=2)
axes_sys[1].legend()
axes_sys[1].set_facecolor((0.95, 0.95, 0.95))
axes_sys[1].grid(True, linestyle='--', color='white', linewidth=1)

# Plot 3
axes_sys[2].plot(dt * time_s, u_traj[:, 0], label='traction force')
# axes[1].axhline(y=para_star[1], label = 'true velocity', color='r', linestyle='--', linewidth=2)
axes_sys[2].legend()
axes_sys[2].set_facecolor((0.95, 0.95, 0.95))
axes_sys[2].grid(True, linestyle='--', color='white', linewidth=1)

# Set x-axis label only on the last plot
axes_sys[2].set_xlabel('Time[s]')

plt.tight_layout()

# --------- RLS subplots (3 rows, 1 column) -----------
fig_rls, axes_rls = plt.subplots(3, 1, figsize=(6, 6))

# Plot 1
# axes[0].plot(time_state, v_traj_opt, label='velocity (OPT)')
axes_rls[0].plot(dt * time, para_traj[:, 0], label='estimated drag coefficient')
axes_rls[0].axhline(y=para_star[0], label = 'true drag coefficient', color='r', linestyle='--', linewidth=2)
axes_rls[0].set_title("RLS Estimation with SMID")
axes_rls[0].legend()
axes_rls[0].set_facecolor((0.95, 0.95, 0.95))
axes_rls[0].grid(True, linestyle='--', color='white', linewidth=1)

# Plot 2
# axes[1].plot(time_state, D_traj_opt, label='distance (OPT)')
axes_rls[1].plot(dt * time, para_traj[:, 1], label='estimated velocity')
axes_rls[1].axhline(y=para_star[1], label = 'true velocity', color='r', linestyle='--', linewidth=2)
axes_rls[1].legend()
axes_rls[1].set_facecolor((0.95, 0.95, 0.95))
axes_rls[1].grid(True, linestyle='--', color='white', linewidth=1)

# Plot 3
axes_rls[2].plot(dt * time, bound_traj, label='error bound')
# axes[1].axhline(y=para_star[1], label = 'true velocity', color='r', linestyle='--', linewidth=2)
axes_rls[2].legend()
axes_rls[2].set_facecolor((0.95, 0.95, 0.95))
axes_rls[2].grid(True, linestyle='--', color='white', linewidth=1)

# Set x-axis label only on the last plot
axes_rls[2].set_xlabel('Time[s]')

# Improve spacing
plt.tight_layout()
plt.show()
