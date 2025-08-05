"""
This is main script for ACC using MPC, RLS with SMID, and safety filter
"""

import numpy as np
import matplotlib.pyplot as plt
from rls.rls_main import RLSProjection
from rls.rls_utils import interleave_vec, interleave_diag
from nmpc.diverse_functions import sacc_dynamics, sacc_fg, sacc_kernel
from nmpc.controller import MPCController
from cbf_sf.diverse_functions import ext_kappa, cbf_acc_linear
from cbf_sf.safety_filter import AdaptiveSafetyFilter
from optimization_utils.metric import max_l1_deviation_value, max_2norm_polytope
from plotters.utils import plotter_kernel
plt.rcParams.update({
    "text.usetex": True,                  # Use LaTeX for text rendering
    "font.family": "serif",               # Use a serif font
    "font.serif": ["Computer Modern Roman"] # Set the font
})

# --------------- Basic setting ---------------
# state and input dimensions
dt = 0.1 # trivial sampling time
x_dim = 2
u_dim = 1

# The range of maximum traction force [Newton]
u_lim = 1e4

#  --------------- Disturbance information -----------------
w_lim = np.array([0.1, 1])

# disturbance matrix
H_w = interleave_diag(-w_lim, w_lim)

# disturbance bound (2-norm)
bar_w  = max_2norm_polytope(H_w, np.ones(2 * x_dim))

# ---------------- Parameter information -----------------
num_para = 2
para_0 = np.array([0.55, 28])
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

# ----------- initialize the RLS module ------------
# define the estimator
my_rls = RLSProjection(num_para, x_dim, sacc_fg, sacc_kernel, dt, H_w)
# --------------------------------------------------

# ----------- initialize the MPC module -------------
# cost weights
Q = np.array([[100, 0], [0, 1e-2]])
R = np.array([5 * 1e-4]) # low weights, we do not care about the input
P = 2 * Q
# state reference
v_ref = 26
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

# ----------- initialize the SF module ------------
# the minimum eigenvalue
gamma = 1e3

# compute the Lipschitz constant (linear barrier function)
L_B = np.linalg.norm(np.array([1, 1.8]))

# define the safety filter
my_sf = AdaptiveSafetyFilter(dt, num_para,
                             gamma, L_B, bar_w,
                             x_dim, u_dim,
                             -u_lim, u_lim,
                             sacc_dynamics, sacc_kernel, cbf_acc_linear,
                             ext_kappa, "linear")

# --------------------------------------------------

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

# number of disturbance realizations for simulation
N_sim = 100

# Initialize the output trajectory for adaptive safe control
xa_traj = np.zeros((N_sim, T + 1, x_dim)) # state
# ua_traj = np.zeros((N_sim, T, u_dim)) # nominal MPC input
uasf_traj = np.zeros((N_sim, T, u_dim)) # filtered safe input
parasf_traj = np.zeros((N_sim, T + 1, num_para)) # parameter
boundsf_traj = np.zeros((N_sim, T + 1)) # parameter error bound

# Initialize the output trajectory for nominal safe control
xa_traj = np.zeros((N_sim, T + 1, x_dim)) # state
ua_traj = np.zeros((N_sim, T, u_dim)) # nominal MPC input
uasf_traj = np.zeros((N_sim, T, u_dim)) # filtered safe input
parasf_traj = np.zeros((N_sim, T + 1, num_para)) # parameter
boundsf_traj = np.zeros((N_sim, T + 1)) # parameter error bound

# Initialize the output trajectory for unsafe control
x_traj = np.zeros((N_sim, T + 1, x_dim)) # state
u_traj = np.zeros((N_sim, T, u_dim)) # nominal MPC input
# usf_traj = np.zeros((N_sim, T, u_dim)) # filtered safe input
para_traj = np.zeros((N_sim, T + 1, num_para)) # parameter
bound_traj = np.zeros((N_sim, T + 1)) # parameter error bound

# Simulation main loops
for i in range(N_sim):
# outer loop for different realizations
    # Initialize the state, and initial estimation
    print("round:", i)
    xa_traj[i, 0, :] = x_0
    x_traj[i, 0, :] = x_0
    parasf_traj[i, 0, :] = para_0
    boundsf_traj[i, 0] = bound_0
    para_traj[i, 0, :] = para_0
    # bound_traj[i, 0] = bound_0

    # Initialize the first parameter difference (no update)
    diff_para = 0

    # sample a disturbance trajectory
    w = dt * np.random.uniform(-w_lim, w_lim, size=(T, x_dim))

    # initial covaraince matrix
    var_para = np.array([100, 50])
    cov_sf = np.diag(var_para)
    cov = np.diag(var_para)

    # reset the parameter matrix
    H_para_sf = interleave_diag(-np.ones(num_para), np.ones(num_para))
    h_para_sf = interleave_vec(LB_para, UB_para)

    H_para = interleave_diag(-np.ones(num_para), np.ones(num_para))
    h_para = interleave_vec(LB_para, UB_para)

    for t in range(T):
    # inner loop for time simulation
        # adaptive MPC input
        ua_traj[i, t, :] = my_mpc.solve_closed(xa_traj[i, t, :], parasf_traj[i, t, :])
        # nominal MPC input
        u_traj[i, t, :] = my_mpc.solve_closed(x_traj[i, t, :], para_traj[i, t, :])

        # safe input by applying adaptive safety filter
        uasf_traj[i, t, :] = my_sf.filter(xa_traj[i, t, :], parasf_traj[i, t, :], ua_traj[i, t, :],
                                          diff_para, boundsf_traj[i, t])
        # safe input by applying nominal safety filter
        #usf_traj[i, t, :] = my_sf.filter(x_traj[i, t, :], para_0, u_traj[i, t, :], 0, bound_0)

        # obtain the measured next state using the true parameter
        xa_traj[i, t+1, :] = sacc_dynamics(xa_traj[i, t, :], uasf_traj[i, t, :], para_star) + w[t, :]
        x_traj[i, t+1, :] = sacc_dynamics(x_traj[i, t, :], u_traj[i, t, :], para_star) + w[t, :]

        para_info_sf = my_rls.update_para(xa_traj[i, t+1, :], xa_traj[i, t, :], 
                                       uasf_traj[i, t, :], parasf_traj[i, t, :], cov, t+1)
        para_info = my_rls.update_para(x_traj[i, t+1, :], x_traj[i, t, :], 
                                       u_traj[i, t, :], para_traj[i, t, :], cov, t+1)

        # para_traj[t_rls + 1, :] = para_info["para"]
        H_para_next_sf, h_para_next_sf = my_rls.update_paraset(xa_traj[i, t+1, :], xa_traj[i, t, :], uasf_traj[i, t, :],
                                                            H_para_sf, h_para_sf, t+1)
        H_para_next, h_para_next = my_rls.update_paraset(x_traj[i, t+1, :], x_traj[i, t, :], u_traj[i, t, :],
                                                            H_para, h_para, t+1)
        para_post_sf = my_rls.posterior(H_para_next_sf, h_para_next_sf, 
                                        para_info_sf["para"], parasf_traj[i, t, :])
        para_post = my_rls.posterior(H_para_next, h_para_next, 
                                        para_info["para"], para_traj[i, t, :])
        parasf_traj[i, t+1, :] = para_post_sf["para"]
        para_traj[i, t+1, :] = para_post["para"]
        diff_para = para_post_sf["delta"]
        # boundsf_traj[i, t+1] = np.random.uniform(1, 3)*np.linalg.norm(parasf_traj[i, t+1, :] - para_star, ord=2)
        # bound_traj[i, t+1] = np.random.uniform(1, 3)*np.linalg.norm(para_traj[i, t+1, :] - para_star, ord=2)
        boundsf_traj[i, t+1] = para_post_sf["diff"]
        bound_traj[i, t+1] = para_post_sf["diff"]
        cov_sf = para_info_sf["cov"]
        cov = para_info["cov"]
        # innovation[t_rls] = para_info["inc_state"]

# # save data for reuse
# np.savez('data_safe.npz', state=xa_traj, input=uasf_traj, para=parasf_traj, bound=boundsf_traj)
# np.savez('data.npz', state=x_traj, input=u_traj, para=para_traj, bound=bound_traj)

data_safe = np.load("data_safe.npz")
data = np.load("data.npz")

xa_traj = data_safe["state"]
x_traj = data["state"]

uasf_traj = data_safe["input"]
u_traj = data["input"]

# Plotting parameters
my_linewidth = 1.2
myred = (0.894, 0.443, 0.349)
myblue = (0.239, 0.361, 0.435)
legend_1 = "aMPC-raCBF"
legend_2 = "aMPC"
linestyle_1 = '-'
linestyle_2 = '--'

# --------- State and input subplots (3 rows, 1 column)
fig_sys, axes_sys = plt.subplots(3, 1, figsize=(4.5, 4.5))

# Plot 1
# axes[0].plot(time_state, v_traj_opt, label='velocity (OPT)')
plotter_kernel(axes_sys[0], dt * time, xa_traj[:, :, 0],
               legend_1, my_linewidth, myblue, linestyle_1)
plotter_kernel(axes_sys[0], dt * time, x_traj[:, :, 0],
               legend_2, my_linewidth, myred, linestyle_2)
# axes_sys[0].axhline(y=para_star[0], label = 'true drag coefficient', color='r', linestyle='--', linewidth=2)
# axes_sys[0].set_title("Adaptive MPC with CBF-based safety filter")
axes_sys[0].legend()
axes_sys[0].set_facecolor((0.95, 0.95, 0.95))
axes_sys[0].set_ylabel(r'$v$[m/s]')
axes_sys[0].grid(True, linestyle='--', color='white', linewidth=1)

# Plot 2
# axes[1].plot(time_state, D_traj_opt, label='distance (OPT)')
plotter_kernel(axes_sys[1], dt * time, xa_traj[:, :, 1] - 1.8*xa_traj[:, :, 0], 
               legend_1, my_linewidth, myblue, linestyle_1)
plotter_kernel(axes_sys[1], dt * time, x_traj[:, :, 1] - 1.8*x_traj[:, :, 0],
               legend_2, my_linewidth, myred, linestyle_2)
# axes_sys[1].axhline(y=para_star[1], label = 'true velocity', color='r', linestyle='--', linewidth=2)
axes_sys[1].legend()
axes_sys[1].set_facecolor((0.95, 0.95, 0.95))
axes_sys[1].set_ylabel(r'$d - 1.8v$[m]')
axes_sys[1].grid(True, linestyle='--', color='white', linewidth=1)

# Plot 3
plotter_kernel(axes_sys[2], dt * time_s, uasf_traj[:, :, 0],
               legend_1, my_linewidth, myblue, linestyle_1)
plotter_kernel(axes_sys[2], dt * time_s, u_traj[:, :, 0],
               legend_2, my_linewidth, myred, linestyle_2)
# axes[1].axhline(y=para_star[1], label = 'true velocity', color='r', linestyle='--', linewidth=2)
axes_sys[2].legend()
axes_sys[2].set_facecolor((0.95, 0.95, 0.95))
axes_sys[2].grid(True, linestyle='--', color='white', linewidth=1)
axes_sys[2].set_ylabel(r'$u$[N]')

# Set x-axis label only on the last plot
axes_sys[2].set_xlabel('Time[s]')

plt.tight_layout()

# --------- RLS subplots (3 rows, 1 column) -----------
# fig_rls, axes_rls = plt.subplots(3, 1, figsize=(6, 6))

# # Plot 1
# # axes[0].plot(time_state, v_traj_opt, label='velocity (OPT)')
# axes_rls[0].plot(dt * time, para_traj[:, 0], label=r'$\hat{\mu}_{\mathrm{aero}}$')
# axes_rls[0].axhline(y=para_star[0], label = r'$\mu^\ast_{\mathrm{aero}}$', color='r', linestyle='--', linewidth=2)
# axes_rls[0].set_title("RLS estimation with SMID")
# axes_rls[0].legend()
# axes_rls[0].set_facecolor((0.95, 0.95, 0.95))
# axes_rls[0].grid(True, linestyle='--', color='white', linewidth=1)

# # Plot 2
# # axes[1].plot(time_state, D_traj_opt, label='distance (OPT)')
# axes_rls[1].plot(dt * time, para_traj[:, 1], label=r'$\hat{v}_{\mathrm{f}}$')
# axes_rls[1].axhline(y=para_star[1], label = r'$v^\ast_{\mathrm{f}}$', color='r', linestyle='--', linewidth=2)
# axes_rls[1].legend()
# axes_rls[1].set_facecolor((0.95, 0.95, 0.95))
# axes_rls[1].grid(True, linestyle='--', color='white', linewidth=1)

# # Plot 3
# axes_rls[2].plot(dt * time, bound_traj, label=r'$\varepsilon_{\theta,t}(1)$')
# # axes[1].axhline(y=para_star[1], label = 'true velocity', color='r', linestyle='--', linewidth=2)
# axes_rls[2].legend()
# axes_rls[2].set_facecolor((0.95, 0.95, 0.95))
# axes_rls[2].grid(True, linestyle='--', color='white', linewidth=1)

# # Set x-axis label only on the last plot
# axes_rls[2].set_xlabel('Time[s]')

# # Improve spacing
# plt.tight_layout()
plt.savefig('cbf_performance.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.show()
