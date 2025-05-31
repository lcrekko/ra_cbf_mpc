"""
This is regret simulation for adaptive linaer control
"""

import numpy as np
import matplotlib.pyplot as plt
from rls.rls_main import RLS_constant
from rls.rls_utils import interleave_vec, interleave_diag
from nmpc.diverse_functions import linear_dynamics, linear_f, linear_kernel
from nmpc.controller import MPCController
from plotters.simulation import SimulateRegret
plt.rcParams.update({
    "text.usetex": True,                  # Use LaTeX for text rendering
    "font.family": "serif",               # Use a serif font
    "font.serif": ["Computer Modern Roman"] # Set the font
})

# -------------- Basics ---------------

# state and input dimensions
dt = 1 # trivial sampling time
x_dim = 2
u_dim = 1

u_lim = 1

# disturbance information
w_lim = 0.1

# disturbance matrix
H_w = interleave_diag(-w_lim * np.ones(x_dim), w_lim * np.ones(x_dim))

# print("H_w:", H_w)

# --------------- Parameter ---------------
num_para = 3
para_0 = np.array([0.0, 0.0, 0.0])
para_star = np.array([0.8, 0.45, -0.18])

# parameter bound 
LB_para = [-1, -1, -1]
UB_para = [1, 1, 1]

# parameter matrix
H_para = interleave_diag(-np.ones(num_para), np.ones(num_para))
h_para = interleave_vec(LB_para, UB_para)

# print("H_para:", H_para)
# print("h_para:", h_para)

# --------------- Test dynamics and kernel function ---------------
# initial state
x_0 = np.array([2, 3])  # specify the initial state

# --------------- MPC Setting ----------------
# prediction horizon
N = 10

# weighting matrices
Q = np.array([[1, 0], [0, 1]]) # the distance D is not heavily penalized
R = np.array([[1]]) # the input weakly penalized
P = np.array([[1.467, 0.207], [0.207, 1.731]])

# reference state and input (set-point tracking)
x_ref = np.array([0, 0])
u_ref = np.array([0])

# ------------- MPC --------------

# initialize the MPC controller
my_mpc = MPCController(N, dt,
                       x_dim, u_dim,
                       x_ref, u_ref,
                       -u_lim, u_lim,
                       Q, R, P,
                       linear_dynamics, num_para)

# ------------- RLS --------------
my_mu = 10
my_rls = RLS_constant(num_para, my_mu, linear_kernel, linear_f, dt, H_w)

T_sim = 100
N_sim = 100
time_regret = dt * np.arange(0, T_sim)
w_sim = np.random.uniform(-w_lim, w_lim, size=(x_dim, T_sim, N_sim))

my_sim = SimulateRegret(my_mpc, my_rls, linear_dynamics,
                        x_0, para_star, u_dim,
                        Q, R,
                        dt, T_sim)

# regret = np.zeros((N_sim, T_sim))

# for i in range(T_sim):
#     out_sim_nominal = my_sim.nominal_mpc_sim(w_sim[:,:,i])
#     out_sim_learning = my_sim.learning_mpc_sim(w_sim[:,:,i], para_0, H_para, h_para)
#     regret[i, :] = my_sim.regret_final(out_sim_nominal["cost"], out_sim_learning["cost"])

# np.save('regret.npy', regret)

# reload the data for fast plotting
regret = np.load('regret.npy')

mean_curve = np.mean(regret, axis=0)
max_curve = np.max(regret, axis=0)
min_curve = np.min(regret, axis=0)

# Plotting
fig_w = 5
fig_h = fig_w * 0.5 * (np.sqrt(5) - 1)
plt.figure(figsize=(fig_w, fig_h))
plt.plot(time_regret, mean_curve, label='Mean', color='blue')
plt.fill_between(time_regret, min_curve, max_curve,
                 color='lightblue', alpha=0.5, label='Min-Max Envelope')
# plt.title(r'Function $f(t) = 0.3047 \sqrt{t} + w(t)$ with Bounded Stochastic Noise')
plt.xlabel(r'$T$', fontsize=20)
plt.ylabel(r'$\mathrm{Reg}_T$', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
plt.grid(True)
plt.tight_layout()

plt.savefig("regret.pdf", format="pdf", dpi=300, bbox_inches='tight')
plt.show()
