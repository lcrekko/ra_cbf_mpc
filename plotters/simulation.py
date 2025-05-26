"""
simulation.py

This module contains the simulation function to generate the regret data
for a specific disturbance realization and a given common initial state.
"""

import numpy as np
from nmpc.diverse_functions import acc_dynamics, acc_kernel, acc_f, acc_g

def simulate_regret(controller, rls_estimator, 
                    x0, para_0, para_star, H_para, h_para,
                    Q, R, 
                    disturbance, dt, T_max):
    """
    This function simulates the system dynamics over a specified time horizon
    using the MPC controller and the RLS estimator to generate regret data.
    Parameters: 
        controller: MPCController instance [class instance]
        rls_estimator: RLS_constant instance [class instance]

        x0: initial state vector [nparray]
        para_0: initial parameter estimate [nparray]
        para_star: true parameter values [nparray]
        H_para: parameter matrix for RLS [nparray 2D]
        h_para: parameter bounds for RLS [nparray 1D]

        Q: state weighting matrix [nparray 2D]
        R: input weighting matrix [nparray 1D]
        
        disturbance: disturbance vector to be applied [nparray X_DIM * T_max]
        dt: sampling time [float]
        T_max: maximum simulation time [int]
    """
    # Initialize state and parameter estimates with dummy values
    x_alg = x0
    x_opt = x0
    para = para_0
    H = H_para
    h = h_para

    # Initialize the cumulative regret
    regret = np.zeros(T_max)

    for t in range(T_max):
        # Get the current disturbance
        d = disturbance[:, t]

        # ----------- Control Input Computation -----------
        u_alg = controller.solve_closed(x_alg, para) # certainty-equivalent learning-based mpc
        
        u_opt = controller.solve_closed(x_opt, para_star) # expert mpc

        # ----------- State Update ------------
        # Both updates use the true dynamics
        x_next_alg = acc_dynamics(x_alg, u_alg, dt, para_star, mode="SIM")

        x_next_opt = acc_dynamics(x_opt, u_opt, dt, para_star, mode="SIM")

        # ----------- RLS Update -----------
        # update the parameter estimate and parameter set estiamte
        para_prior, _ = rls_estimator.update_para(x_next_alg, x_alg, u_alg, para, t+1)
        H, h = rls_estimator.update_paraset(x_next_alg, x_alg, u_alg, H, h, t+1)
        para = rls_estimator.projection(H, h, para_prior)

        # Update the state for the next iteration
        x_alg = x_next_alg
        x_opt = x_next_opt

