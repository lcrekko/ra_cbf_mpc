"""
simulation.py

This module contains the simulation function to generate the regret data
for a specific disturbance realization and a given common initial state.
"""

import numpy as np
from nmpc.diverse_functions import acc_dynamics
from nmpc.controller import MPCController
from rls.rls_main import RLS_constant

def simulate_regret(controller: MPCController, rls_estimator: RLS_constant,
                    x0: np.ndarray, para_0, para_star, H_para, h_para,
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
    dim_x = x0.shape[0]  # Dimension of the state vector
    x_alg = np.zeros((dim_x, T_max+1))  # Adaptive control state
    x_opt = np.zeros((dim_x, T_max+1))
    x_alg[:, 0] = x0  # Set the initial state for adaptive control
    x_opt[:, 0] = x0  # Set the initial state for optimal control
    para = para_0
    H = H_para
    h = h_para

    # Initialize the cumulative regret
    regret = np.zeros(T_max)

    for t in range(T_max):
        # Get the current disturbance
        d = disturbance[:, t]

        # ----------- Control Input Computation -----------
        u_alg = controller.solve_closed(x_alg[:, t], para) # certainty-equivalent learning-based mpc
        u_alg = np.atleast_1d(u_alg)
        
        u_opt = controller.solve_closed(x_alg[:, t], para_star) # expert mpc
        u_opt = np.atleast_1d(u_opt)

        # ----------- State Update ------------
        # Both updates use the true dynamics and the common disturbance
        x_alg[:, t+1] = acc_dynamics(x_alg[:, t], u_alg, dt, para_star, mode="SIM") + dt * d

        x_opt[:, t+1] = acc_dynamics(x_opt[:, t], u_opt, dt, para_star, mode="SIM") + dt * d

        # ----------- RLS Update -----------
        # update the parameter estimate and parameter set estiamte
        para_prior, _ = rls_estimator.update_para(x_alg[:, t+1], x_alg[:, t], u_alg, para, t+1)
        H_new, h_new = rls_estimator.update_paraset(x_alg[:, t+1], x_alg[:, t], u_alg, H, h, t+1)
        para = rls_estimator.projection(H, h, para_prior)

        # ----------- Regret Calculation -----------
        cost_alg = x_alg[:, t].T @ Q @ x_alg[:, t] + u_alg.T @ R @ u_alg
        cost_opt = x_opt[:, t].T @ Q @ x_opt[:, t] + u_opt.T @ R @ u_opt

        regret[t] = regret[t-1] + cost_alg - cost_opt

        # Update the matrices for describing the uncertainty set
        H = H_new
        h = h_new

        return {"regret": regret, "x_alg": x_alg, "x_opt": x_opt}

