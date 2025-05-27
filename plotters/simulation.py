"""
simulation.py

This module contains the simulation function to generate the regret data
for a specific disturbance realization and a given common initial state.
"""

import numpy as np
from nmpc.diverse_functions import acc_dynamics
from nmpc.controller import MPCController
from rls.rls_main import RLS_constant

class SimulateRegret():
    """
    This is the class for simulating the regret, it has 4 functions

    1) initialization
    1) nominal MPC simulator
    2) adaptive MPC simulator
    3) regret calculator
    """
    def __init__(self, controller: MPCController, rls_estimator: RLS_constant,
                 x0: np.ndarray, para_star, dim_u,
                 Q, R,
                 dt, T_max):
        """
        The initialization function that used to specify common input
        arguments for the MPC simulations
        Parameters: 
        
        1) controller: MPCController instance [class instance]
        2) rls_estimator: RLS_constant instance [class instance]

        3) x0: initial state vector [nparray]
        4) para_star: true parameter values [nparray]
        5) dim_u: input dimension [int]

        6) Q: state weighting matrix [nparray]
        7) R: input weighting matrix [nparray]
        
        8) dt: sampling time [float]
        9) T_max: simulation step [int]
        """

        # passing parameters
        self.mpc = controller
        self.rls = rls_estimator
        self.x0 = x0
        self.Q = Q
        self.R = R
        self.para_star = para_star
        self.dt = dt
        self.T = T_max

        # other useful constants
        self.dim_x = x0.shape[0]
        self.dim_u = dim_u
    
    def nominal_mpc_sim(self, disturbance):
        """
        This function simulates the nominal MPC and returns the 
        closed-loop state and input trajectories, and
        the accumulative cost trajectory

        Input:
            1) disturbance: disturbance vector to be applied [nparray X_DIM * T_max]
        
        Output: dictionary
            1) x_traj
            2) u_traj
            3) cost
        """
        # Initialize the state and input vector
        x_opt = np.zeros((self.dim_x, self.T+1))
        u_opt = np.zeros((self.dim_u, self.T))

        # Initialize the accumulative cost
        cost_opt = np.zeros(self.T)

        # Set the initial state for optimal control
        x_opt[:, 0] = self.x0

        # ----------- MPC simulation ----------
        for t in range(self.T):
            # Get the current disturbance
            d = disturbance[:, t]
            
            # Compute the input
            u_opt[:, t] = self.mpc.solve_closed(x_opt[:, t], self.para_star) # expert mpc

            # State update
            x_opt[:, t+1] = acc_dynamics(x_opt[:, t], u_opt[:, t],
                                         self.dt, self.para_star, mode="SIM") + self.dt * d
            cost_opt[t] = x_opt[:, t].T @ self.Q @ x_opt[:, t] + u_opt[:, t].T @ self.R @ u_opt[:, t]
        
        return {"x_traj": x_opt, "u_traj": u_opt, "cost": cost_opt}
    

