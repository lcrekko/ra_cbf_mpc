"""
simulation.py

This module contains the simulation function to generate the regret data
for a specific disturbance realization and a given common initial state.
"""

import numpy as np
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
    def __init__(self, controller: MPCController, rls_estimator: RLS_constant, dynamics,
                 x0: np.ndarray, para_star, dim_u,
                 Q, R,
                 dt, T_max):
        """
        The initialization function that used to specify common input
        arguments for the MPC simulations
        Parameters: 
        
        1) controller: MPCController instance [class instance]
        2) rls_estimator: RLS_constant instance [class instance]
        3) dynamics: the dynamic function

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
        self.dynamics = dynamics
        self.x0 = x0
        self.Q = Q
        self.R = R
        self.para_star = para_star
        self.dt = dt
        self.T = T_max

        # get state and input dimensions
        self.dim_x = x0.shape[0]
        self.dim_u = dim_u

        # get parameter dimension
        self.dim_para = para_star.shape[0]
    
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
            x_opt[:, t+1] = self.dynamics(x_opt[:, t], u_opt[:, t],
                                        self.para_star, self.dt, mode="SIM") + self.dt * d
            # Compute the cost
            cost_opt[t] = x_opt[:, t].T @ self.Q @ x_opt[:, t] + u_opt[:, t].T @ self.R @ u_opt[:, t]
        
        return {"x_traj": x_opt, "u_traj": u_opt, "cost": cost_opt}
    
    def learning_mpc_sim(self, disturbance, para_0, H_para, h_para):
        """
        This function simulates the nominal MPC and returns the 
        closed-loop state and input trajectories, and
        the accumulative cost trajectory

        Input:
            1) disturbance: disturbance vector to be applied [nparray X_DIM * T_max]
            2) para_0: initial parameter estimate [nparray]
            3) H_para: initial H matrix describing the parameter set
            4) h_para: initial h vector describing the parameter set
        
        Output: dictionary
            1) x_traj
            2) u_traj
            3) cost
        """
        # Initialize the state and input vector
        x_alg = np.zeros((self.dim_x, self.T+1))
        u_alg = np.zeros((self.dim_u, self.T))

        # Initialize the parameter estimate vector
        para_est = np.zeros((self.dim_para, self.T+1))

        # Initialize the accumulative cost
        cost_alg = np.zeros(self.T)

        # Set the initial state for optimal control
        x_alg[:, 0] = self.x0

        # Set the initial parameter estimate
        para_est[:, 0] = para_0
        H_prior = H_para
        h_prior = h_para

        # ----------- MPC and RLS simulation ----------
        for t in range(self.T):
            # Get the current disturbance
            d = disturbance[:, t]
            
            # Compute the input
            u_alg[:, t] = self.mpc.solve_closed(x_alg[:, t], para_est[:, t])

            # State update
            x_alg[:, t+1] = self.dynamics(x_alg[:, t], u_alg[:, t],
                                         self.para_star, self.dt, mode="SIM") + self.dt * d
            # Compute the cost
            cost_alg[t] = x_alg[:, t].T @ self.Q @ x_alg[:, t] + u_alg[:, t].T @ self.R @ u_alg[:, t]

            # ---------- RLS update ----------
            # prior parameter estimate
            para_prior, _ = self.rls.update_para(x_alg[:, t+1], x_alg[:, t], u_alg[:, t], para_est[:, t], t+1)
            # update the parameter set
            H_post, h_post = self.rls.update_paraset(x_alg[:, t+1], x_alg[:, t], u_alg[:, t], H_prior, h_prior, 1)
            # projection for the posterior parameter estimate
            para_est[:, t+1] = self.rls.projection(H_post, h_post, para_prior)
            
            # update the prior matrix
            H_prior, h_prior = H_post, h_post
        
        return {"x_traj": x_alg, "u_traj": u_alg, "cost": cost_alg, "para_est": para_est}
    
    def regret_final(self, cost_opt, cost_alg):
        """
        Compute the final regret using cumulative sum
        """
        return np.cumsum(cost_alg - cost_opt)
    

