"""
safety_filter.py

This module contains
(1) the basic SafetyFilter class that
implements a safety filter that generates a safe input for discrete-time systems;

The filter uses CasADi for symbolic modeling and the optimizer within CasADi.
"""

import casadi as ca
import numpy as np

class AdaptiveSafetyFilter():
    """
    This is the adaptive safety filter class, it has two parts
    1. Initialization and defining the NLP optimization problem
    2. Solve the NLP and return the filtered safe control input
    """
    def __init__(self, dt, num_para, 
                 gamma, L_B, bar_w,
                 x_dim, u_dim,
                 u_min, u_max,
                 dynamics, cbf, kappa) -> None:
        """
        Initializing the safety filter.

        Parameters:
        1. dt: sampling time
        2. num_para: [int] number of parameters in the dynamics model
        3. gamma: the minimum eigenvalue of the weighting matrix
        4. L_B: Lipschitz constant of the CBF
        5. bar_w: disturbance norm
        6. x_dim: [int] dimension of the state vector
        7. u_dim: [int] dimension of the control input
        8. u_min: input limit (lower bound)
        9. u_max: input limit (upper bound)
        10. dynamics: [function] parametric system model
        11. cbf: [function] control barrier function
        12. kappa: [function] the extended class-kappa function
        """

        # Create an Opti instance
        self.opti = ca.Opti()

        # Decision variables: control input
        self.usf = self.opti.variable(u_dim)

        # ---------- Parameters -----------
        # 1. nominal input in the objective function
        self.unom = self.opti.parameter(u_dim)
        # 2. system state
        self.x = self.opti.parameter(x_dim)
        # 3. parameter of the dynamics model
        self.para = self.opti.parameter(num_para)
        # 4. increment
        self.diff = self.opti.parameter(1)
        # 5. error bound
        self.bound = self.opti.parameter(1)