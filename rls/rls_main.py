"""
This script is the main file for RLS (recursive least squares) algorithm.

Additional projection operation imposed by set-membership identification (SMI) is also implemented.
"""

import numpy as np
from optimization_utils.metric import project_onto_feasible_set


class RLS_constant:
    def __init__(self, n_theta, my_mu, my_kernel, f, dt, H_w):
        """
        This is the initialization of the RLS class with a constant covariance

        List of parameters:
        0. n_theta: dimension of the unknown parameter
        1. my_mu: the constant learning rate
        2. my_kernel: the kernel function
        3. f: nominal function in x^+ = f(x) + g(x)u
        4. dt: sampling time
        5. H_w: the matrix describing the disturbance polytope
        """

        # Assign the values and functions
        self.n_theta = n_theta
        self.mu = my_mu
        self.kernel = my_kernel
        self.f = f
        self.dt = dt
        self.H_w = H_w
    
    def update_para(self, x_now, x_pre, u_pre, theta_pre, t: int):
        """
        This is the parameter update function

        List of parameters:
        1. x_now: the current state (x_t)
        2. x_pre: the previous state (x_{t-1})
        3. u_pre: the previous input (u_{t-1})
        4. theta_pre: the previous parameter estimate (hat{theta}_{t-1})
        5. t: the running time [integer type]
        """
        # compute the kernel value
        phi_t = self.kernel(x_pre, u_pre, self.dt)

        # compute the learning rate
        if t == 0:
            mu_t = 0
        else:
            mu_t = np.min( [self.mu, 1 / (np.linalg.norm(phi_t, 2) + 1e-6) ] )

        # compute the estimated state
        # u_polish = u_pre.flatten()

        hat_x_now = self.f(x_pre, u_pre, self.dt) + phi_t.T @ theta_pre

        # parameter update
        theta_add = mu_t * phi_t @ (x_now - hat_x_now)
        theta_now  = theta_pre + theta_add

        return theta_now, theta_add
    
    def update_paraset(self, x_now, x_pre, u_pre, H_theta_pre, h_theta_pre, t: int):
        """
        This is the function used to update the set-membership estimate

        List of parameters:
        1. x_now: the current state (x_t)
        2. x_pre: the previous state (x_{t-1})
        3. u_pre: the previous input (u_{t-1})
        4. H_theta_pre: the old parameter matrix (H_theta)
        5. h_theta_pre: the old parameter vector (h_theta)
        6. t: the running time [integer type]
        """
        if t == 0:
            H_theta_new, h_theta_new = H_theta_pre, h_theta_pre
        else:
            # compute the bias
            # u_polish = np.asarray(u_pre).flatten()

            bias_t = x_now - self.f(x_pre, u_pre, self.dt)

            # compute the added rows for the new matrix and vector
            H_theta_add = -self.H_w @ self.kernel(x_pre, u_pre, self.dt).T
            h_theta_add = np.ones(self.H_w.shape[0]) - self.H_w @ bias_t

            # append and get the new matrix
            H_theta_new = np.vstack((H_theta_pre, H_theta_add))
            h_theta_new = np.hstack((h_theta_pre, h_theta_add))

        return H_theta_new, h_theta_new
    
    def projection(self, H_theta_new, h_theta_new, theta_prior):
        """
        This is the projection operation on the prior parameter estimate

        List of parameters:
        1. H_theta_new: the updated matrix H_theta
        2. h_theta_new: the updated vector h_theta
        3. theta_prior: the prior parameter estimate
        """
        return project_onto_feasible_set(H_theta_new, h_theta_new, theta_prior)



