"""
diverse_functions.py

This module contains various basic functions used in the simulation

1. model of the dynamical system

Due to that the original dynamics are in continuous time, sampling time
is included as an additional parameter, with default value

dt = 0.01

and the Euler discretization method is used

The dynamical system also contains the current estimated unknown parameters,
the number of the parameters depends on the used model.


Two models will be considered:

1) Adaptive cruise control (number of parameters: 3)

state: x_1 -> velocity
        x_2 -> distance between the leading vehicle
input: u -> acceleration
2) Adaptive pitch control of aircraft (number of parameters: 2)


"""

import casadi as ca
import numpy as np

## ----------------------- Adaptive Cruise Control ----------------------------

def acc_dynamics(x, u, dt = 0.01, para=np.zeros(3), mode="SIM"):
    """
    Compute the next state based on the NOMINAL discrete-time model.

    Parameters:
        x: casadi.SX or MX, current state vector.
        u: casadi.SX or MX, control input vector.
        bias: ndarray, estimated system parameters
        mode: "NLP" or "SIM" depending on the purpose of usage:
            1) "NLP" for optimization in MPC
            2) "SIM" for simulation and general Numpy based calculations

    Returns:
        x_next: casadi.SX or MX, next state vector. (Or just numpy array)
    """
    m = 500.0 # the mass of the vehicle [kg]
    v_0 = 36.0 # the speed of the leading vehicle on the autoweg [m /s ]
    if mode == "NLP":
        dot_v = 0 - 1 / m * (para[0] + para[1] * x[0] + para[2] * (x[0] ** 2)) + 1 / m * u
        dot_D = v_0 - x[0]
        x_next = ca.vertcat(x[0] + dot_v * dt, x[1] + dot_D * dt)
    elif mode == "SIM":
        dot_v = 0 - 1 / m * (para[0] + para[1] * x[0] + para[2] * (x[0] ** 2)) + 1 / m * u[0]
        dot_D = v_0 - x[0]
        x_next = np.array([x[0] + dot_v * dt, x[1] + dot_D * dt])
    else:
        raise ValueError("Invalid input! Please use 'NLP' for optimizatoin or 'SIM' for simulation.")

    return x_next

def acc_f(x, dt = 0.01, mode="SIM"):
    """
    This is the drift term of the dynamics

    Input
    1) x: state
    2) dt: sampling time
    """
    v_0 = 36 # front vehicle speed

    dot_v_x = 0
    dot_D_x = v_0 - x[0]

    if mode == "NLP":
        return ca.vertcat(x[0] + dt * dot_v_x, x[1] + dt * dot_D_x)
    elif mode == "SIM":
        return np.array([x[0] + dt * dot_v_x, x[1] + dt * dot_D_x])
    else:
        raise ValueError("Invalid input! Please use 'NLP' for optimizatoin or 'SIM' for simulation.")

def acc_g(x, dt = 0.01, mode="SIM"):
    """
    This is the input coupling term of the dynamics

    Input
    1) x: state
    2) dt: sampling time
    """
    m = 500.0 # vehicle mass

    dot_v_u = 1 / m
    dot_D_u = 0

    if mode == "NLP":
        return ca.vertcat(dt * dot_v_u, dt * dot_D_u)
    elif mode == "SIM":
        g_prior = np.array([dt * dot_v_u, dt * dot_D_u])
        g_post = g_prior.reshape(g_prior.shape[0], 1)

        return g_post
    else:
        raise ValueError("Invalid input! Please use 'NLP' for optimizatoin or 'SIM' for simulation.")

def acc_kernel(x, dt = 0.01, mode="SIM"):
    """
    This is the kernel of the dynamics

    Input
    1) x: state
    2) dt: sampling time

    REMARK: be careful about the transpose given in the system dynamics.
            So, here it needs to be tranposed again

    """
    if mode == "NLP":
        row1 = ca.horzcat(1.0, 0.0)
        row2 = ca.horzcat(x[0], 0.0)
        row3 = ca.horzcat(x[0]**2, 0.0)
        return ca.vertcat(row1, row2, row3)
    elif mode == "SIM":
        return dt * np.array([[1.0, 0.0], [x[0], 0.0], [x[0]**2, 0.0]])
    else:
        raise ValueError("Invalid input! Please use 'NLP' for optimizatoin or 'SIM' for simulation.")



