"""
diverse_functions.py

This module contains various basic functions used in the simulation

1. model of the dynamical system

Due to that the original dynamics are in continuous time, sampling time
is included as an additional parameter, with default value

dt = 0.1

and the Euler discretization method is used

The dynamical system also contains the current estimated unknown parameters,
the number of the parameters depends on the used model.


Two models will be considered:

0) Adaptive linear control (an artificial numerical example)

1) Adaptive cruise control (number of parameters: 3)

state: x_1 -> velocity
        x_2 -> distance between the leading vehicle
input: u -> acceleration
2) Adaptive pitch control of aircraft (number of parameters: 2)


"""

import casadi as ca
import numpy as np

## ----------------------- Linear Adaptive Control ----------------------------
def linear_dynamics(x, u, para: np.ndarray = np.zeros(3), dt = 1, mode="SIM"):
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
    if mode == "NLP":
        x_1_next = (0.5+0.042*para[0]+0.015*para[1])*x[0] + (0.2+0.019*para[1])*x[1] + 0.04*para[2]*u[0]
        x_2_next = (-0.1+0.072*para[0]+0.009*para[1])*x[0] + (0.6+0.03*para[0]+0.035*para[1])*x[1] + (0.5+0.054*para[2])*u[0]
        x_next = ca.vertcat(dt * x_1_next, dt * x_2_next)
    elif mode == "SIM":
        x_1_next = (0.5+0.042*para[0]+0.015*para[1])*x[0] + (0.2+0.019*para[1])*x[1] + 0.04*para[2]*u[0]
        x_2_next = (-0.1+0.072*para[0]+0.009*para[1])*x[0] + (0.6+0.03*para[0]+0.035*para[1])*x[1] + (0.5+0.054*para[2])*u[0]
        x_next = np.array([dt * x_1_next, dt * x_2_next])
    else:
        raise ValueError("Invalid input! Please use 'NLP' for optimizatoin or 'SIM' for simulation.")

    return x_next

def linear_f(x, u, dt = 1, mode="SIM"):
    """
    This is the drift term of the dynamics

    Input
    1) x: state
    2) u: input
    """
    if mode == "NLP":
        f_x_1 = 0.5*x[0] + 0.2*x[1]
        f_x_2 = -0.1*x[0] + 0.6*x[1] + 0.5*u[0]
        return ca.vertcat(dt * f_x_1, dt * f_x_2)
    elif mode == "SIM":
        f_x_1 = 0.5*x[0] + 0.2*x[1]
        f_x_2 = -0.1*x[0] + 0.6*x[1] + 0.5*u[0]
        return dt * np.array([f_x_1, f_x_2])
    else:
        raise ValueError("Invalid input! Please use 'NLP' for optimizatoin or 'SIM' for simulation.")

def linear_kernel(x, u, dt = 1, mode="SIM"):
    """
    This is the kernel function of the dynamics

    Input
    1) x: state
    2) u: input
    """
    if mode == "NLP":
        row1 = ca.horzcat(dt * 0.042*x[0], dt * (0.072*x[0] + 0.03*x[1]))
        row2 = ca.horzcat(dt * (0.015*x[0]+0.019*x[1]), dt * (0.009*x[0] + 0.035*x[1]))
        row3 = ca.horzcat(dt * 0.04*u[0], dt * 0.054*u[0])
        return ca.vertcat(row1, row2, row3)
    elif mode == "SIM":
        return dt * np.array([[0.042*x[0], 0.072*x[0] + 0.03*x[1]],
                              [0.015*x[0]+0.019*x[1], 0.009*x[0] + 0.035*x[1]],
                              [0.04*u[0], 0.054*u[0]]])
    else:
        raise ValueError("Invalid input! Please use 'NLP' for optimizatoin or 'SIM' for simulation.")


## ----------------------- Adaptive Cruise Control ----------------------------

def acc_dynamics(x, u, para: np.ndarray = np.zeros(3), dt = 0.1, mode="SIM"):
    """
    Compute the next state based on the NOMINAL discrete-time model.

    Parameters:
        x: casadi.SX or MX, current state vector.
        u: casadi.SX or MX, control input vector.
        para: ndarray, estimated system parameters
        dt: float, sampling time (default is 0.1 seconds)
        mode: "NLP" or "SIM" depending on the purpose of usage:
            1) "NLP" for optimization in MPC
            2) "SIM" for simulation and general Numpy based calculations

    Returns:
        x_next: casadi.SX or MX, next state vector. (Or just numpy array)
    """
    m = 1650.0 # the mass of the vehicle [kg]
    v_0 = 18 # the speed of the leading vehicle on the autoweg [m /s ]
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

def acc_f(x, u, dt = 0.1, mode="SIM"):
    """
    This is the drift term of the dynamics

    Input
    1) x: state
    2) dt: sampling time
    """
    m = 1650.0 # vehicle mass
    v_0 = 18 # front vehicle speed
    

    dot_v_x = u[0] / m
    dot_D_x = v_0 - x[0]

    if mode == "NLP":
        return ca.vertcat(x[0] + dt * dot_v_x, x[1] + dt * dot_D_x)
    elif mode == "SIM":
        return np.array([x[0] + dt * dot_v_x, x[1] + dt * dot_D_x])
    else:
        raise ValueError("Invalid input! Please use 'NLP' for optimizatoin or 'SIM' for simulation.")
    

def acc_kernel(x, u, dt = 0.1, mode="SIM"):
    """
    This is the kernel of the dynamics

    Input
    1) x: state
    2) dt: sampling time

    REMARK: be careful about the transpose given in the system dynamics.
            So, here it needs to be tranposed again

    """
    m = 1650.0 # vehicle mass

    if mode == "NLP":
        row1 = ca.horzcat(1.0 / m, 0.0)
        row2 = ca.horzcat(x[0] / m, 0.0)
        row3 = ca.horzcat(x[0]**2 / m, 0.0)
        return ca.vertcat(row1, row2, row3)
    elif mode == "SIM":
        return (1 / m) * dt * np.array([[1.0, 0.0], [x[0], 0.0], [x[0]**2, 0.0]])
    else:
        raise ValueError("Invalid input! Please use 'NLP' for optimizatoin or 'SIM' for simulation.")



