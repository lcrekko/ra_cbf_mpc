"""
diverse_functions.py

This module contains various basic functions used in CBF based safety filter

Two basic functions are needed

1. extended class kappa function
2. the barrier function

"""

import numpy as np
import casadi as ca
from casadi import SX, MX
from typing import Union

# ------------- Extended class kappa function --------------
def ext_kappa(x, mode: str):
    """
    This is the extended class kappa function.

    Parameters:
    1. x
    2. mode: "linear", "arctan", "tanh"

    Output: y
    """
    if mode == "linear":
        return 0.5 * x
    elif mode == "arctan":
        return 0.5 * np.arctan(x)
    elif mode == "tanh":
        return 0.5 * np.tanh(x)
    else:
        raise ValueError("Invalid input! Please use [linear], [arctan] or [tanh]")


# ------------- Control barrier function --------------
def cbf_acc(x: np.ndarray):
    """
    This is the control barrier function for adaptive cruise control.

    Here we use a quadratic barrier function

    Parameters:
    1. x: state
        x[0]: velocity
        x[1]: distance
    """
    # set the buffer
    buffer = 0.1
    # compute the margin
    margin = x[1] - 1.8 * x[0]
    
    return buffer ** 2 if margin >= buffer else 2 * buffer * margin - margin ** 2

def opt_cbf_acc(x, mode):
    """
    This is the control barrier function for adaptive cruise control
    used in safety filter optimization problem.

    Here we use a quadratic barrier function.

    Parameters:
    1. x: state
        x[0]: velocity
        x[1]: distance
    2. mode: "SIM" or "NLP"
    """
    buffer = 0.1
    margin = x[1] - 1.8 * x[0]

    if mode == "SIM":
        return buffer ** 2 if margin >= buffer else 2 * buffer * margin - margin ** 2
    elif mode == "NLP": 
        return ca.if_else(margin >= buffer, buffer ** 2, 
                          2 * buffer * margin - margin ** 2)
    else:
        raise ValueError("Invalid input! Please use 'NLP' for optimizatoin or 'SIM' for simulation.")
