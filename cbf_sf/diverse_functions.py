"""
diverse_functions.py

This module contains various basic functions used in CBF based safety filter

Two basic functions are needed

1. extended class kappa function
2. the barrier function

"""

import numpy as np

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
def cbf_acc(x: np.ndarray, mode: str):
    """
    This is the control barrier function for adaptive cruise control.

    Parameters:
    1. x: state
        x[0]: velocity
        x[1]: distance
    2. mode: "linear", "quadratic"
    """
    if mode == "linear":
        return x[1] - 1.8 * x[0]
    elif mode == "quadratic":
        # set the buffer
        buffer = 0.1
        # compute the margin
        margin = x[1] - 1.8 * x[0]
        return buffer ** 2 if margin >= buffer else 2 * buffer * margin - margin ** 2
