"""
This is metric computation module, the included functions are all used to compute
metric-related quantities of a given set (e.g., a polytope). Computation of these
quantities are usually directly formulated as optimization algorithms and we use
GUROBI to solve the optimization.

The included modules are

1. 2-norm based projection
2. 1-norm based most-distant-point finder

(more things are added in the future if needed)
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB

def project_onto_feasible_set(A, b, x_star):
    """
    Solves: min_x ||x - x_star||_2^2 s.t. A x <= b
    
    Parameters:
        A : np.ndarray of shape (m, n)
        b : np.ndarray of shape (m,)
        x_star : np.ndarray of shape (n,)
        
    Returns:
        x_proj : np.ndarray of shape (n,) - projected vector
    """
    # ------- defensive conversion --------
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x_star = np.asarray(x_star, dtype=float)

    # build the model
    n = x_star.shape[0]
    model = gp.Model()
    model.setParam("OutputFlag", 0)  # Silence solver output

    # Variables
    x = model.addMVar(shape=n, lb=-GRB.INFINITY, name="x")

    # Objective: minimize ||x - x_star||^2 = (x - x_star)^T (x - x_star)
    obj = (x - x_star) @ (x - x_star)
    model.setObjective(obj, GRB.MINIMIZE)

    # Constraints: A x <= b
    model.addConstr(A @ x <= b, name="Ax_leq_b")

    # Solve
    model.optimize()

    if model.status == GRB.OPTIMAL:
        return x.X
    else:
        raise ValueError("Projection QP was infeasible or did not solve optimally.")


def max_l1_deviation_value(A, b, x_star, tol=1e-6):
    """
    Returns  max_x ||x - x_star||_1  subject to  A x <= b,
    using Gurobi general ABS constraints in a type-safe way.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    x_star = np.asarray(x_star, dtype=float)

    # --- check x_star feasibility ------------------------------------------
    if np.any(A @ x_star - b > tol):
        raise ValueError("x_star is infeasible (A x_star ≤ b violated).")

    n = A.shape[1]
    model = gp.Model()
    model.Params.OutputFlag = 0        # silent

    # primal variables
    x = model.addMVar(n, lb=-GRB.INFINITY, name="x")

    # difference and abs-value auxiliary vars
    d = model.addMVar(n, lb=-GRB.INFINITY, name="d")   # d_i = x_i - x*_i
    z = model.addMVar(n, lb=0.0, name="z")   # for use of z_i = |d_i|

    # d_i = x_i - x*_i
    model.addConstr(d == x - x_star, name="diff")

    # z_i = |d_i|
    for i in range(n):
        model.addGenConstrAbs(z[i], d[i], name=f"abs_{i}") # type: ignore

    # polyhedral feasibility
    model.addConstr(A @ x <= b, name="polytope")

    # objective: maximise Σ z_i
    model.setObjective(z.sum(), GRB.MAXIMIZE)

    model.optimize()

    if model.status == GRB.OPTIMAL:
        return float(model.ObjVal)
    else:
        raise RuntimeError(
            f"Gurobi did not reach OPTIMAL status (code {model.status})."
        )
