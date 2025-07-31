"""
safety_filter.py

This module contains
(1) the basic SafetyFilter class that
implements a safety filter that generates a safe input for discrete-time systems;

The filter uses CasADi for symbolic modeling and the optimizer within CasADi.
"""

import casadi as ca
import numpy as np