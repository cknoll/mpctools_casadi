# -*- coding: utf-8 -*-
"""
cknoll: 2019-08-07 17:26:13

2019-08-07 19:21:59

generate a casadi-compatible module from the model of Bolor K. Dariimaa
"""

import pickle

import matplotlib.pyplot as plt

import numpy as np
from numpy import pi
import sympy as sp
import mpctools as mpc
from ipydex import IPS, activate_ips_on_exception
import symbtools as st
import casadi


activate_ips_on_exception()

fname = "model_dariimaa.pcl"
with open(fname, "rb") as pfile:
    pdict = pickle.load(pfile)
    print(fname, "read")

# load variables from the dict
q_symbs = pdict['symbols']
params = pdict['parameters']
params_values = pdict['parameter_values']
qdd_part_lin_num = pdict['qdd_part_lin_num']
Anum = pdict['Anum']
Bnum = pdict['Bnum']

# state
# phi1, phi2, phi3, x_wagen, ... <Geschwindigkeiten>
xx = q1, q2, q3, q4, q1d, q2d, q3d, q4d = q_symbs[:-4]

# input
a = pdict['a']

# original:
q1dd_expr, q2dd_expr, q3dd_expr, q4dd_expr = qdd_part_lin_num[-4:]

### convert sympy expression
q1dd_fnc = sp.lambdify([q1, q2, q3, q4, q1d, q2d, q3d, q4d, a], q1dd_expr)
q2dd_fnc = sp.lambdify([q1, q2, q3, q4, q1d, q2d, q3d, q4d, a], q2dd_expr)
q3dd_fnc = sp.lambdify([q1, q2, q3, q4, q1d, q2d, q3d, q4d, a], q3dd_expr)
q4dd_fnc = sp.lambdify([q1, q2, q3, q4, q1d, q2d, q3d, q4d, a], q4dd_expr)


# define the model
def model_rhs(state, u):
    x1, x2, x3, x4, x5, x6, x7, x8 = state # q1, q2, q3, q4, q1d, q2d, q3d, q4d
    acc, = u
    x1d = x5
    x2d = x6
    x3d = x7
    x4d = x8

    x5d = q1dd_fnc(x1, x2, x3, x4, x5, x6, x7, x8, acc)
    x6d = q2dd_fnc(x1, x2, x3, x4, x5, x6, x7, x8, acc)
    x7d = q3dd_fnc(x1, x2, x3, x4, x5, x6, x7, x8, acc)
    x8d = q4dd_fnc(x1, x2, x3, x4, x5, x6, x7, x8, acc)

    return np.array([x1d, x2d, x3d, x4d, x5d, x6d, x7d, x8d])


T_total = 2.0
Nt = 100
Delta = T_total/Nt

Nx = 8
Nu = 1

xa = np.array([pi, pi, pi, 0.0,   0.0, 0.0, 0.0, 0.0])
xb = np.array([pi, pi, pi*.4, 0.0,   0.0, 0.0, 0.0, 0.0])


# vdp = mpc.DiscreteSimulator(model_rhs, Delta, [Nx, Nu], ["x", "u"])


# Then get casadi function for rk4 discretization.
ode_rk4_casadi = mpc.getCasadiFunc(model_rhs, [Nx, Nu], ["x", "u"], funcname="F", rk4=True, Delta=Delta, M=1)

# Define stage cost and terminal weight.
Q = np.eye(Nx)
R = np.eye(Nu)


def lfunc(x, u):
    """Standard quadratic stage cost."""
    x = x - xb
    return mpc.mtimes(x.T, Q, x) + mpc.mtimes(u.T, R, u)


l = mpc.getCasadiFunc(lfunc, [Nx, Nu], ["x", "u"], funcname="l")


P = 100*Q  # Terminal penalty.


def Pffunc(x):
    """Quadratic terminal penalty."""
    x = x - xb
    return mpc.mtimes(x.T, P, x)


Pf = mpc.getCasadiFunc(Pffunc, [Nx], ["x"], funcname="Pf")


# Bounds
umax = 20

lb = {"u": -np.ones((Nu,))*umax, "x": np.array([2, 2, -1,  -1,  -15, -15, -15, -5])}
ub = {"u":  np.ones((Nu,))*umax, "x": np.array([4, 4, 4,  1,  15, 15, 15, 5])}

# Make optimizers.
x0 = xa
Ndict = {"x": Nx, "u": Nu, "t": Nt}

np.random.seed(1004)


tt1 = np.arange(0, Delta*(Nt+1), Delta)

# columns
xxa = xa.reshape(-1, 1)
xxb = xa.reshape(-1, 1)

# row
scale_t_01 = tt1.reshape(1, -1) / T_total

# xx_guess = (xxa*(1 - scale_t_01) + xxb*scale_t_01).T

xx_guess = (xa[:, np.newaxis]*(1 - scale_t_01) + xb[:, np.newaxis]*scale_t_01).T


# guess = {"x": np.random.rand(Nt+1, Nx)*0, "u": np.random.rand(Nt, Nu)*0}
guess = {"x": xx_guess, "u": np.random.rand(Nt, Nu)*1}

solver = mpc.nmpc(f=ode_rk4_casadi, N=Ndict, l=l, x0=x0, Pf=Pf, lb=lb, ub=ub, verbosity=5, guess=guess)


# initial and final condition
solver.fixvar("x", 0, xa)
solver.fixvar("x", Nt, xb)

# solver._ControlSolver__solver


# Solve nlp
solver.solve()

# Print stats.
print(solver.stats["status"])


uu = solver.vardict["u"]
xx = solver.vardict["x"]


plt.plot(tt1[:-1], uu)
plt.figure()
plt.plot(tt1, xx, label="a")
plt.legend()
plt.show()

