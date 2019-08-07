# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 16:28:22 2018

@author: Patrick
"""

# based on http://nbviewer.jupyter.org/github/cknoll/beispiele/blob/master/zweifachpendel_nq2_np2_ruled_manif.ipynb


"""
cknoll: 2019-08-07 17:26:13

Dieses Skript generiert ein Casadi-Taugliches Modell aus dem gepickelt Modell von B. K. Dariimaa. 
"""

import sys
import pickle
from joblib import dump

import numpy as np
import sympy as sp
from ipydex import IPS
import symbtools as st
import symbtools.modeltools as mt
from symbtools import mpctools as mpct


sys.path.append("../..")
from plotter import Plotter
from ani import create_animation
from ocp import OptimalControlProblem
from mpc import MPCSolver, optimize_QR


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


q1dd_expr, q2dd_expr, q3dd_expr, q4dd_expr = qdd_part_lin_num[-4:]



x_dot = sp.Matrix(qdd_part_lin_num)


rhs_casasi = mpct.create_casadi_func(x_dot, xx, [a], name="triple_pendulum_rhs_cs")



# dump({'x_dot_str': x_dot_str, 'x_dim': len(x_dot), 'u_dim': 1}, 'examples/triple_pend_cart_pl_bkd.str')



sol_dict = {}
np.random.seed(7)
u_guess = np.random.normal(loc=0.0, scale=4*9.81/3, size=(175, 1))
x_guess = np.random.normal(loc=0.0, scale=10/3, size=(176, 8))
solver_kwargs = {'N': 175, 'u_guess': u_guess, 'x_guess': x_guess}

solver_kwargs_updates = {'rhc_xN_unltd':    {'receding': 'unltd'},
                         'rhc_xN_global':   {'receding': 'global'},
                         'rhc_XNF_unltd':   {'receding': 'unltd'}}
xu_dict = {'x_0':  'x_0',
           'x_1':  'x_2',
           'x_2':  'x_4',
           'x_3':  'x_6',
           'x_4':  'x_1',
           'x_5':  'x_3',
           'x_6':  'x_5',
           'x_7':  'x_7',
           'u':    'u_0'}

ocp = OptimalControlProblem.load('triple_pend_cart_pl_xN')

xa = [pi, pi, pi, 0, 0.0, 0.0, 0.0, 0.0]
xb = [pi, pi, 0, 0, 0.0, 0.0, 0.0, 0.0]


IPS()

exit()
