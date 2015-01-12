# MPC for a multivariable system.

# Imports.
import numpy as np
import mpc_tools_casadi as mpc

# Define continuous time model.
Acont = np.matrix([[0,1],[0,-1]])
Bcont = np.matrix([[0],[10]])
n = Acont.shape[0] # Number of states.
p = Bcont.shape[1] # Number of control elements

# Discretize.
dt = .025
N = 200
t = np.arange(N+1)*dt
(Adisc,Bdisc) = mpc.c2d(Acont,Bcont,dt)
A = [Adisc]
B = [Bdisc]

# Bounds on u.
umax = 1
ulb = [np.array([-umax])]
uub = [np.array([umax])]

# Define Q and R matrices and q penalty for periodic solution.
Q = [np.diag([1,0])]
q = [np.zeros((n,1))]
R = [np.eye(p)]

# Initial condition.
x0 = np.matrix([[10],[0]])

# Solve linear MPC problem.
(x,u) = mpc.lmpc(A,B,x0,N,Q,q,R,ulb,uub)

# Plot things.
mpc.mpcplot(x,u,t,np.zeros(x.shape),xinds=[0])