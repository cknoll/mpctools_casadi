# MPC for a multivariable system.

# Imports.
import numpy as np
import mpc_tools_casadi as mpc

# Define continuous time model.
Acont = np.matrix([[0,1],[0,-1]])
Bcont = np.matrix([[0],[10]])
n = Acont.shape[0] # Number of states.
m = Bcont.shape[1] # Number of control elements

# Discretize.
dt = .025
N = 20
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
R = [np.eye(m)]

# Initial condition.
x0 = np.matrix([[10],[0]])

nsim = 100
xcl = np.zeros((n,nsim))
xcl[0,:] = x0
ucl = np.zeros((m,nsim))
for k in range(nsim):
    # Solve linear MPC problem.
    (x,u) = mpc.lmpc(A,B,x0,N,Q,R,q=q,ulb=ulb,uub=uub)
    xcl[:,k] = x[:,0]
    ucl[:,k] = u[:,0]
    x0 = dot(A,x0) + dot(B,u[0])
# Plot things.
mpc.mpcplot(x,u,t,np.zeros(x.shape),xinds=[0])
