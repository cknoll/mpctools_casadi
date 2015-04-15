# Periodic MPC for single-variable system.

# Imports.
import numpy as np
import scipy.signal as spsignal
import mpctools.legacy as mpc

# Define optimal periodic solution.
T = 1
def f(t): return spsignal.sawtooth(2*np.pi/T*t + np.pi/2,.5)

# Define continuous time model.
Acont = np.array([[-1]])
Bcont = np.array([[10]])
n = Acont.shape[0] # Number of states.
p = Bcont.shape[1] # Number of control elements

# Discretize.
dt = .01
N = 250
t = np.arange(N+1)*dt
(Adisc,Bdisc) = mpc.util.c2d(Acont,Bcont,dt)
A = [Adisc]
B = [Bdisc]

# Bounds on u.
umax = 1
ulb = [np.array([-umax])]
uub = [np.array([umax])]
bounds = dict(uub=uub,ulb=ulb)

# Define Q and R matrices and q penalty for periodic solution.
R = [np.eye(p)]
Q = [np.eye(n)]
q = [None]*(N+1) # Preallocate.
for k in range(N+1):
    q[k] = -Q[k % len(Q)]*f(t[k])

# Initial condition.
x0 = np.array([-2])

# Solve linear MPC problem.
solution = mpc.linear.lmpc(A,B,x0,N,Q,R,q=q,bounds=bounds)
x = solution["x"]
u = solution["u"]

# Plot things.
fig = mpc.plots.mpcplot(x,u,t,f(t[np.newaxis,:]),timefirst=False)
fig.show()