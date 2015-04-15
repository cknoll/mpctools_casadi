# MPC for a multivariable system.

# Imports.
import numpy as np
import mpctools.legacy as mpc

# Define continuous time model.
Acont = np.array([[0,1],[0,-1]])
Bcont = np.array([[0],[10]])
n = Acont.shape[0] # Number of states.
m = Bcont.shape[1] # Number of control elements

# Discretize.
dt = .025
N = 20
(Adisc,Bdisc) = mpc.util.c2d(Acont,Bcont,dt)
A = [Adisc]
B = [Bdisc]

# Bounds on u.
umax = 1
ulb = [np.array([-umax])]
uub = [np.array([umax])]
bounds = dict(uub=uub,ulb=ulb)

# Define Q and R matrices and q penalty for periodic solution.
Q = [np.diag([1,0])]
q = [np.zeros((n,1))]
R = [np.eye(m)]

# Initial condition.
x0 = np.array([10,0])

nsim = 100
t = np.arange(nsim+1)*dt
xcl = np.zeros((n,nsim+1))
xcl[:,0] = x0
ucl = np.zeros((m,nsim))
for k in range(nsim):
    # Solve linear MPC problem.
    sol = mpc.linear.lmpc(A,B,x0,N,Q,R,q=q,bounds=bounds,verbosity=0)
    print "Iteration %d Status: %s" % (k,sol["status"])
    xcl[:,k] = sol["x"][:,0]
    ucl[:,k] = sol["u"][:,0]
    x0 = np.dot(A[0],x0) + np.dot(B[0],sol["u"][:,0])
xcl[:,nsim] = x0 # Store final state.

# Plot things. Since time is along the second dimension, we must specify
# timefirst = False.
fig = mpc.plots.mpcplot(xcl,ucl,t,np.zeros(xcl.shape),xinds=[0],
                        timefirst=False)
fig.show()
