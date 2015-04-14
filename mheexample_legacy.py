# Example for nonlinear MPC with Casadi.

# Imports.
import numpy as np
from numpy import random
from scipy import linalg
import mpctools.legacy.tools as mpc
import mpctools.util as mpcutil
import matplotlib.pyplot as plt

random.seed(927) # Seed random number generator.

verb = 2
doPlots = True

# Grab matrix multiplication function.
mtimes = mpcutil.mtimes

# Problem parameters.
Delta = .1
Nt = 50
t = np.arange(Nt+1)*Delta

Nx = 3
Nu = 2
Ny = 2
Nw = Nx
Nv = Ny

Acont = np.array([[-1,1,0],[0,-2,2],[0,0,-.5]])
Bcont = np.array([[1,0],[0,1],[1,1]])
Ccont = np.array([[1,1,0],[0,1,1]])

(Adisc,Bdisc) = mpcutil.c2d(Acont,Bcont,Delta)
Cdisc = Ccont

# Discrete-time example
A = mpcutil.DMatrix(Adisc) # Cast to Casadi matrix type.
B = mpcutil.DMatrix(Bdisc)
C = mpcutil.DMatrix(Cdisc)

Fdiscrete = lambda x,u,w : [mtimes(Adisc,x) + mtimes(Bdisc,u) + w]
F = [mpc.getCasadiFuncGeneralArgs(Fdiscrete,[Nx,Nu,Nw],["x","u","w"],"F")]

Hdiscrete = lambda x : [mtimes(Cdisc,x)]
H = [mpc.getCasadiFuncGeneralArgs(Hdiscrete,[Nx],["x"],"H")]

# Noise covariances.
Q = .01*np.diag([.1,.25,.05])
Qhalf = linalg.cholesky(Q,lower=True)
Qinv = linalg.inv(Q)

R = np.diag([.5,.25])
Rhalf = linalg.cholesky(R,lower=True)
Rinv = linalg.inv(R)

# First simulate the noisy system.
x0 = np.array([1,2,3])
x0hat = np.array([0,0,0])

omega = 2*np.pi/(Nt*Delta)
u = np.vstack((np.sin(omega*t),np.cos(omega*t))).T # Use periodic input.
w = Qhalf.dot(random.randn(Nw,Nt)).T
v = Rhalf.dot(random.randn(Nv,Nt)).T

x = np.zeros((Nt+1,Nx))
x[0,:] = x0
y = np.zeros((Nt,Ny))

for k in range(Nt):
    thisy = H[0]([x[k,:]])[0]
    y[k,:] = np.squeeze(thisy) + v[k,:]
    xnext = F[0]([x[k,:],u[k,:],w[k,:]])[0]
    x[k+1,:] = np.squeeze(xnext)
    
# Plot simulation.
if doPlots:
    f = plt.figure()
    
    # State
    ax = f.add_subplot(2,1,1)
    for i in range(Nx):
        ax.plot(t,x[:,i],label="$x_{%d}$" % (i,))
    ax.set_ylabel("$x$")
    ax.legend()
    
    # Measurement
    ax = f.add_subplot(2,1,2)
    for i in range(Ny):
        ax.plot(t[:-1],y[:,i],label="$y_{%d}$" % (i,))
    ax.set_ylabel("$y$")
    ax.set_xlabel("Time")
    ax.legend()
    f.tight_layout(pad=.5)

# Now we're ready to try some state estimation. First define stage cost and prior.
l = lambda w,v : [mtimes(w.T,Qinv,w) + mtimes(v.T,Rinv,v)]
l = [mpc.getCasadiFuncGeneralArgs(l,[Nw,Nv],["w","v"],"l")]
lx = lambda x: [10*mtimes(x.T,x)]
lx = mpc.getCasadiFuncGeneralArgs(lx,[Nx],["x"],"lx")

N = {"t" : Nt, "x" : Nx, "y" : Ny, "u" : Nu}

out = mpc.nmhe(F,H,u,y,l,N,lx,x0hat,verbosity=5)

west = out["w"]
vest = out["v"]
xest = out["x"]
xerr = xest - x

# Plot estimation.
if doPlots:
    f = plt.figure()
    
    # State
    ax = f.add_subplot(2,1,1)
    for i in range(Nx):
        ax.plot(t,xest[:,i],label=r"$\hat{x}_{%d}$" % (i,))
    ax.set_ylabel("$x$")
    ax.legend()
    
    # Measurement
    ax = f.add_subplot(2,1,2)
    for i in range(Nx):
        ax.plot(t,xerr[:,i],label=r"$\hat{x}_{%d} - x_{%d}$" % (i,i))
    ax.set_ylabel("Error")
    ax.set_xlabel("Time")
    ax.legend()
    f.tight_layout(pad=.5)