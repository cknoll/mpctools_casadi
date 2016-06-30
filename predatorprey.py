import mpctools as mpc
import numpy as np
import matplotlib.pyplot as plt
from numpy import random

random.seed(927)

# Sizes.
Nx = 3
Nu = 1
Ny = 1
Delta = 0.1

# Pick coefficients.
a = 0.5 # Kill rate for prey.
b = 1 # Birth rate for prey.
c = 0.25 # Birth rate for predators.
d = 1 # Death rate for predators.
N0prey = 50 # Scale factor for prey.
N0pred = 20 # Scale factor for predators.
def ode(x, u):
    """Predator/prey dynamics."""
    [Nprey, Ntag, Npred] = x[:]
    Nprey /= N0prey
    Ntag /= N0prey
    Npred /= N0pred
    [Rtag] = u[:]
    dxdt = [
        N0prey*(-a*Npred*Nprey + b*Nprey - Rtag*Nprey),
        N0prey*(-a*Npred*Ntag + Rtag*Nprey),
        N0pred*(c*Npred*(Nprey + Ntag) - d*Npred),
    ]
    return np.array(dxdt)
def measurement(x):
    """Returns fraction of tagged animals."""
    Nprey = x[0]
    Ntag = x[1]
    return np.array([Ntag/(Nprey + Ntag)])

# Convert to Casadi functions and simulator.    
model = mpc.DiscreteSimulator(ode, Delta, [Nx,Nu], ["x","u"])
f = mpc.getCasadiFunc(ode, [Nx,Nu], ["x","u"], "f", rk4=True, Delta=Delta)
h = mpc.getCasadiFunc(measurement, [Nx], ["x"], "h")
x0 = np.array([N0prey, 0, N0pred])

# Simulate dynamics.
Nt = 100
x = np.NaN*np.ones((Nt + 1, Nx))
x[0,:] = x0
u = 0.1*np.ones((Nt, Nu))
y = np.NaN*np.ones((Nt + 1, Ny))
noise = 0.05*random.randn(Nt + 1, Ny) # Multiplicative noise term.
for t in xrange(Nt + 1):
    # Round x and take measurement.
    x[t,:] = np.maximum(np.round(x[t,:]*(1 + noise[t,:])), 0)
    y[t] = measurement(x[t,:])    
    
    # Simulate step.
    if t < Nt:
        x[t + 1,:] = model.sim(x[t,:], u[t,:])

# Now try MHE.
def stagecost(w, v):
    """Stage cost for measurement and state noise."""
    return 0.1*mpc.mtimes(w.T, w) + 10*mpc.mtimes(v.T, v)
def prior(dx):
    """Prior weight function."""
    return 100*mpc.mtimes(dx.T, dx)
l = mpc.getCasadiFunc(stagecost, [Nx,Ny], ["w","v"])
lx = mpc.getCasadiFunc(prior, [Nx], ["dx"])

N = dict(x=Nx, u=Nu, y=Ny, w=Nx, v=Ny, t=Nt)
guess = dict(x=np.tile(x0, (Nt + 1, 1)))
lb = dict(x=0*np.ones((Nt + 1, Nx))) # Lower bounds are ones.
mhe = mpc.nmhe(f, h, u, y, l, N, lx, x0, lb=lb, guess=guess, wAdditive=True)
mhe.solve()
xhat = mhe.vardict["x"]
yhat = np.array([measurement(xhat[i,:]) for i in xrange(Nt + 1)])

# Make a plot.
t = np.arange(Nt + 1)*Delta
labels = ["Untagged Prey", "Tagged Prey", "Predators", "Tag Fraction"]
data = np.concatenate((x, y), axis=1)
datahat = np.concatenate((xhat, yhat), axis=1)
[fig, ax] = plt.subplots(nrows=len(labels))
for (i, label) in enumerate(labels):
    ax[i].plot(t, data[:,i], color="green", label="Actual")
    ax[i].plot(t, datahat[:,i], color="red", label="Estimated")
    ax[i].set_ylabel(labels[i])
    if i == 0:
        ax[i].legend(loc="upper right")
ax[i].set_xlabel("Time")
mpc.plots.showandsave(fig, "predatorprey.pdf")
