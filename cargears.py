# Simple car with multiple gears.
import numpy as np
import mpctools as mpc

# Model parameters.
Nt = 50
Nx = 1
Nu = 2
Ngears = 3
Delta = 0.1 # s
omegamax = 500 # Rad/s
c = 0.01 # m/s
r = 0.15 # m
lam = 1
udiscrete = np.array([False, True])

# Model functions.
def gear2ratio(g):
    """Converts gear number (between 0 and Ngears) to gear ratio."""
    return Ngears + 1 - g

def ode(x, u):
    """Continuous-time acceleration for car."""
    [velocity] = x[:Nx] # State is velocity.
    [omega, gear] = u[:Nu] # Engine angular velocity and gear number
    ratio = gear2ratio(gear)
    accel = c*ratio*omega*np.tanh(lam*(r*omega/ratio - velocity))
    return np.array([accel])
f = mpc.getCasadiFunc(ode, [Nx, Nu], ["Nx", "Nu"], rk4=True, Delta=Delta)

# Steady state and stage cost.
gearss = Ngears
omegass = 0.5*omegamax
uss = np.array([omegass, gearss])
velocityss = r*omegass/gear2ratio(gearss)
def stagecost(x, u):
    """Quadratic stage cost."""
    l = (10*(x[0]/velocityss - 1)**2 + 0*(u[0]/omegass - 1)**2
         + 0*(u[1]/gearss - 1)**2)
    return l
l = mpc.getCasadiFunc(stagecost, [Nx, Nu], ["Nx", "Nu"])
def termcost(x):
    """Quadratic terminal cost."""
    return 10*stagecost(x, uss)
Pf = mpc.getCasadiFunc(termcost, [Nx], ["Nx"])

# Generate a guess.
x0 = np.array([0]) # Car starts from rest.
x = np.zeros((Nt + 1, Nx))
u = np.tile([omegass, Ngears], (Nt, 1))
x[0,:] = x0
for k in range(Nt):
    x[k + 1,:] = np.squeeze(f(x[k,:], u[k,:]))

# Build controller and optimize.
N = dict(x=Nx, u=Nu, t=Nt)
lb = dict(u=np.array([0, 1]))
ub = dict(u=np.array([omegamax, Ngears]))
guess = dict(x=x, u=u)
cont = mpc.nmpc(f, l, N, x0, lb, ub, guess, Pf=Pf, udiscrete=udiscrete)
cont.solve()

# Make a plot.
t = Delta*np.arange(Nt + 1)
fig = mpc.plots.mpcplot(cont.vardict["x"], cont.vardict["u"], t)
mpc.plots.showandsave(fig, "cargears.pdf")
