# Car driving on an icy hill.
import numpy as np
import mpctools as mpc

# Define sizes and hill functions.
Nx = 2
Nu = 1
Nc = 2
Nt = 60
Delta = 0.25
umax = 0.5

def hill(x):
    """Normal distribution as hill."""
    return np.exp(-x**2/2)
dhill = mpc.util.getScalarDerivative(hill, vectorize=False)

# Define ODE and get LQR.
def ode(x, u):
    """
    Model dx/dt = f(x,u)
    
    x is [position, velocity]
    u is [external thrust]
    """
    dh = dhill(x[0])
    sintheta = dh/np.sqrt(1 + dh**2)
    costheta = 1/np.sqrt(1 + dh**2)
    dxdt = [x[1], (u[0] - sintheta)*costheta]
    return np.array(dxdt)
model = mpc.DiscreteSimulator(ode, Delta, [Nx, Nu], ["x", "u"])
odecasadi = mpc.getCasadiFunc(ode, [Nx, Nu], ["x", "u"], "f")

# Get LQR.
xss = np.zeros((Nx,))
uss = np.zeros((Nu,))
fcasadi = mpc.getCasadiFunc(ode, [Nx, Nu], ["x", "u"], funcname="f")
ss = mpc.util.getLinearizedModel(fcasadi, [xss,uss], ["A","B"], Delta=Delta)
ss["Q"] = np.eye(Nx)
ss["R"] = np.eye(Nu)
[ss["K"], ss["Pi"]] = mpc.util.dlqr(ss["A"], ss["B"], ss["Q"], ss["R"])

# Build controller.
def l(x, u):
    """Quadratic stage cost x'Qx + u'Ru."""
    return mpc.mtimes(x.T, ss["Q"], x) + mpc.mtimes(u.T, ss["R"], u)
lcasadi = mpc.getCasadiFunc(l, [Nx, Nu], ["x", "u"], "l", scalar=False)

def Vf(x):
    """Quadratic terminal penalty."""
    return mpc.mtimes(x.T, ss["Pi"], x)
Vfcasadi = mpc.getCasadiFunc(Vf, [Nx], ["x"], "Vf", scalar=False)

lb = dict(u=-umax*np.ones((Nt, Nu)))
ub = dict(u=umax*np.ones((Nt, Nu)))

N = {"t" : Nt, "x" : Nx, "u" : Nu, "c" : Nc}

x0 = np.array([-1, 0])

controller = mpc.nmpc(f=odecasadi, l=lcasadi, Pf=Vfcasadi, N=N, lb=lb, ub=ub,
                      x0=x0, Delta=Delta, verbosity=4)

# Solve and plot open-loop trajectory.
controller.solve()
t = np.arange(Nt + 1)*Delta
mpc.plots.mpcplot(controller.vardict["x"], controller.vardict["u"], t)
