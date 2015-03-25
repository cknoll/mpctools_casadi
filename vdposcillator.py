# Control of the Van der Pol oscillator.
import mpc_tools_casadi as mpc
import numpy as np

# Define model and get simulator.
Delta = .5
Nx = 2
Nu = 1
def ode(x,u):
    return [(1 - x[1]*x[1])*x[0] - x[1] + u, x[0]]

# Create a simulator.
vdp = mpc.OneStepSimulator(ode, Delta, Nx, Nu)

# Then get nonlinear casadi functions and a linearization.
ode_casadi = mpc.getCasadiFunc(ode, Nx, Nu, name="f")
lin = mpc.getLinearization(ode_casadi,[0,0],[0],Delta=Delta)

# Define stage cost and terminal weight.
lfunc = lambda x,u: [mpc.mtimes(x.T,x) + mpc.mtimes(u.T,u)]
l = mpc.getCasadiFunc(lfunc, Nx, Nu, name="l")

Pffunc = lambda x: [10*mpc.mtimes(x.T,x)]
Pf = mpc.getCasadiFunc(Pffunc, Nx, name="Pf")

# Create linear discrete-time model for comparison.
Ffunc = lambda x,u: [mpc.mtimes(mpc.DMatrix(lin["A"]),x) + mpc.mtimes(mpc.DMatrix(lin["B"]),u)]
F = mpc.getCasadiFunc(Ffunc, Nx, Nu, name="F")

# Bounds on u.
bounds = dict(uub=[1],ulb=[-.75])

# Make optimizers.
x0 = np.array([0,1])
Nt = 20
commonargs = dict(
    N=Nt,
    verbosity=0,
    l=[l],
    x0=x0,
    Pf=Pf,
    bounds=bounds,
    returnTimeInvariantSolver=True,
)
solvers = {}
solvers["lmpc"] = mpc.nmpc(F=[F],**commonargs)
solvers["nmpc"] = mpc.nmpc(F=[ode_casadi],timemodel="colloc",M=4,Delta=Delta,**commonargs)

# Now simulate.
Nsim = 20
times = Delta*Nsim*np.linspace(0,1,Nsim+1)
x = {}
u = {}
for method in solvers.keys():
    x[method] = np.zeros((Nsim+1,Nx))
    x[method][0,:] = x0
    u[method] = np.zeros((Nsim,Nu))
    for t in range(Nsim):
        solvers[method].fixvar("x",0,x[method][t,:])
        sol = solvers[method].solve()
        print "%5s %d: %s" % (method,t,sol["status"])
        u[method][t,:] = sol["u"][0,:]
        x[method][t+1,:] = vdp.sim(x[method][t,:],u[method][t,:])
    mpc.mpcplot(x[method].T,u[method].T,times,title=method)
        
        
