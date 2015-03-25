# Control of the Van der Pol oscillator using mpc_tools_casadi.
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

# Then get nonlinear casadi functions and rk4 discretization.
ode_casadi = mpc.getCasadiFunc(ode, Nx, Nu, name="f")
ode_rk4_casadi = mpc.getRungeKutta4(ode_casadi, Delta)

# Define stage cost and terminal weight.
lfunc = lambda x,u: [mpc.mtimes(x.T,x) + mpc.mtimes(u.T,u)]
l = mpc.getCasadiFunc(lfunc, Nx, Nu, name="l")

Pffunc = lambda x: [10*mpc.mtimes(x.T,x)]
Pf = mpc.getCasadiFunc(Pffunc, Nx, name="Pf")

# Bounds on u.
bounds = dict(uub=[1],ulb=[-.75])

# Make optimizers.
x0 = np.array([0,1])
Nt = 20
solver = mpc.nmpc(F=[ode_rk4_casadi],N=Nt,verbosity=0,l=[l],x0=x0,Pf=Pf,bounds=bounds,returnTimeInvariantSolver=True)

# Now simulate.
Nsim = 20
times = Delta*Nsim*np.linspace(0,1,Nsim+1)
x = np.zeros((Nsim+1,Nx))
x[0,:] = x0
u = np.zeros((Nsim,Nu))
for t in range(Nsim):
    # Fix initial state.
    solver.fixvar("x",0,x[t,:])
    
    # Solve nlp.
    sol = solver.solve()
    
    # Print stats.
    print "%d: %s" % (t,sol["status"])
    u[t,:] = sol["u"][0,:]
    
    # Simulate.
    x[t+1,:] = vdp.sim(x[t,:],u[t,:])
    
# Plots.
mpc.mpcplot(x.T,u.T,times)
