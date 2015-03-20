# Control of the Van der Pol oscillator.
import mpc_tools_casadi as mpc
import numpy as np
import matplotlib.pyplot as plt

# Define model and get simulator.
Delta = .1
Nx = 2
Nu = 2
M = 1
def ode(x,u):
    return [-x[0] - (1+u[0])*x[1], (1+u[0])*x[0] + x[1] + u[1]]
#    return [-x[0] - (1+u)*x[1], (1+u)*x[0] + (M*x[1]**3 - x[1])/(1 + M*x[1]**2)]
x0 = np.array([2,2])

# Create a simulator.
vdp = mpc.OneStepSimulator(ode, Delta, Nx, Nu)

# Then get nonlinear casadi functions and a linearization.
ode_casadi = mpc.getCasadiFunc(ode, Nx, Nu, name="f")
lin = mpc.getLinearization(ode_casadi,np.zeros((Nx,)),np.ones((Nu,)),Delta=Delta)

# Define stage cost and terminal weight.
lfunc = lambda x,u: [100*mpc.mtimes(x.T,x) + mpc.mtimes(u.T,u)]
l = mpc.getCasadiFunc(lfunc, Nx, Nu, name="l")

Pffunc = lambda x: [1000*mpc.mtimes(x.T,x)]
Pf = mpc.getCasadiFunc(Pffunc, Nx, name="Pf")

# Create linear discrete-time model for comparison.
Ffunc = lambda x,u: [mpc.mtimes(mpc.DMatrix(lin["A"]),x) + mpc.mtimes(mpc.DMatrix(lin["B"]),u)]
F = mpc.getCasadiFunc(Ffunc, Nx, Nu, name="F")

# Bounds on u.
bounds = dict(uub=[np.array([1,1])],ulb=[np.array([-1,-1])])

# Make optimizers.
Nt = 25
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
solvers["lmpc"] = mpc.nmpc(F=[F],**commonargs) # LMPC doesn't work because the linearization is uncontrollable.
solvers["nmpc"] = mpc.nmpc(F=[ode_casadi],timemodel="rk4",M=2,Delta=Delta,**commonargs)

# Now simulate.
Nsim = 100
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
        if sol["status"] != "Solve_Succeeded":
            break
        else:
            solvers[method].saveguess(sol)
        u[method][t,:] = sol["u"][0,:]
        x[method][t+1,:] = vdp.sim(x[method][t,:],u[method][t,:])
    
    # Plot in time.    
    mpc.mpcplot(x[method].T,u[method].T,times,title="Time Series (%s)" % (method,))
    
    # Plot in phase space.
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x[method][:,0],x[method][:,1],'-k')
    ax.set_xlabel("$x_0$")
    ax.set_ylabel("$x_1$")
    ax.axvline(color="grey")
    ax.axhline(color="grey")
    fig.canvas.set_window_title("Phase Space (%s)" % (method,))
        
        
