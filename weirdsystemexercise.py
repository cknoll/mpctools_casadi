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
model = mpc.OneStepSimulator(ode, Delta, Nx, Nu)

# Then get nonlinear casadi functions and a linearization.
ode_casadi = mpc.getCasadiFunc(ode, Nx, Nu, name="f")
lin = mpc.getLinearization(ode_casadi,np.zeros((Nx,)),np.zeros((Nu,)),Delta=Delta)

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
solvers["LMPC"] = mpc.nmpc(F=[F],**commonargs)
solvers["NMPC"] = mpc.nmpc(F=[ode_casadi],timemodel="colloc",M=2,Delta=Delta,**commonargs)

# Now simulate.
Nsim = 100
tplot = Delta*Nsim*np.linspace(0,1,Nsim+1)
x = {}
u = {}
tplot = Nsim*Delta*np.linspace(0,1,Nsim+1)
for method in solvers.keys():
    x[method] = np.zeros((Nsim+1,Nx))
    x[method][0,:] = x0
    u[method] = np.zeros((Nsim,Nu))
    for t in range(Nsim):
        # Update initial condition and solve OCP.
        solvers[method].fixvar("x",0,x[method][t,:])
        sol = solvers[method].solve()
        
        # Print status and make sure solver didn't fail.        
        print "%5s %d: %s" % (method,t,sol["status"])
        if sol["status"] != "Solve_Succeeded":
            break
        else:
            solvers[method].saveguess(sol)
        u[method][t,:] = sol["u"][0,:]
        x[method][t+1,:] = model.sim(x[method][t,:],u[method][t,:])
        
        # We can stop early if we are already close to the origin.
        if np.sum(x[method][t+1,:]**2) < 1e-4:
            print "%s at origin after %d iterations." % (method,t)
            break

# Prepare some plots.
phasefig = plt.figure()
phaseax = phasefig.add_subplot(1,1,1)
phaseax.set_xlabel("$x_0$")
phaseax.set_ylabel("$x_1$")
phaseax.axvline(color="grey")
phaseax.axhline(color="grey")
phasefig.canvas.set_window_title("Phase Space")

timefig = plt.figure(figsize=(6,10))
timexax = []
for i in range(Nx):
    timexax.append(timefig.add_subplot(Nx + Nu,1,i+1))
    timexax[-1].set_ylabel("$x_{%d}$" % (i,))
timeuax = []
for i in range(Nu):
    timeuax.append(timefig.add_subplot(Nx + Nu,1,i+Nx+1))
    timeuax[-1].set_ylabel("$u_{%d}$" % (i,))
timeuax[-1].set_xlabel("Time")
timefig.canvas.set_window_title("Time Series")

colors = {"LMPC":"red", "NMPC":"green"}

for method in solvers.keys():
    # Plot in time.
    uplot = np.vstack((u[method],u[method][-1,:]))
    for i in range(Nu):
        timeuax[i].step(tplot,uplot[:,i],color=colors.get(method,"k"))
    for i in range(Nx):
        timexax[i].plot(tplot,x[method][:,i],color=colors.get(method,"k"))
    
    # Plot in phase space.
    phaseax.plot(x[method][:,0],x[method][:,1],'-k',label=method,color=colors.get(method,"k"))

# Some beautification.
for ax in timexax + timeuax:
    mpc.zoomaxis(ax,yscale=1.1)
timefig.tight_layout(pad=.5)
timefig.savefig("weirdsystemtimeseries.pdf")
phasefig.tight_layout(pad=.5)
phaseax.legend(loc="upper right")
phasefig.savefig("weirdsystemphasespace.pdf")
        
        
