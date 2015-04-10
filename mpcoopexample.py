# Closed-loop MPC for a multivariable system. This is basically identical to
# mpcexampleclosedloop.py except that we use both the "functional" way as in
# the original file, and a small object-oriented interface that is better.

# Imports.
import numpy as np
import mpc_tools_casadi as mpc
import time

# Define continuous time model.
Acont = np.array([[0,1],[0,-1]])
Bcont = np.array([[0],[10]])
Nx = Acont.shape[0] # Number of states.
Nu = Bcont.shape[1] # Number of control elements
Nd = None # No disturbances.

# Discretize.
dt = .025
Nt = 20
(Adisc,Bdisc) = mpc.c2d(Acont,Bcont,dt)
A = [Adisc]
B = [Bdisc]

# Bounds on u.
umax = 1
ulb = [np.array([-umax])]
uub = [np.array([umax])]
bounds = dict(uub=uub,ulb=ulb)

# Define Q and R matrices and q penalty for periodic solution.
Q = [np.diag([1,0])]
q = [np.zeros((Nx,1))]
R = [np.eye(Nu)]

# First define alternate SX versions that may be faster.
F_func = lambda x,u : mpc.mtimes(Adisc,x) + mpc.mtimes(Bdisc,u)
F_SX = [mpc.getCasadiFuncGeneralArgs(F_func,[Nx,Nu],["x","u"],"F",scalar=True)]

l_func = lambda x,u : mpc.mtimes(x.T,mpc.DMatrix(Q[0]),x) + mpc.mtimes(u.T,mpc.DMatrix(R[0]),u)
l_SX = [mpc.getCasadiFuncGeneralArgs(l_func,[Nx,Nu],["x","u"],"l",scalar=True)]

Pf_func = lambda x: mpc.mtimes(x.T,mpc.DMatrix(Q[0]),x)
Pf_SX = mpc.getCasadiFuncGeneralArgs(Pf_func,[Nx],["x"],"Pf",scalar=True)

# Convert everything to MX function form.
F = [mpc.getCasadiFunc(lambda x,u : list(F_func(x,u)),Nx,Nu,Nd,"F")]
l = [mpc.getCasadiFunc(lambda x,u : [l_func(x,u)],Nx,Nu,Nd,"l")]
Pf = mpc.getCasadiFunc(lambda x : [Pf_func(x)],Nx,0,0,"Pf")

# Now make TimeInvariantSolver object.
nsim = 100
verbosity = 0
solver = mpc.nmpc(F,l,[0,0],Nt,Pf,bounds,verbosity=verbosity,returnTimeInvariantSolver=True)
solver_SX = mpc.nmpc(F_SX,l_SX,[0,0],Nt,Pf_SX,bounds,verbosity=verbosity,returnTimeInvariantSolver=True)

# Solve.
t = np.arange(nsim+1)*dt
xcl = {}
ucl = {}
solvetimes = {}
tottimes = {}
for method in ["nmpc","lmpc","solver","solver_SX"]:
    starttime = time.clock()
    x0 = np.array([10,0])
    xcl[method] = np.zeros((Nx,nsim+1))
    xcl[method][:,0] = x0
    ucl[method] = np.zeros((Nu,nsim))
    solvetimes[method] = np.zeros((nsim,))    
    for k in range(nsim):
        # Solve linear MPC problem.
        if method == "nmpc":
            sol = mpc.nmpc(F,l,x0,Nt,Pf,bounds=bounds,verbosity=verbosity)
        elif method == "lmpc":
            sol = mpc.lmpc(A,B,x0,Nt,Q,R,q=q,bounds=bounds,verbosity=verbosity)
        elif method == "solver" or method == "solver_SX":
            if method == "solver":            
                solver.fixvar("x",0,x0)
                sol = solver.solve()
            else:
                solver_SX.fixvar("x",0,x0)
                sol = solver.solve()
            
            # Need to flip these because time comes first for these guys.
            sol["x"] = sol["x"].T
            sol["u"] = sol["u"].T
        else:
            raise ValueError("Invalid choice for method.")
        
        print "%s Iteration %d Status: %s" % (method,k,sol["status"])
        xcl[method][:,k] = sol["x"][:,0]
        ucl[method][:,k] = sol["u"][:,0]
        solvetimes[method][k] = sol["ipopttime"]
        x0 = sol["x"][:,1]
    
    xcl[method][:,nsim] = x0 # Store final state.
    endtime = time.clock()
    tottimes[method] = endtime - starttime

    # Plot things.
    fig = mpc.mpcplot(xcl[method],ucl[method],t,np.zeros(xcl[method].shape),xinds=[0])
    fig.canvas.set_window_title(method)    
    fig.show()

for m in tottimes.keys():
    print "%15s: %10.5g s total time, %10.5g s avg. sol. time" % (m,tottimes[m],np.mean(solvetimes[m]))