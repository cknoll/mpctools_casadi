# Nonlinear control of a CSTR.

import cstrmodel
import mpc_tools_casadi as mpc
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import time

plt.close("all")

# Steady-state values.
cs = .878
Ts = 324.5
hs = .659
Fs = .1
Tcs = 300
F0s = .1

eps = 1e-8 # Use this as a small number.

# Get discrete-time model.
cstr = cstrmodel.CSTR(Delta=.2)    
#cstr.Delta = .25

# Update the steady-state values a few times to make sure they don't move.
for i in range(10):
    [[cs],[Ts],[hs]] = cstr.sim([cs,Ts,hs],[Tcs,Fs],[F0s]).tolist()

Nx = cstr.Nx()
Nu = cstr.Nu()
Ny = cstr.Ny()

ss = cstr.getLinearization(cs,Ts,hs,Tcs,Fs,F0s)
A = ss["A"]
B = ss["B"]
Bp = ss["Bp"]
C = ss["C"]

xs = np.array([cs,Ts,hs])
us = np.array([Tcs,Fs])
ds = np.array([F0s])

# First, simulate the system, and then see if the various discretization
# methods work well enough for the system.

# Define nonlinear model, but use the disturbance to denote the setpoint.
Np = Nx + Nu + cstr.Nd()
def model(x,u,d):
    p = d # We use d to store parameters.
    c = x[0] + p[0]
    T = x[1] + p[1]
    h = x[2] + p[2]
    Tc = u[0] + p[3]
    F = u[1] + p[4]
    F0 = p[5]
    return cstr.ode(c,T,h,Tc,F,F0)

ps = np.array([cs,Ts,hs,Tcs,Fs,F0s])

model_casadi = [mpc.getCasadiFunc(model,Nx,Nu,Np,name="cstr")]
#model_integrator = [mpc.getCasadiIntegrator(model,cstr.Delta,Nx,Nu,Np,name="cstr")]
# This doesn't work, which is a problem.

# Weighting matrices for controller.
Q = .5*np.matrix(np.diag(np.array(xs).flatten()))**-2
R = 2*np.matrix(np.diag(np.array(us).flatten()))**-2

[K, Pi] = cstrmodel.dlqr(A,B,Q,R)

# Define casadi functions.
Flinear = lambda x,u: [mpc.mtimes(mpc.DMatrix(A),x) + mpc.mtimes(mpc.DMatrix(B),u)]
Flinear = [mpc.getCasadiFunc(Flinear,Nx,Nu,name="F")]

l = lambda x,u: [mpc.mtimes(x.T,mpc.DMatrix(Q),x) + mpc.mtimes(u.T,mpc.DMatrix(R),u)]
l = [mpc.getCasadiFunc(l,Nx,Nu,name="l")]

Pf = lambda x: [mpc.mtimes(x.T,mpc.DMatrix(Pi),x)]
Pf = mpc.getCasadiFunc(Pf,Nx,name="Pf")

Nt = 10

# Control bounds.
umax = np.array([.01*Ts,.1*Fs])
bounds = dict(uub=[umax],ulb=[-umax])

# Define plotting function.
def cstrplot(x,u,ysp=None,contVars=[],ulb=None,uub=None,title=None):
    u = np.concatenate((u,u[-1:,:]))
    t = np.arange(0,x.shape[0])*cstr.Delta
    ylabelsx = ["$c$ (mol/L)", "$T$ (K)", "$h$ (m)"]
    ylabelsu = ["$T_c$ (K)", "$F$ (kL/min)"]
    
    fig = plt.figure(figsize=(6,10))
    for i in range(Nx):
        ax = fig.add_subplot(Nx + Nu,1,i+1)
        ax.plot(t,np.array(x[:,i]).flatten(),'-ok')
        if i in contVars:
            ax.step(t,np.array(ysp[:,i]).flatten(),'-g',where="post")
        ax.set_ylabel(ylabelsx[i])
    for i in range(Nu):
        ax = fig.add_subplot(Nx + Nu,1,i+1+Nx)
        ax.step(t,np.array(u[:,i]).flatten(),'-k',where="post")
        if uub is not None:
            ax.plot([t[0],t[-1]],uub[i]*np.ones(2,),'-r')
        if ulb is not None:
            ax.plot([t[0],t[-1]],ulb[i]*np.ones(2,),'-r')
        ax.set_ylabel(ylabelsu[i])
    ax.set_xlabel("Time (min)")
    fig.tight_layout(pad=.5)
    if title is not None:
        fig.canvas.set_window_title(title)
    return fig

# Build a solver for the linear and nonlinear models.
x0 = np.array([.05*cs,.75*Ts,.5*hs])
solvers = {}
solvers["lmpc"] = mpc.nmpc(N=Nt,verbosity=0,F=Flinear,l=l,x0=x0,Pf=Pf,returnTimeInvariantSolver=True,bounds=bounds)
solvers["nmpc"] = mpc.nmpc(N=Nt,verbosity=0,F=model_casadi,l=l,x0=x0,Pf=Pf,timemodel="rk4",M=1,Delta=cstr.Delta,returnTimeInvariantSolver=True,bounds=bounds,d=[ps])

# First see what happens if we try to start up the reactor under no control.
Nsim = 40

xnocontrol = np.zeros((Nsim+1,Nx))
xnocontrol[0,:] = x0
unocontrol = np.tile(us,(Nsim,1))
for t in range(Nsim):
    xnocontrol[t+1,:] = cstr.sim(xnocontrol[t,:],unocontrol[t,:],ds,matrix=False)
cstrplot(xnocontrol,unocontrol,title="Uncontrolled Startup")

# Now simulate the process under control.
xcl = {}
ucl = {}
for method in solvers.keys():
    xcl[method] = np.zeros((Nsim+1,Nx))
    xcl[method][0,:] = x0
    thisx = x0    
    ucl[method] = np.zeros((Nsim,Nu))
    xsp = np.tile(xs,(Nsim+1,1))
    #xsp[Nsim/2:,0] = 1.5*cs # Make a setpoint change in the second half.    
    for t in range(Nsim):
        thisxs = xsp[t,:]
        if method == "nmpc":
            thisp = np.concatenate((thisxs,us,ds))
            solvers[method].changeparvals("d",[thisp])
        solvers[method].fixvar("x",0,thisx - thisxs)
        sol = solvers[method].solve()
        print "%10s %3d: %s" % (method,t,sol["status"])
        if sol["status"] != "Solve_Succeeded":
            break
        else:
            solvers[method].saveguess(sol)
            
        if t == 0:
            cstrplot(sol["x"] + xs,sol["u"] + us,title="Initial prediction (%s)" % method)            
            
        thisu = sol["u"][0,...].copy() + us
        ucl[method][t,:] = thisu
        thisx = cstr.sim(thisx,thisu,ds,matrix=False)
        xcl[method][t+1,:] = thisx
    cstrplot(xcl[method],ucl[method],ysp=xsp,title=method,contVars=range(Nx),uub=bounds["uub"][0] + us,ulb=bounds["ulb"][0] + us)



# Now try control with the nonlinear solver.
#raise NotImplementedError("Work in progress.")

########
#kwargs = {}
#kwargs["rk4"] = dict(F=model_casadi,l=l,x0=xs,Pf=Pf,timemodel="rk4",M=1,Delta=cstr.Delta,returnTimeInvariantSolver=True)
##kwargs["colloc"] = dict(F=model_casadi,l=l,x0=xs,Pf=Pf,timemodel="colloc",M=2,Delta=cstr.Delta,returnTimeInvariantSolver=True)
##kwargs["int"] = dict(F=model_integrator,l=l,x0=xs,Pf=Pf,returnTimeInvariantSolver=True)
#
#solver = {}
#guesser = {}
#x0 = [0,0,0]
#for k in kwargs.keys():
#    solver[k] = mpc.nmpc(N=Nt,verbosity=3,**kwargs[k])
#    solver[k].fixvar("x",0,x0)    
#    guesser[k] = mpc.nmpc(N=1,verbosity=0,**kwargs[k])
#    guesser[k].fixvar("x",0,x0)
#    #solver[k].storeguesser(mpc.nmpc(N=1,verbosity=0,**kwargs[k]))
#    #solver[k].storeguesser(guesser[k])
#    
#
## Simulate with a prespecified control profile.
#u = np.zeros((Nt,Nu))
#u[:,0] = .02*Tcs*(np.sin(2*np.pi*np.arange(Nt)/50) - .75)
#u[:,1] = .01*Fs*np.cos(2*np.pi*np.arange(Nt)/50)
#
#x = np.zeros((Nt+1,Nx))
#xs_ = xs.A.flatten()
#us_ = us.A.flatten()
#ds_ = ds.A.flatten()
#for t in range(Nt):
#    x[t+1,:] = np.squeeze(cstr.sim(x[t,:] + xs_, u[t,:] + us_, ds_)) - xs_
#    
#cstrplot(x,u,title="Exact")
#
## Now fix control elements and solve as nlp. Generate a good (actually exact)
## guesss by solving a series of single-stage optimizations.
#solvetimes = {}
#solution = {}
#for method in solver.keys():
#    starttime = time.clock()
#    for t in range(Nt):
#        solver[method].fixvar("u",t,u[t,:])
#        guesser[method].fixvar("u",0,u[t,:])
#        guess = guesser[method].solve()
#        #guess = solver[method].calcguess(t)
#        print "%s %d: %s" % (method,t,guess["status"])
#        if guess["status"] == "Solve_Succeeded":
#            solver[method].guess["x",t+1] = guess["x"][1,:]
#            if "xc" in guess.keys():
#                solver[method].guess["xc",t] = guess["xc"][0,...]
#            guesser[method].fixvar("x",0,guess["x"][1,:])
#    
#    finalguess = mpc.casadiStruct2numpyDict(solver[method].guess)    
#    solution[method] = solver[method].solve()
#    solution[method]["xerr"] = (solution[method]["x"] + xs.A.T)/(x + xs.A.T) - 1
#    solvetimes[method] = time.clock() - starttime
#    cstrplot(solution[method]["x"],solution[method]["u"],title=method)
#    cstrplot(finalguess["x"],finalguess["u"],title=method + " Guess")
#print "\nSolution times:"
#print "\n".join([("  %10s took %10.4g s" % (method,solvetime)) for (method,solvetime) in solvetimes.items()])