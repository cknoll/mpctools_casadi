# Example 1.11 (plus some extras) from Rawlings and Mayne.

import example1_11cstrmodel as cstrmodel
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
cstr = cstrmodel.CSTR()    
cstr.Delta = 1

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

xs = np.matrix([[cs],[Ts],[hs]])
us = np.matrix([[Tcs],[Fs]])
ds = np.matrix([[F0s]])

# First, simulate the system, and then see if the various discretization
# methods work well enough for the system.

def model(x,u):
    c = x[0] + cs
    T = x[1] + Ts
    h = x[2] + hs
    Tc = u[0] + Tcs
    F = u[1] + Fs
    F0 = F0s
    return cstr.ode(c,T,h,Tc,F,F0)

model_casadi = [mpc.getCasadiFunc(model,Nx,Nu,name="cstr")]
model_integrator = [mpc.getCasadiIntegrator(model,cstr.Delta,Nx,Nu,name="cstr")]

# Weighting matrices for controller.
Q = np.matrix(np.diag(np.array(xs).flatten()))**-2
R = np.matrix(np.diag(np.array(us).flatten()))**-2

[K, Pi] = cstrmodel.dlqr(A,B,Q,R)

# Define casadi functions.
l = lambda x,u: [mpc.mtimes(x.T,mpc.DMatrix(Q),x) + mpc.mtimes(u.T,mpc.DMatrix(R),u)]
l = [mpc.getCasadiFunc(l,Nx,Nu,name="l")]

Pf = lambda x: [mpc.mtimes(x.T,mpc.DMatrix(Pi),x)]
Pf = mpc.getCasadiFunc(Pf,Nx,name="Pf")

Nt = 50

kwargs = {}
kwargs["rk4"] = dict(F=model_casadi,l=l,x0=xs,Pf=Pf,timemodel="rk4",M=1,Delta=cstr.Delta,returnTimeInvariantSolver=True)
kwargs["colloc"] = dict(F=model_casadi,l=l,x0=xs,Pf=Pf,timemodel="colloc",M=2,Delta=cstr.Delta,returnTimeInvariantSolver=True)
kwargs["int"] = dict(F=model_integrator,l=l,x0=xs,Pf=Pf,returnTimeInvariantSolver=True)

solver = {}
guesser = {}
x0 = [0,0,0]
for k in kwargs.keys():
    solver[k] = mpc.nmpc(N=Nt,verbosity=3,**kwargs[k])
    solver[k].fixvar("x",0,x0)    
    guesser[k] = mpc.nmpc(N=1,verbosity=0,**kwargs[k])
    guesser[k].fixvar("x",0,x0)

# Define plotting function.
def cstrplot(x,u,ysp=None,contVars=[],title=None):
    x = x.T
    u = u.T
    u = np.concatenate((u,u[:,-1:]),axis=1)
    t = np.arange(0,x.shape[1])*cstr.Delta
    ylabelsx = ["$c$ (mol/L)", "$T$ (K)", "$h$ (m)"]
    ylabelsu = ["$T_c$ (K)", "$F$ (kL/min)"]
    
    fig = plt.figure(figsize=(6,10))
    for i in range(Nx):
        ax = fig.add_subplot(Nx + Nu,1,i+1)
        ax.plot(t,np.array(x[i,:]).flatten() + xs[i,0],'-ok')
        if i in contVars:
            ax.step(t,np.array(ysp[i,:]).flatten() + xs[i,0],'-r',where="post")
        ax.set_ylabel(ylabelsx[i])
    for i in range(Nu):
        ax = fig.add_subplot(Nx + Nu,1,i+1+Nx)
        ax.step(t,np.array(u[i,:]).flatten() + us[i,0],'-k',where="post")
        ax.set_ylabel(ylabelsu[i])
    ax.set_xlabel("Time (min)")
    fig.tight_layout(pad=.5)
    if title is not None:
        fig.canvas.set_window_title(title)
    return fig

# Simulate with a prespecified control profile.
u = np.zeros((Nt,Nu))
u[:,0] = .02*Tcs*(np.sin(2*np.pi*np.arange(Nt)/50) - .75)
u[:,1] = .01*Fs*np.cos(2*np.pi*np.arange(Nt)/50)

x = np.zeros((Nt+1,Nx))
xs_ = xs.A.flatten()
us_ = us.A.flatten()
ds_ = ds.A.flatten()
for t in range(Nt):
    x[t+1,:] = np.squeeze(cstr.sim(x[t,:] + xs_, u[t,:] + us_, ds_)) - xs_
    
cstrplot(x,u,title="Exact")

# Now fix control elements and solve as nlp. Generate a good (actually exact)
# guesss by solving a series of single-stage optimizations.
solvetimes = {}
solution = {}
for method in solver.keys():
    starttime = time.clock()
    for t in range(Nt):
        solver[method].fixvar("u",t,u[t,:])
        guesser[method].fixvar("u",0,u[t,:])
        guess = guesser[method].solve()
        print "%s %d: %s" % (method,t,guess["status"])
        if guess["status"] == "Solve_Succeeded":
            solver[method].guess["x",t+1] = guess["x"][1,:]
            if "xc" in guess.keys():
                solver[method].guess["xc",t] = guess["xc"][0,...]
            guesser[method].fixvar("x",0,guess["x"][1,:])
        
    solution[method] = solver[method].solve()
    solution[method]["xerr"] = (solution[method]["x"] + xs.A.T)/(x + xs.A.T) - 1
    solvetimes[method] = time.clock() - starttime
    cstrplot(solution[method]["x"],solution[method]["u"],title=method)
print "\nSolution times:"
print "\n".join([("  %10s took %10.4g s" % (method,solvetime)) for (method,solvetime) in solvetimes.items()])

# Plot relative error in states.
fig = plt.figure(figsize=(6,6))
t = np.arange(0,x.shape[0])*cstr.Delta
ylabels = ["Error in $c$", "Error in $T$", "Error in $h$"]
for i in range(Nx):
    ax = fig.add_subplot(Nx,1,i+1)
    for m in solution.keys():
        ax.plot(t,np.squeeze(solution[m]["xerr"][:,i]),label=m)
    ax.set_ylabel(ylabels[i])
    ax.legend(loc="upper center",ncol=len(solution.keys()))
ax.set_xlabel("Time")
fig.tight_layout(pad=.5)

#s = solver["rk4"]
#g = solution["rk4"]
#import pdb; pdb.set_trace()
#s.saveguess(g)

def doPart(part,graphfilename=None,disturbanceModel=True,nonlinearModel=True):
    """
    Does a part of the homework assignment.
    
    part should be an integer between -1 and 4.
    """

    # Define disturbance model.
    if part >= 0:
        Nd = Ny
    else: # For first case, we use a bad disturbance model.
        Nd = Nu
        
    Bd = np.matrix(np.zeros((Nx,Nd)))    
    Cd = np.matrix(np.zeros((Ny,Nd)))
    
    # Make some part-specific changes.
    if part == -1:
        # Use bad disturbance model that doesn't remove offset.
        Cd[0,0] = 1
        Cd[2,1] = 1
    elif part == 2:
         # Use different disturbance model.
        Cd[0,0] = 1
        Bd[1,1] = 1 # State disturbance for x.
        Bd[:,2] = Bp
    else:
        # Use book disturbance model.
        Cd[0,0] = 1
        Cd[2,1] = 1
        Bd[:,2] = Bp
    
    # Check rank condition for augmented system.
    svds = linalg.svd(np.vstack((np.hstack((np.matrix(np.eye(Nx)) - A, -Bd)),
                                 np.hstack((C, Cd)))),compute_uv=False)
    rank = sum(svds > eps)
    if rank < Nx + Nd:
        print "***Warning: augmented system is not detectable!"
    
    Qw = np.matrix(linalg.block_diag(eps*np.eye(Nx),np.eye(Nd)))
    Rv = eps*np.matrix(np.diag(np.array(xs).flatten()**2))
    [Aaug, Baug, Caug] = cstrmodel.augment(A,B,C,Bd,Cd)
    
    # Get Kalman filter.
    [L, P] = cstrmodel.dlqe(Aaug, Caug, Qw, Rv)
    Lx = L[:Nx,:]
    Ld = L[Nx:,:]
    
    # Now simulate things.
    Nsim = 50
    
    t = np.arange(Nsim+1)*cstr.Delta
    x = np.matrix(np.zeros((Nx,Nsim+1)))
    u = np.matrix(np.zeros((Nu,Nsim+1)))
    y = np.matrix(np.zeros((Ny,Nsim+1)))
    err = y.copy()
    v = y.copy()
    xhat = x.copy() # State estimate after measurement.
    xhatm = xhat.copy() # State estimate prior to measurement.
    dhat = np.matrix(np.zeros((Nd,Nsim+1)))
    dhatm = dhat.copy()
    
    # Pick disturbance and setpoint.
    d = np.matrix(.1*(t >= 10)*F0s)
    ysp = np.matrix(np.zeros((Ny,Nsim+1)))
    if part == 3:
        ysp[0,:] = -.05*(t >= 5)*cs
        ysp[2,:] = .1*(t >= 15)*hs
        
    # Initial condition and setpoint.
    x[:,0] = np.matrix([[0],[0],[0]])
    if part == 1:
        contVars = [1,2]
    else:
        contVars = [0,2]
    
    H = np.matrix(np.zeros((Nu,Ny)))
    for i in range(len(contVars)):
        H[i,contVars[i]] = 1
    
    # Random numbers.
    np.random.seed(0)
    if part == 4:
        v = 5*np.sqrt(Rv)*np.matrix(np.random.randn(Ny,Nsim+1))
    else:
        v = np.matrix(np.zeros((Ny,Nsim+1)))
    
    for n in range(Nsim + 1):
        # Take plant measurement.
        y[:,n] = C*x[:,n] + v[:,n]
        
        # Update state estimate with measurement.
        err[:,n] = y[:,n] - C*xhatm[:,n] - Cd*dhatm[:,n]
        xhat[:,n] = xhatm[:,n] + Lx*err[:,n]
        if disturbanceModel:
            dhat[:,n] = dhatm[:,n] + Ld*err[:,n]
        
        if n != Nsim: # Make sure we aren't at the last timestep.
            # Steady-state target.
            [xsp, usp] = cstrmodel.sstarg(A,B,C,Q,R,ysp[:,n],H,
                                            dhat[:,n],Bd,Cd,unique=True)
            
            # Regulator.
            u[:,n] = K*(xhat[:,n] - xsp) + usp
            
            # Simulate. 
            if nonlinearModel:
                # System may be unstable, so we break if this fails.
                try:
                    x[:,n+1] = cstr.sim(x[:,n]+xs, u[:,n]+us, d[:,n]+ds) - xs
                except:
                    print "***Error: integrator failed!"
                    break
            else:
                x[:,n+1] = A*x[:,n] + B*u[:,n] + Bp*d[:,n]
            
            # Advance state estimate.
            xhatm[:,n+1] = A*xhat[:,n] + Bd*dhat[:,n] + B*u[:,n]
            if disturbanceModel:
                dhatm[:,n+1] = dhat[:,n]
        else:
             # Duplicate last value to make stairstep plot easier.
            u[:,Nsim] = u[:,Nsim-1]
        
    # Now make plots.
    ylabelsx = ["$c$ (mol/L)", "$T$ (K)", "$h$ (m)"]
    ylabelsu = ["$T_c$ (K)", "$F$ (kL/min)"]
    
    fig = plt.figure(figsize=(6,10))
    for i in range(Nx):
        ax = fig.add_subplot(Nx + Nu,1,i+1)
        ax.plot(t,np.array(x[i,:]).flatten() + xs[i,0],'-ok')
        if i in contVars:
            ax.step(t,np.array(ysp[i,:]).flatten() + xs[i,0],'-r',where="post")
        ax.set_ylabel(ylabelsx[i])
    for i in range(Nu):
        ax = fig.add_subplot(Nx + Nu,1,i+1+Nx)
        ax.step(t,np.array(u[i,:]).flatten() + us[i,0],'-k',where="post")
        ax.set_ylabel(ylabelsu[i])
    ax.set_xlabel("Time (min)")
    
    fig.tight_layout(pad=.5)
    if graphfilename is not None:
        fig.savefig("ex1-11" + graphfilename)
        
    return dict(x=x,xhat=xhat,xhatm=xhatm,dhat=dhat,dhatm=dhat,y=y,u=u,
                err=err,d=d,ysp=ysp,v=v)

plt.close("all")
parts = [-1,0] #[-1,0,1,2,3,4]
for p in parts:
    filename = ("part%d.pdf" % p).replace("-","m")
    doPart(p,filename)
    
    
