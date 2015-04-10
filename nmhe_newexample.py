# Example 4.27 in Rawlings and Mayne.
import mpc_tools_casadi as mpc
import casadi
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import linalg
from numpy import random

random.seed(927) # Seed random number generator.

verb = 2
doPlots = True
fullInformation = False # Whether to use full information estimation or MHE.

# Problem parameters.
Nt = 10 # Horizon length
Delta = 0.25 # Time step
Nsim = 80 # Length of the simulation
tplot = np.arange(Nsim+1)*Delta

Nx = 3
Nu = 1
Ny = 1
Nw = Nx
Nv = Ny

sigma_v = 0.25 # Standard deviation of the measurements
sigma_w = 0.001 # Standard deviation for the process noise
sigma_p = .5 # Standard deviation for prior

# Make covariance matrices.
P = np.diag((sigma_p*np.ones((Nx,)))**2) # Covariance for prior.
Q = np.diag((sigma_w*np.ones((Nw,)))**2)
R = np.diag((sigma_v*np.ones((Nv,)))**2)

x_0 = np.array([1.0,0.0,4.0])
x0 = np.array([0.5,0.05,0.0])

# Parameters of the system
k1 = 0.5
k_1 = 0.05
k2 = 0.2
k_2 = 0.01
RT = 32.84

# Continuous-time models.
def ode(x,u,w=[0,0,0]): # We define the model with u, but there isn't one.
    [cA, cB, cC] = x[:Nx]    
    rate1 = k1*cA - k_1*cB*cC    
    rate2 = k2*cB**2 - k_2*cC
    return np.array([-rate1 + w[0], rate1 - 2*rate2 + w[1], rate1 + rate2 + w[2]])    

def measurement(x):
    return RT*(x[0] + x[1] + x[2])
        
# Need to use getCasadiFuncGeneralArgs because arguments are different.
ode_casadi = mpc.getCasadiFuncGeneralArgs(ode,[Nx,Nu,Nw],["x","u","w"],"F",scalar=True)
model = mpc.OneStepSimulator(ode, Delta, Nx, Nu, Nd=0, Nw=Nw)    

# Convert continuous-time f to explicit discrete-time F with RK4.
F = mpc.getRungeKutta4(ode_casadi,Delta,M=2,argsizes=[Nx,Nu,Nw])
H = mpc.getCasadiFuncGeneralArgs(measurement,[Nx],["x"],"H",scalar=True)

# Define stage costs.
l = lambda w,v: sigma_w**-2*casadi.sum_square(w) + sigma_v**-2*casadi.sum_square(v)
lx = lambda x: mpc.mtimes(x.T,linalg.inv(P),x)

l = mpc.getCasadiFuncGeneralArgs(l,[Nw,Nv],["w","v"],"l",scalar=True)
lx = mpc.getCasadiFuncGeneralArgs(lx,[Nx],["x"],"lx",scalar=True)

# First simulate everything.
w = sigma_w*random.randn(Nsim,Nw)
v = sigma_v*random.randn(Nsim,Nv)

usim = np.zeros((Nsim,Nu)) # This is just a dummy input.
xsim = np.zeros((Nsim+1,Nx))
xsim[0,:] = x0
ysim = np.zeros((Nsim,Ny))

for t in range(Nsim):
    ysim[t] = measurement(xsim[t]) + v[t,:]    
    xsim[t+1,:] = model.sim(xsim[t,:],u=usim[t,:],w=w[t,:])
    
# Plots.
colors = ["red","blue","green"]
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for i in range(Nx):
    ax.plot(tplot,xsim[:,i],color=colors[i])
ax.set_xlabel("Time")
ax.set_ylabel("Concentration")    

# Now do estimation. We're just going to use full-information estimation.
xhat_ = np.zeros((Nsim+1,Nx))
xhat = np.zeros((Nsim,Nx))
vhat = np.zeros((Nsim,Nv))
what = np.zeros((Nsim,Nw))
x0bar = x_0
xhat[0,:] = x0bar
guess = {}
initialtime = time.clock()
for t in range(Nsim):
    # Define sizes of everything.    
    N = {"x":Nx, "y":Ny, "t":min(t+1,Nt), "u":Nu}
    if fullInformation:
        N["t"] = t+1
        tmin = 0
    else:
        tmin = max(0,t+1-Nt)
    tmax = t+1        
    lb = {"x":np.zeros((tmax - tmin + 1,Nx))}  

    # Call solver.
    starttime = time.clock()
    sol = mpc.nmhe_new(f=F,h=H,u=usim[tmin:tmax,:],y=ysim[tmin:tmax,:],l=l,N=N,
                    lx=lx,x0bar=x0bar,verbosity=0,guess=guess,lb=lb)
    print "%3d (%10.5g s): %s" % (t,time.clock() - starttime,sol["status"])
    if sol["status"] != "Solve_Succeeded":
        break
    xhat[t,:] = sol["x"][-2,...]     # This is xhat( t  | t )
    xhat_[t+1,:] = sol["x"][-1,...]  # This is xhat(t+1 | t )
    what[t,:] = sol["w"][-1,...]
    vhat[t,:] = sol["v"][-1,...]
        
    # Save stuff to use as a guess. Cycle the guess.
    guess = {}
    for k in ["x","w","v"]:
        guess[k] = sol[k].copy()
    
    # Do some different things if not using full information estimation.    
    if not fullInformation and t + 1 >= Nt:
        for k in guess.keys():
            guess[k] = guess[k][1:,...] # Get rid of oldest measurement.
            
        # Do EKF to update prior covariance, but don't take EKF state.
        [P, x0bar, _, _] = mpc.ekf(F,H,x=sol["x"][0,...],
            u=usim[tmin,:],w=sol["w"][0,...],y=ysim[tmin,:],P=P,Q=Q,R=R)
        
        # Need to redefine arrival cost.
        lx = lambda x: [mpc.mtimes(x.T,linalg.inv(P),x)]
        lx = mpc.getCasadiFuncGeneralArgs(lx,[Nx],["x"],"lx")      
    
     # Add final guess state for new time point.
    for k in guess.keys():
        guess[k] = np.concatenate((guess[k],guess[k][-1:,...]))
                
print "Simulation took %.5g s." % (time.clock() - initialtime)

# Add to plots.
for i in range(Nx):
    ax.plot(tplot[:-1],xhat[:,i],"o",color=colors[i],markeredgecolor="none",markersize=3)
mpc.zoomaxis(ax, xscale=1.05,yscale=1.05)
fig.savefig("nmhe_exercise.pdf")
