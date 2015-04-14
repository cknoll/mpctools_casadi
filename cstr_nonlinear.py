# Example 1.11 from Rawlings and Mayne with linear and nonlinear control.
import mpctools as mpc
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib import gridspec
import time

# Decide whether to use casadi SX objects or MX. I would expect SX to be faster
# (i.e. set useCasadiSX to True), but it seems MX does much better for this
# problem (i.e. set useCasadiMX to False). Not sure why.
useCasadiSX = False

# Define some parameters and then the CSTR model.
Nx = 3
Nu = 2
Nd = 1
Ny = Nx
Nid = Ny # Number of integrating disturbances.
Nw = Nx + Nid # Noise on augmented state.
Nv = Ny # Noise on outputs.
Delta = 1
eps = 1e-6 # Use this as a small number.

T0 = 350
c0 = 1
r = .219
k0 = 7.2e10
E = 8750
U = 54.94
rho = 1000
Cp = .239
dH = -5e4

def cstrmodel(c,T,h,Tc,F,F0):
    # ODE for CSTR.
    rate = k0*c*np.exp(-E/T)
        
    dxdt = np.array([
        F0*(c0 - c)/(np.pi*r**2*h) - rate,
        F0*(T0 - T)/(np.pi*r**2*h)
            - dH/(rho*Cp)*rate
            + 2*U/(r*rho*Cp)*(Tc - T),    
        (F0 - F)/(np.pi*r**2)
    ])
    return dxdt

# Steady-state values.
cs = .878
Ts = 324.5
hs = .659
Fs = .1
Tcs = 300
F0s = .1

def ode(x,u,d):
    # Grab the states, controls, and disturbance.
    [c, T, h] = x[0:Nx]
    [Tc, F] = u[0:Nu]
    [F0] = d[0:Nd]
    return cstrmodel(c,T,h,Tc,F,F0)

def ode_rk4(x,u,d):
    return mpc.util.rk4(ode,x,[u,d],Delta,2)

# Turn into casadi function and simulator.
ode_casadi = mpc.getCasadiFunc(ode,[Nx,Nu,Nd],["x","u","d"],"ode",True)
ode_rk4_casadi = mpc.getCasadiFunc(ode_rk4,[Nx,Nu,Nd],["x","u","d"],"ode_rk4",True)

cstr = mpc.OneStepSimulator(ode, Delta, Nx, Nu, Nd, vector=True)

# Update the steady-state values a few times to make sure they don't move.
for i in range(10):
    [cs,Ts,hs] = cstr.sim([cs,Ts,hs],[Tcs,Fs],[F0s]).tolist()
xs = np.array([cs,Ts,hs])
xaugs = np.concatenate((xs,np.zeros((Nid,))))
us = np.array([Tcs,Fs])
ds = np.array([F0s])
ps = np.concatenate((ds,xs,us))

# Define augmented model for state estimation.    
def ode_augmented(x,u,d=ds):
    # Grab states, estimated disturbances, controls, and actual disturbance.
    [c, T, h] = x[0:Nx]
    dhat = x[Nx:Nx+Nid]
    [Tc, F] = u[0:Nu]
    [F0] = d[0:Nd]
    
    dxdt = mpc.util.vertcat([cstrmodel(c,T,h,Tc,F+dhat[2],F0)] + [0]*Ny)
    return dxdt
cstraug = mpc.OneStepSimulator(ode_augmented, Delta, Nx + Nid, Nu, Nd, vector=True)

def ode_augmented_rk4(x,u,d=ds):
    return mpc.util.rk4(ode_augmented,x,[u,d],Delta,2)

def ode_estimator_rk4(x,u,w=np.zeros((Nx+Nid,)),d=ds):
    return mpc.util.rk4(ode_augmented,x,[u,d],Delta,2) + w

def measurement(x,d=ds):
    [c, T, h] = x[0:Nx]
    dhat = x[Nx:Nx+Nid]
    return np.array([c + dhat[0], T, h + dhat[1]])
ys = measurement(xaugs)

# Turn into casadi functions.
ode_augmented_casadi = mpc.getCasadiFunc(ode_augmented,[Nx+Nid,Nu,Nd],["xaug","u","d"],"ode_augmented")
ode_augmented_rk4_casadi = mpc.getCasadiFunc(ode_augmented_rk4,[Nx+Nid,Nu,Nd],["xaug","u","d"],"ode_augmented_rk4")
ode_estimator_rk4_casadi = mpc.getCasadiFunc(ode_estimator_rk4,[Nx+Nid,Nu,Nw,Nd],["xaug","u","w","d"],"ode_augmented_rk4")
measurement_casadi = mpc.getCasadiFunc(measurement,[Nx+Nid,Nd],["xaug","d"],"measurement")

# Now get a linearization at this steady state.
ss = mpc.util.getLinearization(ode_casadi, xs, us, ds, Delta)
A = ss["A"]
B = ss["B"]
Bp = ss["Bp"]
C = np.eye(Nx)

# Weighting matrices for controller.
Q = .5*np.diag(xs**-2)
R = 2*np.diag(us**-2)

[K, Pi] = mpc.util.dlqr(A,B,Q,R)

# Get nonlinear controller object.
Nt = 5

def stagecost(x,u,xsp,usp):
    # Return deviation variables.
    dx = x[:Nx] - xsp[:Nx]
    du = u - usp
    
    # Calculate stage cost.
    return mpc.mtimes(dx.T,Q,dx) + mpc.mtimes(du.T,R,du)
l = mpc.getCasadiFunc(stagecost,[Nx+Nid,Nu,Nx+Nid,Nu],["x","u","x_sp","u_sp"],funcname="l")

def costtogo(x,xsp):
    # Deviation variables.
    dx = x[:Nx] - xsp[:Nx]
    
    # Calculate cost to go.
    return mpc.mtimes(dx.T,Pi,dx)
Pf = mpc.getCasadiFunc(costtogo,[Nx+Nid,Nx+Nid],["x","s_xp"],funcname="Pf")

ubounds = np.array([.05*Tcs, .5*Fs])
bounds = dict(uub=[us + ubounds],ulb=[us - ubounds])
lb = {"u" : np.tile(us - ubounds, (Nt,1))}
ub = {"u" : np.tile(us + ubounds, (Nt,1))}

N = {"x":Nx+Nid, "u":Nu, "p":Nd, "t":Nt}
p = np.tile(ds, (Nt,1)) # Parameters for system.
sp = {"x" : np.tile(xaugs, (Nt+1,1)), "u" : np.tile(us, (Nt,1))}
guess = sp.copy()
x0 = xs
xaug0 = xaugs
nmpc = mpc.nmpc(ode_augmented_rk4_casadi,l,N,xaug0,lb,ub,guess,Pf=Pf,sp=sp,p=p,
    verbosity=0,timelimit=60,runOptimization=False,scalar=useCasadiSX)

# Define augmented system with disturbance model.
ss_augmented = mpc.util.getLinearization(ode_augmented_casadi, xaugs, us, ds, Delta)
Aaug = ss_augmented["A"]
Baug = ss_augmented["B"]
Caug = mpc.util.getLinearization(measurement_casadi, xaugs)["A"]

# Extract the various submatrices.
Bd = Aaug[:Nx,Nx:]
Cd = Caug[:,Nx:]

# Check rank condition for augmented system.
svds = linalg.svdvals(np.bmat([[np.eye(Nx) - A, -Bd],[C,Cd]]))
rank = sum(svds > 1e-10)
if rank < Nx + Nid:
    print "***Warning: augmented system is not detectable!"

# Build augmented estimator matrices.
Qw = eps*np.eye(Nx + Nid)
Qw[-1,-1] = 1
Rv = eps*np.diag(xs**2)
Qwinv = linalg.inv(Qw)
Rvinv = linalg.inv(Rv)

# Get Kalman filter.
[L, P] = mpc.util.dlqe(Aaug, Caug, Qw, Rv)

# Define stage costs for estimator.
def lest(w,v): return mpc.mtimes(w.T,Qwinv,w) + mpc.mtimes(v.T,Rvinv,v) 
lest = mpc.getCasadiFunc(lest,[Nw,Nv],["w","v"],"l")

#lxest = lambda x: mpc.mtimes(x.T,Qwinv,x)
#lxest = mpc.getCasadiFuncGeneralArgs(lxest,[Nx + Nid],["x"],"lx",True)
#x0bar = xaugs
lxest = None # We shouldn't need a prior.
x0bar = None

# Define plotting function.
def cstrplot(x,u,ysp=None,contVars=[],title=None):
    u = np.concatenate((u,u[-1:,:]))
    t = np.arange(0,x.shape[0])*Delta
    ylabelsx = ["$c$ (mol/L)", "$T$ (K)", "$h$ (m)"]
    ylabelsu = ["$T_c$ (K)", "$F$ (kL/min)"]
    
    gs = gridspec.GridSpec(Nx*Nu,2)    
    
    fig = plt.figure(figsize=(10,6))
    for i in range(Nx):
        ax = fig.add_subplot(gs[i*Nu:(i+1)*Nu,0])
        ax.plot(t,x[:,i],'-ok')
        if i in contVars:
            ax.step(t,ysp[:,i],'-r',where="post")
        ax.set_ylabel(ylabelsx[i])
        mpc.plots.zoomaxis(ax,yscale=1.1)
    ax.set_xlabel("Time (min)")
    for i in range(Nu):
        ax = fig.add_subplot(gs[i*Nx:(i+1)*Nx,1])
        ax.step(t,u[:,i],'-k',where="post")
        ax.set_ylabel(ylabelsu[i])
        mpc.plots.zoomaxis(ax,yscale=1.25)
    ax.set_xlabel("Time (min)")
    fig.tight_layout(pad=.5)
    if title is not None:
        fig.canvas.set_window_title(title)
    return fig

# Now simulate things.
useMeasuredState = False
Nsim = 50
t = np.arange(Nsim+1)*Delta
for linear in [True,False]:
    starttime = time.clock()
    x = np.zeros((Nsim+1,Nx))
    x[0,:] = xs # Start at steady state.    
    
    u = np.zeros((Nsim,Nu))
    usp = np.zeros((Nsim,Nu))
    xaugsp = np.zeros((Nsim,Nx+Nid))
    y = np.zeros((Nsim+1,Ny))
    err = y.copy()
    v = y.copy()
    
    xhatm = np.zeros((Nsim+1,Nx+Nid)) # Augmented state estimate before measurement.
    xhatm[0,:] = xaugs # Start with estimate at steaty state.
    xhat = np.zeros((Nsim,Nx+Nid)) # Augmented state estimate after measurement.
    
    # Pick disturbance, setpoint, and initial condition.
    d = np.zeros((Nsim,Nd))
    d[:,0] = (t[:-1] >= 10)*(t[:-1] <= 30)*.1*F0s
    d += ds
    
    ysp = np.tile(xs, (Nsim,1))
    contVars = [0,2] # Concentration and height.
    
    # Steady-state target selector matrices.
    if linear:
        H = np.zeros((Nu,Ny))
        for i in range(len(contVars)):
            H[i,contVars[i]] = 1
        G = np.bmat([[np.eye(Nx) - A, -B],[H.dot(C), np.zeros((H.shape[0], Nu))]]).A
    else:
        # Make steady-state target selector bounds.
        sstarglb = {"u" : np.tile(us - ubounds, (1,1))}
        sstargub = {"u" : np.tile(us + ubounds, (1,1))}
        sstargguess = {"u" : np.tile(us, (1,1)), "x" : np.tile(np.concatenate((xs,np.zeros((Nid,)))), (1,1)), "y" : np.tile(xs, (1,1))}        
        sstargp = np.tile(ds, (1,1)) # Parameters for system.
        sstargN = {"x" : Nx + Nid, "u" : Nu, "y" : Ny, "p" : Nd}
        sstarg = mpc.sstarg(ode_augmented_casadi,measurement_casadi,sstargN,
            lb=sstarglb,ub=sstargub,guess=sstargguess,p=sstargp,
            verbosity=0,discretef=False,runOptimization=False,
            scalar=useCasadiSX)
        estimatorguess = {}
        estimatorguess["x"] = np.tile(xaugs, (1,1))
        estimatorguess["v"] = np.zeros((1,Nv))
        estimatorguess["w"] = np.zeros((0,Nw))
    
    for n in range(Nsim):
        # Take plant measurement.
        y[n,:] = measurement(np.concatenate((x[n,:],np.zeros((Nid,))))) + v[n,:]
        
        # Update state estimate with measurement.
        err[n,:] = y[n,:] - measurement(xhatm[n,:])
        
        if linear:
            # Use Kalman Filter
            xhat[n,:] = xhatm[n,:] + L.dot(err[n,:])
            
            # Steady-state target.
            dhat = xhat[n,Nx:]
            rhs = np.concatenate((Bd.dot(dhat), H.dot(ysp[n,:] - ys - Cd.dot(dhat))))
            qsp = linalg.solve(G,rhs) # Similar to "G\rhs" in Matlab.
            xaugsp[n,:Nx] = qsp[:Nx] + xs
            usp[n,:] = qsp[Nx:] + us
            
            # Regulator.
            u[n,:] = K.dot(xhat[n,:Nx] - xaugsp[n,:Nx]) + usp[n,:]     
        else:
            print "(%3d) " % (n,), 
            
            # Handle disturbance.
            if useMeasuredState:
                # Just directly measure state and get the correct disturbance.
                x0hat = x[n,:]
                xhat[n,:Nx] = x0hat # Estimate is exact.
                nmpc.par["p"] = [d[n,:]]*Nt
                sstarg.par["p",0] = d[n,:]
            else:
                # Do full-information estimation.
                N = {"x":Nx + Nid, "u":Nu, "y":Ny, "p":Nd, "t":n}
                estsol = mpc.nmhe(f=ode_estimator_rk4_casadi,h=measurement_casadi,u=u[:n,:],y=y[:n+1,:],l=lest,N=N,
                    lx=lxest,x0bar=x0bar,p=np.tile(ds,(n+1,1)),verbosity=0,guess=estimatorguess,timelimit=5,scalar=useCasadiSX)
                
                print "Estimator: %s, " % (estsol["status"],),
                #mpc.keyboard()             
                xhat[n,:] = estsol["x"][-1,:] # Update our guess.
                
                # Update estimator guess.
                estimatorguess = {}
                for k in ["x","w","v"]:
                    if k in estsol.keys():
                        estimatorguess[k] = np.concatenate((estsol[k][:,...],estsol[k][-1:,...]))
                    elif n == 0 and k == "w":
                        estimatorguess[k] = np.zeros((1,Nw))
                    else:
                        raise KeyError("Entry '%s' missing from estimator solution. Something went wrong." % (k,))
                
                dguess = ds
                sstarg.par["p",0] = ds                
                
            # Use nonlinear steady-state target selector.
            xtarget = np.concatenate((ysp[n,:],xhat[n,Nx:])) # Setpoint for augmented state.
            x0hat = xhat[n,:]
            
            if n == 0:
                uguess = us
            else:
                uguess = u[n-1,:]
            sstarg.fixvar("y",0,ysp[n,contVars],contVars)
            sstarg.guess["x",0] = xtarget
            sstarg.fixvar("x",0,xhat[n,Nx:],range(Nx,Nx+Nid)) # Fix the augmented states.

            sstarg.guess["u",0] = uguess
            sstarg.solve()
            
            xaugsp[n,:] = np.squeeze(sstarg.var["x",0,:])
            usp[n,:] = np.squeeze(sstarg.var["u",0,:])

            print "Target: %s, " % (sstarg.stats["status"],), 
            if sstarg.stats["status"] != "Solve_Succeeded":
                import pdb; pdb.set_trace()
                break

            # Now use nonlinear MPC controller.
            nmpc.par["x_sp"] = [xaugsp[n,:]]*(Nt + 1)
            nmpc.par["u_sp"] = [usp[n,:]]*Nt
            nmpc.fixvar("x",0,x0hat)            
            nmpc.solve()
            print "Controller: %s, " % (nmpc.stats["status"],), 
            
            nmpc.saveguess()
            u[n,:] = np.squeeze(nmpc.var["u",0])
        
        # Simulate with nonilnear model.
        try:
            x[n+1,:] = cstr.sim(x[n,:], u[n,:], d[n,:])
        except:
            print "***Error during simulation!"
            break
        
        # Advance state estimate.
        if not linear:
            xhatm[n+1,:] = cstraug.sim(xhat[n,:], u[n,:], ds)
            print ""
        else:
            xhatm[n+1,:] = Aaug.dot(xhat[n,:] - xaugs) + Baug.dot(u[n,:] - us) + xaugs
    
    # Take final measurement.
    y[Nsim,:] = measurement(np.concatenate((x[Nsim,:],np.zeros((Nid,))))) + v[Nsim,:]
    endtime = time.clock()
    print "%s Took %.5g s." % ("Linear" if linear else "Nonlinear", endtime - starttime,)
       
    fig = cstrplot(x,u,ysp=None,contVars=[],title="Linear" if linear else "Nonlinear")
fig.savefig("cstr_nonlinear.pdf")
