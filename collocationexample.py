import mpc_tools_casadi as mpc
import numpy as np
import casadi
import colloc
import matplotlib.pyplot as plt

# Build model.
k00 = -2
k11 = -.5
k10 = 1

f = lambda x, u: [k00*x[0], k11*x[1] + k10*x[0]]
Nx = 2
Nu = 1
Nd = 0
f = mpc.getCasadiFunc(f,Nx,Nu,Nd,"f")
x0 = [1, 1]
Delta = .5

# Pick horizon and number of collocation points.
Nt = 10
Nc = 10

[r,A,B,q] = colloc.colloc(Nc, True, True)

verbosity = 3
[VAR, LB, UB, GUESS] = mpc.getCasadiVars(Nx,Nu,Nt,Nc)    
[CON, CONLB, CONUB] = mpc.getCollocationConstraints(f,VAR,Delta)
CON = mpc.flattenlist(CON)

# Initial condition.
GUESS["x",0,:] = x0
LB["x",0,:] = x0
UB["x",0,:] = x0

nlpObj = casadi.MX(0) # Start with dummy objective.
nlpCon = casadi.vertcat(CON)

# Create solver and stuff.
[OPTVAR, obj, status, solver] = mpc.callSolver(VAR, LB, UB, GUESS, nlpObj, nlpCon, CONLB, CONUB, verbosity)
x = np.hstack(OPTVAR["x",:])
u = np.hstack(OPTVAR["u",:])
z = np.hstack(OPTVAR["xc",:])

# Plot some stuff.
tx = Delta*np.arange(Nt + 1)
tz = np.hstack([Delta*(n + r[1:-1]) for n in range(Nt)])
tfine = np.linspace(0,Nt*Delta,250)
xfine = np.zeros((2,len(tfine)))

# Analytical solution.
xfine[0,:] = x0[0]*np.exp(k00*tfine)

xfine[1,:] = xfine[0,:]/x0[0]*k10/(k00 - k11)
xfine[1,:] += (x0[1] - xfine[1,0])*np.exp(k11*tfine) 

# Plots.
f = plt.figure()
for i in range(2):
    ax = f.add_subplot(2,1,1+i)
    ax.plot(tfine,xfine[i,:],'-k',label="Analytical")
    ax.plot(tx,x[i,:],'o',label="State Points",markerfacecolor="k",markeredgecolor="k")
    ax.plot(tz,z[i,:],'o',label="Collocation Points",markerfacecolor="none",markeredgecolor="r")
    ax.set_ylabel("$x_{%d}$" % (i,))    
    ax.legend(loc="upper right")
ax.set_xlabel("Time")


