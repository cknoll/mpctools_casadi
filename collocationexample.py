import mpc_tools_casadi as mpc
import numpy as np
import casadi
import colloc
import matplotlib.pyplot as plt

# Build model.
f = lambda x, u: [-x[0], -.5*x[1]]
Nx = 2
Nu = 1
Nd = 0
f = mpc.getCasadiFunc(f,Nx,Nu,Nd,"f")
x0 = [1, 1]
Delta = 1

# Pick horizon and number of collocation points.
Nt = 5
Nc = 10

[r,A,B,q] = colloc.colloc(Nc, True, True)

verbosity = 3

[VAR, CON, CONLB, CONUB] = mpc.getCollocationConstraints(f,Delta,Nt,Nc)
LB = VAR(-np.inf)
UB = VAR(np.inf)
GUESS = VAR(0)

# Initial condition.
GUESS["x",0,:] = x0
LB["x",0,:] = x0
UB["x",0,:] = x0


nlpObj = casadi.MX(0) # Start with dummy objective.
nlpCon = casadi.vertcat(CON)

# Create solver and stuff.
nlp = casadi.MXFunction(casadi.nlpIn(x=VAR),casadi.nlpOut(f=nlpObj,g=nlpCon))
solver = casadi.NlpSolver("ipopt",nlp)
solver.setOption("print_level",verbosity)
solver.setOption("print_time",verbosity > 2)   
solver.init()

solver.setInput(GUESS,"x0")
solver.setInput(LB,"lbx")
solver.setInput(UB,"ubx")
solver.setInput(CONLB,"lbg")
solver.setInput(CONUB,"ubg")

# Solve.    
solver.evaluate()
status = solver.getStat("return_status")
if verbosity > 0:
    print("Solver Status:", status)

# Get solution.
OPTVAR = VAR(solver.getOutput("x"))
x = np.hstack(OPTVAR["x",:])
u = np.hstack(OPTVAR["u",:])
z = np.hstack(OPTVAR["z",:])

# Plot some stuff.
tx = Delta*np.arange(Nt + 1)
tz = np.hstack([Delta*(n + r[1:-1]) for n in range(Nt)])
tfine = np.linspace(0,Nt*Delta,250)
xfine = x0[0]*np.exp(-tfine)

f = plt.figure()
ax = f.add_subplot(1,1,1)
ax.plot(tfine,xfine,'-k',label="Analytical")
ax.plot(tx,x[0,:],'o',label="State Points",markerfacecolor="k",markeredgecolor="k")
ax.plot(tz,z[0,:],'o',label="Collocation Points",markerfacecolor="none",markeredgecolor="r")
ax.legend(loc="upper right")
ax.set_xlabel("Time")
ax.set_ylabel("$x$")

