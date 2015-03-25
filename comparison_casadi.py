# Control of the Van der Pol oscillator using pure casadi.
import casadi
import casadi.tools as ctools
import numpy as np
import matplotlib.pyplot as plt

# Define model and get simulator.
Delta = .5
Nx = 2
Nu = 1
def ode(x,u):
    return [(1 - x[1]*x[1])*x[0] - x[1] + u, x[0]]

# Define symbolic variables.
x = casadi.MX.sym("x",Nx)
u = casadi.MX.sym("u",Nu)

# Make integrator object.
ode_integrator = casadi.MXFunction(casadi.daeIn(x=x,p=u),casadi.daeOut(ode=casadi.vertcat(ode(x,u))))
vdp = casadi.Integrator("cvodes",ode_integrator)
vdp.setOption("abstol",1e-8)
vdp.setOption("reltol",1e-8)
vdp.setOption("tf",Delta)
vdp.init()

# Then get nonlinear casadi functions and rk4 discretization.
ode_casadi = casadi.MXFunction([x,u],[casadi.vertcat(ode(x,u))])
ode_casadi.init()

[k1] = ode_casadi([x,u])
[k2] = ode_casadi([x + Delta/2*k1,u])
[k3] = ode_casadi([x + Delta/2*k2,u])
[k4] = ode_casadi([x + Delta*k3,u])
xrk4 = x + Delta/6*(k1 + 2*k2 + 2*k3 + k4)    
ode_rk4_casadi = casadi.MXFunction([x,u],[xrk4])
ode_rk4_casadi.init()

# Define stage cost and terminal weight.
lfunc = casadi.mul([x.T,x]) + casadi.mul([u.T,u])
l = casadi.MXFunction([x,u],[lfunc])
l.init()

Pffunc = casadi.mul([x.T,x])
Pf = casadi.MXFunction([x],[Pffunc])
Pf.init()

# Bounds on u.
uub = 1
ulb = -.75

# Make optimizers.
x0 = np.array([0,1])
Nt = 20

# Create variables struct.
var = ctools.struct_symMX([(
    ctools.entry("x",shape=(Nx,),repeat=Nt+1),
    ctools.entry("u",shape=(Nu,),repeat=Nt),
)])
varlb = var(-np.inf)
varub = var(np.inf)
varguess = var(0)

# Adjust the relevant constraints.
for t in range(Nt):
    varlb["u",t,:] = ulb
    varub["u",t,:] = uub

# Now build up constraints and objective.
obj = casadi.MX(0)
con = []
for t in range(Nt):
    con.append(ode_rk4_casadi([var["x",t],var["u",t]])[0] - var["x",t+1])
    obj += l([var["x",t],var["u",t]])[0]
obj += Pf([var["x",Nt]])[0]

# Build solver object.
con = casadi.vertcat(con)
conlb = np.zeros((Nx*Nt,))
conub = np.zeros((Nx*Nt,))

nlp = casadi.MXFunction(casadi.nlpIn(x=var),casadi.nlpOut(f=obj,g=con))
solver = casadi.NlpSolver("ipopt",nlp)
solver.setOption("print_level",0)
solver.setOption("print_time",False)  
solver.setOption("max_cpu_time",60)
solver.init()

solver.setInput(conlb,"lbg")
solver.setInput(conub,"ubg")

# Now simulate.
Nsim = 20
times = Delta*Nsim*np.linspace(0,1,Nsim+1)
x = np.zeros((Nsim+1,Nx))
x[0,:] = x0
u = np.zeros((Nsim,Nu))
for t in range(Nsim):
    # Fix initial state.    
    varlb["x",0,:] = x[t,:]
    varub["x",0,:] = x[t,:]
    varguess["x",0,:] = x[t,:]
    solver.setInput(varguess,"x0")
    solver.setInput(varlb,"lbx")
    solver.setInput(varub,"ubx")    
    
    # Solve nlp.    
    solver.evaluate()
    status = solver.getStat("return_status")
    optvar = var(solver.getOutput("x"))
    
    # Display stats.
    print "%d: %s" % (t,status)
    u[t,:] = optvar["u",0,:]
    
    # Simulate.
    vdp.setInput(x[t,:],"x0")
    vdp.setInput(u[t,:],"p")
    vdp.evaluate()
    x[t+1,:] = np.array(vdp.getOutput("xf")).flatten()
    vdp.reset()
    
# Plots.
fig = plt.figure()
numrows = max(Nx,Nu)
numcols = 2

# u plots.
u = np.concatenate((u,u[-1:,:])) # Repeat last element for stairstep plot.
for i in range(Nu):
    ax = fig.add_subplot(numrows,numcols,numcols*(i+1))
    ax.step(times,u[:,i],"-k")
    ax.set_xlabel("Time")
    ax.set_ylabel("Control %d" % (i + 1))

# x plots.    
for i in range(Nx):
    ax = fig.add_subplot(numrows,numcols,numcols*(i+1) - 1)
    ax.plot(times,x[:,i],"-k",label="System")
    ax.set_xlabel("Time")
    ax.set_ylabel("State %d" % (i + 1))

fig.tight_layout(pad=.5)