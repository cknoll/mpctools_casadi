# Control of the Van der Pol
# oscillator using pure casadi.
import casadi
import casadi.tools as ctools
import numpy as np
import matplotlib.pyplot as plt

#<<ENDCHUNK>>

# Define model and get simulator.
Delta = .5
Nt = 20
Nx = 2
Nu = 1
def ode(x,u):
    dxdt = [
        (1 - x[1]*x[1])*x[0] - x[1] + u,
        x[0]]
    return np.array(dxdt)

#<<ENDCHUNK>>

# Define symbolic variables.
x = casadi.SX.sym("x",Nx)
u = casadi.SX.sym("u",Nu)

# Make integrator object.
ode_integrator = casadi.SXFunction(
    casadi.daeIn(x=x,p=u),
    casadi.daeOut(ode=casadi.vertcat(ode(x,u))))
vdp = casadi.Integrator("cvodes",ode_integrator)
vdp.setOption("abstol",1e-8)
vdp.setOption("reltol",1e-8)
vdp.setOption("tf",Delta)
vdp.init()

#<<ENDCHUNK>>

# Then get nonlinear casadi functions
# and rk4 discretization.
ode_casadi = casadi.SXFunction(
    [x,u],[casadi.vertcat(ode(x,u))])
ode_casadi.init()

[k1] = ode_casadi([x,u])
[k2] = ode_casadi([x + Delta/2*k1,u])
[k3] = ode_casadi([x + Delta/2*k2,u])
[k4] = ode_casadi([x + Delta*k3,u])
xrk4 = x + Delta/6*(k1 + 2*k2 + 2*k3 + k4)    
ode_rk4_casadi = casadi.SXFunction([x,u],[xrk4])
ode_rk4_casadi.init()

#<<ENDCHUNK>>

# Define stage cost and terminal weight.
lfunc = (casadi.mul([x.T,x])
    + casadi.mul([u.T,u]))
l = casadi.SXFunction([x,u],[lfunc])
l.init()

Pffunc = casadi.mul([x.T,x])
Pf = casadi.SXFunction([x],[Pffunc])
Pf.init()

#<<ENDCHUNK>>

# Bounds on u.
uub = 1
ulb = -.75

#<<ENDCHUNK>>

# Make optimizers.
x0 = np.array([0,1])

# Create variables struct.
var = ctools.struct_symSX([(
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
obj = casadi.SX(0)
con = []
for t in range(Nt):
    con.append(ode_rk4_casadi([var["x",t],
        var["u",t]])[0] - var["x",t+1])
    obj += l([var["x",t],var["u",t]])[0]
obj += Pf([var["x",Nt]])[0]

# Build solver object.
con = casadi.vertcat(con)
conlb = np.zeros((Nx*Nt,))
conub = np.zeros((Nx*Nt,))

nlp = casadi.SXFunction(casadi.nlpIn(x=var),
    casadi.nlpOut(f=obj,g=con))
solver = casadi.NlpSolver("ipopt",nlp)
solver.setOption("print_level",0)
solver.setOption("print_time",False)  
solver.setOption("max_cpu_time",60)
solver.init()

solver.setInput(conlb,"lbg")
solver.setInput(conub,"ubg")

#<<ENDCHUNK>>

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
    
    #<<ENDCHUNK>>    
    
    # Solve nlp.    
    solver.evaluate()
    status = solver.getStat("return_status")
    optvar = var(solver.getOutput("x"))
    
    #<<ENDCHUNK>>    
    
    # Print stats.
    print "%d: %s" % (t,status)
    u[t,:] = optvar["u",0,:]
    
    #<<ENDCHUNK>>    
    
    # Simulate.
    vdp.setInput(x[t,:],"x0")
    vdp.setInput(u[t,:],"p")
    vdp.evaluate()
    x[t+1,:] = np.array(
        vdp.getOutput("xf")).flatten()
    vdp.reset()

#<<ENDCHUNK>>
    
# Plots.
fig = plt.figure()
numrows = max(Nx,Nu)
numcols = 2

# u plots. Need to repeat last element
# for stairstep plot.
u = np.concatenate((u,u[-1:,:]))
for i in range(Nu):
    ax = fig.add_subplot(numrows,
        numcols,numcols*(i+1))
    ax.step(times,u[:,i],"-k")
    ax.set_xlabel("Time")
    ax.set_ylabel("Control %d" % (i + 1))

# x plots.    
for i in range(Nx):
    ax = fig.add_subplot(numrows,
        numcols,numcols*(i+1) - 1)
    ax.plot(times,x[:,i],"-k",label="System")
    ax.set_xlabel("Time")
    ax.set_ylabel("State %d" % (i + 1))

fig.tight_layout(pad=.5)
import mpctools.plots # Need to grab one function to show plot.
mpctools.plots.showandsave(fig,"comparison_casadi.pdf")
