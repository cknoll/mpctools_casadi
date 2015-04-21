# Compares various model forms for a linear problem.

# Imports.
import numpy as np
import mpctools as mpc
import mpctools.legacy as mpc_old

verb = 2

# Problem parameters.
Delta = .1
Nt = 50
t = np.arange(Nt+1)*Delta

Nx = 2
Nu = 2
Nd = 0

xsp = np.zeros((Nt+1,Nx))

# We're going to solve the same problem using linear mpc, nonlinear mpc
# starting with an exact discrete-time model, and nonlinear mpc starting from
# a continuous-time model. In theory, the results should be identical.

Acont = np.array([[-1,1],[0,-1]])
Bcont = np.eye(2)

(A,B) = mpc.util.c2d(Acont,Bcont,Delta)

# Bounds, initial condition, and stage costs.
ulb = np.array([-1,-1])
uub = np.array([1,1])
lb = {"u" : np.tile(ulb,(Nt,1))}
ub = {"u" : np.tile(uub,(Nt,1))}
bounds = {"uub" : [uub], "ulb" : [ulb]}
x0 = [10,10]

Q = np.eye(Nx)
R = np.eye(Nx)

def lfunc(x,u): return mpc.mtimes(x.T,Q,x) + mpc.mtimes(u.T,R,u)
l = mpc.getCasadiFunc(lfunc,[Nx,Nu],["x","u"],"l")

def Pffunc(x): return mpc.mtimes(x.T,Q,x)
Pf = mpc.getCasadiFunc(Pffunc,[Nx],["x"],"Pf")

# Solve problem with linear mpc and plot.
print "=Linear="
opt_lmpc = mpc_old.linear.lmpc([A],[B],x0,Nt,[Q],[R],bounds=bounds,
                               verbosity=verb)
fig_lmpc = mpc.plots.mpcplot(opt_lmpc["x"],opt_lmpc["u"],t,xsp.T,
                             xinds=[0,1],timefirst=False)
fig_lmpc.canvas.set_window_title("Linear MPC")

# Discrete-time example
def Fdiscrete(x,u): return mpc.mtimes(A,x) + mpc.mtimes(B,u)
F = mpc.getCasadiFunc(Fdiscrete,[Nx,Nu],["x","u"],"F")

print "\n=Exact Discretization="
N = {"x":Nx, "u":Nu, "t":Nt}
opt_dnmpc = mpc.nmpc(F,l,N,x0,lb,ub,Pf=Pf,verbosity=verb)
fig_dnmpc = mpc.plots.mpcplot(opt_dnmpc["x"],opt_dnmpc["u"],t,xsp,
                              xinds=[0,1])
fig_dnmpc.canvas.set_window_title("Discrete-time NMPC")

# Continuous time interfaces in nmpc.
def f(x,u): return mpc.mtimes(Acont,x) + mpc.mtimes(Bcont,u)

Mrk4 = 5
Mcolloc = 5

def F_rk4(x,u): return mpc.util.rk4(f,x,[u],Delta=Delta,M=Mrk4)
F_rk4 = mpc.getCasadiFunc(F_rk4,[Nx,Nu],["x","u"],"F_rk4")    

print "\n=RK4 Discretization="
opt_crk4nmpc = mpc.nmpc(F_rk4,l,N,x0,lb,ub,Pf=Pf,verbosity=verb)
fig_crk4nmpc = mpc.plots.mpcplot(opt_crk4nmpc["x"],opt_crk4nmpc["u"],t,
                                 xsp,xinds=[0,1])
fig_crk4nmpc.canvas.set_window_title("Continuous-time NMPC (RK4)")

print "\n=Collocation Discretization="
Ncolloc = N.copy()
Ncolloc["c"] = Mcolloc
opt_ccollocnmpc = mpc.nmpc(F,l,N,x0,lb,ub,Pf=Pf,verbosity=verb,Delta=Delta)
fig_ccollocnmpc = mpc.plots.mpcplot(opt_ccollocnmpc["x"],
                                    opt_ccollocnmpc["u"],t,xsp,xinds=[0,1])
fig_ccollocnmpc.canvas.set_window_title("Continuous-time NMPC (Collocation)")

# Discrete-time but with Casadi's integrators. This is slow, but it may be
# necessary if your ODE is difficult to discretize.
print "\n=Casadi Integrator Discretization="
F_integrator = mpc.tools.getCasadiIntegrator(f,Delta,[Nx,Nu],["x","u"],"int_f")
opt_integrator = mpc.nmpc(F_integrator,l,N,x0,lb,ub,Pf=Pf,verbosity=verb,
                          scalar=False)
fig_integrator = mpc.plots.mpcplot(opt_integrator["x"],
                                   opt_integrator["u"],t,xsp,xinds=[0,1])
fig_integrator.canvas.set_window_title("NMPC with Casadi Integrators")

