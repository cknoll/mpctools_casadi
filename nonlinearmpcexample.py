# Example for nonlinear MPC with Casadi.

# Imports.
import numpy as np
import mpc_tools_casadi as mpc

verb = 2

# Problem parameters.
Delta = .1
N = 50
t = np.arange(N+1)*Delta

Nx = 2
Nu = 2
Nd = 0

xsp = np.zeros((Nx,N+1))

# We're going to solve the same problem using linear mpc, nonlinear mpc starting
# with a discrete-time model, and nonlinear mpc starting from a continuous-time
# model. In theory, the results should be identical.

Acont = np.array([[-1,1],[0,-1]])
Bcont = np.eye(2)

(Adisc,Bdisc) = mpc.c2d(Acont,Bcont,Delta)
A = [Adisc]
B = [Bdisc]

# Bounds, initial condition, and stage costs.
bounds = {"ulb" : [-1,-1], "uub" : [1,1]}
x0 = [10,10]

Q = [np.eye(2)]
R = [np.eye(2)]

l = lambda x,u : [mpc.mtimes(x.T,mpc.DMatrix(Q[0]),x) + mpc.mtimes(u.T,mpc.DMatrix(R[0]),u)]
l = [mpc.getCasadiFunc(l,Nx,Nu,Nd,"l")]
Pf = lambda x: [mpc.mtimes(x.T,mpc.DMatrix(Q[0]),x)]
Pf = mpc.getCasadiFunc(Pf,Nx,0,0,"Pf")

# Solve problem with linear mpc and plot.
opt_lmpc = mpc.lmpc(A,B,x0,N,Q,R,bounds=bounds,verbosity=verb)
fig_lmpc = mpc.mpcplot(opt_lmpc["x"],opt_lmpc["u"],t,xsp,xinds=[0,1])
fig_lmpc.canvas.set_window_title("Linear MPC")
fig_lmpc.show()

# Discrete-time example
Adisc = mpc.DMatrix(Adisc) # Cast to Casadi matrix type.
Bdisc = mpc.DMatrix(Bdisc)

Fdiscrete = lambda x,u : list(mpc.mtimes(Adisc,x) + mpc.mtimes(Bdisc,u))
F = [mpc.getCasadiFunc(Fdiscrete,Nx,Nu,Nd,"F")]

opt_dnmpc = mpc.nmpc(F,l,x0,N,Pf,bounds,d=None,verbosity=verb)
fig_dnmpc = mpc.mpcplot(opt_dnmpc["x"],opt_dnmpc["u"],t,xsp,xinds=[0,1])
fig_dnmpc.canvas.set_window_title("Discrete-time NMPC")
fig_dnmpc.show()

# Continuous time interfaces in nmpc.
Acont = mpc.DMatrix(Acont) # Cast to Casadi matrix type.
Bcont = mpc.DMatrix(Bcont)

f = lambda x,u : list(mpc.mtimes(Acont,x) + mpc.mtimes(Bcont,u))
fcasadi = [mpc.getCasadiFunc(f,Nx,Nu,Nd)]
Mrk4 = 5
Mcolloc = 5

opt_crk4nmpc = mpc.nmpc(fcasadi,l,x0,N,Pf,bounds,d=None,verbosity=verb,timemodel="rk4",M=Mrk4,Delta=Delta)
fig_crk4nmpc = mpc.mpcplot(opt_crk4nmpc["x"],opt_crk4nmpc["u"],t,xsp,xinds=[0,1])
fig_crk4nmpc.canvas.set_window_title("Continuous-time NMPC (RK4)")
fig_crk4nmpc.show()

opt_ccollocnmpc = mpc.nmpc(fcasadi,l,x0,N,Pf,bounds,d=None,verbosity=verb,timemodel="colloc",M=Mcolloc,Delta=Delta)
fig_ccollocnmpc = mpc.mpcplot(opt_ccollocnmpc["x"],opt_ccollocnmpc["u"],t,xsp,xinds=[0,1])
fig_ccollocnmpc.canvas.set_window_title("Continuous-time NMPC (Collocation)")
fig_ccollocnmpc.show()

# Discrete-time but with Casadi's integrators.
Fintegrator = [mpc.getCasadiIntegrator(f,Delta,Nx,Nu,Nd)]
opt_integrator = mpc.nmpc(F,l,x0,N,Pf,bounds,d=None,verbosity=verb)
fig_integrator = mpc.mpcplot(opt_integrator["x"],opt_integrator["u"],t,xsp,xinds=[0,1])
fig_integrator.canvas.set_window_title("NMPC with Casadi Integrators")
fig_integrator.show()

