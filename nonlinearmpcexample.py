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

l = lambda x,u : [mpc.mtimes(x.T,mpc.np2mx(Q[0]),x) + mpc.mtimes(u.T,mpc.np2mx(R[0]),u)]
l = [mpc.getCasadiFunc(l,2,2,0,"l")]
Pf = lambda x: mpc.mtimes(x.T,mpc.np2mx(Q[0]),x)

# Solve problem with linear mpc and plot.
opt_lmpc = mpc.lmpc(A,B,x0,N,Q,R,ulb=bounds["ulb"],uub=bounds["uub"],verbosity=verb)
fig_lmpc = mpc.mpcplot(opt_lmpc["x"],opt_lmpc["u"],t,xsp,xinds=[0,1])
fig_lmpc.canvas.set_window_title("Linear MPC")
fig_lmpc.show()

# Discrete-time example
Adisc = mpc.np2mx(Adisc) # Cast to Casadi MX object.
Bdisc = mpc.np2mx(Bdisc)

Fdiscrete = lambda x,u : list(mpc.mtimes(Adisc,x) + mpc.mtimes(Bdisc,u))
F = [mpc.getCasadiFunc(Fdiscrete,Nx,Nu,Nd,"F")]

opt_dnmpc = mpc.nmpc(F,l,Pf,x0,N,bounds,d=None,verbosity=verb)
fig_dnmpc = mpc.mpcplot(opt_dnmpc["x"],opt_dnmpc["u"],t,xsp,xinds=[0,1])
fig_dnmpc.canvas.set_window_title("Discrete-time NMPC")
fig_dnmpc.show()

# Continuous-time example with RK4 discretization.
Acont = mpc.np2mx(Acont) # Cast to Casadi MX object.
Bcont = mpc.np2mx(Bcont)

f = lambda x,u : list(mpc.mtimes(Acont,x) + mpc.mtimes(Bcont,u))
fcasadi = mpc.getCasadiFunc(f,Nx,Nu,Nd)
F = [mpc.getRungeKutta4(fcasadi,Delta,5)]

opt_cnmpc = mpc.nmpc(F,l,Pf,x0,N,bounds,d=None,verbosity=verb)
fig_cnmpc = mpc.mpcplot(opt_cnmpc["x"],opt_cnmpc["u"],t,xsp,xinds=[0,1])
fig_cnmpc.canvas.set_window_title("Continuous-time NMPC")
fig_cnmpc.show()
