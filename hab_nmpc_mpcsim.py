# This is a hot-air balloon example
#
# ToDo:
# (1) Add soft output constraints and expose tuning weights
# (2) Make the fuel an integer MV
# (3) Constrain the predicted altitude to h >= 0
# (4) Expose the mhe tuning parameters
# (5) Linearized LQG version
#
# Tom Badgwell 05/11/17

from mpctools import mpcsim as sim
import mpctools as mpc
import numpy as np
from scipy import linalg

useCasadiSX = True

def runsim(k, simcon, opnclsd):

    print "runsim: iteration %d -----------------------------------" % k

    # Unpack stuff from simulation container.

    mvlist = simcon.mvlist
    cvlist = simcon.cvlist
    xvlist = simcon.xvlist
    oplist = simcon.oplist
    deltat = simcon.deltat
    nf = oplist[0]
    doolpred = oplist[1]
    fuelincrement = oplist[2].value

    # Check for changes.

    chsum = 0

    for var in mvlist:
        chsum += var.chflag
        var.chflag = 0
    
    for var in xvlist:
        chsum += var.chflag
        var.chflag = 0
    
    for var in cvlist:
        chsum += var.chflag
        var.chflag = 0
    
    for var in oplist:
        chsum += var.chflag
        var.chflag = 0
    
    # Grab bounds.

    uub = [mvlist[0].maxlim, mvlist[1].maxlim]
    ulb = [mvlist[0].minlim, mvlist[1].minlim]
    yub = [cvlist[0].maxlim, cvlist[1].maxlim, cvlist[2].maxlim]
    ylb = [cvlist[0].minlim, cvlist[1].minlim, cvlist[2].minlim]
    
    # Initialize values on first execution or when something changes.

    if (k == 0 or chsum > 0):

        print "runsim: initialization"

        # Define problem size parameters.

        Nx   = 3            # number of states
        Nu   = 2            # number of inputs
        Ny   = 3            # number of outputs
        Nid  = Ny           # number of integrating disturbances

        Nw   = Nx + Nid     # number of augmented states
        Nv   = Ny           # number of output measurements
        Nf   = cvlist[0].Nf # length of NMPC future horizon
        Nmhe = 60           # length of MHE past horizon

        psize = (Nx, Nu, Ny, Nid, Nw, Nv, Nf, Nmhe)

        # Define sample time in minutes.
        
        Delta = deltat

        # Set the initial values.

        y0 = np.zeros((Ny,))
        x0 = np.zeros((Nx,))
        u0 = np.zeros((Nu,))
        for i in range(Ny):
            y0[i] = cvlist[i].value
        for i in range(Nx):
            x0[i] = xvlist[i].value
        for i in range(Nu):
            u0[i] = mvlist[i].value
#        for i in range(Nd):
#            d0[i] = dvlist[i].value

        # Define scaling factors
            
        h0 = 1.1e4         # altitude scaling factor            (m)
        T0 = 288.2         # air temperature at takeoff         (K)
        f0 = 3672          # fuel flowrate at takeoff           (sccm)
        p0 = 100.0         # vent position                      (%)
        t0 = 33.49         # time scale factor                  (s)
        #tc = 100.0         # reference trajectory time constant (s) # Apparently not used.
        
        # Define parameters for hot-air balloon

        alpha  = 2.549     # balloon number
        omega  = 20.0      # drag coefficient
        beta   = 0.1116    # heat transfer coefficient
        gamma  = 5.257     # atmosphere number
        delta  = 0.2481    # temperature drop-off coefficient
        lambde = 1.00      # vent coefficient

        # Define ode for the hot-air balloon .

        def ode(x, u):
            """ODE for hot air balloon system."""
            f     = (1 + 0.03*u[0])*100/f0
            term1 = alpha*(1 - delta*x[0])**(gamma - 1)
            term2 = (1 - (1 - delta*x[0])/x[2])
            term3 = beta*(x[2] -1 + delta*x[0])
            term4 = (1 + lambde*u[1]/p0)
            term5 = omega*x[1]*np.fabs(x[1])
            dx1dt = x[1]
            dx2dt = term1*term2 - 0.5 - term5
            dx3dt = -term3*term4 + f
            dxdt = np.array([dx1dt, dx2dt, dx3dt])
            return dxdt

        # Create simulator.

        hab = mpc.DiscreteSimulator(ode, Delta, [Nx,Nu], ["x","u"])

        # Initialize the steady-state values

        ys = y0
        xs = x0
        us = u0
        xaugs = np.concatenate((xs, np.zeros((Nid,))))
        
        # Define augmented model for state estimation.    

        # We need to define two of these because Ipopt isn't smart enough to throw out
        # the 0 = 0 equality constraints. ode_disturbance only gives dx/dt for the
        # actual states, and ode_augmented appends the zeros so that dx/dt is given for
        # all of the states. It's possible that Ipopt may figure this out by itself,
        # but we should just be explicit to avoid any bugs.    

        def ode_disturbance(x, u):

            # For this case there are no input disturbances.

            dxdt = ode(x[:Nx], u)
            return dxdt

        def ode_augmented(x, u):

            # Need to add extra zeros for derivative of disturbance states.

            dxdt = mpc.vcat([ode_disturbance(x, u), np.zeros((Nid,))])
            return dxdt
 
        habaug = mpc.DiscreteSimulator(ode_augmented, Delta,
                                        [Nx+Nid,Nu], ["x","u"])
        

        # Only the first three states are measured

        Cx = np.array([[1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]])

        def measurement(x):

            # For this case all of the disturbances are output disturbances.

            xc    = x[:Nx]
            dhat  = x[Nx:Nx+Nid]
            yd = mpc.mtimes(Cx, xc) + dhat
            ym = yd
            ym[0] = yd[0]*h0
            ym[1] = yd[1]*h0/t0
#           ym[2] = yd[2]
            ym[2] = yd[2]*T0 - 273.2
            return ym

        # Turn into casadi functions.

        ode_disturbance_casadi = mpc.getCasadiFunc(ode_disturbance,
                                   [Nx+Nid,Nu], ["x","u"],"ode_disturbance")
        ode_augmented_rk4_casadi = mpc.getCasadiFunc(ode_augmented,
                                   [Nx+Nid,Nu],["x","u"],"ode_augmented_rk4",
                                     rk4=True,Delta=Delta,M=2)

        def ode_estimator_rk4(x,u,w=np.zeros((Nx+Nid,))):
            return ode_augmented_rk4_casadi(x, u) + w

        ode_estimator_rk4_casadi = mpc.getCasadiFunc(ode_estimator_rk4,
                                   [Nx+Nid,Nu,Nw], ["x","u","w"],
                                   "ode_estimator_rk4", scalar=False)

        measurement_casadi = mpc.getCasadiFunc(measurement,
                             [Nx+Nid], ["x"], "measurement")

        # Weighting matrices for controller.

        Qy  = np.diag([cvlist[0].qvalue, cvlist[1].qvalue, cvlist[2].qvalue])
        Qx  = mpc.mtimes(Cx.T,Qy,Cx)
        R   = np.diag([mvlist[0].rvalue, mvlist[1].rvalue])
        S   = np.diag([mvlist[0].svalue, mvlist[1].svalue])

        # Define control stage cost.
        lbslack = np.array([[cv.lbslack for cv in cvlist]])
        ubslack = np.array([[cv.ubslack for cv in cvlist]])
        def stagecost(x, u, xsp, usp, Deltau, s):
            dx = x[:Nx] - xsp[:Nx]
            du = u - usp
            slb = s[:Ny]
            sub = s[Ny:]
            slack = mpc.mtimes(lbslack, slb) + mpc.mtimes(ubslack, sub)
            return (mpc.mtimes(dx.T,Qx,dx) + .1*mpc.mtimes(du.T,R,du)
                + mpc.mtimes(Deltau.T,S,Deltau) + slack)

        largs = ["x","u","x_sp","u_sp","Du","s"]
        l = mpc.getCasadiFunc(stagecost,
            [Nx+Nid,Nu,Nx+Nid,Nu,Nu,2*Ny],largs,funcname="l",scalar=False)

        # Define cost to go.

        def costtogo(x,xsp):
            dx = x[:Nx] - xsp[:Nx]
            return mpc.mtimes(dx.T, Qx, dx)
        Pf = mpc.getCasadiFunc(costtogo,[Nx+Nid,Nx+Nid],["x","x_sp"],
                               funcname="Pf", scalar=False)

        # Build augmented estimator matrices.

        Qw = np.diag([xv.mnoise for xv in xvlist] + [xv.dnoise for xv in xvlist])
        Rv = np.diag([cv.mnoise for cv in cvlist])
        Qwinv = linalg.inv(Qw)
        Rvinv = linalg.inv(Rv)

        # Define stage costs for estimator.

        def lest(w,v):
            return mpc.mtimes(w.T,Qwinv,w) + mpc.mtimes(v.T,Rvinv,v)
                      
        lest = mpc.getCasadiFunc(lest,[Nw,Nv],["w","v"],"l",scalar=False)

        # Don't use a prior.

        lxest = None
        x0bar = None

        # Make NMHE solver.

        uguess = np.tile(us,(Nmhe,1))
        xguess = np.tile(xaugs,(Nmhe+1,1))
        yguess = np.tile(ys,(Nmhe+1,1))
        nmheargs = {
            "f" : ode_estimator_rk4_casadi,
            "h" : measurement_casadi,
            "u" : uguess,
            "y" : yguess,
            "l" : lest,
            "N" : {"x":Nx + Nid, "u":Nu, "y":Ny, "t":Nmhe},
            "lx" : lxest,
            "x0bar" : x0bar,
            "verbosity" : 0,
            "guess" : {"x":xguess, "y":yguess, "u":uguess},
            "timelimit" : 60,
            "casaditype" : "SX" if useCasadiSX else "MX",
        }
        estimator = mpc.nmhe(**nmheargs)

        # Declare ydata and udata. Note that it would make the most sense to declare
        # these using collection.deques since we're always popping the left element or
        # appending a new element, but for these sizes, we can just use a list without
        # any noticable slowdown.

        if (k == 0):
            ydata = [ys]*Nmhe
            udata = [us]*(Nmhe-1)
        else:
            ydata = simcon.ydata
            udata = simcon.udata

        # Make steady-state target selector.

        Rss = R
        Qyss = Qy
        
        def sstargobj(y, y_sp, u, u_sp, Q, R, s):
            dy = y - y_sp
            du = u - u_sp
            slb = s[:Ny]
            sub = s[Ny:]
            slack = mpc.mtimes(lbslack, slb) + mpc.mtimes(ubslack, sub)
            return mpc.mtimes(dy.T,Q,dy) + mpc.mtimes(du.T,R,du) + slack

        phiargs = ["y", "y_sp", "u", "u_sp", "Q", "R", "s"]
        phi = mpc.getCasadiFunc(sstargobj, [Ny,Ny,Nu,Nu,(Ny,Ny),(Nu,Nu),2*Ny],
                                phiargs, scalar=False)

        # Add slacked output constraints.
        def outputcon(x, s):
            """Softened output constraints."""
            y = measurement(x)
            slb = s[:Ny]
            sub = s[Ny:]
            terms = (
                np.array(ylb) - y - slb,
                y - np.array(yub) - sub,
            )
            return np.concatenate(terms)
        outputcon_casadi = mpc.getCasadiFunc(outputcon, [Nx + Nid, 2*Ny],
                                             ["x", "s"], funcname="e")

        sstargargs = {
            "f" : ode_disturbance_casadi,
            "h" : measurement_casadi,
            "lb" : {"u" : np.tile(ulb, (1,1))},
            "ub" : {"u" : np.tile(uub, (1,1))},
            "guess" : {
                "u" : np.tile(us, (1,1)),
                "x" : np.tile(np.concatenate((xs,np.zeros((Nid,)))), (1,1)),
                "y" : np.tile(ys, (1,1)),
            },
            "N" : {"x" : Nx + Nid, "u" : Nu, "y" : Ny, "f" : Nx, "e" : 2*Ny,
                   "s" : 2*Ny},
            "phi" : phi,
            "inferargs" : True,
            "e" : outputcon_casadi,
            "extrapar" : {"R" : Rss, "Q" : Qyss, "y_sp" : ys, "u_sp" : us},
            "verbosity" : 0,
            "discretef" : False,
            "casaditype" : "SX" if useCasadiSX else "MX",
        }
        targetfinder = mpc.sstarg(**sstargargs)
    
        # Make NMPC solver.

        duub = [ mvlist[0].roclim,  mvlist[1].roclim]
        dulb = [-mvlist[0].roclim, -mvlist[1].roclim]
        lb = {"u" : np.tile(ulb, (Nf,1)), "Du" : np.tile(dulb, (Nf,1))}
        ub = {"u" : np.tile(uub, (Nf,1)), "Du" : np.tile(duub, (Nf,1))}
        N = {"x": Nx + Nid, "u": Nu, "t": Nf, "s": 2*Ny, "e": 2*Ny}
        sp = {"x" : np.tile(xaugs, (Nf+1,1)), "u" : np.tile(us, (Nf,1))}
        guess = sp.copy()
        xaug0 = xaugs
        nmpcargs = {
            "f" : ode_augmented_rk4_casadi,
            "l" : l,
            "inferargs" : True,
            "N" : N,
            "x0" : xaug0,
            "uprev" : us,
            "lb" : lb,
            "ub" : ub,
            "guess" : guess,
            "Pf" : Pf,
            "sp" : sp,
            "e" : outputcon_casadi,
            "verbosity" : 0,
            "timelimit" : 60,
            "casaditype" : "SX" if useCasadiSX else "MX",
        }
        controller = mpc.nmpc(**nmpcargs)

        # Store values in simulation container
        simcon.proc = [hab]
        simcon.mod = (us, xs, ys, estimator, targetfinder, controller, habaug,
                      measurement, psize)
        simcon.ydata = ydata
        simcon.udata = udata
        simcon.extra["quanterr"] = 0

    # Get stored values
    #TODO: these should be dictionaries or NamedTuples.
    hab           = simcon.proc[0]
    (us, xs, ys, estimator, targetfinder, controller, habaug, measurement, psize) = simcon.mod
    (Nx, Nu, Ny, Nid, Nw, Nv, Nf, Nmhe) = psize
    ydata         = simcon.ydata
    udata         = simcon.udata

    # Get variable values
    x_km1 = xvlist.asvec()
    u_km1 = mvlist.asvec()
#    print "x_km1 = ", x_km1
#    print "u_km1 = ", u_km1
#    print "d_km1 = ", d_km1

    # Advance the process

    x_k = hab.sim(x_km1, u_km1)

    # Constrain the altitude state

    if (x_k[0] < 0.0): x_k[0] = 0.0

    # Take plant measurement

    y_k = measurement(np.concatenate((x_k,np.zeros((Nid,)))))

    if (nf.value > 0.0):

        for i in range(0, Ny):
            y_k[i] += nf.value*np.random.normal(0.0, cvlist[i].noise)
    
    # Do Nonlinear MHE.

    ydata.append(y_k)
    udata.append(u_km1) 
    estimator.par["y"] = ydata
    estimator.par["u"] = udata
    estimator.solve()
    estsol = mpc.util.casadiStruct2numpyDict(estimator.var)        

    print "runsim: estimator status - %s" % (estimator.stats["status"])
    xaughat_k = estsol["x"][-1,:]
    xhat_k = xaughat_k[:Nx]
    dhat_k = xaughat_k[Nx:]

    yhat_k = measurement(np.concatenate((xhat_k, dhat_k)))
    ydata.pop(0)
    udata.pop(0)    
    estimator.saveguess()        

    # Initialize the input
    u_k = u_km1

    # Update open and closed-loop predictions
    for field in ["olpred", "clpred"]:
        mvlist.vecassign(u_k, field, index=0)
#        dvlist.vecassign(d_km1, field, index=0)
        xvlist.vecassign(xhat_k, field, index=0)
        cvlist.vecassign(yhat_k, field, index=0)
    
    xof_km1 = np.concatenate((xhat_k,dhat_k))

    # Need to be careful about this forecasting. Temporarily aggressive control
    # could cause the system to go unstable if continued indefinitely, and so
    # this simulation might fail. If the integrator fails at any step, then we
    # just return NaNs for future predictions. Also, if the user doesn't want
    # predictions, then we just always skip them.
    predictionOkay = bool(doolpred.value)
    for i in range(0,(Nf - 1)):
        if predictionOkay:
            try:
                xof_k = habaug.sim(xof_km1, u_km1)
            except RuntimeError: # Integrator failed.
                predictionOkay = False
        if predictionOkay:
            # Take measurement.
            yof_k = measurement(xof_k)
            
            # Stop forecasting if bounds are exceeded.
            if np.any(yof_k > yub) or np.any(yof_k < ylb):
                predictionOkay = False
        else:
            xof_k = np.NaN*np.ones((Nx+Nid,))
            yof_k = np.NaN*np.ones((Ny,))

        for field in ["olpred", "clpred"]:
            mvlist.vecassign(u_k, field, index=(i + 1))
#            dvlist.vecassign(d_km1, field, index=(i + 1))
            xvlist.vecassign(xof_k[:Nx], field, index=(i + 1)) # Note [:Nx].
            cvlist.vecassign(yof_k[:Ny], field, index=(i + 1))
    
        xof_km1 = xof_k
    
    # calculate mpc input adjustment in control is on

    if (opnclsd.status.get() == 1):

        # Use nonlinear steady-state target selector

        ysp_k = [cvlist[0].setpoint, cvlist[1].setpoint, cvlist[2].setpoint]
        usp_k = [mvlist[0].target, mvlist[1].target]
        xtarget = np.concatenate((x_km1,dhat_k))

        # Previously had targetfinder.par["p",0] = d_km1, but this shouldn't
        # be because the target finder should be using the same model as the
        # controller and doesn't get to know the real disturbance.
        targetfinder.guess["x",0] = xtarget
        targetfinder.fixvar("x",0,dhat_k,range(Nx,Nx+Nid))
        targetfinder.par["y_sp",0] = ysp_k
        targetfinder.par["u_sp",0] = usp_k
        targetfinder.guess["u",0] = u_km1
        targetfinder.solve()

        xaugss = np.squeeze(targetfinder.var["x",0,:])
        uss = np.squeeze(targetfinder.var["u",0,:])

        print "runsim: target status - %s (Obj: %.5g)" % (targetfinder.stats["status"],targetfinder.obj) 
        
        # Now use nonlinear MPC controller.

        controller.par["x_sp"] = [xaugss]*(Nf + 1)
        controller.par["u_sp"] = [uss]*Nf
        controller.par["u_prev"] = [u_km1]
        controller.fixvar("x",0,xaughat_k)            
        controller.solve()
        print "runsim: controller status - %s (Obj: %.5g)" % (controller.stats["status"],controller.obj) 

        controller.saveguess()
        sol = mpc.util.casadiStruct2numpyDict(controller.var)
        
        # Apply quantization.
        
        if fuelincrement > 1e-3:
            minfuel = ulb[0]
            maxfuel = uub[0]
            
            # Use cumulative rounding strategy.
            quantum = (maxfuel - minfuel)*fuelincrement
            wantfuel = (sol["u"][:,0] - minfuel)/quantum
            wantsofar = 0
            getfuel = []
            getsofar = simcon.extra["quanterr"]/quantum
            for want in wantfuel:
                wantsofar += want
                get = round(wantsofar - getsofar)
                getfuel.append(get)
                getsofar += get
            simcon.extra["quanterr"] += (getfuel[0] - wantfuel[0])*quantum
            sol["u"][:,0] = minfuel + np.array(getfuel)*quantum
            
            # Re-simulate the x trajectory.
            for i in xrange(sol["u"].shape[0]):
                sol["x"][i + 1,:] = habaug.sim(sol["x"][i,:], sol["u"][i,:])
        else:
            simcon.extra["quanterr"] = 0
        
        print "runsim: quantization offset: %g" % simcon.extra["quanterr"]
        u_k = np.squeeze(sol["u"][0,:])

        # Update closed-loop predictions

        mvlist.vecassign(u_k, "clpred", index=0)
        xvlist.vecassign(xhat_k, "clpred", index=0)
        cvlist.vecassign(yhat_k, "clpred", index=0)

        for i in range(Nf - 1):
            mvlist.vecassign(sol["u"][i+1,:], "clpred", index=(i + 1))
            xvlist.vecassign(sol["x"][i+1,:Nx], "clpred", index=(i + 1))
            xcl_k = sol["x"][i+1,:]
            ycl_k = measurement(xcl_k)
            cvlist.vecassign(ycl_k[:Ny], "clpred", index=(i + 1))

    else:
        # Track the cv setpoints if the control is not on.
        cvlist[0].setpoint = y_k[0]
        cvlist[1].setpoint = y_k[1]
        cvlist[2].setpoint = y_k[2]

    # Store variable values
    mvlist.vecassign(u_k)
    xvlist.vecassign(x_k)
    xvlist.vecassign(xhat_k, "est")
#    dvlist[0].est = dhat_k
    cvlist.vecassign(y_k)
    cvlist.vecassign(yhat_k, "est")
    cvlist.vecassign(dhat_k, "dist")
    simcon.ydata    = ydata
    simcon.udata    = udata

# set up hab mpc example

simname = 'Hot-Air Ballon Example'

# define variables

MVmenu=["value","rvalue","svalue","target","maxlim","minlim","roclim","pltmax","pltmin"]
XVmenu=["mnoise","dnoise","pltmax","pltmin"]
CVmenu=["setpoint","qvalue","maxlim","minlim","mnoise","noise","pltmax","pltmin","lbslack","ubslack"]
DVmenu=["value","pltmax","pltmin"]

MV1 = sim.MVobj(name='f', desc='fuel flow setpoint', units='(%)',
            pltmin=-5.0, pltmax=105.0, minlim=0.0, maxlim=100.0, svalue=9.0e-4,
            rvalue=9.0e-7, value=0.0, target=0.0, Nf=60, menu=MVmenu)

MV2 = sim.MVobj(name='p', desc='top vent position', units='(%)', 
            pltmin=0.0, pltmax=100.0, minlim=1.0, maxlim=99.0, svalue=1.0e-4,
            rvalue=1.0e-4, value=0.0, target=0.0, Nf=60, menu=MVmenu)

CV1 = sim.CVobj(name='h', desc='altitude', units='(m)', 
            pltmin=-300.0, pltmax=7300.0, minlim=0.0, maxlim=7000.0, qvalue=1.0, noise=1.0,
            value=0.0, setpoint=0.0, Nf=60, lbslack=1000, ubslack=1000, menu=CVmenu)

CV2 = sim.CVobj(name='v', desc='vertical velocity', units='(m/s)', 
            pltmin=-25.0, pltmax=25.0, minlim=-23.0, maxlim=23.0, qvalue=0.0, noise=1.0,
            value=0.0, setpoint=0.0, Nf=60, lbslack=1000, ubslack=1000, menu=CVmenu)

CV3 = sim.CVobj(name='T', desc='bag temperature', units='(degC)', 
            pltmin=55.0, pltmax=125.0, minlim=60.0, maxlim=120.0, qvalue=0.0, noise=0.1,
            value=85.0, setpoint=85.0, Nf=60, lbslack=1000, ubslack=1000, menu=CVmenu)

XV1 = sim.XVobj(name='h', desc='dim. altitude', units='', 
            pltmin=-0.1, pltmax=0.7, mnoise=1, dnoise=1,
            value=0.0, Nf=60, menu=XVmenu)

XV2 = sim.XVobj(name='v', desc='dim. vertical velocity', units='', 
            pltmin=-0.08, pltmax=0.08, mnoise=1, dnoise=1,
            value=0.0, Nf=60, menu=XVmenu)

XV3 = sim.XVobj(name='T', desc='dim. bag temperature', units='', 
            pltmin=1.19, pltmax=1.4, mnoise=1, dnoise=1,
            value=1.244, Nf=60, menu=XVmenu)

# define options

NF = sim.Option(name='NF', desc='Noise Factor', value=0.0)
OL = sim.Option(name="OL Pred.", desc="Open-Loop Predictions", value=1)
fuel = sim.Option(name="Fuel increment", desc="Fuel increment", value=0)

# load up variable lists

MVlist = [MV1, MV2]
XVlist = [XV1, XV2, XV3]
CVlist = [CV1, CV2, CV3]
OPlist = [NF, OL, fuel]
DeltaT = 0.5
N      = 120
refint = 100
simcon = sim.SimCon(simname=simname,
                    mvlist=MVlist, cvlist=CVlist, xvlist=XVlist,
                    oplist=OPlist, N=N, refint=refint, runsim=runsim, deltat=DeltaT)

# build the GUI and start it up.
plotspacing = dict(left=0.075, top=0.95, bottom=0.05, right=0.99, hspace=0.5)
sim.makegui(simcon, plotspacing=plotspacing)
