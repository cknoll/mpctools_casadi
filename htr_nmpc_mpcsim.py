# This is a fired heater example.
#

from mpctools import mpcsim as sim
import mpctools as mpc
import numpy as np
from scipy import linalg

useCasadiSX = True

def readmat(filename):

    # Read array from text file into a list.

    with open(filename,'r') as f:
        X = []
        for line in f:
            sline = line.split()
            Xrow = [float(s) for s in sline]
            X.append(Xrow)

    return X

def runsim(k, simcon, opnclsd):

    print "runsim: iteration %d -----------------------------------" % k

    # Unpack stuff from simulation container.

    mvlist = simcon.mvlist
    dvlist = simcon.dvlist
    cvlist = simcon.cvlist
    oplist = simcon.oplist
    deltat = simcon.deltat
    nf = oplist[0]
    doolpred = oplist[1]

    #TODO:  change this to not reinitialize - just change stuff

    # Check for changes.

    chsum = 0

    for var in mvlist:
        chsum += var.chflag
        var.chflag = 0
    
    for var in dvlist:
        chsum += var.chflag
        var.chflag = 0
    
    for var in cvlist:
        chsum += var.chflag
        var.chflag = 0
    
    for var in oplist:
        chsum += var.chflag
        var.chflag = 0
    
    # Grab bounds.

    uub = [mvlist[0].maxlim, mvlist[1].maxlim, mvlist[2].maxlim]
    ulb = [mvlist[0].minlim, mvlist[1].minlim, mvlist[2].minlim]
    yub = [cvlist[0].maxlim, cvlist[1].maxlim, cvlist[2].maxlim, cvlist[3].maxlim, cvlist[4].maxlim]
    ylb = [cvlist[0].minlim, cvlist[1].minlim, cvlist[2].minlim, cvlist[3].minlim, cvlist[4].minlim]
    
    # Initialize values on first execution or when something changes.

    if (k == 0 or chsum > 0):

        print "runsim: initialization"

        # Define problem size parameters.

        Nx   = 10           # number of states
        Nu   = 3            # number of inputs
        Nd   = 2            # number of measured disturbances
        Ny   = 5            # number of outputs
        Nid  = Ny           # number of integrating disturbances

        Nw   = Nx + Nid     # number of augmented states
        Nv   = Ny           # number of output measurements
        Nf   = cvlist[0].Nf # length of NMPC future horizon
        Nmhe = 60           # length of MHE past horizon

        psize = [Nx,Nu,Nd,Ny,Nid,Nw,Nv,Nf,Nmhe]

        # Define sample time in minutes.
        
        Delta = deltat

        # Read in the fired heater model continuous state-space matrices.
        # The matrices were generate in Matlab.
        
        Ax = np.array(readmat('ssfhtr_Ax.txt'))
        Bu = np.array(readmat('ssfhtr_Bu.txt'))
        Bd = np.array(readmat('ssfhtr_Bd.txt'))
        Cx = np.array(readmat('ssfhtr_Cx.txt'))

        # Set the initial values.

        y0 = np.zeros((Ny,))
        x0 = np.zeros((Nx,))
        u0 = np.zeros((Nu,))
        d0 = np.zeros((Nd,))
        for i in range(0,Ny-1):
            y0[i] = cvlist[i].value
        for i in range(0,Nu-1):
            u0[i] = mvlist[i].value
        for i in range(0,Nd-1):
            d0[i] = dvlist[i].value
            
        # Define ode for the fired heater simulation.

        def ode(x,u,d):

            # ODE for fired heater.

            deltax = x - x0
            deltau = u - u0
            deltad = d - d0
            dxdt = np.dot(Ax,deltax) + np.dot(Bu,deltau) + np.dot(Bd,deltad)  
            return dxdt

        # Create casadi function and simulator.

        ode_casadi = mpc.getCasadiFunc(ode,[Nx,Nu,Nd],["x","u","d"],"ode")
        htr = mpc.DiscreteSimulator(ode, Delta, [Nx,Nu,Nd], ["x","u","d"])

        # Initialize the steady-state values

        ys = y0
        xs = x0
        us = u0
        ds = d0
        
        # Define augmented model for state estimation.    

        # We need to define two of these because Ipopt isn't smart enough to throw out
        # the 0 = 0 equality constraints. ode_disturbance only gives dx/dt for the
        # actual states, and ode_augmented appends the zeros so that dx/dt is given for
        # all of the states. It's possible that Ipopt may figure this out by itself,
        # but we should just be explicit to avoid any bugs.    

        def ode_disturbance(x,u,d=ds):

            # For this case there are no input disturbances.

            dxdt = htr(x,u,d)
            return dxdt

        def ode_augmented(x,u,d=ds):

            # Need to add extra zeros for derivative of disturbance states.

            dxdt = np.concatenate((ode_disturbance(x,u,d),np.zeros((Nid,))))
            return dxdt
 
        htraug = mpc.DiscreteSimulator(ode_augmented, Delta,
                                        [Nx+Nid,Nu,Nd], ["xaug","u","d"])

        def measurement(x,d=ds):

            # For this case all of the disturbances are output disturbances.

            deltax = x - x0
            dhat   = x[Nx:Nx+Nid]
            deltay = np.dot(Cx,deltax) + dhat
            ym     = deltay + y0
            return ym

        # Turn into casadi functions.

        ode_disturbance_casadi = mpc.getCasadiFunc(ode_disturbance,
                                   [Nx+Nid,Nu,Nd],["xaug","u","d"],"ode_disturbance")
        ode_augmented_casadi = mpc.getCasadiFunc(ode_augmented,
                               [Nx+Nid,Nu,Nd],["xaug","u","d"],"ode_augmented")
        ode_augmented_rk4_casadi = mpc.getCasadiFunc(ode_augmented,
                                   [Nx+Nid,Nu,Nd],["xaug","u","d"],"ode_augmented_rk4",
                                     rk4=True,Delta=Delta,M=2)

        def ode_estimator_rk4(x,u,w=np.zeros((Nx+Nid,)),d=ds):
            return ode_augmented_rk4_casadi(x, u, d) + w

        def ode_estimator(x,u,w=np.zeros((Nx+Nid,)),d=ds):
            return ode_augmented_casadi(x, u, d) + w

        ode_estimator_rk4_casadi = mpc.getCasadiFunc(ode_estimator_rk4,
                                   [Nx+Nid,Nu,Nw,Nd], ["xaug","u","w","d"],
                                   "ode_estimator_rk4", scalar=False)

        measurement_casadi = mpc.getCasadiFunc(measurement,
                             [Nx+Nid,Nd], ["xaug","d"], "measurement")

        # Weighting matrices for controller.

        Q = np.diag([cvlist[0].qvalue, cvlist[1].qvalue, cvlist[2].qvalue,
                     cvlist[3].qvalue, cvlist[4].qvalue])
        R = np.diag([mvlist[0].rvalue, mvlist[1].rvalue, mvlist[2].rvalue])
        S = np.diag([mvlist[0].svalue, mvlist[1].svalue, mvlist[2].svalue])

        # Now calculate the cost-to-go.

        [K, Pi] = mpc.util.dlqr(Ax,Bu,Q,R)

        # Define control stage cost.

        def stagecost(x,u,xsp,usp,Deltau):
            dx = x[:Nx] - xsp[:Nx]
            du = u - usp
            return (mpc.mtimes(dx.T,Q,dx) + .1*mpc.mtimes(du.T,R,du)
                + mpc.mtimes(Deltau.T,S,Deltau))

        largs = ["x","u","x_sp","u_sp","Du"]
        l = mpc.getCasadiFunc(stagecost,
            [Nx+Nid,Nu,Nx+Nid,Nu,Nu],largs,funcname="l",scalar=False)

        # Define cost to go.

        def costtogo(x,xsp):
            dx = x[:Nx] - xsp[:Nx]
            return mpc.mtimes(dx.T,Pi,dx)
        Pf = mpc.getCasadiFunc(costtogo,[Nx+Nid,Nx+Nid],["x","s_xp"],
                               funcname="Pf", scalar=False)

        # Build augmented estimator matrices.

        Qw = np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        Rv = np.diag([cvlist[0].mnoise, cvlist[1].mnoise, cvlist[2].mnoise,
                      cvlist[3].mnoise, cvlist[4].mnoise])
        Qwinv = linalg.inv(Qw)
        Rvinv = linalg.inv(Rv)

        # Define stage costs for estimator.

        def lest(w,v):
            return mpc.mtimes(w.T,Qwinv,w) + mpc.mtimes(v.T,Rvinv,v)
                      
        lest = mpc.getCasadiFunc(lest,[Nw,Nv],["w","v"],"l",scalar=False)

        # Don't use a prior.

        lxest = None
        x0bar = None

        # Check if the augmented system is detectable. (Rawlings and Mayne, Lemma 1.8)

        Aaug = mpc.util.getLinearizedModel(ode_augmented_casadi,[xaugs, us, ds],
                                       ["A","B","Bp"], Delta)["A"]
        Caug = mpc.util.getLinearizedModel(measurement_casadi,[xaugs, ds],
                                       ["C","Cp"])["C"]
        Oaug = np.vstack((np.eye(Nx,Nx+Nid) - Aaug[:Nx,:], Caug))
        svds = linalg.svdvals(Oaug)
        rank = sum(svds > 1e-8)
        if rank < Nx + Nid:
            print "***Warning: augmented system is not detectable!"

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
            "N" : {"x":Nx + Nid, "u":Nu, "y":Ny, "p":Nd, "t":Nmhe},
            "lx" : lxest,
            "x0bar" : x0bar,
            "p" : np.tile(ds,(Nmhe+1,1)),
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

        contVars = [0,2]
        Rss = np.zeros((Nu,Nu))
        Qss = np.zeros((Ny,Ny))
        Qss[contVars,contVars] = 1 # Only care about controlled variables.

        def sstargobj(y,y_sp,u,u_sp,Q,R):
            dy = y - y_sp
            du = u - u_sp
            return mpc.mtimes(dy.T,Q,dy) + mpc.mtimes(du.T,R,du)

        phiargs = ["y","y_sp","u","u_sp","Q","R"]
        phi = mpc.getCasadiFunc(sstargobj, [Ny,Ny,Nu,Nu,(Ny,Ny),(Nu,Nu)],
                                phiargs, scalar=False)

        sstargargs = {
            "f" : ode_disturbance_casadi,
            "h" : measurement_casadi,
            "lb" : {"u" : np.tile(ulb, (1,1)), "y" : np.tile(ylb, (1,1))},
            "ub" : {"u" : np.tile(uub, (1,1)), "y" : np.tile(yub, (1,1))},
            "guess" : {
                "u" : np.tile(us, (1,1)),
                "x" : np.tile(np.concatenate((xs,np.zeros((Nid,)))), (1,1)),
                "y" : np.tile(xs, (1,1)),
            },
            "p" : np.tile(ds, (1,1)), # Parameters for system.
            "N" : {"x" : Nx + Nid, "u" : Nu, "y" : Ny, "p" : Nd, "f" : Nx},
            "phi" : phi,
            "funcargs" : dict(phi=phiargs),
            "extrapar" : {"R" : Rss, "Q" : Qss, "y_sp" : ys, "u_sp" : us},
            "verbosity" : 0,
            "discretef" : False,
            "casaditype" : "SX" if useCasadiSX else "MX",
        }
        targetfinder = mpc.sstarg(**sstargargs)

        # Make NMPC solver.

        duub = [ mvlist[0].roclim,  mvlist[1].roclim,  mvlist[2].roclim]
        dulb = [-mvlist[0].roclim, -mvlist[1].roclim, -mvlist[2].roclim]
        lb = {"u" : np.tile(ulb, (Nf,1)), "Du" : np.tile(dulb, (Nf,1)),
              "y" : np.tile(ylb, (Nf,1))}
        ub = {"u" : np.tile(uub, (Nf,1)), "Du" : np.tile(duub, (Nf,1)),
              "y" : np.tile(yub, (Nf,1))}
        N = {"x":Nx+Nid, "u":Nu, "p":Nd, "t":Nf}
        p = np.tile(ds, (Nf,1)) # Parameters for system.
        sp = {"x" : np.tile(xaugs, (Nf+1,1)), "u" : np.tile(us, (Nf,1))}
        guess = sp.copy()
        xaug0 = xaugs
        nmpcargs = {
            "f" : ode_augmented_rk4_casadi,
            "l" : l,
            "funcargs" : dict(l=largs),
            "N" : N,
            "x0" : xaug0,
            "uprev" : us,
            "lb" : lb,
            "ub" : ub,
            "guess" : guess,
            "Pf" : Pf,
            "sp" : sp,
            "p" : p,
            "verbosity" : 0,
            "timelimit" : 60,
            "casaditype" : "SX" if useCasadiSX else "MX",
        }
        controller = mpc.nmpc(**nmpcargs)

        if k == 0:
            # Initialize variables
            x_k      = np.zeros(Nx)
            xhat_k   = np.zeros(Nx)
            dhat_k   = np.zeros(Nid)

            # Store initial values for variables
            xvlist.vecassign(xs)
            xvlist.vecassign(xs, "est")
            mvlist.vecassign(us)
            dvlist.vecassign(dhat_k[0], "est")

        # Store values in simulation container
        simcon.proc = [htr]
        simcon.mod = []
        simcon.mod.append(us)
        simcon.mod.append(xs)
        simcon.mod.append(ys)
        simcon.mod.append(ds)
        simcon.mod.append(estimator)
        simcon.mod.append(targetfinder)
        simcon.mod.append(controller)
        simcon.mod.append(htraug)
        simcon.mod.append(measurement)
        simcon.mod.append(psize)
        simcon.ydata = ydata
        simcon.udata = udata

    # Get stored values
    #TODO: these should be dictionaries.
    htr          = simcon.proc[0]
    us            = simcon.mod[0]
    xs            = simcon.mod[1]
    ys            = simcon.mod[2]
    ds            = simcon.mod[3]
    estimator     = simcon.mod[4]
    targetfinder  = simcon.mod[5]
    controller    = simcon.mod[6]
    htraug        = simcon.mod[7]
    measurement   = simcon.mod[8]
    psize         = simcon.mod[9]
    Nx            = psize[0]
    Nu            = psize[1]
    Nd            = psize[2]
    Ny            = psize[3]
    Nid           = psize[4]
    Nw            = psize[5]
    Nv            = psize[6]
    Nf            = psize[7]
    Nmhe          = psize[8]
    ydata         = simcon.ydata
    udata         = simcon.udata

    # Get variable values
    x_km1 = xvlist.asvec()
    u_km1 = mvlist.asvec()
    d_km1 = dvlist.asvec()    

    # Advance the process

    x_k = htr.sim(x_km1, u_km1, d_km1)

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
        dvlist.vecassign(d_km1, field, index=0)
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
                xof_k = htraug.sim(xof_km1, u_km1, ds)
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
            dvlist.vecassign(d_km1, field, index=(i + 1))
            xvlist.vecassign(xof_k[:Nx], field, index=(i + 1)) # Note [:Nx].
            cvlist.vecassign(yof_k[:Ny], field, index=(i + 1))
    
        xof_km1 = xof_k
    
    # calculate mpc input adjustment in control is on

    if (opnclsd.status.get() == 1):

        # Use nonlinear steady-state target selector

        ysp_k = [cvlist[0].setpoint, cvlist[1].setpoint, cvlist[2].setpoint,
                 cvlist[3].setpoint, cvlist[4].setpoint]
        usp_k = [mvlist[0].target, mvlist[1].target, mvlist[2].target]
        xtarget = np.concatenate((ysp_k,dhat_k))

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
        u_k = np.squeeze(controller.var["u",0])

        # Update closed-loop predictions

        sol = mpc.util.casadiStruct2numpyDict(controller.var)

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
    dvlist.vecassign(dhat_k, "est")
    cvlist.vecassign(y_k)
    cvlist.vecassign(yhat_k, "est")
    simcon.ydata    = ydata
    simcon.udata    = udata

# set up htr mpc example

simname = 'Fired Heater NMPC Example'

# define variables

MVmenu=["value","rvalue","svalue","maxlim","minlim","roclim","pltmax","pltmin"]
DVmenu=["value","pltmax","pltmin"]
XVmenu=["mnoise","noise","pltmax","pltmin"]
CVmenu=["setpoint","qvalue","maxlim","minlim","mnoise","noise","pltmax","pltmin"]
FVmenu=["mnoise","noise","pltmax","pltmin"]

MV1 = sim.MVobj(name='f1sp', desc='mv - pass 1 flow setpoint', units='(bph)',
            pltmin=0.0, pltmax=200.0, minlim=5.0, maxlim=195.0,
            value=100.0, target=100.0, Nf=60, menu=MVmenu)

MV2 = sim.MVobj(name='f2sp', desc='mv - pass 2 flow setpoint', units='(bph)', 
            pltmin=0.0, pltmax=200.0, minlim=5.0, maxlim=195.0,
            value=100.0, target=100.0, Nf=60, menu=MVmenu)

MV3 = sim.MVobj(name='fgsp', desc='fg - fuel gas flow setpoint', units='(scfh)', 
            pltmin=0.0, pltmax=200.0, minlim=5.0, maxlim=195.0,
            value=100.0, target=100.0, Nf=60, menu=MVmenu)

DV1 = sim.MVobj(name='t1in', desc='dv - pass 1 inlet temp', units='(degf)', 
            pltmin=520.0, pltmax=560.0,
            value=540.0, Nf=60, menu=DVmenu)

DV2 = sim.MVobj(name='t2in', desc='dv - pass 2 inlet temp', units='(degf)', 
            pltmin=520.0, pltmax=560.0,
            value=540.0, Nf=60, menu=DVmenu)

CV1 = sim.XVobj(name='toc', desc='cv - combined outlet temp', units='(degf)', 
            pltmin=700.0, pltmax=800.0, minlim=705.0, maxlim=795.0, noise=1.0,
            value=750.0, setpoint=750.0, Nf=60, menu=CVmenu)

CV2 = sim.XVobj(name='foc', desc='cv - combined outlet flow', units='(bph)', 
            pltmin=0.0, pltmax=400.0, minlim=10.0, maxlim=390.0, noise=1.0,
            value=200.0, setpoint=200.0, Nf=60, menu=CVmenu)

CV3 = sim.XVobj(name='dpt', desc='cv - delta pass temp', units='(degf)', 
            pltmin=-10.0, pltmax=10.0, minlim=-9.5, maxlim=9.5, noise=0.1,
            value=0.0, setpoint=0.0, Nf=60, menu=CVmenu)

CV4 = sim.XVobj(name='t1s', desc='cv - pass 1 tubeskin temp', units='(degf)', 
            pltmin=850, pltmax=950, minlim=800.0, maxlim=920.0, noise=0.5,
            value=900.0, setpoint=900.0, Nf=60, menu=CVmenu)

CV5 = sim.XVobj(name='t2s', desc='cv - pass 2 tubeskin temp', units='(degf)', 
            pltmin=850, pltmax=950, minlim=800.0, maxlim=920.0, noise=0.5,
            value=900.0, setpoint=900.0, Nf=60, menu=CVmenu)

# define options

NF = sim.Option(name='NF', desc='Noise Factor', value=0.0)
OL = sim.Option(name="OL Pred.", desc="Open-Loop Predictions", value=1)

# load up variable lists

MVlist = [MV1, MV2, MV3]
DVlist = [DV1, DV2]
XVlist = []
CVlist = [CV1, CV2, CV3, CV4, CV5]
OPlist = [NF, OL]
DeltaT = 1.0
N      = 120
refint = 100
simcon = sim.SimCon(simname=simname,
                    mvlist=MVlist, dvlist=DVlist, cvlist=CVlist, xvlist=XVlist,
                    oplist=OPlist, N=N, refint=refint, runsim=runsim, deltat=DeltaT)

# build the GUI and start it up

sim.makegui(simcon)
