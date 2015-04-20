# This is an example of how to interface mpc-sim with mpc-tools-casadi.
#

import mpcsim as sim
import numpy as np
import mpc_tools_casadi as mpc
import random as rn

def runsim(k, simcon, opnclsd, options):

    print "runsim: iteration %d " % k

    # unpack stuff from simulation container

    mvlist = simcon.mvlist
    dvlist = simcon.dvlist
    cvlist = simcon.cvlist
    xvlist = simcon.xvlist
    deltat = simcon.deltat

    # get mv, cv

    mv = mvlist[0]
    cv = cvlist[0]
    xv = xvlist[0]

    # check for changes

    chsum = 0
    for mv in mvlist:
        chsum += mv.chflag
        mv.chflag = 0
    for dv in dvlist:
        chsum += dv.chflag
        dv.chflag = 0
    for cv in cvlist:
        chsum += cv.chflag
        cv.chflag = 0
    for xv in xvlist:
        chsum += xv.chflag
        xv.chflag = 0

    chsum += options.chflag
    options.chflag = 0

    # initialize values on first execution or when something changes

    if (k == 0 or chsum > 0):

        # Define continuous time model.

        Acont = np.array([[-0.9]])
        Bcont = np.array([[10.0]])
        Nx = Acont.shape[0] # Number of states.
        Nu = Bcont.shape[1] # Number of control elements
        Nd = None # No disturbances.

        # Discretize.

        (Adisc,Bdisc) = mpc.c2d(Acont,Bcont,deltat)
        A = [Adisc]
        B = [Bdisc]

        # Define Q and R matrices and q penalty for periodic solution.

        Q = [np.diag([cv.qvalue])]
        q = [np.zeros((Nx,1))]
        R = [np.diag([mv.rvalue])]

        # Create discrete process

        Fproc = lambda x,u : (mpc.mtimes(Adisc,x) + mpc.mtimes(Bdisc,u))
        simcon.proc = Fproc

        # Create discrete controller (possibly with model mismatch)

        Amod = Adisc
        Bmod = options.gfac*Bdisc
        simcon.gain = Bmod/(1-Amod)
        Fmod = lambda x,u : (mpc.mtimes(Amod,x) + mpc.mtimes(Bmod,u))
        simcon.mod = Fmod
        Fdiscrete = lambda x,u : list(mpc.mtimes(Amod,x) + mpc.mtimes(Bmod,u))
        simcon.F = [mpc.getCasadiFunc(Fdiscrete,Nx,Nu,Nd,"F")]
        l = lambda x,u : [mpc.mtimes(x.T,mpc.DMatrix(Q[0]),x)
                  + mpc.mtimes(u.T,mpc.DMatrix(R[0]),u)]
        simcon.l = [mpc.getCasadiFunc(l,Nx,Nu,Nd,"l")]
        Pf = lambda x: [mpc.mtimes(x.T,mpc.DMatrix(Q[0]),x)]
        simcon.Pf = mpc.getCasadiFunc(Pf,Nx,0,0,"Pf")

        # initialize the state

        simcon.xmk = cv.value

    # increment the process

    xpkm1    = xv.value
    ukm1     = mv.value + mv.dist
    dxpkm1   = xpkm1 - xv.ref
    dukm1    = ukm1 - mv.ref
    dxpk     = simcon.proc(dxpkm1, dukm1) + cv.dist
    xpk      = dxpk + xv.ref
    ypk      = xpk + cv.dist

    # zero out the disturbances

    if (mv.dist != 0.0): mv.dist = 0.0
    if (xv.dist != 0.0): xv.dist = 0.0

    # add noise if desired

    if (options.fnoise > 0.0):

        xpk += options.fnoise*rn.uniform(-xv.noise,xv.noise) 
        ypk += options.fnoise*rn.uniform(-cv.noise,cv.noise) 
    
    # store values

    xv.value = xpk
    cv.value = ypk

    # initialize input

    uk = ukm1

    # increment the model

    xmkm1   = simcon.xmk
    dxmkm1  = xmkm1 - xv.ref
    dukm1   = ukm1 - mv.ref
    dxmk    = simcon.mod(dxmkm1, dukm1)
    xmk     = dxmk + xv.ref
    ymk     = xmk
    simcon.xmk = xmk

    # simple bias feedback

    cv.bias = (ypk - ymk)

#    print 'cv.bias = ', cv.bias

    # update future predictions

    mv.olpred[0] = uk
    xv.olpred[0] = xmk + cv.bias
    cv.olpred[0] = ymk + cv.bias
    mv.clpred[0] = uk
    xv.clpred[0] = xmk + cv.bias
    cv.clpred[0] = ymk + cv.bias
    duk          = uk - mv.ref

    for i in range(0,(xv.Nf-1)):

       dxmkp1 = simcon.mod(dxmk, duk)
       mv.olpred[i+1] = uk
       xv.olpred[i+1] = dxmkp1 + xv.ref + cv.bias
       cv.olpred[i+1] = dxmkp1 + cv.ref + cv.bias
       mv.clpred[i+1] = uk
       xv.clpred[i+1] = dxmkp1 + xv.ref + cv.bias
       cv.clpred[i+1] = dxmkp1 + cv.ref + cv.bias
       dxmk = dxmkp1

    # set xv target, limits same as cv limits

    xv.maxlim = cv.maxlim
    xv.minlim = cv.minlim

    # calculate mpc input adjustment in control is on

    if (opnclsd.status.get() == 1):

        # calculate steady state

        cv.sstarg = cv.setpoint - cv.bias
        xv.sstarg = cv.sstarg
        mv.sstarg = (xv.sstarg - xv.ref)/simcon.gain + mv.ref

        # set mv, xv bounds

        ulb = [np.array([mv.minlim - mv.sstarg])]
        uub = [np.array([mv.maxlim - mv.sstarg])]
        xlb = [np.array([xv.minlim - xv.sstarg - cv.bias])]
        xub = [np.array([xv.maxlim - xv.sstarg - cv.bias])]
        bounds = dict(uub=uub,ulb=ulb,xub=xub,xlb=xlb)

        # solve for new input and state

        alg   = mpc.nmpc(simcon.F,simcon.l,[0],xv.Nf,simcon.Pf,
                bounds,verbosity=0,returnTimeInvariantSolver=True)
        dxss = xmk - xv.sstarg
        alg.fixvar("x",0,dxss)
        sol = alg.solve()
        sol["u"] = sol["u"].T
        sol["x"] = sol["x"].T
        duk  = sol["u"][0,0] 
        uk   = duk + mv.sstarg
        
        # update future predictions

        for i in range(0,(xv.Nf)):

            mv.clpred[i] = sol["u"][0,i] + mv.sstarg
            xv.clpred[i] = sol["x"][0,i] + xv.sstarg + cv.bias
            cv.clpred[i] = sol["x"][0,i] + cv.sstarg + cv.bias

        print "runsim: control status - %s" % sol["status"]

    # load current input

    mv.value = uk

# set up siso mpc example

simname = 'SISO MPC Example'

# define variables

MV = sim.MVobj(name='MV', desc='manipulated variable', units='m3/h  ', 
               pltmin=0.0, pltmax=4.0, minlim=0.5, maxlim=3.5,
               value=2.0, Nf=60)
CV = sim.CVobj(name='CV', desc='controlled variable ', units='degC  ', 
               pltmin=0.0, pltmax=50.0, minlim=5.0, maxlim=45.0,
               value=25.0, setpoint=25.0, noise=0.1, Nf=60)
XV = sim.XVobj(name='XV', desc='state variable ', units='degC  ', 
               pltmin=0.0, pltmax=50.0, 
               value=25.0, setpoint=25.0, noise=0.1, Nf=60)


# load up variable lists

MVlist = [MV]
DVlist = []
CVlist = [CV]
XVlist = [CV]
DeltaT = .10
N      = 120
refint = 10.0
simcon = sim.SimCon(simname=simname,
                    mvlist=MVlist, dvlist=DVlist, cvlist=CVlist, xvlist=[XV],
                    N=N, refint=refint, runsim=runsim, deltat=DeltaT)

# build the GUI and start it up

sim.makegui(simcon)
