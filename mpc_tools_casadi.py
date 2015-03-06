from __future__ import print_function
import numpy as np
import scipy.linalg
import casadi
import casadi.tools as ctools
import matplotlib.pyplot as plt
import time
import colloc

"""
Functions for solving MPC problems using Casadi and Ipopt.

The main function is lmpc, which is analogous to the mpc-tools function of the
same name. However, this function is currently missing a lot of the "advanced"
functionality of Octave lmpc, e.g. soft constraints, solver tolerances, and
returning lagrange multipliers.

There is now a function for nonlinear MPC. This works with nonlinear discrete-
time models. There is also a function to discretize continuous-time models using
a 4th-order Runge-Kutta method.

Most other functions are just convenience wrappers to replace calls like
long.function.name((args,as,tuple)) with f(args,as,args), although these are
largely unnecessary.
"""

# =================================
# Linear
# =================================

def lmpc(A,B,x0,N,Q,R,q=None,r=None,M=None,xlb=None,xub=None,ulb=None,uub=None,D=None,G=None,d=None,verbosity=5):
    """
    Solves the canonical linear MPC problem using a discrete-time model.
    
    Inputs are discrete-time state-space model, objective function weights, and
    input constraints. Output is a tuple (x,u) with the optimal state and input
    trajectories.    
    
    The actual optimization problem is as follows:
    
        min \sum_{k=0}^N        0.5*x[k]'*Q[k]*x[k] + q[k]'*x[k]      
            + \sum_{k=0}^{N-1}  0.5*u[k]'*R[k]*u[k] + r[k]'*u[k]
                                 + x[k]'*M[k]*u[k]
    
        s.t. x[k+1] = A[k]*x[k] + B[k]*u[k]               k = 0,...,N-1
             ulb[k] <= u[k] <= uub[k]                     k = 0,...,N-1
             xlb[k] <= x[k] <= xlb[k]                     k = 0,...,N
             D[k]*u[k] - G[k]*x[k] <= d[k]                k = 0,...,N-1
    
    A, B, Q, R, M, D, and G should be lists of numPy matrices. x0 should be a
    numPy vector. q, r, xlb, xub, ulb, uub, and d should be lists of numpy vectors.
    
    All of these lists are accessed modulo their respective length;hus,
    time-invariant models can be lists with one element, while time-varying
    periodic model with period T should have T elements.
    
    All arguments are optional except A, B,x0, N, Q, and R. If any of D, G, or d
    are given, then all must be given.    
    
    Optional argument verbosity controls how much solver output there is. This
    value must be an integer between 0 and 12 (inclusive). Higher numbers
    indicate more verbose output.    
    
    Return value is a dictionary. Entries "x" and "u" are 2D arrays with the first
    index corresponding to individual states and the second index corresponding
    to time. Entry "status" is a string with solver status.
    """    
    starttime = time.clock()    
    
    # Get shapes.    
    n = A[0].shape[0]
    p = B[0].shape[1]
    
    # Sort out arguments.
    if xlb is None:
        xlb = [-np.inf*np.ones((n,1))]
    if xub is None:
        xub = [np.inf*np.ones((n,1))]
    if ulb is None:
        ulb = [-np.inf*np.ones((p,1))]
    if uub is None:
        uub = [np.inf*np.ones((p,1))]
    
    argcheck = [D is None, G is None, d is None]    
    if any(argcheck) and not all(argcheck):
        raise ValueError("D, G, and d must be specified all or none.")
    
    # Define NLP variables.
    VAR = ctools.struct_symMX([(
        ctools.entry("x",shape=(n,1),repeat=N+1),
        ctools.entry("u",shape=(p,1),repeat=N),
    )])
    LB = VAR(-np.Inf) # Default all bounds to +/- Inf.
    UB = VAR(np.Inf)
    qpF = casadi.MX(0) # Start with dummy objective.
    qpG = [None]*N # Preallocate, although we could just append.
    
    # First handle objective/constraint terms that aren't optional.    
    for k in range(N):
        if k != N:
            LB["u",k,:] = ulb[k % len(ulb)]
            UB["u",k,:] = uub[k % len(uub)]
            
            qpF += .5*mtimes(VAR["u",k].T,R[k % len(R)],VAR["u",k]) 
            qpG[k] = mtimes(A[k % len(A)],VAR["x",k]) + mtimes(B[k % len(B)],VAR["u",k]) - VAR["x",k+1]
        
        if k == 0:
            LB["x",0,:] = x0
            UB["x",0,:] = x0
        else:
            LB["x",k,:] = xlb[k % len(xlb)]
            UB["x",k,:] = xub[k % len(xub)]
        
        qpF += .5*mtimes(VAR["x",k].T,Q[k % len(Q)],VAR["x",k])
    
    conlb = np.zeros((n*N,))
    conub = np.zeros((n*N,))

    # Now check optional stuff.
    if q is not None:
        for k in range(N):
            qpF += mtimes(q[k % len(q)].T,VAR["x",k])
    if r is not None:
        for k in range(N-1):
            qpF += mtimes(r[k % len(r)].T,VAR["u",k])
    if M is not None:
        for k in range(N-1):
            qpF += mtimes(VAR["x",k].T,M[k % len(M)],VAR["u",k])                
    
    if D is not None:
        for k in range(N-1):
            qpG.append(mtimes(D[k % len(d)],VAR["u",k]) 
                        - mtimes(G[k % len(d)],VAR["x"]) - d[k % len(d)])
        s = (D[0].shape[0]*N, 1) # Shape for inequality RHS vector.
        conlb = np.concatenate(conlb,-np.inf*np.ones(s))
        conub = np.concatenate(conub,np.zeros(s))
    
    # Make qpG into a single large vector.     
    qpG = casadi.vertcat(qpG) 
    
    # Create solver and stuff.
    nlp = casadi.MXFunction(casadi.nlpIn(x=VAR),casadi.nlpOut(f=qpF,g=qpG))
    solver = casadi.NlpSolver("ipopt",nlp)
    solver.setOption("print_level",verbosity)
    solver.setOption("print_time",verbosity > 2)   
    solver.init()

    solver.setInput(LB,"lbx")
    solver.setInput(UB,"ubx")
    solver.setInput(conlb,"lbg")
    solver.setInput(conub,"ubg")
    
    # Solve.    
    solver.evaluate()
    status = solver.getStat("return_status")
    if verbosity > 0:
        print("Solver Status:", status)
    
    # Get solution.
    OPTVAR = VAR(solver.getOutput("x"))
    x = np.hstack(OPTVAR["x",:,:])
    u = np.hstack(OPTVAR["u",:,:])
    
    # Add singleton dimension if n = 1 or p = 1.
    if n == 1:    
        x = x.reshape(1,N+1)
    if p == 1:
        u = u.reshape(1,N)
    
    endtime = time.clock()
    if verbosity > 1:
        print("Took %g s." % (endtime - starttime))
    
    return {"x" : x, "u" : u, "status" : status, "time" : endtime - starttime}

def c2d(Ac,Bc,Delta):
    """
    Converts continuous-time model (Ac,Bc) to discrete time (Ad,Bd) with sample time Delta.
    
    Inputs Ac and Bc should be numpy matrices of the appropriate size. Delta should be a
    scalar. Output is a tuple with discretized (Ad,Bd).
    """
    n = Ac.shape[0]
    m = Bc.shape[1]
    
    D = scipy.linalg.expm(Delta*np.vstack((np.hstack((Ac, Bc)), np.zeros((m,m+n)))))
    Ad = D[0:n,0:n];
    Bd = D[0:n,n:];
    
    return (Ad,Bd)
  
def np2mx(x):
    """
    Casts a numpy array or matrix x to a casadi MX variable.
    """
    return casadi.MX(casadi.DMatrix(x))
    
def mtimes(*args):
    """
    Convenience wrapper for casadi.tools.mul.
    
    Matrix multiplies all of the given arguments and returns the result.
    """
    return ctools.mul(args)
    
def vcat(*args):
    """
    Convenience wrapper for np.vstack.
    
    Vertically concatenates all arguments and returns the result.
    
    Accepts variable number of arguments instead of a single tuple.
    """
    return np.vstack(args)
    
def hcat(*args):
    """
    Convenience wrapper for np.hstack.
    
    Horizontally concatenates all arguments and returns the result.    
    
    Accepts variable number of arguments instead of a single tuple.
    """
    return np.hstack(args)
# =================================
# Nonlinear
# =================================
    
# Nonlinear functions adapted from scripts courtesy of Lars Petersen.

def getCasadiFunc(f,Nx,Nu=0,Nd=0,name=None):
    """
    Takes a function handle and turns it into a Casadi function.
    
    f should be defined to take three keyword arguments x, u, and d. It should
    be written so that x, u, and d are python LISTs of length Nx, Nu, and Nd
    respectively. It must return a LIST of length Nx.
    """
    
    # Create symbolic variables.
    x  = casadi.MX.sym("x",Nx) # States
    args = {"x" : x}
    invar = [x]
    if Nu > 0:
        u  = casadi.MX.sym("u",Nu) # Control
        args["u"] = u
        invar += [u]
        if Nd > 0: # Only consider disturbances if there are control inputs.
            d  = casadi.MX.sym("d",Nd) # Other (e.g. parameters or disturbances)
            args["d"] = d
            invar += [d]
    
    # Create symbolic function.
    outvar = [casadi.vertcat(f(**args))]
    fcasadi = casadi.MXFunction(invar,outvar)
    fcasadi.setOption("name",name if name is not None else "f")
    fcasadi.init()
    
    return fcasadi

def getModelArgs(f):
    """
    Checks number and sizes of arguments for the casadi function f.
    
    Returns a tuple (Nx, Nu, Nd) with sizes for x, u, and d. The first argument
    is always x, the second u (if present), and the third d (if present).
    
    The function may take more than three arguments, but these aren't returned.
    """
    sizes = []
    for i in range(f.getNumInputs()):
        sizes.append(f.getInput(i).size())
    sizes += [0]*(3 - len(sizes))
    return tuple(sizes[:3])

def getRungeKutta4(f,Delta,M=1,name=None):
    """
    Uses RK4 to discretize xdot = f(x,u,d) with M points and timestep Delta.

    f must be a Casadi SX function with 1, 2, or 3 inputs in the order shown.
    All inputs must be vectors.
    """

    # First find out how many arguments f takes.
    (Nx, Nu, Nd) = getModelArgs(f)
    
    # Now make relevant symbolic variables.
    x0 = casadi.MX.sym("x0",Nx)
    zoh = []  # Zero-order hold variables.
    if Nu > 0:    
        u = casadi.MX.sym("u",Nu)
        zoh += [u]
        if Nd > 0: # Only consider disturbances if there are control inputs.
            d = casadi.MX.sym("d",Nd)
            zoh += [d]
        
    h = Delta/float(M) # h in Runge-Kutta.
    
    # Do M RK4 steps.
    x = x0
    for j in range(M):
        [k1] = f([x] + zoh)
        [k2] = f([x + h/2*k1] + zoh)
        [k3] = f([x + h/2*k2] + zoh)
        [k4] = f([x + h*k3] + zoh)
        x += h/6*(k1 + 2*k2 + 2*k3 + k4)
    
    # Build casadi function and initialize.
    F = casadi.MXFunction([x0] + zoh,[x])
    F.setOption("name",name if name is not None else "F")
    F.init()
    
    return F
        
def nmpc(F,l,x0,N,Pf=None,bounds={},d=None,verbosity=5,guess={}):
    """
    Solves a nonlinear MPC problem using a discrete-time model.
    
    Inputs are discrete-time state-space model, stage costs, terminal weight,
    initial state, prediction horizon, and any known disturbances. Output is a
    tuple (x,u) with the optimal state and input trajectories.    
    
    The actual optimization problem is as follows:
    
        min \sum_{k=0}^{N} l[k](x[k],u[k]) + Pf(x[N])     
    
        s.t. x[k+1] = F(x[k],u[k],d[k])   k = 0,...,N-1
             ulb[k] <= u[k] <= uub[k]     k = 0,...,N-1
             xlb[k] <= x[k] <= xlb[k]     k = 0,...,N
    
    F and l should be lists of Casadi functions. Pf is just a single Casadi
    function. Each F must take two or three arguments. Each l must take two,
    and Pf must take one.
    
    All of these lists are accessed modulo their respective length; thus,
    time-invariant models can be lists with one element, while time-varying
    periodic model with period T should have T elements.
    
    Input bounds is a dictionary that can contain box constraints on x or u.
    The corresponding entries are "xlb", "xub", "ulb", and "uub". Bounds must
    be lists of vectors of appropriate size.
    
    d should be a list of "disturbances" known a-priori. It can be None to
    indicate that they are all zero, or that they they simply aren't present.
    
    Optional argument verbosity controls how much solver output there is. This
    value must be an integer between 0 and 12 (inclusive). Higher numbers
    indicate more verbose output.

    guess is a dictionary with optional keys "x" and "u". If provided, these
    should be Nx by N+1 and Nu by N arrays respectively. These are fed to the
    solver as an initial guess. These points need not be feasible, but it
    helps if they are.
    
    Return value is a dictionary. Entries "x" and "u" are 2D arrays with the first
    index corresponding to individual states and the second index corresponding
    to time. Entry "status" is a string with solver status.
    """
    starttime = time.clock()    
    
    # Get shapes.    
    (Nx, Nu, Nd) = getModelArgs(F[0])
    
    # Check what bounds were supplied.
    defaultbounds = [("xlb",Nx,-np.Inf),("xub",Nx,np.Inf),("ulb",Nu,-np.Inf),("uub",Nu,np.Inf)]
    for (k,n,M) in defaultbounds:
        if k not in bounds:
            bounds[k] = [M*np.ones((n,))]
    
    # Define NLP variables.
    VAR = ctools.struct_symMX([(
        ctools.entry("x",shape=(Nx,1),repeat=N+1),
        ctools.entry("u",shape=(Nu,1),repeat=N),
    )])
    
    # Decide whether we need to include d or not.    
    if Nd > 0:
        if d is None:
            d = [[0]*Nd] # All zero if unspecified.
        Z = [[VAR["x",k], VAR["u",k], d[k % len(d)]] for k in range(N)]
    else:
        Z = [[VAR["x",k], VAR["u",k]] for k in range(N)]
    
    # Preallocate.
    GUESS = VAR(0) # Guess all variables zero.
    LB = VAR(-np.Inf) # Default all bounds to +/- Inf.
    UB = VAR(np.Inf)
    nlpObj = casadi.MX(0) # Start with dummy objective.
    nlpCon = [None]*N
    getBounds = lambda var,k : bounds[var][k % len(bounds[var])]    
    
    # Steps 0 to N-1.    
    for k in range(N):
        # Model and objective.        
        nlpCon[k] = F[k % len(F)](Z[k])[0] - VAR["x",k+1]
        nlpObj += l[k % len(l)]([VAR["x",k],VAR["u",k]])[0]        
        
        # Variable bounds.
        LB["x",k,:] = getBounds("xlb",k)
        UB["x",k,:] = getBounds("xub",k)
        LB["u",k,:] = getBounds("ulb",k)
        UB["u",k,:] = getBounds("uub",k)
    
    # Adjust bounds for x0 and xN.
    LB["x",0,:] = x0
    UB["x",0,:] = x0
    LB["x",N,:] = getBounds("xlb",N)
    UB["x",N,:] = getBounds("xub",N)
    if Pf is not None:
        nlpObj += Pf([VAR["x",N]])[0]
     
    # Bounds for constraints (all are equality constraints).
    conlb = np.zeros((Nx*N,))
    conub = np.zeros((Nx*N,))
    
    # Make constraints into a single large vector.     
    nlpCon = casadi.vertcat(nlpCon) 
    
    # Worry about user-supplied guesses.
    if "x" in guess:    
        for k in range(N+1):
            GUESS["x",k,:] = guess["x"][:,k]
    if "u" in guess:
        for k in range(N):
            GUESS["u",k,:] = guess["u"][:,k]
    
    # Create solver and stuff.
    nlp = casadi.MXFunction(casadi.nlpIn(x=VAR),casadi.nlpOut(f=nlpObj,g=nlpCon))
    solver = casadi.NlpSolver("ipopt",nlp)
    solver.setOption("print_level",verbosity)
    solver.setOption("print_time",verbosity > 2)   
    solver.init()

    solver.setInput(GUESS,"x0")
    solver.setInput(LB,"lbx")
    solver.setInput(UB,"ubx")
    solver.setInput(conlb,"lbg")
    solver.setInput(conub,"ubg")
    
    # Solve.    
    solver.evaluate()
    status = solver.getStat("return_status")
    if verbosity > 0:
        print("Solver Status:", status)
    
    # Get solution.
    OPTVAR = VAR(solver.getOutput("x"))
    x = np.hstack(OPTVAR["x",:,:])
    u = np.hstack(OPTVAR["u",:,:])
    
    # Add singleton dimension if n = 1 or p = 1.
    if Nx == 1:    
        x = x.reshape(1,N+1)
    if Nu == 1:
        u = u.reshape(1,N)
    
    endtime = time.clock()
    if verbosity > 1:
        print("Took %g s." % (endtime - starttime))    
    
    return {"x" : x, "u" : u, "status" : status, "time" : endtime - starttime}        
    
 
def getCollocationConstraints(f,Delta,Nt,Nc,d=None):
    """
    Returns constraints for collocation of ODE f in variables x and u.
    
    Nt is the number of time points and Nc is the number of collocation points
    for each time period. Delta is the timestep between time points.
    
    If provided, d is a list of disturbances at each time point. It is acccessed
    mod length, so periodic disturbances with period < Nt can be used.
    
    Returns a struct of variables and a list of all the constraints.
    """

    # Get shapes.        
    (Nx, Nu, Nd) = getModelArgs(f)
    
    # Define NLP variables.
    VAR = ctools.struct_symMX([(
        ctools.entry("x",shape=(Nx,1),repeat=Nt+1), # States at time points.
        ctools.entry("z",shape=(Nx,Nc),repeat=Nt), # Collocation points.
        ctools.entry("u",shape=(Nu,1),repeat=Nt), # Control actions.
    )])
    
    # Get collocation weights.
    [r,A,B,q] = colloc.colloc(Nc, True, True)
    
    # Preallocate.
    CON = []
    
    for k in range(Nt):
        # Decide about disturbances.        
        if d is not None:
            thisd = [d[k % len(d)]]
        else:
            thisd = []
        
        # Build a convenience list.
        zaug = [VAR["x",k]] + [VAR["z",k,:,j] for j in range(Nc)] + [VAR["x",k+1]]
        
        # Loop through interior points.        
        for j in range(1,len(zaug) - 1):
            thisargs = [zaug[j],VAR["u",k]] + thisd
            thiscon = Delta*f(thisargs)[0] # Start with function evaluation.
            
            #import pdb; pdb.set_trace()
            # Add collocation weights.
            for jprime in range(len(zaug)):
                thiscon -= A[j,jprime]*zaug[jprime]
            
            CON.append(thiscon)
    
    CONLB = np.zeros((Nx*Nt*Nc,))
    CONUB = CONLB.copy()
        
    return [VAR, CON, CONLB, CONUB]
    
   
# =================================
# Plotting
# =================================


    
def mpcplot(x,u,t,xsp=None,fig=None,xinds=None,uinds=None,tightness=.5):
    """
    Makes a plot of the state and control trajectories for an mpc problem.
    
    Inputs x and u should be n by N+1 and p by N numpy arrays. xsp if provided
    should be the same size as x. t should be a numpy N+1 vector.
    
    If given, fig is the matplotlib figure handle to plot everything. If not
    given, a new figure is used.
    
    xinds and uinds are optional lists of indices to plot. If not given, all
    indices of x and u are plotted.
    
    Returns the figure handle used for plotting.
    """
    
    # Process arguments.
    if xinds is None:
        xinds = range(x.shape[0])
    if uinds is None:
        uinds = range(u.shape[0])
    if fig is None:
        fig = plt.figure()
    if xsp is None:
        xlspec = "-k"
        ulspec = "-k"
        plotxsp = False
    else:
        xlspec = "-g"
        ulspec = "-b"
        plotxsp = True
    
    # Figure out how many plots to make.
    numrows = max(len(xinds),len(uinds))
    if numrows == 0: # No plots to make.
        return None
    numcols = 2
    
    # u plots.
    u = np.hstack((u,u[:,-1:])) # Repeat last element for stairstep plot.
    for i in range(len(uinds)):
        uind = uinds[i]
        a = fig.add_subplot(numrows,numcols,numcols*(i+1))
        a.step(t,np.squeeze(u[uind,:]),ulspec)
        a.set_xlabel("Time")
        a.set_ylabel("Control %d" % (uind + 1))
    
    # x plots.    
    for i in range(len(xinds)):
        xind = xinds[i]
        a = fig.add_subplot(numrows,numcols,numcols*(i+1) - 1)
        a.hold("on")
        a.plot(t,np.squeeze(x[xind,:]),xlspec,label="System")
        if plotxsp:
            a.plot(t,np.squeeze(xsp[xind,:]),"--r",label="Setpoint")
            plt.legend(loc="best")
        a.set_xlabel("Time")
        a.set_ylabel("State %d" % (xind + 1))
    
    # Layout tightness.
    if not tightness is None:
        fig.tight_layout(pad=tightness)        
    
    return fig