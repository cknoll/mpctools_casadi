from __future__ import print_function, division # Grab some handy Python3 stuff.
import numpy as np
import scipy.linalg
import casadi
import casadi.tools as ctools
import matplotlib.pyplot as plt
import time
import itertools
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

# Developer notes:
#
#
# - Documentation is very important. Try to use descriptive variable names as
#   much as possible, and make sure every function has a docstring.
#
#
# - We use the print function here, so any calls to print must have the
#   arguments surrounded by parentheses.
#
#
# - Since casadi is actively being developed, we should avoid duplicated code
#   as much as possible. If you ever find yourself copy/pasting more than a few
#   lines of code in this module, consider writing a separate function.
#
#   An exception to this rule is the lmpc function. In its current form, it is
#   a lot like nmpc. Ideally casadi will eventually allow us to solve QPs
#   explicitly instead of solving it as a general NLP, so we keep this a
#   separate function now because at some point it will become fundamentally
#   different from nmpc.

# =================================
# Linear
# =================================


def lmpc(A,B,x0,N,Q,R,q=None,r=None,M=None,bounds={},D=None,G=None,d=None,verbosity=5):
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
    m = B[0].shape[1]
    
    # Fill in default bounds.
    bounds = fillBoundsDict(bounds,n,m)
    getBounds = lambda var,k : bounds[var][k % len(bounds[var])]    
    
    # Define NLP variables.
    [VAR, LB, UB, GUESS] = getCasadiVars(n,m,N)
    
    # Preallocate.
    qpF = casadi.MX(0) # Start with dummy objective.
    qpG = [None]*N # Preallocate, although we could just append.
    
    # First handle objective/constraint terms that aren't optional.    
    for k in range(N):
        if k != N:
            LB["u",k,:] = getBounds("ulb",k)
            UB["u",k,:] = getBounds("uub",k)
            
            qpF += .5*mtimes(VAR["u",k].T,R[k % len(R)],VAR["u",k]) 
            qpG[k] = mtimes(A[k % len(A)],VAR["x",k]) + mtimes(B[k % len(B)],VAR["u",k]) - VAR["x",k+1]
        
        if k == 0:
            LB["x",0,:] = x0
            UB["x",0,:] = x0
        else:
            LB["x",k,:] = getBounds("xlb",k)
            UB["x",k,:] = getBounds("xub",k)
        
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
    [OPTVAR,obj,status,solver] = callSolver(VAR,LB,UB,GUESS,qpF,qpG,conlb,conub,verbosity)
    x = np.hstack(OPTVAR["x",:,:])
    u = np.hstack(OPTVAR["u",:,:])
    
    # Add singleton dimension if n = 1 or p = 1.
    x = atleastnd(x)
    u = atleastnd(u)

    endtime = time.clock()
    if verbosity > 1:
        print("Took %g s." % (endtime - starttime))
    
    return {"x" : x, "u" : u, "status" : status, "time" : endtime - starttime, "obj" : obj}


def atleastnd(arr,n=2):
    """
    Adds an initial singleton dimension to arrays with fewer than n dimensions.
    """
    
    if len(arr.shape) < n:
        arr = arr.reshape((1,) + arr.shape)
    
    return arr

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

def getRungeKutta4(f,Delta,M=1,d=False,name=None):
    """
    Uses RK4 to discretize xdot = f(x,u) with M points and timestep Delta.

    A disturbance argument can be specified by setting d=True. If present, d
    must come last in the list of arguments (i.e. f(x,u,d))

    f must be a Casadi SX function with inputs in the proper order.
    """

    # First find out how many arguments f takes.
    fargs = getModelArgSizes(f,d=d)
    [Nx, Nu, Nd] = [fargs[a] for a in ["x","u","d"]]
    
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
        
def fillBoundsDict(bounds,Nx,Nu):
    """
    Fills a given bounds dictionary with any unspecified defaults.
    """
    
    # Check what bounds were supplied.
    defaultbounds = [("xlb",Nx,-np.Inf),("xub",Nx,np.Inf),("ulb",Nu,-np.Inf),("uub",Nu,np.Inf)]
    for (k,n,M) in defaultbounds:
        if k not in bounds:
            bounds[k] = [M*np.ones((n,))]
            
    return bounds
    
def nmpc(F,l,x0,N,Pf=None,bounds={},d=None,verbosity=5,guess={},timemodel="discrete",M=None,Delta=None):
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
    
    timemodel is a string to decide how to handle time. It must be one of
    "discrete", "rk4", or "colloc" and the argument M must be given the number
    of intermediate points and Delta must have the timestep if something other 
    than "discrete" is chosen. For the 'colloc' model, only the first entry in
    F is taken and used for the entire horizon.    
    
    Return value is a dictionary. Entries "x" and "u" are 2D arrays with the first
    index corresponding to individual states and the second index corresponding
    to time. Entry "status" is a string with solver status.
    """
    starttime = time.clock()    
    
    # Figure out what user wants as time model.
    if timemodel not in ["discrete","rk4","colloc"]:
        raise ValueError("Invalid choice for timemodel!")
    elif timemodel != "discrete" and (M is None or Delta is None):
        raise ValueError("Must supply M and Delta for '%s' timemodel." % (timemodel,))
    if timemodel == "colloc":
        Nc = M
    else:
        Nc = None
        
    if timemodel == "rk4":
        F = [getRungeKutta4(f,Delta,M) for f in F]
    
    # Get shapes.
    Fargs = getModelArgSizes(F[0],d=(d is not None),z=False)
    [Nx, Nu, Nd] = [Fargs[a] for a in ["x","u","d"]]    
    
    # Check what bounds were supplied.
    bounds = fillBoundsDict(bounds,Nx,Nu)
    getBounds = lambda var,k : bounds[var][k % len(bounds[var])]    
    
    # Define NLP variables.
    [VAR, LB, UB, GUESS] = getCasadiVars(Nx,Nu,N,Nc)
    
    
    # Get constraints.
    if timemodel == "colloc":
        [nlpCon, conlb, conub] = getCollocationConstraints(F[0],VAR,Delta,d)
    else:
        [nlpCon, conlb, conub] = getDiscreteTimeConstraints(F,VAR,d)
    
    # Need to flatten everything.
    nlpCon = flattenlist(nlpCon)
    conlb.shape = (conlb.size,)
    conub.shape = (conlb.size,)
    
    # Steps 0 to N-1.
    nlpObj = casadi.MX(0) # Start with dummy objective.    
    for k in range(N):
        # Model and objective.        
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
    
    # Make constraints into a single large vector.     
    nlpCon = casadi.vertcat(nlpCon) 
    
    # Worry about user-supplied guesses.
    if "x" in guess:    
        for k in range(N+1):
            GUESS["x",k,:] = guess["x"][:,k]
    if "u" in guess:
        for k in range(N):
            GUESS["u",k,:] = guess["u"][:,k]
    
    # Call solver and stuff.
    [OPTVAR,obj,status,solver] = callSolver(VAR,LB,UB,GUESS,nlpObj,nlpCon,conlb,conub,verbosity)
    
    x = np.hstack(OPTVAR["x",:,:])
    u = np.hstack(OPTVAR["u",:,:])
    
    # Add singleton dimension if Nx = 1 or Nu = 1.
    x = atleastnd(x)
    u = atleastnd(u)
    
    endtime = time.clock()
    if verbosity > 1:
        print("Took %g s." % (endtime - starttime))    
    
    optDict = {"x" : x, "u" : u, "status" : status, "time" : endtime - starttime, "obj" : obj}

    # Return collocation points if present.
    if Nc is not None:
        xc = np.hstack(OPTVAR["xc",:,:,:])
        optDict["xc"] = xc
    
    return optDict      

def getModelArgSizes(f,u=True,d=False,z=False,error=True):
    """
    Checks number and sizes of arguments for the casadi function f.
    
    Returns a dictionary with sizes for x, z, u, and d. Arguments must always
    be specified in the order (x,z,u,d), with the presence or absence of each
    argument specified by the corresponding optional argument. x must always
    be present.
    
    If the number of suggested arguments does not match the number of arguments
    taken by f, then an error is raised. You can turn off this error by setting
    the optional argument error=False.
    """
    
    # Figure out what arguments f is supposed to have.
    sizes = {}
    args = []
    for (var,tf) in [("x",True), ("z",z), ("u",u), ("d",d)]:
        if tf:
            args += [var]
        else:
            sizes[var] = None
    
    # Figure out what arguments f does have.
    argsizes = []
    for i in range(f.getNumInputs()):
        argsizes.append(f.getInput(i).size())
        
    # Error of something doesn't match.
    if len(args) != len(argsizes) and error:
        raise ValueError("Arguments of f are inconsistent with supplied arguments!")
    
    # Store sizes.
    for (arg, size) in zip(args, argsizes):
        sizes[arg] = size
    
    return sizes

def getVarShapes(f,var,disturbancemodel=False,error=False):
    """
    Returns a dictionary of various sizes for the model f and variables var.
    
    If error is true, issues a ValueError for size inconsistencies.
    """
    issueError = False    
    
    # First check if there are algebraic variables.
    algebraic = "z" in var.keys()    
    
    # Get shapes.
    fargs = getModelArgSizes(f,d=disturbancemodel,z=algebraic)
    [Nx, Nu, Nd, Nz] = [fargs[a] for a in ["x","u","d","z"]]        
    
    # Check sizes and find number of collocation points.
    colloc = "xc" in var.keys()
    sizes = getCasadiVarsSizes(var,colloc,algebraic)
    for (N, k) in [(Nx,"x"), (Nu,"u"), (Nz,"z")]:
        if k in sizes and N != sizes[k]:
            print("*** Warning: size mismatch for %s!" % (k,))
            issueError = error
    sizes["d"] = Nd
        
    # Raise error if requested.        
    if issueError:
        raise ValueError("Inconsistent sizes for one or more variables.")
    
    return sizes
 
def getCollocationConstraints(f,var,Delta,d=None):
    """
    Returns constraints for collocation of ODE f in variables x and u.
    
    var should be a casadi struct_symMX object, e.g. the output of
    getCasadiVars.

    The sizes of variables and the number of time points are inferred from f
    and var. Make sure that the model f and variables var have consistent sizes.    
    
    If provided, d is a list of disturbances at each time point. It is acccessed
    mod length, so periodic disturbances with period < Nt can be used.
    
    Returns a a list of all the constraints and bounds for the constraints.
    """

    # Get shapes.
    shapes = getVarShapes(f,var,disturbancemodel=(d is not None))
    [Nx, Nu, Nd, Nc, Nt] = [shapes[k] for k in ["x","u","d","c","t"]]
        
    # Get collocation weights.
    [r,A,B,q] = colloc.colloc(Nc, True, True)
    
    # Preallocate. CON will be a list of lists.
    CON = []
    
    for k in range(Nt):
        # Decide about disturbances.        
        if d is not None:
            thisd = [d[k % len(d)]]
        else:
            thisd = []
        
        # Build a convenience list.
        xaug = [var["x",k]] + [var["xc",k,:,j] for j in range(Nc)] + [var["x",k+1]]
        thesecon = []
        
        # Loop through interior points.        
        for j in range(1,Nc+2):
            thisargs = [xaug[j],var["u",k]] + thisd
            thiscon = Delta*f(thisargs)[0] # Start with function evaluation.
            
            # Add collocation weights.
            for jprime in range(len(xaug)):
                thiscon -= A[j,jprime]*xaug[jprime]
            thesecon.append(thiscon)
        CON.append(thesecon)
    
    # Return bounds a a NumPy array. This is more convenient, and we choose
    # this order so we can flatten in C order (NumPy's default).
    CONLB = np.zeros((Nt,Nc+1,Nx))
    CONUB = CONLB.copy()
        
    return [CON, CONLB, CONUB]

def getDiscreteTimeConstraints(F,var,d=None):
    """
    Returns constraints for discrete-time model F in variables x and u.
    
    var should be a casadi struct_symMX object, e.g. the output of
    getCasadiVars.

    The sizes of variables and the number of time points are inferred from f
    and var. Make sure that the model f and variables var have consistent sizes.    
    
    If provided, d is a list of disturbances at each time point. It is acccessed
    mod length, so periodic disturbances with period < Nt can be used.
    
    Returns a list of three elements [con, conlb, conub]. con is a list with
    each element a list of constraints for each time point. conlb and conub
    are 3D numpy arrays with conlb[t,n,x] giving the constraint bound at time
    t for constraint n and element x.
    """
    
    # Get shapes.
    shapes = getVarShapes(F[0],var,disturbancemodel=(d is not None))
    [Nx, Nu, Nd, Nt] = [shapes[k] for k in ["x","u","d","t"]]    
    
    # Decide whether we need to include d or not.    
    if Nd is not None and Nd > 0:
        if d is None:
            d = [[0]*Nd] # All zero if unspecified.
        Z = [[var["x",k], var["u",k], d[k % len(d)]] for k in range(Nt)]
    else:
        Z = [[var["x",k], var["u",k]] for k in range(Nt)]
   
    nlpCon = []
    for k in range(Nt):
        nlpCon.append([F[k % len(F)](Z[k])[0] - var["x",k+1]])
    
    conlb = np.zeros((Nt,1,Nx))
    conub = conlb.copy()
    
    return [nlpCon, conlb, conub]
    
def getCasadiVars(Nx,Nu,Nt,Nc=None,Nz=None):
    """
    Returns a casadi struct_symMX with the appropriate variables.
    
    [var, lb, ub guess] = getCasadiVars(Nx,Nu,Nt,Nc=None,Nz=None)    
    
    Also returns objects with default bounds: -inf for lower bounds,
    0 for initial guess, and inf for upper bounds.
    
    Variables are accessed as var[*,t] where t is the time point and * is one
    of the following variable names:
    
    "x" : system states.
    "u" : control inputs.
    "xc" : collocation points (not present if Nc is None)
    "z" : algebraic variables (not present if Nz is None)
    "zc" : collocation albegraic variables (not present if Nc or Nz are None)
    """
    
    args = (
        ctools.entry("x",shape=(Nx,1),repeat=Nt+1), # States at time points.
        ctools.entry("u",shape=(Nu,1),repeat=Nt), # Control actions.
    )
    if Nc is not None:
        args += (ctools.entry("xc",shape=(Nx,Nc),repeat=Nt),) # Collocation points.
    if Nz is not None:
        args += (ctools.entry("z",shape=(Nz,1),repeat=Nt+1),) # Algebraic variables.
        if Nc is not None:
            args += (ctools.entry("zc",shape=(Nz,Nc),repeat=Nt),) # Coll. Alg. variables.
    
    VAR =  ctools.struct_symMX([args])
    LB = VAR(-np.inf)
    UB = VAR(np.inf)
    GUESS = VAR(0)
    
    return [VAR, LB, UB, GUESS]

def getCasadiVarsSizes(var,colloc=False,algebraic=False):
    """
    Checks the entries of a casadi struct_symMX for the proper fields and returns sizes.
    
    The return value is a dictionary. The only keys correspond to elements that
    are actually present in the variable struct.
    
    Raises a ValueError if something is missing.
    """
    
    # Check entries.
    needSizeKeys = ["x", "u"]
    needVarKeys = ["x", "u"]
    if colloc:
        needSizeKeys.append("c")
        needVarKeys.append("xc")
    if algebraic:
        needSizeKeys.append("z")
        needVarKeys.append("z")
        if colloc:
            needVarKeys.append("zc")
    givenKeys = var.keys()
    for k in needVarKeys:
        if k not in givenKeys:
            raise ValueError("Entry %s missing from var! Consider using getCasadiVars for proper structure." % (k,))
            
    # Grab sizes for everything. Format is "sizename" : ("varname", dimension)
    sizes = {"x": ("x",0), "u": ("u",0), "c": ("xc",1), "z": ("z",0)}
    for k in sizes.keys():
        (v,i) = sizes.pop(k) # Get variable name and index.
        if k in needSizeKeys: # Save size if we need it.
            sizes[k] = var[v,0].shape[i]
            
    sizes["t"] = len(var["x"]) - 1 # Time is slightly different.
            
    return sizes
    
def callSolver(var,varlb,varub,varguess,obj,con,conlb,conub,verbosity=5):
    """
    Calls ipopt to solve an NLP.
    
    Arguments are mostly self-explainatory. var, varlb, varub, and varguess
    should be casadi struct_symMX objects, e.g. the outputs of getCasadiVars.
    obj should be a scalar casadi MX object, and con should be a vector casadi
    MX object (possibly from calling casadi.vertcat on a list of constraints).
    conlb and conub should be numpy vectors of the appropriate size.
    
    Returns the optimal variables, the objective functiona status string, and
    the solver object.
    """
    
    nlp = casadi.MXFunction(casadi.nlpIn(x=var),casadi.nlpOut(f=obj,g=con))
    solver = casadi.NlpSolver("ipopt",nlp)
    solver.setOption("print_level",verbosity)
    solver.setOption("print_time",verbosity > 2)   
    solver.init()

    solver.setInput(varguess,"x0")
    solver.setInput(varlb,"lbx")
    solver.setInput(varub,"ubx")
    solver.setInput(conlb,"lbg")
    solver.setInput(conub,"ubg")
    
    # Solve.    
    solver.evaluate()
    status = solver.getStat("return_status")
    if verbosity > 0:
        print("Solver Status:", status)
     
    optvar = var(solver.getOutput("x"))
    obj = float(solver.getOutput("f"))   
     
    return [optvar, obj, status, solver]

def flattenlist(l,depth=1):
    """
    Flattens a nested list of lists of the given depth.
    
    E.g. flattenlist([[1,2,3],[4,5],[6]]) returns [1,2,3,4,5,6]. Note that
    all sublists must have the same depth.
    """
    for i in range(depth):
        l = list(itertools.chain.from_iterable(l))
    return l
   
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
     