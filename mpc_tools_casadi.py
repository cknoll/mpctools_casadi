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
a 4th-order Runge-Kutta method. Collocation is also available.

Finally, there is a draft version of NMHE. These functions use different
input arguments, but I think this way is better than how the MPC functions are
done, and so the MPC function call syntax may change in the future.

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
    for k in range(N+1):
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
                        - mtimes(G[k % len(d)],VAR["x",k]) - d[k % len(d)])
        s = (D[0].shape[0]*N, 1) # Shape for inequality RHS vector.
        conlb = np.concatenate(conlb,-np.inf*np.ones(s))
        conub = np.concatenate(conub,np.zeros(s))
    
    # Make qpG into a single large vector.     
    qpG = casadi.vertcat(qpG) 
    
    # Create solver and stuff.
    ipoptstart = time.clock()
    [OPTVAR,obj,status,solver] = callSolver(VAR,LB,UB,GUESS,qpF,qpG,conlb,conub,verbosity=verbosity)
    ipoptend = time.clock()
    x = np.hstack(OPTVAR["x",:,:])
    u = np.hstack(OPTVAR["u",:,:])
    
    # Add singleton dimension if n = 1 or p = 1.
    x = atleastnd(x)
    u = atleastnd(u)
    
    optVars = {"x" : x, "u" : u, "status" : status}
    optVars["obj"] = obj
    optVars["ipopttime"] = ipoptend - ipoptstart
    
    endtime = time.clock()
    if verbosity > 1:
        print("Took %g s." % (endtime - starttime))
    optVars["time"] = endtime - starttime
    
    return optVars

# =================================
# Building CasADi Functions
# =================================

def getCasadiFunc(f,Nx,Nu=0,Nd=0,name=None):
    """
    Takes a function handle and turns it into a Casadi function.
    
    f should be defined to take three keyword arguments x, u, and d. It should
    be written so that x, u, and d are python LISTs of length Nx, Nu, and Nd
    respectively. It must return a LIST of length Nx.
    """
    
    # Create symbolic variables.
    [args, invar, _] = __getCasadiSymbols(Nx,Nu,Nd)    
    
    # Create symbolic function.
    outvar = [casadi.vertcat(f(**args))]
    fcasadi = casadi.MXFunction(invar,outvar)
    fcasadi.setOption("name",name if name is not None else "f")
    fcasadi.init()
    
    return fcasadi

def getCasadiFuncGeneralArgs(f,varsizes,varnames=None,funcname="f"):
    """
    Takes a function handle and turns it into a Casadi function.
    
    f should be defined to take a specified number of arguments and return a
    LIST of outputs. varnames, if specified, gives names to each of the inputs,
    but this is unnecessary. sizes should be a list of how many elements are
    in each one of the inputs.
    
    This version is more general because it lets you specify arbitrary
    arguments, but you have to make sure you do everything properly.
    """
    
    # Create symbolic variables.
    if varnames is None:
        varnames = ["x%d" % (i,) for i in range(len(varsizes))]
    elif len(varsizes) != len(varnames):
        raise ValueError("varnames must be the same length as varsizes!")    
    args = [casadi.MX.sym(name,size) for (name,size) in zip(varnames,varsizes)]
    
    # Now evaluate function and make a Casadi object.    
    fval = [casadi.vertcat(f(*args))]
    fcasadi = casadi.MXFunction(args,fval)
    fcasadi.setOption("name",funcname)
    fcasadi.init()
    
    return fcasadi

def getCasadiIntegrator(f,Delta,Nx,Nu=0,Nd=0,name=None,dae=False,abstol=1e-8,reltol=1e-8):
    """
    Gets a Casadi integrator for function f from 0 to Delta.
    
    Set d to True or False to decide whether there is a disturbance model.
    """
    
    if dae:
        raise NotImplementedError("DAE systems not implemented yet.")
    
    # Create symbolic variables for integrator I(x0,p).
    [args, invar, zoh] = __getCasadiSymbols(Nx,Nu,Nd,combine=True)
    x0 = args["x"]
    fode = casadi.vertcat(f(**args))   
    
    # Build ODE and integrator.
    invar = casadi.daeIn(x=x0,p=casadi.vertcat(zoh))
    outvar = casadi.daeOut(ode=fode)
    ode = casadi.MXFunction(invar,outvar)
    
    integrator = casadi.Integrator("cvodes",ode)
    integrator.setOption("abstol",abstol)
    integrator.setOption("reltol",reltol)
    integrator.setOption("tf",Delta)
    integrator.setOption("name","int_f")
    integrator.init()
    
    # Now do the subtle bits. Integrator has arguments x0 and p, but we need
    # arguments x, u, d.
    
    # Create symbolic variables for function F(x,u,d).
    [args, invar, zoh] = __getCasadiSymbols(Nx,Nu,Nd,combine=False)  
    
    wrappedIntegrator = integrator(x0=args["x"],p=casadi.vertcat(zoh))
    F = casadi.MXFunction(invar,wrappedIntegrator)
    F.setOption("name",name if name is not None else "F")
    F.init()
    
    return F

def getLinearization(f,xs,us=None,ds=None,Delta=None):
        """
        Returns linear state-space model for the given function.
        """
        # Put together function arguments.
        args = []
        N = {}
        for (arg,name) in [(xs,'x'),(us,'u'),(ds,'d')]:
            if arg is not None:
                arg = np.array(arg)
                args.append(arg)
                N[name] = arg.shape[0]
            else:
                N[name] = 0
        
        # Evalueate function.
        fs = np.array(f(args)[0]) # Column vector.        
        
        # Get Jacobian.
        jacobian = f.fullJacobian()
        jacobian.init()
        Js = np.array(jacobian(args)[0])        
        
        A = Js[:,:N["x"]]
        B = Js[:,N["x"]:N["x"]+N["u"]]
        Bp = Js[:,N["x"]+N["u"]:N["x"]+N["u"]+N["d"]]
        
        if Delta is not None:
            [A, B, Bp, f] = c2d_augmented(A,B,Bp,fs,Delta)
        
        return {"A": A, "B": B, "Bp": Bp, "f": fs}

# =================================
# Nonlinear
# =================================
    
# Nonlinear functions adapted from scripts courtesy of Lars Petersen.

def nmpc(F,l,x0,N,Pf=None,bounds={},d=None,verbosity=5,guess={},timemodel="discrete",M=None,Delta=None,returnTimeInvariantSolver=False):
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
    
    Alternatively, if returnTimeInvariantSolver is set to True, the function will
    return a TimeInvariantSolver object. This is just a lightweight wrapper for
    a casadi solver object with some convenience methods. This is useful for
    closed-loop simulations of time-invariant systems because all you have to
    do is change the initial condition, cycle your guess by one period, and
    re-solve.
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
        F = [getRungeKutta4(f,Delta,M,d=(d is not None)) for f in F]
    
    # Get shapes.
    Fargs = getModelArgSizes(F[0],d=(d is not None),z=False)
    [Nx, Nu, Nd] = [Fargs[a] for a in ["x","u","d"]]    
    
    # Check what bounds were supplied.
    bounds = fillBoundsDict(bounds,Nx,Nu)
    getBounds = lambda var,k : bounds[var][k % len(bounds[var])]    
    
    # Define NLP variables.
    [VAR, LB, UB, GUESS] = getCasadiVars(Nx,Nu,N,Nc)
    
    # Decide aboud disturbances.
    if d is not None:
        PAR = ctools.struct_symMX([(ctools.entry("d",shape=d[0].shape,repeat=N),)])
        dsym = PAR["d"]
        PARVAL = PAR(0)
        for t in range(N):
            PARVAL["d",t] = d[t % len(d)]
    else:
        dsym = None
        PAR = None
        PARVAL = None
    
    # Get constraints.
    if timemodel == "colloc":
        [nlpCon, conlb, conub] = getCollocationConstraints(F[0],VAR,Delta,dsym)
    else:
        [nlpCon, conlb, conub] = getDiscreteTimeConstraints(F,VAR,dsym)
    
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
    
    # Now decide what to do based on what the user has said.
    if returnTimeInvariantSolver:
        solver = TimeInvariantSolver(VAR,LB,UB,GUESS,nlpObj,nlpCon,conlb,conub,par=PAR,verbosity=verbosity,parval=PARVAL)    
        return solver
    
    # Call solver and stuff.
    [OPTVAR,obj,status,solver] = callSolver(VAR,LB,UB,GUESS,nlpObj,nlpCon,conlb,conub,par=PAR,verbosity=verbosity,parval=PARVAL)
    
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

def __getCasadiSymbols(Nx,Nu=None,Nd=None,combine=False):
    """
    Returns symbolic variables of appropriate sizes.

    [args, invar, zoh] = __getCasadiSymbols(Nx,Nu,Nd)

    args is a dictionary of the input variables. invar is a list in [x,u,d]
    order. zoh is a list of the variables that are constant on a given
    interval (i.e. u and d).
    
    If combine is true, u and d will be combined into a single parameter p.
    The entries args['u'] and args['d'] will access the appropriate entries of
    p, and invar/zoh will contain only p and not the individual u and d. This
    is useful, e.g. for building integrators.
    """
    
    # Create symbolic variables.
    x  = casadi.MX.sym("x",Nx) # States
    args = {"x" : x}
    zoh = []
    if combine:
        Np = Nu + Nd
        if Np > 0:
            p = casadi.MX.sym("p",Np)
            zoh = [p]
            if Nu > 0:
                args["u"] = p[:Nu]
                if Nd > 0:
                    args["d"] = p[Nu:]
            elif Nd > 0:
                args["d"] = p[:Nd]
        else:
            zoh = []
    else:
        if Nu > 0:
            u  = casadi.MX.sym("u",Nu) # Control
            args["u"] = u
            zoh += [u]
        if Nd > 0:
            d  = casadi.MX.sym("d",Nd) # Other (e.g. parameters or disturbances)
            args["d"] = d
            zoh += [d]
    invar = [x] + zoh
    
    return [args, invar, zoh]

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

# =================================
# Constraint Building
# =================================
 
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

    # Get symbolic variables.
    [args, _, zoh] = __getCasadiSymbols(Nx,Nu,Nd)
    x0 = args["x"]
    
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

# =================================
# MHE and Functions
# =================================

# Note that the MHE function uses a different paradigm than the MPC functions
# above. I think this method is better, so eventually the MPC functions should
# be rewritten. Basically, the MHE functions were written to be much more
# flexible with respect to function arguments, what variables are present,
# etc., and so it permits a lot more functionality in less code.

def nmhe(f,h,u,y,l,N,lx=None,x0hat=None,bounds={},g=None,p=None,verbosity=5,guess={},largs=["w","v"]):
    """
    Solves nonlinear MHE problem.
    
    N muste be a dictionary with at least entries "x", "y", and "t". "w" may be
    specified, but it is assumed to be equal to "x" if not given. "v" is always
    taken to be equal to "y". If parameters are present, you must also specify
    a "p" entry.
    
    u, y, and p must be 2D arrays with the time dimension first. Note that y
    should have N["t"] + 1 rows, and u and p should have N["t"] rows.
    """
    # Check specified sizes.
    try:
        for i in ["t","x","y"]:
            if N[i] <= 0:
                N[i] = 1
        if "w" not in N:
            N["w"] = N["x"]
        N["v"] = N["y"] 
    except KeyError:
        raise KeyError("Invalid or missing entries in N dictionary!")
        
    # Now get the shapes of all the variables that are present.
    allShapes = __generalVariableShapes(N)
    
    # Build Casadi symbolic structures. These need to be separate because one
    # is passed as a set of variables and one is a set of parameters.
    parNames = set(["u","p","y"])
    parStruct = __casadiSymStruct(allShapes.copy(),parNames)(0)
        
    varNames = set(["x","z","w","v","xc","zc"])
    varStruct = __casadiSymStruct(allShapes.copy(),varNames)
    varlbStruct = varStruct(-np.inf)
    varubStruct = varStruct(np.inf)
    varguessStruct = varStruct(0)
    
    # Now we fill up the parameters.
    for (name,val) in [("u",u),("p",p),("y",y)]:
        if name in parStruct.keys():
            for t in range(N["t"]):
                parStruct[name,t] = val[t,:]
    
    # Now smush everything back together to get the constraints.
    varAndPar = {}
    for k in parStruct.keys():
        varAndPar[k] = parStruct[k]
    for k in varStruct.keys():
        varAndPar[k] = varStruct[k]
    
    # Buid up constraints.
    if "z" in allShapes.keys(): # Need to decide about algebraic constraints.
        Ng = N["z"]
    else:
        Ng = None
    constraints = __generalDiscreteConstraints(varAndPar,N["t"],f=f,Nf=N["x"],
        g=g,Ng=Ng,h=h,Nh=N["y"],l=l,largs=largs)
    
    con = []
    conlb = np.zeros((0,))
    conub = conlb.copy()
    for f in ["state","measurement","algebra"]:
        if f in constraints.keys():
            con += flattenlist(constraints[f]["con"])
            conlb = np.concatenate([conlb,constraints[f]["lb"].flatten()])
            conub = np.concatenate([conub,constraints[f]["ub"].flatten()])
    con = casadi.vertcat(con)
    
    obj = casadi.MX(0)
    for t in range(N["t"]):
        obj += constraints["cost"][t]
    if lx is not None and x0hat is not None:
        obj += lx([varStruct["x",0] - x0hat])[0]
        
    # Now call the solver.
    solverReturn = callSolver(varStruct,varlbStruct,varubStruct,varguessStruct,
        obj,con,conlb,conub,verbosity=verbosity)
    
    # Should probably do a bit more post-processing to make these return
    # values a bit more user-friendly.                         
    return solverReturn

def __generalVariableShapes(sizeDict):
    """
    Generates variable shapes from the size dictionary N.
    
    The keys of N must be a subset of
        ["x","z","u","w","p","y","v","c"]    
    
    If present, "c" will specify collocation, at which point extra variables
    will be created.
    
    Each entry in the returned dictionary will be a 3-element tuple. The first
    is the number of copies for each variable, the second is the variable's
    shape, and the third is a boolean flag that is True to indicate the
    variable is pointwise and False otherwise. 
    """

    # Figure out what variables are supplied.
    allsizes = set(["x","z","u","w","p","y","v","c","t"])
    givensizes = allsizes.intersection(sizeDict.keys())
    
    # Make sure we were given a time.
    try:
        Nt = sizeDict["t"]
    except KeyError:
        raise KeyError("Entry 't' must be provided!")
    
    # Now we're going to build a data structure that says how big each of the
    # variables should be. The first entry 1 if there should be N+1 copies of
    # the variable and 0 if there should be N. The second is a tuple of sizes.
    allvars = {
        "x" : (1,("x",)),
        "z" : (1,("z",)),
        "u" : (0,("u",)),
        "w" : (0,("w",)),
        "p" : (0,("p",)),
        "y" : (0,("y",)),
        "v" : (0,("v",)),
        "xc": (0,("x","c")),
        "zc": (0,("z","c")),        
    }    
    
    # Now loop through all the variables, and if we've been given all of the
    # necessary sizes, then add that variable
    shapeDict = {}
    for (v, (t0, shapeinds)) in allvars.items():
        if givensizes.issuperset(shapeinds):
            shape = [sizeDict[i] for i in shapeinds]
            if len(shape) == 0:
                shape = [1,1]
            elif len(shape) == 1:
                shape += [1]
            shapeDict[v] = {"repeat" : Nt + t0, "shape" : tuple(shape)}
    
    return shapeDict
    

def __generalDiscreteConstraints(var,Nt,f=None,Nf=0,g=None,Ng=0,h=None,Nh=0,t0=0,l=None,largs=[]):
    """
    Creates general state evolution constraints for the following system:
    
       x^+ = f(x,z,u,w,p)                        \n
       g(x,z,p) = 0                              \n
       y = h(x,z,p) + v                          \n
       
    The variables are intended as follows:
    
        x: differential states                  \n
        z: algebraic states                     \n
        u: control actions                      \n
        w: state disturbances                   \n
        p: fixed system parameters              \n
        y: meadured outputs                     \n
        v: noise on outputs
    
    Also builds up a list of stage costs l(...). Note that if l is given, its
    arguments must be specified in largs as a tuple of variable names.
    
    In principle, you can use the variables for whatever you want, but they
    must show up in the proper order. We do very little checking of this, so
    if this function errors, make sure you are passing the proper variables.
    
    var should be a dictionary with entries "x", "u", "y", etc., that give
    either casadi variables or data values. Data must be accessed as follows:

        var["x"][t][k] gives the kth state of x at time t
        
    In particular, if struct is a casadi.tools.struct_symMX object, then
    setting var["x"] = struct["x"] will suffice. If you want to use your own
    data structure, so be it.
    
    Note that f, g, and h should be LISTS of functions that will be accessed
    modulo length. For a time-invariant system, the lists should only have
    one element. Note that in principle, if you can define your time-varying
    system using parameters p, this is perferrable to actually having different
    f, g, and h at each time point. However, for logical conditions, parameters
    might not be sufficient, at which point you will have to use the
    time-varying list.    
    
    Returns a dictionary with entries "state", "algebra", and "measurement".
    Note that the relevant fields will be missing if f, g, or h are set to
    None. Each entry in the return dictionary will be a list of lists, although
    for this class of constraint, each of those lists will have only one
    element. The list of stage costs is in "cost". This one is just a list.
    """
    
    # Figure out what variables are supplied.
    allvars = set(["x","z","u","w","p","y","v"])
    givenvars = allvars.intersection(var.keys())
    
    # Now sort out arguments for each function.
    isGiven = lambda v: v in givenvars # Membership function.
    args = {
        "f" : filter(isGiven,["x","z","u","w","p"]),
        "g" : filter(isGiven, ["x","z","p"]),
        "h" : filter(isGiven, ["x","z","p"]),
        "l" : filter(isGiven, largs)
    }
    getArgs = lambda func, times: [[var[v][t] for v in args[func]] for t in times]
    tintervals = range(t0,t0+Nt)
    tpoints = range(t0,t0+Nt+1)
    
    # Preallocate return dictionary.
    returnDict = {}    
    
    # State evolution f.   
    if f is not None:
        if Nf <= 0:
            raise ValueError("Nf must be a positive integer!")
        fargs = getArgs("f",tintervals)
        state = []
        for t in tintervals:
            thiscon = f[t % len(f)](fargs[t])[0]
            if "x" in givenvars:
                thiscon -= var["x"][t+1]
            state.append([thiscon])
        lb = np.zeros((Nt,Nf))
        ub = lb.copy()
        returnDict["state"] = dict(con=state,lb=lb,ub=ub)
            
    # Algebraic constraints g.
    if g is not None:
        if Ng <= 0:
            raise ValueError("Ng must be a positive integer!")
        gargs = getArgs("g",tpoints)
        algebra = []
        for t in tpoints:
            algebra.append([g[t % len(g)](gargs[t])[0]])
        lb = np.zeros((Nt+1,Ng))
        ub = lb.copy()
        returnDict["algebra"] = dict(con=algebra,lb=lb,ub=ub)
        
    # Measurements h.
    if h is not None:
        if Nh <= 0:
            raise ValueError("Nh must be a positive integer!")
        hargs = getArgs("h",tintervals)
        measurement = []
        for t in tintervals:
            thiscon = h[t % len(h)](hargs[t])[0]
            if "y" in givenvars:
                thiscon -= var["y"][t]
            if "v" in givenvars:
                thiscon += var["v"][t]
            measurement.append([thiscon])
        lb = np.zeros((Nt,Nh))
        ub = lb.copy()
        returnDict["measurement"] = dict(con=measurement,lb=lb,ub=ub)
    
    # Stage costs.
    if l is not None:
        largs = getArgs("l",tintervals)
        cost = []
        for t in tintervals:
            cost.append(l[t % len(l)](largs[t])[0])
        returnDict["cost"] = cost
    
    return returnDict
    
def __casadiSymStruct(allVars,theseVars=None):
    """
    Returns a Casadi sym struct for the variables in allVars.
    
    To use only a subset of variables, set theseVars to the subset of variables
    that you need. If theseVars is not None, then only variable names appearing
    in allVars and theseVars will be created.
    """
    
    # Figure out what names we need.
    varNames = set(allVars.keys())
    if theseVars is not None:
        for v in varNames.difference(theseVars):
            allVars.pop(v)
    
    # Build casadi sym_structMX    
    structArgs = tuple([ctools.entry(name,**args) for (name,args) in allVars.items()])
    return ctools.struct_symMX([structArgs])
    
# =================================
# Solver Interfaces
# =================================                    
    
def callSolver(var,varlb,varub,varguess,obj,con,conlb,conub,par=None,verbosity=5,parval=None):
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
    nlpInputs = {"x" : var}
    if par is not None:
        nlpInputs["p"] = par
    
    nlp = casadi.MXFunction(casadi.nlpIn(**nlpInputs),casadi.nlpOut(f=obj,g=con))
    solver = casadi.NlpSolver("ipopt",nlp)
    solver.setOption("print_level",verbosity)
    solver.setOption("print_time",verbosity > 2)   
    solver.init()

    solver.setInput(varguess,"x0")
    solver.setInput(varlb,"lbx")
    solver.setInput(varub,"ubx")
    solver.setInput(conlb,"lbg")
    solver.setInput(conub,"ubg")
    if parval is not None:
        solver.setInput(parval,"p")
    
    # Solve.    
    solver.evaluate()
    status = solver.getStat("return_status")
    if verbosity > 0:
        print("Solver Status:", status)
     
    optvar = var(solver.getOutput("x"))
    obj = float(solver.getOutput("f"))   
     
    return [optvar, obj, status, solver]

class TimeInvariantSolver(object):
    """
    A simple class to reduce overhead for solving time-invariant problems.
    
    Creates one casadi solver object at the beginning and then just adjusts
    bounds and/or parameters so that re-solving is quick.
    
    This is most helpful for moving-horizon simulations for time-invariant
    systems.
    """
    
    @property
    def var(self):
        return self.__var
        
    @property
    def lb(self):
        return self.__lb
        
    @property
    def ub(self):
        return self.__ub
    
    @property
    def guess(self):
        return self.__guess    
    
    @property
    def con(self):
        return self.__con
    
    @property
    def obj(self):
        return self.__obj
    
    @property
    def conlb(self):
        return self.__conlb
        
    @property
    def conub(self):
        return self.__conub
    
    @property
    def par(self):
        return self.__par
        
    @property
    def verbosity(self):
        return self.__verbosity
        
    @verbosity.setter
    def verbosity(self,v):
        self.__verbosity = min(max(v,0),12)
    
    @property
    def guesser(self):
        return self.__guesser    
    
    def __init__(self,var,varlb,varub,varguess,obj,con,conlb,conub,par=None,verbosity=5,parval=None):
        """
        Store variables, constraints, etc., and generate the solver.
        """
        self.__var = var
        self.__lb = varlb
        self.__ub = varub
        self.__guess = varguess
        
        self.__obj = obj
        self.__con = con
        self.__conlb = conlb
        self.__conub = conub
        
        self.__parsym = par
        self.__par = parval
        
        self.verbosity = verbosity
        self.initializeSolver()
        self.__guesser = None
            
    def initializeSolver(self):
        """
        Recreates the solver object completely.
        
        You shouldn't really ever need to do this manually because it is called
        automatically when the object is first created.
        """
        
        nlpInputs = {"x" : self.var}
        if self.__parsym is not None:
            nlpInputs["p"] = self.__parsym
        nlpInputs = casadi.nlpIn(**nlpInputs)
        nlpOutputs = casadi.nlpOut(f=self.obj,g=self.con)
        nlp = casadi.MXFunction(nlpInputs,nlpOutputs)
        
        solver = casadi.NlpSolver("ipopt",nlp)
        solver.setOption("print_level",self.verbosity)
        solver.setOption("print_time",self.verbosity > 2)   
        solver.init()
    
        solver.setInput(self.guess,"x0")
        solver.setInput(self.lb,"lbx")
        solver.setInput(self.ub,"ubx")
        solver.setInput(self.conlb,"lbg")
        solver.setInput(self.conub,"ubg")
        if self.par is not None:
            solver.setInput(self.par,"p")
        
        self.solver = solver
        
    def solve(self):
        """
        Calls the nlp solver and solves the current problem.
        """
        solver = self.solver        
        
        starttime = time.clock()
        solver.evaluate()
        ipoptendtime = time.clock()
        status = solver.getStat("return_status")
        if self.verbosity > 0:
            print("Solver Status:", status)
         
        optvar = self.var(solver.getOutput("x"))
        optvarDict = casadiStruct2numpyDict(optvar)       
        
        obj = float(solver.getOutput("f"))   
        optvarDict["obj"] = obj
        optvarDict["status"] = status
        optvarDict["ipopttime"] = ipoptendtime - starttime                
        
        endtime = time.clock()
        optvarDict["time"] = endtime - starttime                
        
        return optvarDict            
    
    def saveguess(self,guess,toffset=1):
        """
        Stores the vales from the numpy Dict guess into the guess field.
        
        This is useful to store the results of the previous step as a guess for
        the next step. toffset defaults to 1, which means the time indices are
        shifted by 1, but you can set this to whatever you want.
        """
        for k in set(guess.keys()).intersection(self.guess.keys()):
            val = guess[k].copy()
            
            # These values and the guess structure can potentially have a
            # different number of time points. So, we need to figure out
            # what range of times will be valid for both things. The offset
            # makes things a bit weird.
            tmaxVal = val.shape[0]
            tmaxGuess = len(self.guess[k]) + toffset
            tmax = min(tmaxVal,tmaxGuess)
            
            tmin = max(toffset,0)
            
            # Now actually store the stuff.            
            for t in range(tmin,tmax):
                self.guess[k,t-toffset] = val[t,...]
    
    def fixvar(self,var,t,val):
        """
        Fixes variable var at time t to val.
        """
        self.lb[var,t] = val
        self.ub[var,t] = val
        self.guess[var,t] = val
        
        self.solver.setInput(self.lb,"lbx")
        self.solver.setInput(self.ub,"ubx")
        self.solver.setInput(self.guess,"x0")
    
    def changeparvals(self,name,values):
        """
        Updates time-varying parameter given a list of values.
        
        values should be [p(0), p(1), ...], with numbers referring to time
        points. If parameters are not time-varying, then values should be a
        list with one element.
        """
        
        for t in range(len(self.par[name])):
            self.par[name,t] = values[t % len(values)]
        self.solver.setInput(self.par,"p")
    
    def storeguesser(self,guesser):
        """
        Saves a one-timestep solver used to generate guesses.
        """
        # Should probably do some checking to make sure this a valid entry,
        # i.e. is a TimeInvariantSolver object.
        self.__guesser = guesser
    
    def calcguess(self,t,saveguess=True):
        """
        Calculates a one-step guess from time t to t+1.
        
        The initial state for the one-step subproblem is taken from the current
        guess field. The optimal values are stored to the appropriate guess
        fields.
        
        Note that the guesser must have already been initialized.
        """
        if self.guesser is None:
            raise AttributeError("Guesser has not been initialized!")
            
        # Set the guesser's variable bounds.
        for var in self.guess.keys():
            toffset = len(self.guesser.guess[var]) - 1
            if toffset == 1:
                self.guesser.fixvar(var,0,self.guess[var,t])
            self.guesser.lb[var,toffset] = self.lb[var,t+toffset]
            self.guesser.ub[var,toffset] = self.ub[var,t+toffset]
        
        onestep = self.guesser.solve()
        self.saveguess(onestep, toffset=-t)        
        
        return onestep

# ================================
# Simulation
# ================================

class OneStepSimulator(object):
    """
    Simulates a continuous-time system.
    """
    
    def __init__(self,ode,Delta,Nx,Nu,Nd=0):
        """
        Initialize by specifying model and sizes of everything.
        """
        # Create variables.
        x = casadi.MX.sym("x",Nx)
        p = casadi.MX.sym("p",Nu+Nd)
        u = p[:Nu]
        odeargs = {"x":x, "u":u}
        if Nd > 0:
            d = p[Nu:]
            odeargs["d"] = d
        
        # Save sizes.
        self.Nx = Nx
        self.Nu = Nu
        self.Nd = Nd
        
        # Now define integrator for simulation.
        model = casadi.MXFunction(casadi.daeIn(x=x,p=p),
                                  casadi.daeOut(ode=casadi.vertcat(ode(**odeargs))))
        model.init()
        
        self.__Integrator = casadi.Integrator("cvodes",model)
        self.__Integrator.setOption("tf",Delta)
        self.__Integrator.init()

    def sim(self,x0,u,d=[]):
        """
        Simulate one timestep.
        """
        self.__Integrator.setInput(x0,"x0")
        self.__Integrator.setInput(casadi.vertcat([u,d]),"p")
        self.__Integrator.evaluate()
        xf = self.__Integrator.getOutput("xf")
        self.__Integrator.reset()
        
        return np.array(xf).flatten()
        
    
# =================================
# Plotting
# =================================
    
def mpcplot(x,u,t,xsp=None,fig=None,xinds=None,uinds=None,tightness=.5,title=None):
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
        zoomaxis(a,yscale=1.05)
    
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
        zoomaxis(a,yscale=1.05)
    
    # Layout tightness.
    if not tightness is None:
        fig.tight_layout(pad=tightness)
    if title is not None:
        fig.canvas.set_window_title(title)       
    
    return fig

def zoomaxis(axes=None,xscale=None,yscale=None):
    """
    Zooms the axes by a specified amounts (positive multipliers).
    
    If axes is None, plt.gca() is used.
    """
    # Grab default axes if necessary.
    if axes is None:
        axes = plt.gca()
    
    # Make sure input is valid.
    if (xscale is not None and xscale <= 0) or (yscale is not None and yscale <= 0):
        raise ValueError("Scale values must be strictly positive.")
    
    # Adjust axes limits.
    for (scale,getter,setter) in [(xscale,axes.get_xlim,axes.set_xlim), (yscale,axes.get_ylim,axes.set_ylim)]:
        if scale is not None:
            # Subtract one from each because of how we will calculate things.            
            scale -= 1
   
            # Get limits and change them.
            (minlim,maxlim) = getter()
            offset = .5*scale*(maxlim - minlim)
            setter(minlim - offset, maxlim + offset)

# =================================
# Helper Functions
# =================================

# First, we grab a few things from the CasADi module.
DMatrix = casadi.DMatrix


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

def c2d_augmented(A,B,Bp,f,Delta):
    """
    Discretizes affine system (A,B,Bp,f) with timestep Delta.
    
    This includes disturbances and a potentially nonzero steady-state.
    """
    
    n = A.shape[0]
    m = B.shape[1]
    mp = Bp.shape[1]
    M = m + mp + 1 # Extra 1 is for function column.
    
    D = scipy.linalg.expm(Delta*np.vstack((np.hstack([A, B, Bp, f]),
                                     np.zeros((M,M+n)))))
    Ad = np.matrix(D[0:n,0:n])
    Bd = np.matrix(D[0:n,n:n+m])
    Bpd = np.matrix(D[0:n,n+m:n+m+mp])
    fd = np.matrix(D[0:n,n+m+mp:n+m+mp+1])    
    
    return [Ad,Bd,Bpd,fd]
    
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

def flattenlist(l,depth=1):
    """
    Flattens a nested list of lists of the given depth.
    
    E.g. flattenlist([[1,2,3],[4,5],[6]]) returns [1,2,3,4,5,6]. Note that
    all sublists must have the same depth.
    """
    for i in range(depth):
        l = list(itertools.chain.from_iterable(l))
    return l

def casadiStruct2numpyDict(struct):
    """
    Takes a casadi struct and turns int into a dictionary of numpy arrays.
    
    Access patterns are now as follows:

        struct["var",t,...] = dict["var"][t,...]    
    """ 

    npdict = {}
    for k in struct.keys():
        npdict[k] = listcatfirstdim(struct[k])
        
    return npdict

def listcatfirstdim(l):
    """
    Takes a list of numpy arrays, prepends a dimension, and concatenates.
    """
    
    newl = []
    for a in l:
        a = np.array(a)
        if len(a.shape) == 2 and a.shape[1] == 1:
            a.shape = (a.shape[0],)
        a.shape = (1,) + a.shape
        newl.append(a)
    
    return np.concatenate(newl)    