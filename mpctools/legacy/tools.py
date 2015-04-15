from __future__ import print_function, division # Grab some Python3 stuff.
import numpy as np
import casadi
import casadi.tools as ctools
import time

# Grab other modules from this package.
from .. import colloc
from .. import util
from .. import tools as newtools
import solvers

"""
Contains the old versions of many functions to retain backwards-compatibility.
"""

def getCasadiFunc(f,Nx,Nu=0,Nd=0,name=None,vector=False):
    """
    Takes a function handle and turns it into a Casadi function.
    
    f should be defined to take three keyword arguments x, u, and d. It should
    be written so that x, u, and d are python LISTs of length Nx, Nu, and Nd
    respectively. It must return a LIST of length Nx.
    
    Alternatively, if f is written to accept numpy vectors and return a single
    numpy vector, then set vector=True.
    """
    
    # Create symbolic variables.
    [args, invar, _] = __getCasadiSymbols(Nx,Nu,Nd)    
    
    # Create symbolic function. If not vectorized, then need to call vertcat.
    if vector:
        outvar = [casadi.vertcat([f(**args)])]
    else:
        outvar = [casadi.vertcat(f(**args))]
    
    fcasadi = casadi.MXFunction(invar,outvar)
    fcasadi.setOption("name",name if name is not None else "f")
    fcasadi.init()
    
    return fcasadi


def getCasadiFuncGeneralArgs(f,varsizes,varnames=None,funcname="f",
                             scalar=False):
    """
    Takes a function handle and turns it into a Casadi function.
    
    f should be defined to take a specified number of arguments and return a
    LIST of outputs. varnames, if specified, gives names to each of the inputs,
    but this is unnecessary. sizes should be a list of how many elements are
    in each one of the inputs.
    
    This version is more general because it lets you specify arbitrary
    arguments, but you have to make sure you do everything properly.
    """
    return newtools.getCasadiFunc(f,varsizes,varnames,funcname,scalar)


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


# =================================
# Nonlinear
# =================================
    
# Nonlinear functions adapted from scripts courtesy of Lars Petersen.

def nmpc(F,l,x0,N,Pf=None,bounds={},d=None,verbosity=5,guess={},timemodel="discrete",
         M=None,Delta=None,returnTimeInvariantSolver=False,lDependsOnd=False,timelimit=60):
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
    nlpCon = util.flattenlist(nlpCon)
    conlb.shape = (conlb.size,)
    conub.shape = (conlb.size,)
    
    # Steps 0 to N-1.
    nlpObj = casadi.MX(0) # Start with dummy objective.    
    for k in range(N):
        # Model and objective.
        args = [VAR["x",k],VAR["u",k]]
        if lDependsOnd:
            args.append(PAR["d",k])
        nlpObj += l[k % len(l)](args)[0]        
        
        # Variable bounds.
        LB["x",k,:] = getBounds("xlb",k)
        UB["x",k,:] = getBounds("xub",k)
        LB["u",k,:] = getBounds("ulb",k)
        UB["u",k,:] = getBounds("uub",k)
    
    # Adjust bounds for x0 and xN.
    LB["x",0,:] = x0
    UB["x",0,:] = x0
    GUESS["x",0,:] = x0
    LB["x",N,:] = getBounds("xlb",N)
    UB["x",N,:] = getBounds("xub",N)
    if Pf is not None:
        args = [VAR["x",N]]
        if lDependsOnd:
            args.append(PAR["d",N-1]) # No Nth disturbance; just use N-1.
        nlpObj += Pf(args)[0]
    
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
        solver = solvers.TimeInvariantSolver(VAR,LB,UB,GUESS,nlpObj,nlpCon,
            conlb,conub,par=PAR,verbosity=verbosity,parval=PARVAL,
            timelimit=timelimit)    
        return solver
    
    # Call solver and stuff.
    ipoptstarttime = time.clock()
    [OPTVAR,obj,status,solver] = solvers.callSolver(VAR,LB,UB,GUESS,nlpObj,
        nlpCon,conlb,conub,par=PAR,verbosity=verbosity,parval=PARVAL,
        timelimit=timelimit)
    ipopttime = time.clock() - ipoptstarttime    
    
    x = np.hstack(OPTVAR["x",:,:])
    u = np.hstack(OPTVAR["u",:,:])
    
    # Add singleton dimension if Nx = 1 or Nu = 1.
    x = util.atleastnd(x)
    u = util.atleastnd(u)
    
    endtime = time.clock()
    if verbosity > 1:
        print("Took %g s." % (endtime - starttime))    
    
    optDict = {"x" : x, "u" : u, "status" : status, 
        "time" : endtime - starttime, "obj" : obj}
    optDict["ipopttime"] = ipopttime

    # Return collocation points if present.
    if Nc is not None:
        xc = np.hstack(OPTVAR["xc",:,:,:])
        optDict["xc"] = xc
    
    return optDict      


# Note that the MHE function uses a different paradigm than the MPC functions
# above. I think this method is better, so eventually the MPC functions should
# be rewritten. Basically, the MHE functions were written to be much more
# flexible with respect to function arguments, what variables are present,
# etc., and so it permits a lot more functionality in less code.

def nmhe(f,h,u,y,l,N,lx=None,x0bar=None,lb={},ub={},g=None,p=None,verbosity=5,
         guess={},largs=["w","v"],substitutev=False,timelimit=60,includeFinalPredictor=True):
    """
    Solves nonlinear MHE problem.
    
    N muste be a dictionary with at least entries "x", "y", and "t". "w" may be
    specified, but it is assumed to be equal to "x" if not given. "v" is always
    taken to be equal to "y". If parameters are present, you must also specify
    a "p" entry.
    
    u, y, and p must be 2D arrays with the time dimension first. Note that y
    should have N["t"] + 1 rows, and u and p should have N["t"] rows.
    
    lb and ub should be dictionaries of bounds for the various variables. Each
    entry should have time as the first index (i.e. lb["x"] should be a
    N["t"] + 1 by N["x"] array). guess should have the same structure.    
    
    The return value is a dictionary. Entry "x" is a N["t"] + 1 by N["x"]
    array that gives xhat(k | N["t"]-1) for k = 0,1,...,N["t"]. Thus, to find
    xhat(N["T"]-1 | N["t"]-1), you should get ["x"][-2,:], and to find the
    predition xhat(N["T"] | N["T"]-1), you should grab ["x"][-1,:].
    
    By default, the optimization includes a terminal predictor step. This is
    only relevant to the optimization if there are hard constraints on the
    state. To ignore it, you can set includeFinalPredictor to False. This means
    the final entry for "x" will be the the estimate xhat(N["T"]-1 | N["T"]-1).
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
    allShapes = newtools.__generalVariableShapes(N)
    
    # Build Casadi symbolic structures. These need to be separate because one
    # is passed as a set of variables and one is a set of parameters.
    parNames = set(["u","p","y"])
    parStruct = newtools.__casadiSymStruct(allShapes,parNames,scalar=False)(0)
        
    varNames = set(["x","z","w","v","xc","zc"])
    varStruct = newtools.__casadiSymStruct(allShapes,varNames,scalar=False)
    varlbStruct = varStruct(-np.inf)
    varubStruct = varStruct(np.inf)
    varguessStruct = varStruct(0)
    
    # Get rid of final x variable if we don't care about it.
    if not includeFinalPredictor and "x" in varStruct.keys():
        varlbStruct["x",-1] = varguessStruct["x",-1]
        varubStruct["x",-1] = varguessStruct["x",-1]
    
    dataAndStructure = [(guess,varguessStruct), (lb,varlbStruct),
                        (ub,varubStruct)]
    for (data,structure) in dataAndStructure:
        for v in data.keys():
            for t in range(N["t"]):
                structure[v,t] = data[v][t,...]

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
        g=g,Ng=Ng,h=h,Nh=N["y"],l=l,largs=largs,substitutev=substitutev)
    
    # Remove final state evolution constraint if not needed.
    if not includeFinalPredictor and "state" in constraints.keys():
        constraints["state"]["con"].pop(-1) # Get rid of last constraint.
        constraints["state"]["lb"] = constraints["state"]["lb"][:-1,...]
        constraints["state"]["ub"] = constraints["state"]["ub"][:-1,...]
    
    con = []
    conlb = np.zeros((0,))
    conub = conlb.copy()
    for f in ["state","measurement","algebra"]:
        if f in constraints.keys():
            con += util.flattenlist(constraints[f]["con"])
            conlb = np.concatenate([conlb,constraints[f]["lb"].flatten()])
            conub = np.concatenate([conub,constraints[f]["ub"].flatten()])
    con = casadi.vertcat(con)
    
    obj = casadi.MX(0)
    for t in range(N["t"]):
        obj += constraints["cost"][t]
    if lx is not None and x0bar is not None:
        obj += lx([varStruct["x",0] - x0bar])[0]
        
    # Now call the solver.
    [var, cost, status, solver] = solvers.callSolver(varStruct,varlbStruct,
        varubStruct,varguessStruct,obj,con,conlb,conub,verbosity=verbosity,
        timelimit=60)
    
    returnDict = util.casadiStruct2numpyDict(var)
    if not includeFinalPredictor and "x" in returnDict.keys():
        returnDict["x"] = returnDict["x"][:-1,...]
    returnDict["cost"] = cost
    returnDict["status"] = status    
                  
    return returnDict

           
def fillBoundsDict(bounds,Nx,Nu):
    """
    Fills a given bounds dictionary with any unspecified defaults.
    """
    # Check what bounds were supplied.
    defaultbounds = [("xlb",Nx,-np.Inf),("xub",Nx,np.Inf),
                     ("ulb",Nu,-np.Inf),("uub",Nu,np.Inf)]
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
        raise ValueError("Arguments of f are inconsistent with "
            "supplied arguments!")
    
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
            d  = casadi.MX.sym("d",Nd) # Other (parameters or disturbances)
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
        # Collocation points.
        args += (ctools.entry("xc",shape=(Nx,Nc),repeat=Nt),)
    if Nz is not None:
        # Algebraic variables.
        args += (ctools.entry("z",shape=(Nz,1),repeat=Nt+1),)
        if Nc is not None:
             # Collocation algebraic variables.
            args += (ctools.entry("zc",shape=(Nz,Nc),repeat=Nt),)
    
    VAR =  ctools.struct_symMX([args])
    LB = VAR(-np.inf)
    UB = VAR(np.inf)
    GUESS = VAR(0)
    
    return [VAR, LB, UB, GUESS]


def getCasadiVarsSizes(var,collocation=False,algebraic=False):
    """
    Checks entries of a struct_symMX for the proper fields and returns sizes.
    
    The return value is a dictionary. The only keys correspond to elements that
    are actually present in the variable struct.
    
    Raises a ValueError if something is missing.
    """
    # Check entries.
    needSizeKeys = ["x", "u"]
    needVarKeys = ["x", "u"]
    if collocation:
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
            raise ValueError("Entry %s missing from var! Consider using "
                "getCasadiVars for proper structure." % (k,))
            
    # Grab sizes for everything. Format is "sizename" : ("varname", dimension)
    sizes = {"x": ("x",0), "u": ("u",0), "c": ("xc",1), "z": ("z",0)}
    for k in sizes.keys():
        (v,i) = sizes.pop(k) # Get variable name and index.
        if k in needSizeKeys: # Save size if we need it.
            sizes[k] = var[v,0].shape[i]
            
    sizes["t"] = len(var["x"]) - 1 # Time is slightly different.
            
    return sizes


def getCollocationConstraints(f,var,Delta,d=None):
    """
    Returns constraints for collocation of ODE f in variables x and u.
    
    var should be a casadi struct_symMX object, e.g. the output of
    getCasadiVars.

    The sizes of variables and the number of time points are inferred from f
    and var. Make sure that the model f and variables var have consistent
    sizes.    
    
    If provided, d is a list of disturbances at each time point. It is used
    mod length, so periodic disturbances with period < Nt can be used.
    
    Returns a a list of all the constraints and bounds for the constraints.
    """
    # Get shapes.
    shapes = getVarShapes(f,var,disturbancemodel=(d is not None))
    [Nx, Nu, Nd, Nc, Nt] = [shapes[k] for k in ["x","u","d","c","t"]]
        
    # Get collocation weights.
    [r,A,B,q] = colloc.weights(Nc, True, True)
    
    # Preallocate. CON will be a list of lists.
    CON = []
    
    for k in range(Nt):
        # Decide about disturbances.        
        if d is not None:
            thisd = [d[k % len(d)]]
        else:
            thisd = []
        
        # Build a convenience list.
        xaug = ([var["x",k]] + [var["xc",k,:,j] for j in range(Nc)]
            + [var["x",k+1]])
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
    and var. Make sure that the model f and variables var have consistent
    sizes.    
    
    If provided, d is a list of disturbances at each time point. It is used
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

def __generalDiscreteConstraints(var,Nt,f=None,Nf=0,g=None,Ng=0,h=None,Nh=0,
                                 t0=0,l=None,largs=[],substitutev=False):
    """
    Creates general state evolution constraints for the following system:
    
       x^+ = f(x,z,u,d,w,p)                      \n
       g(x,z,p) = 0                              \n
       y = h(x,z,p) + v                          \n
       
    The variables are intended as follows:
    
        x: differential states                  \n
        z: algebraic states                     \n
        u: control actions                      \n
        d: modeled state disturbances           \n
        w: unmodeled state disturbances         \n
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
    
    If substitutev is set to True, then the measurement constraints will be
    directly inserted into the stage cost so that v is no longer a decision
    variable. This reduces problem size but makes the objective function more
    nonlinear.
    """
    # Figure out what variables are supplied.
    allvars = set(["x","z","u","d","w","p","y","v"])
    givenvars = allvars.intersection(var.keys())
    
    # Now sort out arguments for each function.
    isGiven = lambda v: v in givenvars # Membership function.
    args = {
        "f" : filter(isGiven,["x","z","u","d","w","p"]),
        "g" : filter(isGiven, ["x","z","p"]),
        "h" : filter(isGiven, ["x","z","p"]),
        "l" : filter(isGiven, largs)
    }
    def getArgs(func, times):
        return [[var[v][t] for v in args[func]] for t in times]
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
                if not substitutev:
                    thiscon += var["v"][t]
                else:
                    thiscon *= -1 # Need to flip sign to get v correct.
            measurement.append([thiscon])
        lb = np.zeros((Nt,Nh))
        ub = lb.copy()
        returnDict["measurement"] = dict(con=measurement,lb=lb,ub=ub)
    
    # Stage costs.
    if l is not None:
        # Have to be careful about the arguments for this one because the
        # function may depend on v, but there is no explicit variable v.
        if substitutev:
            largs = []
            for t in tintervals:
                largs.append([])
                for v in args["l"]:
                    if v == "v": # Grab expression that defines v.
                        largs[-1] += measurement[t-t0]
                    else:
                        largs[-1].append(var[v][t])
        else:        
            largs = getArgs("l",tintervals)
        
        cost = []
        for t in tintervals:
            cost.append(l[t % len(l)](largs[t])[0])
        returnDict["cost"] = cost
    
    # If we substituted out v, then we don't need to include the measurement
    # constrants, so just get rid of them.
    if substitutev:
        returnDict.pop("measurement")
    
    return returnDict


def getRungeKutta4(f,Delta,M=1,d=False,name=None,argsizes=None):
    """
    Uses RK4 to discretize xdot = f(x,u) with M points and timestep Delta.

    A disturbance argument can be specified by setting d=True. If present, d
    must come last in the list of arguments (i.e. f(x,u,d)). Alternatively,
    you may specify a list argsizes for a function with an arbitrary number
    of inputs. If the model has, e.g. 3 states, 2 control inputs, 1
    disturbance, and 4 parameters, you would use argsizes=[3,2,1,4].

    f must be a Casadi SX function with inputs in the proper order.
    """
    
    if argsizes is None:
        # First find out how many arguments f takes.
        fargs = getModelArgSizes(f,d=d)
        [Nx, Nu, Nd] = [fargs[a] for a in ["x","u","d"]]
    
        # Get symbolic variables.
        [args, _, zoh] = __getCasadiSymbols(Nx,Nu,Nd)
        x0 = args["x"]
    else:
        x0 = casadi.MX.sym("x",argsizes[0])
        zoh = [casadi.MX.sym("v_%d",i) for i in argsizes[1:]]
    
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


class OneStepSimulator(object):
    """
    Simulates a continuous-time system.
    """
    def __init__(self,ode,Delta,Nx,Nu,Nd=0,Nw=0,vector=False):
        """
        Initialize by specifying model and sizes of everything.
        """
        # Create variables.
        x = casadi.SX.sym("x",Nx)
        p = casadi.SX.sym("p",Nu+Nd+Nw)
        u = p[:Nu]
        odeargs = {"x":x}
        if Nu > 0:
            odeargs["u"] = u
        if Nd > 0:
            d = p[Nu:Nu+Nd]
            odeargs["d"] = d
        if Nw > 0:
            w = p[Nu+Nd:]
            odeargs["w"] = w
        
        # Save sizes. Really we should make these all properties because
        # changing them doesn't do anything.
        self.Nx = Nx
        self.Nu = Nu
        self.Nd = Nd
        self.Nw = Nw
        self.Delta = Delta
        
        # Decide how to call ode.
        if vector:
            f = ode(**odeargs)
        else:
            f = casadi.vertcat(ode(**odeargs))    
        
        # Now define integrator for simulation.
        model = casadi.SXFunction(casadi.daeIn(x=x,p=p),casadi.daeOut(ode=f))
        model.init()
        
        self.__Integrator = casadi.Integrator("cvodes",model)
        self.__Integrator.setOption("tf",Delta)
        self.__Integrator.init()

    def sim(self,x0,u=[],d=[],w=[]):
        """
        Simulate one timestep.
        """
        self.__Integrator.setInput(x0,"x0")
        self.__Integrator.setInput(casadi.vertcat([u,d,w]),"p")
        self.__Integrator.evaluate()
        xf = self.__Integrator.getOutput("xf")
        self.__Integrator.reset()
        
        return np.array(xf).flatten()