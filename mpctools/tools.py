from __future__ import print_function, division # Grab some handy Python3 stuff.
import numpy as np
import casadi
import casadi.tools as ctools
import scipy.linalg
import colloc
import warnings

# Other things from our package.
import util
import solvers

"""
Functions for solving MPC problems using Casadi and Ipopt.
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

# =================================
# MPC and MHE
# =================================

# These functions are rewrites of the legacy MPC and MHE versions to use a more
# common framework. They appear to be strictly better than the functions they
# replace.

def nmpc(f,l,N,x0,lb={},ub={},guess={},g=None,Pf=None,largs=None,sp={},p=None,
    uprev=None,verbosity=5,timelimit=60,Delta=1,runOptimization=True,
    scalar=True):
    """
    Solves nonlinear MPC problem.
    
    N muste be a dictionary with at least entries "x", "u", and "t". If 
    parameters are present, you must also specify a "p" entry, and if algebraic
    states are present, you must provide a "z" entry.
    
    If provided, p must be a 2D array with the time dimension first. It should
    have N["t"] rows and N["p"] columns.
    
    lb and ub should be dictionaries of bounds for the various variables. Each
    entry should have time as the first index (i.e. lb["x"] should be a
    N["t"] + 1 by N["x"] array). guess should have the same structure.    
    
    sp is a dictionary that holds setpoints for x and u. If supplied, the stage
    cost is assumed to be a function l(x,u,x_sp,u_sp). If not supplied, l is
    l(x,u). To explicitly specify a different order of arguments or something
    else, e.g. a dependence on parameters, specify a list of input variables.
    Similarly, Pf is assumed to be Pf(x,x_sp) if a setpoint for x is supplied,
    and it is left as Pf(x) otherwise.    
    
    To include rate of change penalties or constraints, set uprev so a vector
    with the previous u entry. Bound constraints can then be entered with the
    key "Du" in the lb and ub structs. "Du" can also be specified as in largs.    
    
    The return value is a dictionary with the values of the optimal decision
    variables and also some time vectors.
    """     
    # Copy dictionaries so we don't change the user inputs.
    N = N.copy()
    guess = guess.copy()     
   
    # Check specified sizes.
    try:
        for i in ["t","x"]:
            if N[i] <= 0:
                N[i] = 1
    except KeyError:
        raise KeyError("Invalid or missing entries in N dictionary!")
    
    # Make sure these elements aren't present.
    for i in ["y","v"]:
        N.pop(i,None)
    
    # Now get the shapes of all the variables that are present.
    deltaVars = ["u"] if uprev is not None else []
    allShapes = __generalVariableShapes(N,setpoint=sp.keys(),delta=deltaVars)
    if "c" not in N:
        N["c"] = 0    
    
    # Sort out bounds on x0.
    for (d,v) in [(lb,-np.inf), (ub,np.inf), (guess,0)]:
        if "x" not in d.keys():
            d["x"] = v*np.ones((N["t"]+1,N["x"]))
    lb["x"][0,...] = x0
    ub["x"][0,...] = x0
    guess["x"][0,...] = x0
    
    # Build Casadi symbolic structures. These need to be separate because one
    # is passed as a set of variables and one is a set of parameters. Note that
    # if this ends up empty, we just set it to None.
    parNames = set(["p"] + [k + "_sp" for k in sp.keys()]
        + [k + "_prev" for k in deltaVars])
    parStruct = __casadiSymStruct(allShapes,parNames,scalar=scalar)
    if len(parStruct.keys()) == 0:
        parStruct = None
        
    varNames = set(["x","u","xc","zc"] + ["D" + k for k in deltaVars])
    varStruct = __casadiSymStruct(allShapes,varNames,scalar=scalar)

    # Add parameters and setpoints to the guess structure.
    guess["p"] = p
    for v in sp.keys():
        guess[v + "_sp"] = sp[v]
    if uprev is not None:
        # Need uprev to have shape (1, N["u"]).
        uprev = np.array(uprev).flatten()
        uprev.shape = (1,uprev.size)
        guess["u_prev"] = uprev
    
    # Need to decide about algebraic constraints.
    if "z" in allShapes.keys():
        N["g"] = N["z"]
    else:
        N["g"] = None
    N["f"] = N["x"]
    
    # Make initial objective term.    
    if Pf is not None:
        if "x" in sp.keys():
            obj = Pf([varStruct["x",-1],parStruct["x_sp",-1]])[0]
        else:
            obj = Pf([varStruct["x",-1]])[0]
    else:
        obj = casadi.SX(0)
    
    # Decide arguments of l.
    if largs is None:
        largs = ["x","u"]
        if "x" in sp.keys():
            largs.append("x_sp")
        if "u" in sp.keys():
            largs.append("u_sp")
    
    return __optimalControlProblem(N,varStruct,parStruct,lb,ub,guess,obj,
         f=f,g=g,h=None,l=l,largs=largs,Delta=Delta,verbosity=verbosity,
         runOptimization=runOptimization,deltaVars=deltaVars,
         scalar=scalar)


def nmhe(f,h,u,y,l,N,lx=None,x0bar=None,lb={},ub={},guess={},g=None,p=None,
    verbosity=5,largs=["w","v"],timelimit=60,Delta=1,runOptimization=True,
    scalar=True):
    """
    Solves nonlinear MHE problem.
    
    N muste be a dictionary with at least entries "x", "y", and "t". "w" may be
    specified, but it is assumed to be equal to "x" if not given. "v" is always
    taken to be equal to "y". If parameters are present, you must also specify
    a "p" entry.
    
    u, y, and p must be 2D arrays with the time dimension first. Note that y
    and p should have N["t"] + 1 rows, while u should have N["t"] rows.
    
    lb and ub should be dictionaries of bounds for the various variables. Each
    entry should have time as the first index (i.e. lb["x"] should be a
    N["t"] + 1 by N["x"] array). guess should have the same structure.    
    
    The return value is a dictionary. Entry "x" is a N["t"] + 1 by N["x"]
    array that gives xhat(k | N["t"]) for k = 0,1,...,N["t"]. There is no final
    predictor step.
    """
    
    # Copy dictionaries so we don't change the user inputs.
    N = N.copy()
    guess = guess.copy()    
    
    # Check specified sizes.
    try:
        for i in ["x","y"]:
            if N[i] <= 0:
                N[i] = 1
            if N["t"] < 0:
                N["t"] = 0
        if "w" not in N:
            N["w"] = N["x"]
        N["v"] = N["y"] 
    except KeyError:
        raise KeyError("Invalid or missing entries in N dictionary!")
        
    # Now get the shapes of all the variables that are present.
    allShapes = __generalVariableShapes(N,finalx=True,finaly=True)
    if "c" not in N:
        N["c"] = 0    
    
    # Build Casadi symbolic structures. These need to be separate because one
    # is passed as a set of variables and one is a set of parameters.
    parNames = set(["u","p","y"])
    parStruct = __casadiSymStruct(allShapes,parNames,scalar=scalar)
        
    varNames = set(["x","z","w","v","xc","zc"])
    varStruct = __casadiSymStruct(allShapes,varNames,scalar=scalar)

    # Now we fill up the parameters in the guess structure.
    for (name,val) in [("u",u),("p",p),("y",y)]:
        guess[name] = val
    
    # Need to decide about algebraic constraints.
    if "z" in allShapes.keys():
        N["g"] = N["z"]
    else:
        N["g"] = None
    N["h"] = N["y"]
    N["f"] = N["x"]
    
    # Make initial objective term.
    finallargs = []    
    for k in largs:
        if k == "w":
            finallargs.append(np.zeros(N["w"]))
        elif k in parStruct.keys():
            finallargs.append(parStruct[k,-1])
        elif k in varStruct.keys():
            finallargs.append(varStruct[k,-1])
        else:
            raise KeyError("l argument %s is invalid!" % (k,))
    obj = l(finallargs)[0]    
    if lx is not None and x0bar is not None:
        obj += lx([varStruct["x",0] - x0bar])[0]
    
    return __optimalControlProblem(N,varStruct,parStruct,lb,ub,guess,obj,
         f=f,g=g,h=h,l=l,largs=largs,Delta=Delta,verbosity=verbosity,
         runOptimization=runOptimization,scalar=scalar)


def sstarg(f,h,N,phi=None,phiargs=None,lb={},ub={},guess={},g=None,p=None,
    discretef=True,verbosity=5,timelimit=60,runOptimization=True,scalar=True):
    """
    Solves nonlinear steady-state target problem.
    
    N muste be a dictionary with at least entries "x" and "y". If parameters
    are present, you must also specify a "p" entry.
    
    lb and ub should be dictionaries of bounds for the various variables.
    guess should have the same structure.    
    """
    
    # Copy dictionaries so we don't change the user inputs.
    N = N.copy()
    guess = guess.copy()    
    
    # Check specified sizes.
    try:
        for i in ["x","y"]:
            if N[i] <= 0:
                N[i] = 1
    except KeyError:
        raise KeyError("Invalid or missing entries in N dictionary!")
    
    # Now get the shapes of all the variables that are present.
    N["t"] = 1
    allShapes = __generalVariableShapes(N,finalx=False,finaly=False)
    if "c" not in N:
        N["c"] = 0
    
    # Build Casadi symbolic structures. These need to be separate because one
    # is passed as a set of variables and one is a set of parameters.
    parNames = set(["p"])
    parStruct = __casadiSymStruct(allShapes,parNames,scalar=scalar)
        
    varNames = set(["x","z","u","y"])
    varStruct = __casadiSymStruct(allShapes,varNames,scalar=scalar)

    # Now we fill up the parameters in the guess structure.
    guess["p"] = p
    
    # Need to decide about algebraic constraints.
    if "z" in allShapes.keys():
        N["g"] = N["z"]
    else:
        N["g"] = None
    N["h"] = N["y"]
    N["f"] = N["x"]
    
    # Make objective term.    
    if phi is not None and phiargs is not None:
        args = []
        for v in phiargs:
            if v in varStruct.keys():
                args.append(varStruct[v,0])
            elif v in parStruct.keys():
                args.append(parStruct[v,0])
            else:
                raise ValueError("Argumet %s is invalid! Must be 'x', 'u', "
                    "'y', or 'p'!" % (v,))
        obj = phi(args)[0]
    elif scalar:
        obj = casadi.SX(0)
    else:
        obj = casadi.MX(0)
       
    return __optimalControlProblem(N,varStruct,parStruct,lb,ub,guess,obj,
         f=f,g=g,h=h,verbosity=verbosity,discretef=discretef,finalpoint=False,
         runOptimization=runOptimization,scalar=scalar)


def __optimalControlProblem(N,var,par=None,lb={},ub={},guess={},
    obj=None,f=None,g=None,h=None,l=None,largs=[],Delta=1,
    discretef=True,deltaVars=[],finalpoint=True,verbosity=5,
    runOptimization=True,scalar=True):
    """
    General wrapper for an optimal control problem (e.g. mpc or mhe).
    
    struct should be a dictionary of the appropriate parameter or variable
    casadi sym_structs. It should have an entry for each variable/parameter
    that is a list running through time.
    
    Note that only variable fields are taken from lb and ub, but parameter
    values must be specified in guess
    """

    # Initialize things.
    varlb = var(-np.inf)
    varub = var(np.inf)
    varguess = var(0)
    dataAndStructure = [(guess,varguess,"guess"), (lb,varlb,"lb"), (ub,varub,"ub")]
    if par is not None:
        parval = par(0)
        dataAndStructure.append((guess,parval,"par"))
    else:
        parval = None

    # Sort out bounds and parameters.
    for (data,structure,name) in dataAndStructure:
        for v in set(data.keys()).intersection(structure.keys()):
            # Check sizes.            
            if len(structure[v]) < data[v].shape[0]:
                warnings.warn("Extra time points in %s['%s']. Ignoring." % (name,v))
            elif len(structure[v]) > data[v].shape[0]:
                raise IndexError("Too few time points in %s['%s']!" % (name,v))
            
            # Grab data.            
            for t in range(len(structure[v])):
                structure[v,t] = data[v][t,...]
    
    # Smush together variables and parameters to get the constraints.
    struct = {}
    for k in var.keys():
        struct[k] = var[k]
    if par is not None:
        for k in par.keys():
            struct[k] = par[k]
    
    # Double-check some sizes and then get constraints.
    for (func,name) in [(f,"f"), (g,"g"), (h,"h")]:
        if func is None:
            N[name] = 0    
    constraints = __generalConstraints(struct,N["t"],f=f,Nf=N["f"],
        g=g,Ng=N["g"],h=h,Nh=N["h"],l=l,largs=largs,Ncolloc=N["c"],
        Delta=Delta,discretef=discretef,deltaVars=deltaVars,
        finalpoint=finalpoint)
     
    con = []
    conlb = np.zeros((0,))
    conub = conlb.copy()
    for f in ["state","measurement","algebra","delta"]:
        if f in constraints.keys():
            con += util.flattenlist(constraints[f]["con"])
            conlb = np.concatenate([conlb,constraints[f]["lb"].flatten()])
            conub = np.concatenate([conub,constraints[f]["ub"].flatten()])
    con = casadi.vertcat(con)
    
    if obj is None:
        if scalar:
            obj = casadi.SX(0)
        else:
            obj = casadi.MX(0)
    if "cost" in constraints.keys():
        for t in range(N["t"]):
            obj += constraints["cost"][t]
        
    # If we want an optimization, then do some post-processing. Otherwise, just
    # return the solver object.
    if runOptimization:
        # Call the solver.
        [var, cost, status, solver] = solvers.callSolver(var,varlb,varub,
            varguess,obj,con,conlb,conub,par,parval,verbosity=verbosity,
            timelimit=60,scalar=scalar)
        returnDict = util.casadiStruct2numpyDict(var)
        returnDict["cost"] = cost
        returnDict["status"] = status    
        
        # Create an array of time points.
        returnDict["t"] = N["t"]*Delta*np.linspace(0,1,N["t"] + 1)
        if N["c"] > 0:
            r = constraints["colloc.weights"]["r"][1:-1] # Throw out endpoints.        
            r.shape = (1,r.size)
            returnDict["tc"] = returnDict["t"].reshape(
                (returnDict["t"].size,1))[:-1] + Delta*r
    else:
        # Build ControlSolver object and return that.
        returnDict = solvers.ControlSolver(var,varlb,varub,varguess,
            obj,con,conlb,conub,par,parval,verbosity=verbosity,timelimit=60,
            scalar=scalar)
    
    return returnDict


def __generalConstraints(var,Nt,f=None,Nf=0,g=None,Ng=0,h=None,Nh=0,
    l=None,largs=[],Ncolloc=0,Delta=1,discretef=True,deltaVars=[],
    finalpoint=True):
    """
    Creates general state evolution constraints for the following system:
    
       x^+ = f(x,z,u,w,p)                      \n
       g(x,z,w,p) = 0                          \n
       y = h(x,z,p) + v                        \n
       
    The variables are intended as follows:
    
        x: differential states                  \n
        z: algebraic states                     \n
        u: control actions                      \n
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

        var["x"][t][k,...] gives the kth state of x at time t
        
    In particular, if struct is a casadi.tools.struct_symMX object, then
    setting var["x"] = struct["x"] will suffice. If you want to use your own
    data structure, so be it.

    If Ncolloc is not None, var must also have entries "xc" and "zc". Each
    entry must have Ncolloc as the size of the second dimension.
    
    deltaVars should be a list of variables to make constraints for time
    differences. For each entry in this list, var must have the appropriate
    keys, e.g. if deltaVars = ["u"], then var must have "u", "Du", and "u_prev"
    entries or else this will error.    
    
    Returns a dictionary with entries "state", "algebra", and "measurement".
    Note that the relevant fields will be missing if f, g, or h are set to
    None. Each entry in the return dictionary will be a list of lists, although
    for this class of constraint, each of those lists will have only one
    element. The list of stage costs is in "cost". This one is just a list.
    """
    
    # Figure out what variables are supplied.
    givenvars = set(var.keys())    
    givenvarscolloc = givenvars.intersection(["x","z"])    
    
    # Now sort out arguments for each function.
    isGiven = lambda v: v in givenvars # Membership function.
    args = {
        "f" : filter(isGiven, ["x","z","u","w","p"]),
        "g" : filter(isGiven, ["x","z","w","p"]),
        "h" : filter(isGiven, ["x","z","p"]),
        "l" : filter(isGiven, largs)
    }
    # Also define inverse map to get positions of arguments.
    argsInv = {}
    for a in args.keys():
        argsInv[a] = dict([(args[a][j], j) for j in range(len(args[a]))])
    
    # Define some helper variables.    
    def getArgs(func,times,var): 
        return [[var[v][t] for v in args[func]] for t in times]
    tintervals = range(0,Nt)
    tpoints = range(0,Nt + bool(finalpoint))
    
    # Preallocate return dictionary.
    returnDict = {}    
    
    # Decide whether we got the correct stuff for collocation.
    if Ncolloc < 0 or round(Ncolloc) != Ncolloc:
        raise ValueError("Ncolloc must be a nonnegative integer if given.")
    if Ncolloc > 0:
        [r,A,B,q] = colloc.weights(Ncolloc, True, True) # Collocation weights.
        returnDict["colloc.weights"] = {"r":r, "A":A, "B":B, "q":q}
        collocvar = {}
        for v in givenvarscolloc:
            # Make sure we were given the corresponding "c" variables.            
            if v + "c" not in givenvars:
                raise KeyError("Entry %sc not found in vars!" % (v,))
            collocvar[v] = []
            for k in range(Nt):
                collocvar[v].append([var[v][k]]
                    + [var[v+"c"][k][:,j] for j in range(Ncolloc)]
                    + [var[v][k+1 % len(var[v])]])
    
        def getCollocArgs(k,t,j):
            """
            Gets arguments for function k at time t and collocation point j.
            """
            thisargs = []
            for a in args[k]:
                if a in givenvarscolloc:
                    thisargs.append(collocvar[a][t][j])
                else:
                    thisargs.append(var[a][t])        
            return thisargs
    
    # State evolution f.   
    if f is not None:
        if Nf <= 0:
            raise ValueError("Nf must be a positive integer!")
        fargs = getArgs("f",tintervals,var)
        state = []
        for t in tintervals:
            if Ncolloc == 0:
                # Just use discrete-time equations.
                thiscon = f(fargs[t])[0]
                if "x" in givenvars and discretef:
                    thiscon -= var["x"][t+1 % len(var["x"])]
                thesecons = [thiscon] # Only one constraint per timestep.
            else:
                # Need to do collocation stuff.
                thesecons = []
                for j in range(1,Ncolloc+2):
                    thisargs = getCollocArgs("f",t,j)
                    # Start with function evaluation.
                    thiscon = Delta*f(thisargs)[0]
                    
                    # Add collocation weights.
                    if "x" in givenvarscolloc:
                        for jprime in range(len(collocvar["x"][t])):
                            thiscon -= A[j,jprime]*collocvar["x"][t][jprime]
                    thesecons.append(thiscon)
            state.append(thesecons)
        lb = np.zeros((len(tintervals),Ncolloc+1,Nf))
        ub = lb.copy()
        returnDict["state"] = dict(con=state,lb=lb,ub=ub)
            
    # Algebraic constraints g.
    if g is not None:
        if Ng <= 0:
            raise ValueError("Ng must be a positive integer!")
        gargs = getArgs("g",tpoints,var)
        algebra = []
        for t in tpoints:
            if Ncolloc == 0 or t == Nt:
                thesecons = [g(gargs[t])[0]]
            else:
                thesecons = []
                for j in range(Ncolloc+1):
                    thisargs = getCollocArgs("g",t,j)
                    thiscon = g(thisargs)[0]
                    thesecons.append(thiscon)
            algebra.append(thesecons)
        lb = np.zeros((len(tpoints),Ncolloc+1,Ng))
        ub = lb.copy()
        returnDict["algebra"] = dict(con=algebra,lb=lb,ub=ub)
        
    # Measurements h.
    if h is not None:
        if Nh <= 0:
            raise ValueError("Nh must be a positive integer!")
        hargs = getArgs("h",tpoints,var)
        measurement = []
        for t in tpoints:
            thiscon = h(hargs[t])[0]
            if "y" in givenvars:
                thiscon -= var["y"][t]
            if "v" in givenvars:
                thiscon += var["v"][t]
            measurement.append([thiscon])
        lb = np.zeros((len(tpoints),Nh))
        ub = lb.copy()
        returnDict["measurement"] = dict(con=measurement,lb=lb,ub=ub)
    
    # Delta variable constraints.
    if len(deltaVars) > 0:
        deltaconstraints = []
        numentries = 0
        for v in deltaVars:
            if not set([v,"D"+v, v+"_prev"]).issubset(var.keys()):
                raise KeyError("Variable '%s' must also have entries 'D%s' "
                    "and '%s_prev'!" % (v,v,v))
            thisdelta = [var["D" + v][0] - var[v][0] + var[v + "_prev"][0]]
            for t in range(1,len(var[v])):
                thisdelta.append(var["D" + v][t] - var[v][t] + var[v][t-1])
            deltaconstraints.append(thisdelta)
            numentries += len(var[v])*np.product(var[v][0].shape)
        lb = np.zeros((numentries,))
        ub = lb.copy()
        returnDict["delta"] = dict(con=deltaconstraints,lb=lb,ub=ub)
          
    # Stage costs.
    if l is not None:
        largs = getArgs("l",tintervals,var)
        
        cost = []
        for t in tintervals:
            cost.append(l(largs[t])[0])
        returnDict["cost"] = cost
    
    return returnDict


# =====================================
# Building CasADi Functions and Objects
# =====================================

def __generalVariableShapes(sizeDict,setpoint=[],delta=[],finalx=True,
                            finaly=False):
    """
    Generates variable shapes from the size dictionary N.
    
    The keys of N must be a subset of
        ["x","z","u","d","w","p","y","v","c"]    
    
    If present, "c" will specify collocation, at which point extra variables
    will be created.
    
    Each entry in the returned dictionary will be a dictionary of keyword
    arguments repeat and shape to pass to __casadiSymStruct.
    
    Optional argument setpiont is a list of variables that have corresponding
    setpoint parameters. Any variable names in this list will have a
    corresponding entry suffixed with _sp. This is useful, e.g., for control
    problems where you may change the system setpoint.
    
    Optional argument delta is similar, but it defines decision variables to
    calculate the difference between successive time points. This is useful for
    rate-of-change penalties or constraints for control problems.
    
    If N["t"] is 0, then each variable will only have one entry. This is useful
    for steady-state target problems where you only want one x and z variable.
    """
    # Figure out what variables are supplied.
    allsizes = set(["x","z","u","w","p","y","v","c","t"])
    givensizes = allsizes.intersection(sizeDict.keys())
    
    # Make sure we were given a time.
    try:
        Nt = sizeDict["t"]
    except KeyError:
        raise KeyError("Entry 't' must be provided!")
    
    # Need to decide whether to include final point of various entries.
    finalx = bool(finalx)
    finaly = bool(finaly)
    
    # Now we're going to build a data structure that says how big each of the
    # variables should be. The first entry 1 if there should be N+1 copies of
    # the variable and 0 if there should be N. The second is a tuple of sizes.
    allvars = {
        "x" : (Nt+finalx,("x",)),
        "z" : (Nt+finalx,("z",)),
        "u" : (Nt,("u",)),
        "w" : (Nt,("w",)),
        "p" : (Nt+finaly,("p",)),
        "y" : (Nt+finaly,("y",)),
        "v" : (Nt+finaly,("v",)),
        "xc": (Nt,("x","c")),
        "zc": (Nt,("z","c")),        
    }
    # Here we define any "extra" variables like setpoints. The syntax is a
    # tuple with the prefix string, the list of variables, the suffix string,
    # and the number of entries (None to use the default value).
    extraEntries = [
        ("", setpoint, "_sp", None), # These are parameters for sepoints.
        ("D", delta, "", None), # These are variables to calculate deltas.
        ("", delta, "_prev", 1), # This is a parameter for the previous value.    
    ]
    for (prefix, var, suffix, num) in extraEntries:
        for v in set(var).intersection(allvars.keys()):
            thisnum = num if num is not None else allvars[v][0]
            allvars[prefix + v + suffix] = (thisnum, allvars[v][1])
        
    # Now loop through all the variables, and if we've been given all of the
    # necessary sizes, then add that variable
    shapeDict = {}
    for (v, (t, shapeinds)) in allvars.items():
        if givensizes.issuperset(shapeinds):
            shape = [sizeDict[i] for i in shapeinds]
            if len(shape) == 0:
                shape = [1,1]
            elif len(shape) == 1:
                shape += [1]
            shapeDict[v] = {"repeat" : t, "shape" : tuple(shape)}
    
    return shapeDict


def __casadiSymStruct(allVars,theseVars=None,scalar=True):
    """
    Returns a Casadi sym struct for the variables in allVars.
    
    To use only a subset of variables, set theseVars to the subset of variables
    that you need. If theseVars is not None, then only variable names appearing
    in allVars and theseVars will be created.
    """
    # Figure out what names we need.
    allVars = allVars.copy()
    varNames = set(allVars.keys())
    if theseVars is not None:
        for v in varNames.difference(theseVars):
            allVars.pop(v)
    
    # Build casadi sym_structX    
    structArgs = tuple([ctools.entry(name,**args) for (name,args)
        in allVars.items()])
    if scalar:
        struct = ctools.struct_symSX
    else:
        struct = ctools.struct_symMX
    return struct([structArgs])


def getCasadiFunc(f,varsizes,varnames=None,funcname="f",scalar=True):
    """
    Takes a function handle and turns it into a Casadi function.
    
    f should be defined to take a specified number of arguments and return a
    LIST of outputs. varnames, if specified, gives names to each of the inputs,
    but this is unnecessary. sizes should be a list of how many elements are
    in each one of the inputs.
    
    This version is more general because it lets you specify arbitrary
    arguments, but you have to make sure you do everything properly.
    """
    # Decide whether to use SX or MX.    
    if scalar:
        XSym = casadi.SX.sym
        XFunction = casadi.SXFunction
    else:
        XSym = casadi.MX.sym
        XFunction = casadi.MXFunction
    
    # Create symbolic variables.
    if varnames is None:
        varnames = ["x%d" % (i,) for i in range(len(varsizes))]
    elif len(varsizes) != len(varnames):
        raise ValueError("varnames must be the same length as varsizes!")    
    args = [XSym(name,size) for (name,size) in zip(varnames,varsizes)]
    
    # Now evaluate function and make a Casadi object.    
    if scalar:
        fval = [f(*args)]
    else:
        fval = [casadi.vertcat(f(*args))]
    fcasadi = XFunction(args,fval)
    fcasadi.setOption("name",funcname)
    fcasadi.init()
    
    return fcasadi

    
# =============
# Miscellany
# ============

def getCasadiIntegrator(f,Delta,argsizes,argnames=None,funcname="int_f",
                        abstol=1e-8,reltol=1e-8,wrap=True):
    """
    Gets a Casadi integrator for function f from 0 to Delta.
    
    Argsizes should be a list with the number of elements for each input. Note
    that the first argument is assumed to be the differential variables, and
    all others are kept constant.
    
    wrap can be set to False to return the raw casadi Integrator object, i.e.,
    with inputs x and p instead of the arguments specified by the user.
    """
    if len(argsizes) < 1:
        raise IndexError("argsizes must have at least one element!")
    if argnames is None:
        argnames = ["x_%d" for i in range(len(argsizes))]
    
    # Create symbolic variables for integrator I(x0,p).
    x0 = casadi.SX.sym(argnames[0],argsizes[0])
    par = [casadi.SX.sym(argnames[i],argsizes[i]) for i
        in range(1,len(argsizes))]
    fode = f(x0,*par)   
    
    # Build ODE and integrator.
    invar = casadi.daeIn(x=x0,p=casadi.vertcat(par))
    outvar = casadi.daeOut(ode=fode)
    ode = casadi.SXFunction(invar,outvar)
    
    integrator = casadi.Integrator("cvodes",ode)
    integrator.setOption("abstol",abstol)
    integrator.setOption("reltol",reltol)
    integrator.setOption("tf",Delta)
    integrator.setOption("name",funcname)
    integrator.init()
    
    # Now do the subtle bit. Integrator has arguments x0 and p, but we need
    # arguments as given by the user. First we need MX arguments.
    if wrap:
        X0 = casadi.MX.sym(argnames[0],argsizes[0])
        PAR = [casadi.MX.sym(argnames[i],argsizes[i]) for i
            in range(1,len(argsizes))]    
        
        wrappedIntegrator = integrator(x0=X0,p=casadi.vertcat(PAR))
        F = casadi.MXFunction([X0] + PAR,wrappedIntegrator)
        F.setOption("name",funcname)
        F.init()
        
        return F
    else:
        return integrator

class DiscreteSimulator(object):
    """
    Simulates one timestep of a continuous-time system.
    """
    
    @property
    def Delta(self):
        return self.__Delta
        
    @property
    def Nargs(self):
        return self.__Nargs
        
    @property
    def args(self):
        return self.__argnames
        
    def __init__(self,ode,Delta,argsizes,argnames=None):
        """
        Initialize by specifying model and sizes of everything.
        """
        # Make sure there is at least one argument.
        if len(argsizes) == 0:
            raise ValueError("Model must have at least 1 argument.")
        
        # Save sizes.
        self.__Delta = Delta
        self.__Nargs = len(argsizes)        
        
        # Decide argument names.
        if argnames is None:
            argnames = ["x"] + ["p_%d" % (i,) for i in range(1,self.Nargs)]
        
        # Store names and Casadi Integrator object.
        self.__argnames = argnames
        self.__integrator = getCasadiIntegrator(ode,Delta,
                                                argsizes,argnames,wrap=False)

    def sim(self,*args):
        """
        Simulate one timestep.
        """
        # Check arguments.
        if len(args) != self.Nargs:
            raise ValueError("Wrong number of arguments: "
                "%d given; %d expected." % (len(args),self.Nargs))
        self.__integrator.setInput(args[0],"x0")
        if len(args) > 1:
            self.__integrator.setInput(casadi.vertcat(args[1:]),"p")
        
        # Call integrator.        
        self.__integrator.evaluate()
        xf = self.__integrator.getOutput("xf")
        self.__integrator.reset()
        
        return np.array(xf).flatten()

def ekf(f,h,x,u,w,y,P,Q,R,f_jacx=None,f_jacw=None,h_jacx=None):
    """
    Updates the prior distribution P^- using the Extended Kalman filter.
    
    f and h should be casadi functions. f must be discrete-time. P, Q, and R
    are the prior, state disturbance, and measurement noise covariances. Note
    that f must be f(x,u,w) and h must be h(x).
    
    If specified, f_jac and h_jac should be initialized jacobians. This saves
    some time if you're going to be calling this many times in a row, althouth
    it's really not noticable unless the models are very large.
    
    The value of x that should be fed is xhat(k | k-1), and the value of P
    should be P(k | k-1). xhat will be updated to xhat(k | k) and then advanced
    to xhat(k+1 | k), while P will be updated to P(k | k) and then advanced to
    P(k+1 | k). The return values are a list as follows
    
        [P(k+1 | k), xhat(k+1 | k), P(k | k), xhat(k | k)]
        
    Depending on your specific application, you will only be interested in
    some of these values.
    """
    
    # Check jacobians.
    if f_jacx is None:
        f_jacx = f.jacobian(0)
        f_jacx.init()
    if f_jacw is None:
        f_jacw = f.jacobian(2)
        f_jacw.init()
    if h_jacx is None:
        h_jacx = h.jacobian(0)
        h_jacx.init()
        
    # Get linearization of measurement.
    C = np.array(h_jacx([x])[0])
    yhat = np.array(h([x])[0]).flatten()
    
    # Advance from x(k | k-1) to x(k | k).
    xhatm = x                                          # This is xhat(k | k-1)    
    Pm = P                                             # This is P(k | k-1)    
    L = scipy.linalg.solve(C.dot(Pm).dot(C.T) + R, C.dot(Pm)).T          
    xhat = xhatm + L.dot(y - yhat)                     # This is xhat(k | k) 
    P = (np.eye(Pm.shape[0]) - L.dot(C)).dot(Pm)       # This is P(k | k)
    
    # Now linearize the model at xhat.
    w = np.zeros(w.shape)
    A = np.array(f_jacx([xhat,u,w])[0])
    G = np.array(f_jacw([xhat,u,w])[0])
    
    # Advance.
    Pmp1 = A.dot(P).dot(A.T) + G.dot(Q).dot(G.T)       # This is P(k+1 | k)
    xhatmp1 = np.array(f([xhat,u,w])[0]).flatten()     # This is xhat(k+1 | k)    
    
    return [Pmp1, xhatmp1, P, xhat]
    
    