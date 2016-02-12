from __future__ import print_function, division # Grab some handy Python3 stuff.
import scipy.linalg
import casadi
import casadi.tools as ctools
import collections
import numpy as np
import pdb
import itertools
import sys
import os
from contextlib import contextmanager

# First, we grab a few things from the CasADi module.
DM = casadi.DM
MX = casadi.MX
vertcat = casadi.vertcat

# Grab pdb function to emulate Octave/Matlab's keyboard().
keyboard = pdb.set_trace

# Also make a wrapper to numpy's array function that forces float64 data type.
def array(x, dtype=np.float64, **kwargs):
    """
    Wrapper to NumPy's array that forces floating point data type.
    
    Uses numpy.float64 as the default data type instead of trying to infer it.
    See numpy.array for other keyword arguments.
    """
    kwargs["dtype"] = dtype
    return np.array(x, **kwargs)


# Now give the actual functions.
def rk4(f,x0,par,Delta=1,M=1):
    """
    Does M RK4 timesteps of function f with variables x0 and parameters par.
    
    The first argument of f must be var, followed by any number of parameters
    given in a list in order.
    
    Note that var and the output of f must add like numpy arrays.
    """
    h = Delta/M
    x = x0
    j = 0
    while j < M: # For some reason, a for loop creates problems here.       
        k1 = f(x,*par)
        k2 = f(x + k1*h/2,*par)
        k3 = f(x + k2*h/2,*par)
        k4 = f(x + k3*h,*par)
        x = x + (k1 + 2*k2 + 2*k3 + k4)*h/6
        j += 1
    return x


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


def getLinearization(f, xs, us=None, ds=None, Delta=None):
    """
    Less general wrapper to getLinearizedModel.    
    
    Equivalent to
    
        getLinearizedModel(f, [xs,us,ds], ["A","B","Bp"], Delta)
        
    with the corresponding entries omitted if us and/or ds are None.
    """
    args = [xs]
    names = ["A"]
    if us is not None:
        args.append(us)
        names.append("B")
    if ds is not None:
        args.append(ds)
        names.append("Bp")
    return getLinearizedModel(f, args, names, Delta)


def linearizeModel(*args, **kwargs):
    """
    Synonym for getLinearizedModel.
    
    Refer to getLinearizedModel for more details.
    """
    return getLinearizedModel(*args, **kwargs)


def getLinearizedModel(f,args,names=None,Delta=None,returnf=True,forcef=False):
    """
    Returns linear (affine) state-space model for f at the point in args.
    
    Note that f must be a casadi function (e.g., the output of getCasadiFunc).    
    
    names should be a list of strings to specify the dictionary entry for each
    element. E.g., for args = [xs, us] to linearize a model in (x,u), you
    might choose names = ["A", "B"]. These entries can then be accessed from
    the returned dictionary to get the linearized state-space model.
    
    If "f" is not in the list of names, then the return dict will also include
    an "f" entry with the actual value of f at the linearization point. To
    disable this, set returnf=False.
    """
    # Decide names.
    if names is None:
        names = ["A"] + ["B_%d" % (i,) for i in range(1,len(args))]
    
    # Evaluate function.
    fs = np.array(f(args)[0])    
    
    # Now do jacobian.
    jacobians = []
    for i in range(len(args)):
        jac = f.jacobian(i,0) # df/d(args[i]).
        jacobians.append(np.array(jac(args)[0]))
    
    # Decide whether or not to discretize.
    if Delta is not None:
        (A, Bfactor) = c2d(jacobians[0],np.eye(jacobians[0].shape[0]),Delta)
        jacobians = [A] + [Bfactor.dot(j) for j in jacobians[1:]]
        fs = Bfactor.dot(fs)
    
    # Package everything up.
    ss = dict(zip(names,jacobians))
    if returnf and ("f" not in ss or forcef):
        ss["f"] = fs
    return ss    

    
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
    Ad = D[0:n,0:n]
    Bd = D[0:n,n:n+m]
    Bpd = D[0:n,n+m:n+m+mp]
    fd = D[0:n,n+m+mp:n+m+mp+1]   
    
    return [Ad,Bd,Bpd,fd]


def c2dObjective(a,b,q,r,Delta):
    """
    Discretization with continuous objective.

    Converts from continuous-time objective
    
                 / \Delta
        l(x,u) = |        x'qx + u'ru  dt
                 / 0
        dx/dt = ax + bu
    
    to the equivalent
    
        L(x,u) = x'Qx + 2x'Mu + u'Qu
        x^+ = Ax + Bu
        
    in discrete time.
    
    Formulas from Pannocchia, Rawlings, Mayne, and Mancuso (2014).
    """
    # Make sure everything is a matrix.
    for m in [a,b,q,r]:
        try:
            if len(m.shape) != 2:
                raise ValueError("All inputs must be 2D arrays!")
        except AttributeError:
            raise TypeError("All inputs must have a shape attribute!")
            
    # Get sizes.
    Nx = a.shape[1]
    Nu = b.shape[1]
    for (m,s) in [(a,(Nx,Nx)), (b,(Nx,Nu)), (q,(Nx,Nx)), (r,(Nu,Nu))]:
        if m.shape != s:
            raise ValueError("Incorrect sizes for inputs!")
    
    # Now stack everybody up.
    i = [slice(j*Nx,(j+1)*Nx) for j in range(3)] + [slice(3*Nx,3*Nx+Nu)]
    c = np.zeros((3*Nx + Nu,)*2)
    c[i[0],i[0]] = -a.T
    c[i[1],i[1]] = -a.T
    c[i[2],i[2]] = a
    c[i[0],i[1]] = np.eye(Nx)
    c[i[1],i[2]] = q
    c[i[2],i[3]] = b
    
    # Now exponentiate and grab everybody.
    C = scipy.linalg.expm(c*Delta);
    F3 = C[i[2],i[2]]
    G3 = C[i[2],i[3]]
    G2 = C[i[1],i[2]]
    H2 = C[i[1],i[3]]
    K1 = C[i[0],i[3]]
    
    # Then, use formulas.
    A = F3
    B = G3
    Q = F3.T.dot(G2)
    M = F3.T.dot(H2)
    R = r*Delta + b.T.dot(F3.T.dot(K1)) + (b.T.dot(F3.T.dot(K1))).T
    
    return [A,B,Q,R,M]


def dlqr(A,B,Q,R,M=None):
    """
    Get the discrete-time LQR for the given system.
    
    Stage costs are
    
        x'Qx + 2*x'Mu + u'Qu
        
    with M = 0 if not provided.
    """
    # For M != 0, we can simply redefine A and Q to give a problem with M = 0.
    if M is not None:
        RinvMT = scipy.linalg.solve(R,M.T)
        Atilde = A - B.dot(RinvMT)
        Qtilde = Q - M.dot(RinvMT)
    else:
        Atilde = A
        Qtilde = Q
        M = np.zeros(B.shape)
    Pi = scipy.linalg.solve_discrete_are(Atilde,B,Qtilde,R)
    K = -scipy.linalg.solve(B.T.dot(Pi).dot(B) + R, B.T.dot(Pi).dot(A) + M.T)
    
    return [K, Pi]

    
def dlqe(A,C,Q,R):
    """
    Get the discrete-time Kalman filter for the given system.
    """
    P = scipy.linalg.solve_discrete_are(A.T,C.T,Q,R)
    L = scipy.linalg.solve(C.dot(P).dot(C.T) + R, C.dot(P)).T     
    
    return [L, P]

    
def mtimes(*args, **kwargs):
    """
    Smarter version casadi.tools.mtimes.
    
    Matrix multiplies all of the given arguments and returns the result. If all
    inputs are 2D, then passes straight through to casadi's mul. Otherwise,
    uses a sequence of np.dot operations.
    
    Keyword arguments forcedot or forcemtimes can be set to True to pick one
    behavior or another.
    """
    # Pick whether to use mul or dot.
    useMul = kwargs.get("forcemtimes", None)
    if useMul is None:
        useMul = kwargs.get("forcedot", None)
        if useMul is None:
            useMul = True
            for (i,a) in enumerate(args):
                try:
                    shape = a.shape
                except AttributeError:
                    try:
                        shape = np.array(a).shape
                    except:
                        raise AttributeError("Unable to get shape of "
                        "argument %d!" % (i,))
                useMul &= len(shape) == 2
        else:
            useMul = not useMul
    # Now actually do multiplication.
    if useMul:
        ans = ctools.mtimes(args)
    else:
        ans = args[0]
        for (i,a) in enumerate(args[1:]):
            try:
                ans = np.dot(ans,a)
            except ValueError:
                raise ValueError("Wrong alignment for argument %d!" % (i + 1))
    return ans

    
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
        
    Note that if the struct entry is empty, then there will not be a
    corresponding key in the returned dictonary.
    """ 

    npdict = {}
    for k in struct.keys():
        if len(struct[k]) > 0:
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


def smushColloc(t, x, tc, xc, Delta=1, asdict=False):
    """
    Combines point x variables and interior collocation xc variables.
    
    The sizes of each input must be as follows:
     -  t: (Nt+1,)
     -  x: (Nt+1,Nx)
     - tc: (Nt,Nc)
     - xc: (Nt,Nx,Nc)
    with Nt the number of time periods, Nx the number of states in x, and Nc
    the number of collocation points on the interior of each time period. Note
    that if t or tc is None, then they are constructed using a timestep of
    Delta (with default value 1).
    
    Returns arrays T with size (Nt*(Nc+1) + 1,) and X with size 
    (Nt*(Nc+1) + 1, Nx) that combine the collocation points and edge points.
    Also return Tc and Xc which only contain the collocation points.

    If asdict=True, then results are returned in a dictionary. This contains
    fields "t" and "x" with interior collocation and edge points together,
    "tc" and "xc" with just the inter collocation points, and "tp" and "xp"
    which are only the edge points.         
    """
    # Make copies.
    if t is not None:
        t = t.copy()
    if tc is not None:
        tc = tc.copy()
    x = x.copy()
    xc = xc.copy()
    
    # Build t and tc if not given.
    if t is None or tc is None:
        Nt = xc.shape[0]
        if t is None:                
            t = np.arange(0, Nt+1)*Delta
        else:
            t.shape = (t.size,)
        import colloc
        Nc = xc.shape[2]
        [r, _, _, _] = colloc.weights(Nc, include0=False, include1=False)
        r.shape = (r.size,1)
        tc = (t[:-1] + r*Delta).T.copy()
    
    # Add some dimensions to make sizes compatible.
    t.shape = (t.size,1)
    x.shape += (1,)
    
    # Begin the smushing.
    T = np.concatenate((t[:-1],tc),axis=1)    
    X = np.concatenate((x[:-1,...],xc),axis=2)
    
    # Have to do some permuting for X. Order is now (t,c,x).
    X = X.transpose((0,2,1))
    Xc = xc.transpose((0,2,1))
    
    # Now flatten.
    T.shape = (T.size,)
    Tc = tc.flatten()
    X = X.reshape((X.shape[0]*X.shape[1],X.shape[2]))
    Xc = Xc.reshape((Xc.shape[0]*Xc.shape[1],Xc.shape[2]))
    
    # Then add final elements.
    T = np.concatenate((T,t[-1:,0]))
    X = np.concatenate((X,x[-1:,:,0]))
    
    if asdict:
        ret = dict(t=T, x=X, tc=Tc, xc=Xc, tp=np.squeeze(t), xp=np.squeeze(x))
    else:
        ret = [T, X, Tc, Xc]
    return ret


@contextmanager
def stdout_redirected(to=os.devnull):
    """
    context to redirect all Python output, including C code.
    
    Used in a with statement, e.g.,

        with stdout_redirected(to=filename):
            print "from Python"
            ipopt.solve()
    
    will capture both the Python print output and any output of calls to C
    libraries (e.g., IPOPT).
    
    Note that this makes use of CasADi's tools.io.nice_stdout context, which
    means all C output is buffered and then returned all at once. Thus, this is
    only really useful if don't need to see output as it is created.
    """
    import casadi.tools.io as casadiio

    old_stdout = sys.stdout
    with open(to, "w") as new_stdout:
        with casadiio.nice_stdout(): # Buffers C output to Python stdout.
            sys.stdout = new_stdout # Redefine Python stdout.
            try:
                yield # Allow code to be run with the redirected stdout.
            finally:
                sys.stdout = old_stdout # Reset stdout.

   
@contextmanager
def dummy_context(*args):
    """
    Dummy context for a with statement.
    """
    # We need this in solvers.py.
    yield


class ArrayDict(collections.MutableMapping):
    """
    Python dictionary of numpy arrays.

    When instantiating or when setting an item, calls np.array to convert
    everything.
    """
    def __init__(self, *args, **kwargs):
        """
        Creates a dictionary and then wraps everything in np.array.
        """
        self.__arraydict__ = dict() # This is where we actually store things.
        self.update(dict(*args, **kwargs)) # We get this method for free.      
    
    def __setitem__(self, k, v):
        """
        Wraps v with np.array before setting.
        """
        self.__arraydict__[k] = np.array(v)
    
    # The rest of the methods just perform the corresponding dict action.
    def __getitem__(self, k):
        return self.__arraydict__[k]
    
    def __len__(self):
        return len(self.__arraydict__)
    
    def __iter__(self):
        return iter(self.__arraydict__)
        
    def __delitem__(self, k):
        del self.__arraydict__[k]
        
    def __repr__(self):
        return repr(self.__arraydict__)


def strcolor(s, color=None, bold=False):
    """
    Adds ANSI escape sequences to colorize string s.
    
    color must be one of the eight standard colors (RGBCMYKW). Accepts full
    names or one-letter abbreviations.
    
    Keyword bold decides to make string bold.
    """
    colors = dict(_end="\033[0m", _bold="\033[1m", b="\033[94m", c="\033[96m",
        g="\033[92m", k="\033[90m", m="\033[95m", r="\033[91m", w="\033[97m",
        y="\033[93m")
    colors[""] = "" # Add a few defaults.
    colors[None] = ""
    
    # Decide what color user gave.
    c = color.lower()
    if c == "black":
        c = "k"
    elif len(c) > 0:
        c = c[0]
    try:
        c = colors[c]
    except KeyError:
        raise ValueError("Invalid color choice '%s'!" % (color,))
    
    # Build up front and back of string and return.
    front = (colors["_bold"] if bold else "") + c
    back = (colors["_end"] if len(front) > 0 else "")
    return "%s%s%s" % (front, s, back)


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
    if f_jacw is None:
        f_jacw = f.jacobian(2)
    if h_jacx is None:
        h_jacx = h.jacobian(0)
        
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


# Conveinence function for getting derivatives.
def getScalarDerivative(f, nargs=1, wrt=(0,), vectorize=True):
    """
    Returns a function that gives the derivative of the function scalar f.
    
    f must be a function that takes nargs scalar entries and returns a single
    scalar. Derivatives are taken with respect to the variables specified in
    wrt, which must be a tuple of integers. E.g., to take a second derivative
    with respect to the first argument, specify wrt=(0,0).
    
    vectorize is a boolean flag to determine whether or not the function should
    be wrapped with numpy's vectorize. Note that vectorized functions do not
    play well with Casadi symbolics, so set vectorize=False if you wish to
    use the function later on with Casadi symbolics.
    """
    x = [casadi.SX.sym("x" + str(n)) for n in range(nargs)]
    dfdx_expression = f(*x)
    for i in wrt:
        dfdx_expression = casadi.jacobian(dfdx_expression, x[i])
    dfcasadi = casadi.SXFunction("dfdx", x, [dfdx_expression])
    def dfdx(*x):
        return dfcasadi(x)[0]
    if len(wrt) > 1:
        funcstr = "d^%df/%s" % (len(wrt), "".join(["x%d" % (i,) for i in wrt]))
    else:
        funcstr = "df/dx"
    dfdx.__doc__ = "\n%s = %s" % (funcstr, repr(dfdx_expression))
    if vectorize:    
        ret = np.vectorize(dfdx, otypes=[np.float])
    else:
        ret = dfdx
    return ret
    