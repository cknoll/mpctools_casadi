from __future__ import print_function, division # Grab some handy Python3 stuff.
import scipy.linalg
import casadi
import casadi.tools as ctools
import numpy as np
import pdb
import itertools

# First, we grab a few things from the CasADi module.
DMatrix = casadi.DMatrix
MX = casadi.MX
vertcat = casadi.vertcat

# Grab pdb function to emulate Octave/Matlab's keyboard().
keyboard = pdb.set_trace

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
        x += (k1 + 2*k2 + 2*k3 + k4)*h/6
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


def linearizeModel(f,args,names=None,Delta=None):
    """
    Returns linear (affine) state-space model for f at the point point.
    
    This is just a rewrite of getLinearization to take an arbitrary number
    of arguments.
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
        jac.init()
        jacobians.append(np.array(jac(args)[0]))
    
    # Decide whether or not to discretize.
    if Delta is not None:
        (A, Bfactor) = c2d(jacobians[0],np.eye(jacobians[0].shape[0]),Delta)
        jacobians = [A] + [Bfactor.dot(j) for j in jacobians[1:]]
        fs = Bfactor.dot(fs)
    
    # Package everything up.
    ss = dict(zip(names,jacobians))
    if "f" not in ss.keys():
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


def dlqr(A,B,Q,R):
    """
    Get the discrete-time LQR for the given system.
    """
    Pi = scipy.linalg.solve_discrete_are(A,B,Q,R)
    K = -scipy.linalg.solve(B.T.dot(Pi).dot(B) + R, B.T.dot(Pi).dot(A))
    
    return [K, Pi]

    
def dlqe(A,C,Q,R):
    """
    Get the discrete-time Kalman filter for the given system.
    """
    P = scipy.linalg.solve_discrete_are(A.T,C.T,Q,R)
    L = scipy.linalg.solve(C.dot(P).dot(C.T) + R, C.dot(P)).T     
    
    return [L, P]

    
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


def smushColloc(t,x,tc,xc):
    """
    Combines point x variables and interior collocation xc variables.
    
    The sizes of each input must be as follows:
     -  t: (Nt+1,)
     -  x: (Nt+1,Nx)
     - tc: (Nt,Nc)
     - xc: (Nt,Nx,Nc)
    with Nt the number of time periods, Nx the number of states in x, and Nc
    the number of collocation points on the interior of each time period.
    
    Returns arrays T with size (Nt*(Nc+1) + 1,) and X with size 
    (Nt*(Nc+1) + 1, Nx) that combine the collocation points and edge points.
    Also return Tc and Xc which only contain the collocation points.         
    """
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
    
    return [T,X,Tc,Xc]
    
    