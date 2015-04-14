from __future__ import print_function, division # Grab some handy Python3 stuff.
import numpy as np
import casadi
import time
from .. import util
import tools
import solvers

"""
Old linear mpc function. May be rewritten to solve as QP, but probably not.
"""

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
    0
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
    
    If your model is time-invariant, consider using nmpc to get a time-invariant
    solver that will be much faster for repeated solves.
    """        
    starttime = time.clock()    
    
    # Get shapes.    
    n = A[0].shape[0]
    m = B[0].shape[1]
    
    # Fill in default bounds.
    bounds = tools.fillBoundsDict(bounds,n,m)
    getBounds = lambda var,k : bounds[var][k % len(bounds[var])]    
    
    # Define NLP variables.
    [VAR, LB, UB, GUESS] = tools.getCasadiVars(n,m,N)
    
    # Preallocate.
    qpF = casadi.MX(0) # Start with dummy objective.
    qpG = [None]*N # Preallocate, although we could just append.
    
    # First handle objective/constraint terms that aren't optional.    
    for k in range(N+1):
        if k != N:
            LB["u",k,:] = getBounds("ulb",k)
            UB["u",k,:] = getBounds("uub",k)
            
            qpF += .5*util.mtimes(VAR["u",k].T,R[k % len(R)],VAR["u",k]) 
            qpG[k] = (util.mtimes(A[k % len(A)],VAR["x",k]) +
                util.mtimes(B[k % len(B)],VAR["u",k]) - VAR["x",k+1])
        
        if k == 0:
            LB["x",0,:] = x0
            UB["x",0,:] = x0
        else:
            LB["x",k,:] = getBounds("xlb",k)
            UB["x",k,:] = getBounds("xub",k)
        
        qpF += .5*util.mtimes(VAR["x",k].T,Q[k % len(Q)],VAR["x",k])
    
    conlb = np.zeros((n*N,))
    conub = np.zeros((n*N,))

    # Now check optional stuff.
    if q is not None:
        for k in range(N):
            qpF += util.mtimes(q[k % len(q)].T,VAR["x",k])
    if r is not None:
        for k in range(N-1):
            qpF += util.mtimes(r[k % len(r)].T,VAR["u",k])
    if M is not None:
        for k in range(N-1):
            qpF += util.mtimes(VAR["x",k].T,M[k % len(M)],VAR["u",k])                
    
    if D is not None:
        for k in range(N-1):
            qpG.append(util.mtimes(D[k % len(d)],VAR["u",k]) 
                - util.mtimes(G[k % len(d)],VAR["x",k]) - d[k % len(d)])
        s = (D[0].shape[0]*N, 1) # Shape for inequality RHS vector.
        conlb = np.concatenate(conlb,-np.inf*np.ones(s))
        conub = np.concatenate(conub,np.zeros(s))
    
    # Make qpG into a single large vector.     
    qpG = casadi.vertcat(qpG) 
    
    # Create solver and stuff.
    ipoptstart = time.clock()
    [OPTVAR,obj,status,solver] = solvers.callSolver(VAR,LB,UB,GUESS,qpF,qpG,
        conlb,conub,verbosity=verbosity,isQp=True)
    ipoptend = time.clock()
    x = np.hstack(OPTVAR["x",:,:])
    u = np.hstack(OPTVAR["u",:,:])
    
    # Add singleton dimension if n = 1 or p = 1.
    x = util.atleastnd(x)
    u = util.atleastnd(u)
    
    optVars = {"x" : x, "u" : u, "status" : status}
    optVars["obj"] = obj
    optVars["ipopttime"] = ipoptend - ipoptstart
    
    endtime = time.clock()
    if verbosity > 1:
        print("Took %g s." % (endtime - starttime))
    optVars["time"] = endtime - starttime
    
    return optVars