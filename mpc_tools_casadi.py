import numpy as np
import scipy.linalg
import casadi
import casadi.tools
import matplotlib.pyplot as plt

"""
Functions for solving linear MPC problems using Casadi and Ipopt.

The main function is lmpc, which is analogous to the mpc-tools function of the
same name. However, this function is currently missing a lot of the "advanced"
functionality of Octave lmpc, e.g. soft constraints, solver tolerances, and
returning lagrange multipliers.

Most other functions are just convenience wrappers to replace calls like
long.function.name((args,as,tuple)) with f(args,as,args), although these are
largely unnecessary.
"""

def lmpc(A,B,x0,N,Q,R,q=None,r=None,M=None,xlb=None,xub=None,ulb=None,uub=None,D=None,G=None,d=None):
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
    
    Return value is a tuple (x,u) with both x and u as 2D arrays with the first
    index corresponding to individual states and the second index corresponding
    to time.
    """    
    
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
    VAR = casadi.tools.struct_symMX([(
        casadi.tools.entry("x",shape=(n,1),repeat=N+1),
        casadi.tools.entry("u",shape=(p,1),repeat=N),
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
            
            qpF += .5*mtimes(VAR["u",k].T,R,VAR["u",k]) 
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
    solver.init()
    solver.setInput(LB,"lbx")
    solver.setInput(UB,"ubx")
    solver.setInput(conlb,"lbg")
    solver.setInput(conub,"ubg")
    
    # Solve.    
    solver.evaluate()

    # Get solution.
    OPTVAR = VAR(solver.getOutput("x"))
    x = np.hstack(OPTVAR["x",:,:])
    u = np.hstack(OPTVAR["u",:,:])
    
    # Add singleton dimension if n = 1 or p = 1.
    if n == 1:    
        x = x.reshape(1,N+1)
    if p == 1:
        u = u.reshape(1,N)
    
    return (x,u)

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
    return casadi.tools.mul(args)
    
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
    
def mpcplot(x,u,t,xsp=None,fig=None,xinds=None,uinds=None,tightness=.5):
    """
    Makes a plot of the state and control trajectories for an mpc problem.
    
    Inputs x and u should be n by N and p by N numpy arrays. xsp if provided
    should be the same size as x. t should be a numpy N vector.
    
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