import numpy as np
import scipy.linalg
import casadi
import casadi.tools
import matplotlib.pyplot as plt

"""
Functions for solving linear MPC problems using Casadi and Ipopt.

The main function is lmpc, which is analogous to the mpc-tools function of the
same name. However, this function is currently missing a lot of functionality,
e.g. state constraints, non-box input constraints, and mixed objective terms.

Most other functions are just convenience wrappers to replace calls like
long.function.name((args,as,tuple)) with f(args,as,args), although these are
largely unnecessary.
"""

def lmpc(A,B,x0,N,Q,q,R,ulb,uub):
    """
    Solves the canonical linear MPC problem using a discrete-time model.
    
    Inputs are discrete-time state-space model, objective function weights, and
    input constraints. Output is a tuple (x,u) with the optimal state and input
    trajectories.    
    
    The actual optimization problem is as follows:
    
        min \sum_{k=0}^N        0.5*x[k]'*Q[k]*x[k] + q[k]'*x[k]      
            + \sum_{k=0}^{N-1}  0.5*u[k]'*R[k]*u[k]
    
        s.t. x[k+1] = A[k]*x[k] + B[k]*u[k]               k = 0,...,N-1
             ulb[k] <= u[k] <= uub[k]                     k = 0,...,N-1
    
    A, B, Q, and R should be lists of numPy matrices. x0 should be a numPy vector.
    q, ulb, and uub should be lists of numpy vectors.
    
    All of these lists are accessed modulo their respective length;hus,
    time-invariant models can be lists with one element, while time-varying
    periodic model with period T should have T elements.
    
    Return value is 
    """
    
    # Get shapes.    
    n = A[0].shape[0]
    p = B[0].shape[1]
    
    # Define NLP variables.
    VAR = casadi.tools.struct_symMX([(
        casadi.tools.entry("x",shape=(n,1),repeat=N+1),
        casadi.tools.entry("u",shape=(p,1),repeat=N),
    )])
    LB = VAR(-np.Inf) # Default all bounds to +/- Inf.
    UB = VAR(np.Inf)
    F = casadi.MX(0) # Start with dummy objective.
    G = [None]*N # Preallocate, although we don't really need to do this.
    for k in range(N,-1,-1):
        if k != N:
            LB["u",k,:] = ulb[k % len(ulb)]
            UB["u",k,:] = uub[k % len(uub)]
            
            F += .5*mtimes([VAR["u",k].T,R,VAR["u",k]]) 
            G[k] = mtimes(A[k % len(A)],VAR["x",k]) + mtimes(B[k % len(B)],VAR["u",k]) - VAR["x",k+1]
        
        if k == 0:
            LB["x",0,:] = x0
            UB["x",0,:] = x0
        
        F += .5*mtimes([VAR["x",k].T,Q[k % len(Q)],VAR["x",k]]) + mtimes(q[k % len(q)].T,VAR["x",k])                
    
    # Make G into a single large vector.     
    G = casadi.vertcat(G) 
    
    # Create solver and stuff.
    nlp = casadi.MXFunction(casadi.nlpIn(x=VAR),casadi.nlpOut(f=F,g=G))
    solver = casadi.NlpSolver("ipopt",nlp)
    solver.init()
    solver.setInput(LB,"lbx")
    solver.setInput(UB,"ubx")
    solver.setInput(0,"lbg")
    solver.setInput(0,"ubg")
    
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
    
    Accepts variable number of arguments instead of a single tuple.
    """
    return casadi.tools.mul(*args)
    
def vcat(*args):
    """
    Convenience wrapper for np.vstack.
    
    Accepts variable number of arguments instead of a single tuple.
    """
    return np.vstack(args)
    
def hcat(*args):
    """
    Convenience wrapper for np.hstack.
    
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