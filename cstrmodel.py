# Helper stuff for Example 1.11 from Rawlings and Mayne

from __future__ import print_function
from casadi import casadi
import numpy as np
from scipy import linalg

# ******
# CSTR class to facilitate simulation with Casadi
# ******

class CSTR(object):
    """
    Object for CSTR simulation and control.
    """
    # Parameters
    T0 = 350
    c0 = 1
    r = .219
    k0 = 7.2e10
    E = 8750
    U = 54.94
    rho = 1000
    Cp = .239
    dH = -5e4
    
    Delta = 1
    
    # We want these private because they are very important.
    __Nx = 3
    __Nu = 2
    __Nd = 1
    __Ny = 3
    def Nx(self):
        """Number of states."""
        return self.__Nx
    def Nu(self):
        """Number of control inputs."""
        return self.__Nu
    def Nd(self):
        """Number of disturbances."""
        return self.__Nd
    def Ny(self):
        """Number of outputs."""
        return self.__Ny

    # Initialize these guys as None. Will be set upon initialization.
    __Function = None
    __Jacobian = None
    __Integrator = None

    # Build system model.
    def ode(self,c,T,h,Tc,F,F0):
        """
        cstr mass and energy balance.
        """
        rate = self.k0*c*np.exp(-self.E/T)
        
        dxdt = [
            F0*(self.c0 - c)/(np.pi*self.r**2*h) - rate,
            F0*(self.T0 - T)/(np.pi*self.r**2*h)
                - self.dH/(self.rho*self.Cp)*rate
                + 2*self.U/(self.r*self.rho*self.Cp)*(Tc - T),    
            (F0 - F)/(np.pi*self.r**2)
        ]
        
        return dxdt

    def __init__(self,Delta=None):
        """
        Initializes model and does Casadi symbolic stuff.
        """
        if Delta is not None:
            self.Delta = Delta
        self.updateModel()

    def updateModel(self):
        """
        Updates the model with the current parameter values.
        """
        # State variables.
        c = casadi.SX.sym("c")
        T = casadi.SX.sym("T")
        h = casadi.SX.sym("h")

        # Inputs.
        F = casadi.SX.sym("F")
        Tc = casadi.SX.sym("Tc")

        # Disturbances.
        F0 = casadi.SX.sym("F0")        
        
        # Make into a casadi function.
        x = casadi.vertcat([c,T,h])
        u = casadi.vertcat([Tc,F,F0])
        dxdt = self.ode(c,T,h,Tc,F,F0)
        
        # Define function and jacobian as Casadi objects.
        self.__Function = casadi.SXFunction([c,T,h,Tc,F,F0],dxdt)
        self.__Function.init()
        
        self.__Jacobian = self.__Function.fullJacobian()
        self.__Jacobian.init()
        
        # Now define integrator for simulation.
        model = casadi.SXFunction(casadi.daeIn(x=x,p=u),
                                  casadi.daeOut(ode=casadi.vertcat(dxdt)))
        model.init()
        
        self.__Integrator = casadi.Integrator("cvodes",model)
        self.__Integrator.setOption("tf",self.Delta)
        self.__Integrator.init()
        
    def getLinearization(self,cs,Ts,hs,Tcs,Fs,F0s):
        """
        Returns linear state-space model for CSTR system.
        """
        # Now linearize and then discretize model.
        fcont = np.array(self.__Function([cs,Ts,hs,Tcs,Fs,F0s]))[:,np.newaxis]
        Js = np.array(self.__Jacobian([cs,Ts,hs,Tcs,Fs,F0s])[0])       
        Acont = Js[:,:self.__Nx]
        Bcont = Js[:,self.__Nx:self.__Nx+self.__Nu]
        Bpcont = Js[:,self.__Nx+self.__Nu:self.__Nx+self.__Nu+self.__Nd]
        
        [A, B, Bp, f] = c2d(Acont,Bcont,Bpcont,fcont,self.Delta)
        C = np.matrix(np.eye(self.__Nx))
        
        return {"A": A, "B": B, "C": C, "Bp": Bp, "f": f}
        
    def sim(self,x0,u,d,matrix=True):
        """
        Simulates one timestep with the nonlinear models.
        
        Inputs:
            x = [c,T,h]
            u = [Tc,F]
            p = [F0]
        """
        self.__Integrator.setInput(x0,"x0")
        self.__Integrator.setInput(casadi.vertcat([u,d]),"p")
        self.__Integrator.evaluate()
        xf = self.__Integrator.getOutput("xf")
        self.__Integrator.reset()
        
        if matrix:
            return np.matrix(xf)
        else:
            return np.array(xf).flatten()

# *****************
# Helper Functions
# *****************

def c2d(A,B,Bp,f,Delta):
    """
    Discretizes affine system (A,B,Bp,f) with timestep Delta.
    """
    
    n = A.shape[0]
    m = B.shape[1]
    mp = Bp.shape[1]
    M = m + mp + 1 # Extra 1 is for function column.
    
    D = linalg.expm(Delta*np.vstack((np.hstack([A, B, Bp, f]),
                                     np.zeros((M,M+n)))))
    Ad = np.matrix(D[0:n,0:n])
    Bd = np.matrix(D[0:n,n:n+m])
    Bpd = np.matrix(D[0:n,n+m:n+m+mp])
    fd = np.matrix(D[0:n,n+m+mp:n+m+mp+1])    
    
    return [Ad,Bd,Bpd,fd]
    
def dlqr(A,B,Q,R):
    """
    Get the discrete-time LQR for the given system.
    """
    Pi = np.matrix(linalg.solve_discrete_are(A,B,Q,R))
    K = -linalg.solve(B.T*Pi*B + R, B.T*Pi*A)
    
    return [K, Pi]
    
def dlqe(A,C,Q,R):
    """
    Get the discrete-time Kalman filter for the given system.
    """
    P = np.matrix(linalg.solve_discrete_are(A.T,C.T,Q,R))
    L = linalg.solve(C*P*C.T + R, C*P).T 
    
    return [L, P]
    
def augment(A,B,C,Bd,Cd):
    """
    Augments a state-space model (A,B,C) with disturbance model (I,Bd,Cd).
    """
    # Sizes.
    Nx = A.shape[0]
    Nu = B.shape[1]
    Nd = Bd.shape[1]
    
    # Build augmented matrices.
    Aaug = np.vstack((np.hstack((A,Bd)),
                      np.hstack((np.zeros((Nd,Nx)),np.eye(Nd)))))
    Baug = np.vstack((B,np.zeros((Nd,Nu))))
    Caug = np.hstack((C,Cd))
    
    return [Aaug,Baug,Caug]
    
def sstarg(A,B,C,Q,R,ysp,H,d=None,Bd=None,Cd=None,unique=False):
    """
    Solve the steady-state target problem for the given augmented system.
    
    If you know the solution will be unique, setting unique=True will save
    computation time.
    """
    # Sizes.
    Nx = A.shape[0]
    Nu = B.shape[1]
    
    # Calculate rsp.
    rsp = H*ysp
            
    # Build some matrices.
    conA = np.zeros((Nx+Nu,Nx+Nu))
    conA[:Nx,:Nx] = np.matrix(np.eye(Nx)) - A
    conA[:Nx,Nx:] = -B
    conA[Nx:,:Nx] = H*C
    
    if d is None:
        if Bd is not None or Cd is not None:
            raise ValueError("If d is given, Bd and Cd must also be given!")
        conb = np.vstack((np.matrix(np.zeros((Nx,1))),rsp))
    else:
        conb = np.vstack((Bd*d,rsp-H*Cd*d))
        
    objQ = linalg.block_diag(C.T*Q*C,R)
    if d is None:
        objq = np.matrix(np.zeros((Nx+Nu,1)))
    else:
        objq = np.vstack((C.T*Q*(Cd*d - ysp),np.matrix(np.zeros((Nu,1)))))
    
    if not unique:    
        # Build big primal dueal matrix
        LHS = np.vstack((np.hstack((objQ,-conA.T)),
                         np.hstack((-conA,np.zeros((Nx+Nu,Nx+Nu))))))
        RHS = np.vstack((-objq,-conb))
    else:
        # Just find the steady-state value.
        LHS = conA
        RHS = conb
    
    # Use analytical formula.
    optimal = linalg.solve(LHS,RHS,overwrite_a=True,overwrite_b=True)
    
    xstar = optimal[0:Nx]
    ustar = optimal[Nx:Nx+Nu]
    
    return [xstar, ustar]
    
# Some printing functions.
def numberformat(n,nzeros=3):
    """
    Formats a float as a string with proper sig figs.
    """
    s = ("%%.%dg" % nzeros) % (n,)
    # Add trailing period for floats.        
    if round(n) != n and s.find(".") == -1 and n < 10**(nzeros+1):
        s += "."
    
    # Check if there's scientific notation.
    e = s.find("e")
    if e >= 0:
        head = s[0:e]
    else:
        head = s    
    
    # Make sure we show enough sig figs.    
    if head.find(".") >= 0 and len(head) <= nzeros:
        addstr = "0"*(nzeros - len(head) + 1)
    else:
        addstr = ""
    
    if e >= 0: # Need to handle scientific notation.
        s = s.replace("e",addstr + r" \times 10^{")
        for [f,r] in [["{+","{"],["{0","{"],["{-0","{-"]]:
            for i in range(5):
                s = s.replace(f,r)
        s = s + "}"
    else:
        s += addstr
    return s

def printmatrix(A,before="     "):
    """
    Prints a matrix for copy/paste into LaTeX.
    """
    for i in range(A.shape[0]):
        print(before + " & ".join([numberformat(a) for a in 
                                    np.array(A)[i,:].tolist()]) + r" \\")    
    