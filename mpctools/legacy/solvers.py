from __future__ import print_function, division # Grab some handy Python3 stuff.
import numpy as np
import casadi
import time
from .. import util

"""
Solvers and helper functions for legacy mpc-tools-casadi code.
"""

def callSolver(var,varlb,varub,varguess,obj,con,conlb,conub,par=None,
    parval=None,verbosity=5,timelimit=60,isQp=False,runOptimization=True):
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
    solver.setOption("max_cpu_time",timelimit)
    ## This option seems to just crash Python whenever anything bad happens.
    #solver.setOption("check_derivatives_for_naninf","yes")
    solver.setOption("eval_errors_fatal",True)
    if isQp:
        solver.setOption("hessian_constant","yes")
        solver.setOption("jac_c_constant","yes")
        solver.setOption("jac_d_constant","yes")
    solver.init()

    solver.setInput(varguess,"x0")
    solver.setInput(varlb,"lbx")
    solver.setInput(varub,"ubx")
    solver.setInput(conlb,"lbg")
    solver.setInput(conub,"ubg")
    if parval is not None:
        solver.setInput(parval,"p")
    
    # Solve.
    if runOptimization:    
        solver.evaluate()
        status = solver.getStat("return_status")
    else:
        status = "NO_OPTIMIZATION_REQUESTED"
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
    
    def __init__(self,var,varlb,varub,varguess,obj,con,conlb,conub,par=None,verbosity=5,parval=None,timelimit=60):
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
        self.timelimit = timelimit
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
        solver.setOption("max_cpu_time",self.timelimit)
        ## This option seems to just crash Python whenever anything bad happens.
        #solver.setOption("check_derivatives_for_naninf","yes")
        solver.setOption("eval_errors_fatal",True)
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
        optvarDict = util.casadiStruct2numpyDict(optvar)       
        
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
        self.lb[var,t,:] = val
        self.ub[var,t,:] = val
        self.guess[var,t,:] = val
        
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
            self.par[name,t,:] = values[t % len(values)]
        self.solver.setInput(self.par,"p")
    
    def storeguesser(self,guesser):
        """
        Saves a one-timestep solver used to generate guesses.
        """
        # Should probably do some checking to make sure this a valid entry,
        # i.e. is a TimeInvariantSolver object.
        self.guesser = guesser
    
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

        
class TargetSelector(object):
    """
    Finds u given a steady-state x for a discrete- or continuous-time system.
    """
    
    def __init__(self,model,measurement,contvars,Nx,Ny,Nd=0,continuous=True,verbosity=0,Q=None,unique=False,bounds={},timelimit=10):
        """
        Initialize by specifying sizes and model.
        
        Set unique to True if the problem will always have a unique solution.
        Then it will be solved as an unconstrained minimization of the residual,
        which should be better than with tight constraints.
        """
        
        # Save sizes.
        self.Nx = Nx
        self.Ny = Ny
        self.Nd = Nd        
        self.Nu = len(contvars)
        self.bounds = bounds
        
        # Define casadi variables.        
        Np = Nx + Nd # Parameters are xsp and any other disturbance.
        p = casadi.MX.sym("p",Np)
        xsp = p[:Nx]
        d = p[Nx:]
        
        Nz = Nx + self.Nu
        z = casadi.MX.sym("z",Nz)
        x = z[:Nx]
        u = z[Nx:]
        
        # Define constraints symbolically.
        xerr = x - xsp
        if Q is None:
            Q = np.eye(Nx)
        objective = util.mtimes(xerr.T,Q,xerr)
        
        # Enforce steady-state.
        conModel = casadi.vertcat(model(x=x,u=u,d=d)[:Nx-Ny])
        if not continuous:
            conModel -= x[:Nx-Ny]
        
        # Select setpoint things.
        H = np.zeros((self.Nu,Ny))
        for i in range(self.Nu):
            H[i,contvars[i]] = 1       
        
        conMeas = util.mtimes(H,casadi.vertcat(measurement(x)) - casadi.vertcat(measurement(xsp)))       
        constraints = casadi.vertcat([conModel,conMeas])
        
        if unique:
            objective = casadi.MX(0)            
        nlpOut = casadi.nlpOut(f=objective,g=constraints)
        self.unique = unique                
                
        # Make solver object.
        nlp = casadi.MXFunction(casadi.nlpIn(x=z,p=p),nlpOut)
        
        self.solver = casadi.NlpSolver("ipopt",nlp)
        self.solver.setOption("print_level",verbosity)
        self.solver.setOption("print_time",verbosity > 2)
        self.solver.setOption("max_cpu_time",timelimit)
        ## This option seems to just crash Python whenever anything bad happens.
        #self.solver.setOption("check_derivatives_for_naninf","yes")
        self.solver.setOption("eval_errors_fatal",True)
        self.solver.init()
        if not unique:
            self.solver.setInput(np.zeros((Nz-Ny,)),"lbg")
            self.solver.setInput(np.zeros((Nz-Ny,)),"ubg")
        self.solver.setInput(np.zeros((Nz,)),"x0")
       
    def solve(self,xsp,uguess=None,d=[],fixedx=None):
       """
       Solve to find the steady-state value of u.
       """
       
       # Store parameters and guess.
       self.solver.setInput(np.concatenate((xsp,d)),"p")
       if uguess is not None:
           uguess = np.zeros((self.Nu,))
       self.solver.setInput(np.concatenate((xsp,uguess)),"x0")
       
       # Worry about bounds.
       ubx = np.inf*np.ones((self.Nx + self.Nu,))
       lbx = -ubx
       if fixedx is None:
           fixedx = range(self.Nx - self.Ny,self.Nx)
       for i in fixedx:
           ubx[i] = xsp[i]
           lbx[i] = xsp[i]
       if "uub" in self.bounds.keys():
           ubx[self.Nx:] = self.bounds["uub"]
       if "ulb" in self.bounds.keys():
           lbx[self.Nx:] = self.bounds["ulb"]
       self.solver.setInput(ubx,"ubx")
       self.solver.setInput(lbx,"lbx")
       self.solver.evaluate()
       
       z = np.array(self.solver.getOutput("x")).flatten()
       x = z[:self.Nx]
       u = z[self.Nx:]
       objval = float(self.solver.getOutput("f"))
       status = self.solver.getStat("return_status")
       
       return dict(x=x,u=u,r=objval,status=status)
       