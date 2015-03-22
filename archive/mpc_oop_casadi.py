from __future__ import print_function, division # Grab handy Python3 stuff.
import mpc_tools_casadi as mpc
import casadi
import numpy as np
import copy
import time

class Mpc(object):
    """
    Utility class for solving MPC problems.
    
    Allows for saving variables/constraints and efficient re-optimization.
    """
     
    @property
    def Nx(self):
        return self.__Nx
        
    @property
    def Nu(self):
        return self.__Nu
        
    @property
    def Nt(self):
        return self.__Nt
        
    @property
    def Delta(self):
        return self.__Delta
    
    @property
    def Nd(self):
        return self.__Nd
    
    @property
    def Nc(self):
        return self.__Nc
        
    @property
    def Nz(self):
        return self.__Nz
     
    def __init__(self,Nx,Nu,Nt,Delta,Nd=None,Nc=None,Nz=None):
        """
        Initialize with variable sizes.
        
        Nx and Nu are number of states and controls. Nt is the TOTAL number of
        time periods (the prediction horizon for each individual problem is
        specified later). Delta is the timestep.
        
        If present, Nd is the number of disturbances affecting the model. These
        must be passed as parameters later on.        
        
        If present, Nc is the number of collocation points within each time
        period. Must be None if no collocation is to be used.
        
        If present, Nz is the number of algebraic variables (for a DAE
        system). Must be None if system is not a DAE.
        """
        # Numbers that must 
        self.__Nx = checkPosInt(Nx)
        self.__Nu = checkPosInt(Nu)
        self.__Nt = checkPosInt(Nt)
        
        # Timestep.
        if Delta <= 0:
            raise ValueError("Delta must be strictly positive!")
        else:
            self.__Delta = Delta
        
        if Nd is not None:
            Nd = checkPosInt(Nd)
        self.__Nd = Nd
        
        # Now figure out potentially weird stuff.
        varnames = ["x","u"]
        if Nc is not None:
            Nc = checkPosInt(Nc)
            varnames += ["xc"]
        self.__Nc = Nc
        
        if Nz is not None:
            Nz = checkPosInt(Nz)
            varnames += ["z"]
            if Nc is not None:
                varnames += ["zc"]
        self.__Nz = Nz
        
        # Now build casadi variable struct and initialize some bounds.
        [self.__var, self.__lb, self.__ub, self.__guess] = mpc.getCasadiVars(
            Nx,Nu,Nt,Nc,Nz)
        
        self.__history = self.__var(0) # Save system history.
        self.__bounds = mpc.fillBoundsDict({},Nx,Nu)
    
    def setDiscreteModel(self,F,G=None,d=None):
        """"
        Store explicit discrete-time model x^+ = F(x,u).
        
        If disturbances and/or algebraic states are present, the order is
        different:
        
          Disturbances : F(x,u,d)
          Algebraic    : F(x,z,u)
          Both         : F(x,z,u,d)
        
        With algebraic states, G must be specified, and the argument order is
        the same.        
        
        All functions should be specified as LISTS of functions. They will be
        accessed modulo length, which allows for time-invariant systems,
        time-varying systems, and periodic systems.
        """
        # Make sure model is the right size.
        shapes = mpc.getVarShapes(F[0],self.__var,
                                  disturbancemodel=(self.Nd is not None))
        if shapes["x"] != self.Nx:
            raise ValueError("Incorrect size for function F!")
        
        # If they for some reason provide a disturbance when we aren't set up
        # for one, just ignore it.
        if self.Nd is None:
            d = None
        
        # Generate all of the constraints.
        [con, conlb, conub] = mpc.getDiscreteTimeConstraints(F,self.__var,d=d)
        
        self.__con = con
        self.__conlb = conlb
        self.__conub = conub        
            
    def setObjective(self,l,Vf=None):
        """
        Stores stage cost and a terminal penalty.
        
        Both l and Vf must be lists of functions. l must have all of the model
        arguments, i.e., l(x,z,u,d), and Vf should be Vf(x,z).
        """
        
        self.__stagecosts = l
        self.__termpen = Vf
        
    def setBounds(self,xlb=None,xub=None,ulb=None,uub=None):
        """
        Specify box constraints on x and u.
        """
        
        givenbounds = {}
        for (k,v) in [("xlb",xlb), ("xub",xub), ("ulb",ulb), ("uub",uub)]:
            if v is not None:
                givenbounds[k] = v
        self.__bounds = mpc.fillBoundsDict(givenbounds,self.Nx,self.Nu)
        
        # Now save these values to the actual bounds structures.
        for v in ["x","u"]:        
            for t in range(self.Nt):
                self.__lb[v,t] = self.getBound(v + "lb",t)
                self.__ub[v,t] = self.getBound(v + "ub",t)
        self.__lb["x",self.Nt] = self.getBound("xlb",self.Nt)
        self.__ub["x",self.Nt] = self.getBound("xub",self.Nt)
    
    def getBound(self, var,t):
        """
        Returns the bound of var at time t.
        
        var is one of "xlb", "xub", "ulb", or "uub". Algebraic variables to
        be added later.
        """
        
        try:
            bound = self.__bounds[var]
        except KeyError:
            raise KeyError("Invalid choice of var!")
            
        return bound[t % len(bound)]
    
    def solve(self,N,t0=0,x0=None,z0=None,verbosity=5):
        """
        Solve finite-horizon MPC problem with horizon N starting at t0.
        
        If x0 unspecified, pull value from history.
        """
        starttime = time.clock()
        
        t1 = t0 + N
        
        # Get variables.
        var = self.__var
        varlb = copy.deepcopy(self.__guess) # Default to all variables fixed. Will
        varub = copy.deepcopy(self.__guess) # be undone next.
        varguess = copy.deepcopy(self.__guess)
        
        # Un-fix the important variables.
        allvarnames = var.keys()
        finalvarnames = [v for v in ["x","z"] if v in allvarnames]
        nonfinalvarnames = [v for v in allvarnames if v not in finalvarnames]
        for T in range(t0,t1):
            for [vs,t] in [[nonfinalvarnames, T],[finalvarnames, T+1]]:
                for v in vs:
                    varlb[v,t] = self.__lb[v,t]
                    varub[v,t] = self.__ub[v,t]
        
        # Decide initial conditions.
        for [vname,v0] in [["x",x0],["z",z0]]:
            if vname in allvarnames:
                if v0 is None:
                    v0 = self.__history[var,t0]
                varlb[vname,t0] = v0
                varub[vname,t0] = v0
                varguess[vname,t0] = v0
        
        # Extract subset of constraints.
        con = casadi.vertcat(mpc.flattenlist(self.__con[t0:t1]))
        conlb = self.__conlb[t0:t1,...].flatten()
        conub = self.__conub[t0:t1,...].flatten()
        
        # Sort out objective.
        l = self.__stagecosts
        Pf = self.__termpen
        obj = casadi.MX(0)
        for t in range(t0,t1):
            obj += l[t % len(l)]([var["x",t],var["u",t]])[0]
        if Pf is not None:
            obj += Pf([var["x",t1]])[0]
        
        # Call solver.
        ipoptstart = time.clock()
        [optvar,obj,status,solver] = mpc.callSolver(var,varlb,varub,varguess,
            obj,con,conlb,conub,verbosity=verbosity)
        ipoptend = time.clock()
        
        x = np.hstack(optvar["x",t0:t1+1,:])
        u = np.hstack(optvar["u",t0:t1,:])
        
        # Add singleton dimension if Nx = 1 or Nu = 1.
        x = mpc.atleastnd(x)
        u = mpc.atleastnd(u)
        
        endtime = time.clock()
        if verbosity > 1:
            print("Took %g s." % (endtime - starttime))    
        
        optDict = {"x" : x, "u" : u, "status" : status,
                   "time" : endtime - starttime, "obj" : obj,
                   "t0" : t0, "t1" : t1, "__var__" : optvar,
                   "ipopttime" : ipoptend - ipoptstart}
    
        # Return collocation points if present.
        if self.Nc is not None:
            xc = np.hstack(optvar["xc",t0:t1,:,:])
            optDict["xc"] = xc
        
        return optDict
    
    def acceptSolve(self,optDict,Nblock=1):
        """
        Accepts the solution given in optDict and saves it to the object.
        
        optDict should be the output of Mpc.solve.
        
        Nblock is the number of moves to accept. Accepted moves are stored into
        the history field. All values are stored into the guess struct.
        """
        
        # Grab data fields.
        try:
            t0 = optDict["t0"]
            t1 = optDict["t1"]
            var = optDict["__var__"]
        except KeyError as err:
            print("Invalid input. Be sure to use output of Mpc.solve.")
            raise err
        
        # Get list of variables that have a "final" entry.
        allvars = var.keys()
        finalvars = [v for v in allvars if v in ["x","z"]]
        nonfinalvars = [v for v in allvars if v not in finalvars]        
        
        # Save initial block of control actions and next states.
        for dt in range(Nblock):        
            for k in nonfinalvars:
                self.__history[k,t0+dt] = var[k,t0+dt]
            for k in finalvars:
                self.__history[k,t0+dt+1] = var[k,t0+dt+1]
        
        # Save full trajectory as guess.
        for t in range(t0,t1):
            for k in allvars:
                self.__guess[k,t] = var[k,t]
        for k in finalvars:
            self.__guess[k,t1] = var[k,t1]
    
    def setContinuousModel(self,f,g,method="colloc",M=None):
        """
        Store state-space model and specify how to approximate.
        """ 
        #shapes = mpc.getVarShapes(f,self.var,disturbancemodel=(self.Nd is not None))
        
        raise NotImplementedError("Not done yet.")        
                     
# ================
# Helper Functions
# ================

def checkPosInt(i):
    """
    Makes sure i is a positive integer. Issues ValueError otherwise.
    """
    if round(i) != i or i <= 0:
        raise ValueError("Input must be a positive integer!")

    return i        
        