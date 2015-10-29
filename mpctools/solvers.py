from __future__ import print_function, division # Grab some handy Python3 stuff.
import numpy as np
import util
import casadi
import time

"""
Holds solver interfaces, wrappers, and class definitions.
"""

_MAX_VERBOSITY = 100

def setMaxVerbosity(verb=100):
    """
    Sets a module override for maximum verbosity setting.
    """
    global _MAX_VERBOSITY
    _MAX_VERBOSITY = verb


def callSolver(var,varlb,varub,varguess,obj,con,conlb,conub,par=None,
    parval=None,verbosity=5,timelimit=60,isQp=False,runOptimization=True,
    scalar=True):
    """
    Calls ipopt to solve an NLP.
    
    Arguments are mostly self-explainatory. var, varlb, varub, and varguess
    should be casadi struct_symMX objects, e.g. the outputs of getCasadiVars.
    obj should be a scalar casadi MX object, and con should be a vector casadi
    MX object (possibly from calling casadi.vertcat on a list of constraints).
    conlb and conub should be numpy vectors of the appropriate size.
    
    Returns the optimal variables, the objective function, status string, and
    the ControlSolver object.
    """
    solver = ControlSolver(var,varlb,varub,varguess,obj,con,conlb,conub,par,
        parval,verbosity,timelimit,isQp,scalar)
    
    # Solve if requested and get variables.
    if runOptimization:    
        solver.solve()
        status = solver.stats["status"]
        obj = solver.obj
    else:
        status = "NO_OPTIMIZATION_REQUESTED"
        obj = np.inf
    optvar = solver.var
     
    return [optvar, obj, status, solver]
    

class ControlSolver(object):
    """
    A simple class for holding a casadi solver object.
    
    Users have access parameter and guess fields to adjust parameters on the
    fly and reuse past trajectories in subsequent optimizations, etc.
    """
    
    # We use the attributes __varguess and __parval to store the values for
    # guesses and parameters, and the attributes __var and __par for the
    # casadi symbolic structures. Users should never need to access __var and
    # __par, so these aren't properties. We do expose __varguess and __parval.
    # After an optimization, __varval holds the optimal values of variables,
    # and this is exposed through the property var.
    
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
    def conlb(self):
        return self.__conlb
        
    @property
    def conub(self):
        return self.__conub
    
    @property
    def par(self):
        return self.__parval
    
    @property
    def var(self):
        return self.__varval
        
    @property
    def obj(self):
        return self.__objval
    
    @property
    def stats(self):
        return self.__stats
    
    @property
    def verbosity(self):
        return self.__settings["verbosity"]
        
    @verbosity.setter
    def verbosity(self, v):
        v = min(min(max(v, -1), 12), _MAX_VERBOSITY)
        self.__changesettings(verbosity=v)
    
    @property
    def name(self):
        return self.__settings["name"]
        
    @name.setter
    def name(self, n):
        self.__changesettings(name=n)
        
    @property
    def timelimit(self):
        return self.__settings["timelimit"]
        
    @timelimit.setter
    def timelimit(self, t):
        self.__changesettings(timelimit=t)
    
    @property
    def isQP(self):
        return self.__settings["isQP"]
    
    @isQP.setter
    def isQp(self, tf):
        self.__changesettings(isQP=tf)
    
    def __changesetting(self, **settings):
        """
        Changes fields of the settings dictionary and sets update flag.
        """
        self.__settings.update(settings)
        self.__changed = True
    
    
    def __init__(self, var, varlb, varub, varguess, obj, con, conlb, conub,
                 par=None, parval=None, verbosity=5, timelimit=60, isQP=False,
                 scalar=True, name="ControlSolver", ipoptoptions={}):
        """
        Initialize the solver object.
        
        These arguments should be almost identical to callSolver, which is
        simply a functional wrapper for this class.
        """
        
        # First store everybody to the object.
        self.__var = var
        self.__lb = varlb
        self.__ub = varub
        self.__guess = varguess
        
        self.__obj = obj
        self.__con = con
        self.__conlb = conlb
        self.__conub = conub
        
        self.__par = par
        self.__parval = parval
        
        self.__stats = {}
        self.__settings = {} # Need to initialize this.
        self.__changesetting(scalar=scalar, isQP=isQP, name=name,
                               verbosity=verbosity, timelimit=timelimit)
        
        # Now initialize the solver object.
        self.initializeSolver(**ipoptoptions)        
        
    def initializeSolver(self, name=None, **options):
        """
        Recreates the solver object completely.
        
        You shouldn't need to do this manually unless you are changing internal
        ipopt options (via keyword arguments).
        
        For a complete list of ipopt options, use availableIpoptOptions. Note
        that all of these are either strings or floats, and any boolean values
        will likely cause errors.
        """
        if name is not None:
            self.name = name
        
        if self.__settings["scalar"]:
            XFunction = casadi.SXFunction
        else:
            XFunction = casadi.MXFunction
        
        nlpInputs = {"x" : self.__var}
        if self.__par is not None:
            nlpInputs["p"] = self.__par
        nlpOutputs = {"f" : self.__obj, "g" : self.__con}
        
        nlp = XFunction(self.name + "_nlp", casadi.nlpIn(**nlpInputs),
                        casadi.nlpOut(**nlpOutputs))

        # Because of Casadi changes in Version 2.4, we have to build up all the
        # options and pass them to the constructor. We build up a list of
        # defaults first, and then add any user options.
        
        # Print ant time limit options. Note that we must respect Ipopt's
        # limits with print_level, which are different from ours.
        options["print_level"] =  min(12, max(0, self.verbosity))
        options["print_time"] = self.verbosity > 2  
        options["max_cpu_time"] = self.timelimit        
        
        # Note that there is an option "check_derivatives_for_naninf" that in
        # theory would error out if NaNs or Infs are encountered, but it seems
        # to just crash Python whenever anything bad happens.
        options["eval_errors_fatal"] = True
                
        # Options if problem is a QP.
        if self.__settings["isQP"]:
            options["hessian_constant"] = "yes"
            options["jac_c_constant"] = "yes"
            options["jac_d_constant"] = "yes"

        # Finally, create the solver object.
        solver = casadi.NlpSolver(self.name, "ipopt", nlp, options)
        
        ## Prior to V2.4, we could check option names here. Since they're
        ## handled at creation time, I'm not sure how best to trap those
        ## errors. I leave the old code commented below. - MJR (10/29/2015)
#        for (k,v) in kwargs.iteritems():
#            try:
#                solver.setOption(k,v)
#            except RuntimeError as err:
#                msg = [m.replace("printOptions",
#                                 "ControlSolver.availableIpoptOptions")
#                                 for m in err.message.split("\n")[1:]]
#                raise RuntimeError("\n".join(msg))
#        solver.init()
        
        # Finally, save the solver and unset the changed flag.
        self.__solver = solver
        self.__changed = False
    
    def availableIpoptOptions(self, display=True):
        """
        Returns a dictionary of ipopt options and prints web link.
        
        Dictionary entry "__web__" includes the link to the online ipopt
        documentation, which may be better. Set display to False to suppress
        printing.
        """
        names = self.__solver.getOptionNames()
        options = {}
        for n in names:
            o = self.__solver.getOptionDescription(n)
            o = o.replace("(see IPOPT documentation)","").strip()
            options[n] = o
        options["__web__"] = ("http://www.coin-or.org/"
            "Ipopt/documentation/node39.html")
        if display:
            print("See\n\n    %s\n\nfor details about IPOPT options."
                % (options["__web__"],)) 
            print("\nOptions can be set using keyword arguments to"
                " ControlSolver.initializeSolver.\n")
        return options
    
    def solve(self):
        """
        Solve the current solver object.
        """
        # Solve the problem and get optimal variables.
        starttime = time.clock()
        if self.__changed:
            self.initializeSolver()
        solver = self.__solver
        
        # Now set guess and bounds.
        solver.setInput(self.guess,"x0")
        solver.setInput(self.lb,"lbx")
        solver.setInput(self.ub,"ubx")
        solver.setInput(self.conlb,"lbg")
        solver.setInput(self.conub,"ubg")
        if self.par is not None:
            solver.setInput(self.par,"p")
        
        # Need something special to prevent c code from printing; in
        # particular, we want to suppress Ipopt's splash message if
        # verbosity <= -1. Note that this redirection can have some weird
        # side-effects, so that's why we don't do it for verbosity = 0.
        if self.verbosity <= -1:
            printcontext = util.stdout_redirected
        else:
            printcontext = util.dummy_context
        with printcontext():
            solver.evaluate()
        self.__varval = self.__var(solver.getOutput("x"))
        self.__objval = float(solver.getOutput("f"))
        endtime = time.clock()
        
        # Grab some stats.
        status = solver.getStat("return_status")
        if self.verbosity > 0:
            print("Solver Status:", status)
            if status == "NonIpopt_Exception_Thrown":
                print("***Warning: NaN or Inf encountered during function "
                    "evaluation.")
        if self.verbosity > 1:
            print("Took %g s." % (endtime - starttime,))
        self.stats["status"] = status
        self.stats["time"] = endtime - starttime
         
    def saveguess(self,toffset=1):
        """
        Stores the vales from the from the last optimization as a guess.
        
        This is useful to store the results of the previous step as a guess for
        the next step. toffset defaults to 1, which means the time indices are
        shifted by 1, but you can set this to whatever you want.
        """
        for k in self.var.keys():
            # These values and the guess structure can potentially have a
            # different number of time points. So, we need to figure out
            # what range of times will be valid for both things. The offset
            # makes things a bit weird.
            tmaxVal = len(self.var[k])
            tmaxGuess = len(self.guess[k]) + toffset
            tmax = min(tmaxVal,tmaxGuess)
            
            tmin = max(toffset,0)
            
            # Now actually store the stuff.           
            for t in range(tmin,tmax):
                self.guess[k,t-toffset] = self.var[k,t]
    
    def fixvar(self,var,t,val,indices=None):
        """
        Fixes variable var at time t to val.
        
        Indices can be specified as a list to fix only a subset of values.
        """
        
        if indices is None:
            self.lb[var,t] = val
            self.ub[var,t] = val
            self.guess[var,t] = val
        else:
            self.lb[var,t,indices] = val
            self.ub[var,t,indices] = val
            self.guess[var,t,indices] = val
        
        self.__solver.setInput(self.lb,"lbx")
        self.__solver.setInput(self.ub,"ubx")
        self.__solver.setInput(self.guess,"x0")       
