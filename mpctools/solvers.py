from __future__ import print_function, division # Grab some handy Python3 stuff.
import copy
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


def callSolver(solver, verbosity=None):
    """
    Wrapper to ControlSolver.solve() that returns a single dictionary.
    
    This function is mainly for backward-compatibility now that the option
    runOptimization was removed from nmpc, nmhe, etc.
    
    Returns a dictionary with optimal variables as NumPy arrays. Additional
    keys "t", "obj" and "status" are also present. Finally, if the model used
    collocation, an entry "tc" is also present which is an Nt by Nc array of
    time points.
    """
    if verbosity is not None:
        solver.verbosity = verbosity
    solver.solve()

    returnDict = util.casadiStruct2numpyDict(solver.var)
    returnDict["obj"] = solver.obj
    returnDict["status"] = solver.stats["status"]
    
    Delta = solver.misc.get("Delta", 1)
    N = solver.misc["N"]
    returnDict["t"] = Delta*np.arange(N["t"] + 1)
    if "c" in N and N["c"] > 0:
        r = solver.misc["colloc"]["r"][1:-1] # Throw out endpoints.        
        r = r[np.newaxis,:]
        returnDict["tc"] = returnDict["t"][:-1,np.newaxis] + Delta*r    
    
    return returnDict
    

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
    def defaultguess(self):
        return copy.deepcopy(self.__defaultguess)
        
    @defaultguess.setter
    def defaultguess(self, g):
        self.__defaultguess = copy.deepcopy(g)
    
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
    def vardict(self):
        return util.casadiStruct2numpyDict(self.__varval)
    
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
    
    def __changesettings(self, **settings):
        """
        Changes fields of the settings dictionary and sets update flag.
        """
        self.__settings.update(settings)
        self.__changed = True
    
    def __init__(self, var, varlb, varub, varguess, obj, con, conlb, conub,
                 par=None, parval=None, verbosity=5, timelimit=60, isQP=False,
                 casaditype="SX", name="ControlSolver", casadioptions=None,
                 solveroptions=None, misc=None):
        """
        Initialize the solver object.
        
        Arguments are mostly self-explainatory. var, varlb, varub, and varguess
        should be casadi struct_symMX objects, e.g. the outputs of
        getCasadiVars. obj should be a scalar casadi MX object, and con should
        be a vector casadi MX object (possibly from calling casadi.vertcat on a
        list of constraints). conlb and conub should be numpy vectors of the
        appropriate size. misc is a read-only dictionary for storing to hold
        miscellaneous parameters that cannot be changed.
        
        Typically, it's easiest to build these objects using nmpc, nmhe, or
        sstarg from the tools module, all of which return ControlSolver
        objects.
        """
        
        # First store everybody to the object.
        self.__var = var
        self.__lb = varlb
        self.__ub = varub
        self.__guess = varguess
        self.defaultguess = varguess
        
        self.__obj = obj
        self.__con = con
        self.__conlb = conlb
        self.__conub = conub
        
        self.__par = par
        self.__parval = parval
        
        self.__stats = {}
        self.__settings = {} # Need to initialize this.
        self.__changesettings(isQP=isQP, name=name, verbosity=verbosity,
                              timelimit=timelimit)
        if misc is None:
            misc = {}
        self.misc = util.ReadOnlyDict(**misc)
        
        # Now initialize the solver object.
        if casadioptions is None:
            casadioptions = {}
        if solveroptions is None:
            solveroptions = {}
        self.initialize(casadioptions, solveroptions)        
        
    def initialize(self, casadioptions=None, solveroptions=None):
        """
        Recreates the solver object completely.
        
        You shouldn't need to do this manually unless you are changing internal
        casadi or ipopt options (via keyword arguments). Note that casadi
        options are passed as keywords, while ipopt options should be passed
        in a single dictionary as ipopt.
        
        For a complete list of ipopt options, use availableIpoptOptions. Note
        that all of these are either strings or floats, and any boolean values
        will likely cause errors.
        """
        if casadioptions is None:
            casadioptions = {}
        else:
            casadioptions = casadioptions.copy()
        if solveroptions is None:
            solveroptions = {}
        else:
            solveroptions = solveroptions.copy()
        
        nlp = {
            "x" : self.__var,
            "f" : self.__obj,
            "g" : self.__con
        }
        if self.__par is not None:
            nlp["p"] = self.__par
        
        # Print and time limit options. Note that we must respect Ipopt's
        # limits with print_level, which are different from ours.
        solveroptions["print_level"] =  min(12, max(0, self.verbosity))
        solveroptions["max_cpu_time"] = self.timelimit        
        casadioptions["print_time"] = self.verbosity > 2        
        
        # Note that there is an option "check_derivatives_for_naninf" that in
        # theory would error out if NaNs or Infs are encountered, but it seems
        # to just crash Python whenever anything bad happens.
        casadioptions["eval_errors_fatal"] = True
                
        # Choose different function whether QP or not.
        # TODO: allow user to specify solver. - MJR 2/12/2016
        if self.__settings["isQP"]:
            # TODO: handle options for qpoases. - MJR 2/12/2016
            solver = casadi.qpsol(self.name, "qpoases", nlp)
        else:
            casadioptions["ipopt"] = solveroptions
            solver = casadi.nlpsol(self.name, "ipopt", nlp, casadioptions)
        
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
        # TODO: List options for the chosen solver. - MJR 2/12/2016
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
                " ControlSolver.initialize().\n")
        return options
    
    def solve(self):
        """
        Solve the current solver object.
        """
        # Solve the problem and get optimal variables.
        starttime = time.clock()
        if self.__changed:
            self.initialize()
        solver = self.__solver
        
        # Now set guess and bounds.
        solverargs = {
            "x0" : self.guess,
            "lbx" : self.lb,
            "ubx" : self.ub,
            "lbg" : self.conlb,
            "ubg" : self.conub,
        }
        if self.par is not None:
            solverargs["p"] = self.par
        
        # Need something special to prevent c code from printing; in
        # particular, we want to suppress Ipopt's splash message if
        # verbosity <= -1. Note that this redirection can have some weird
        # side-effects, so that's why we don't do it for verbosity = 0.
        if self.verbosity <= -1:
            printcontext = util.stdout_redirected
        else:
            printcontext = util.dummy_context
        with printcontext():
            sol = solver(solverargs)
            stats = solver.stats()            
        self.__varval = self.__var(sol["x"])
        self.__objval = float(sol["f"])
        endtime = time.clock()
        
        # Grab some stats.
        status = stats["return_status"]
        if self.verbosity > 0:
            print("Solver Status:", status)
            if status == "NonIpopt_Exception_Thrown":
                print("***Warning: NaN or Inf encountered during function "
                    "evaluation.")
        if self.verbosity > 1:
            print("Took %g s." % (endtime - starttime,))
        self.stats["status"] = status
        self.stats["time"] = endtime - starttime
         
    def saveguess(self, toffset=None, default=False):
        """
        Stores the vales from the from the last optimization as a guess.
        
        This is useful to store the results of the previous step as a guess for
        the next step. toffset defaults to 1, which means the time indices are
        shifted by 1, but you can set this to whatever you want.
        
        If default is True, then uses the guess stored in self.defaultguess
        instead of the current optimization. Note that toffset defaults to 0
        in this case.
        """
        newguess = self.__defaultguess if default else self.var
        if toffset is None:
            toffset = 0 if default else 1
        for k in self.var.keys():
            # These values and the guess structure can potentially have a
            # different number of time points. So, we need to figure out
            # what range of times will be valid for both things. The offset
            # makes things a bit weird.
            tmaxVal = len(newguess[k])
            tmaxGuess = len(self.guess[k]) + toffset
            tmax = min(tmaxVal,tmaxGuess)
            tmin = max(toffset,0)
            
            # Now actually store the stuff.           
            for t in range(tmin,tmax):
                self.guess[k,t-toffset] = newguess[k,t]
    
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
