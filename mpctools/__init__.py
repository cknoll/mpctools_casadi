"""
Import common functions here to make things a bit easier for the users.

Also check the version of Casadi and make sure it is new enough but not too
new (due to backward incompatibility in Casadi).
"""

# Version for mpctools.
__version__ = "2.2"

# Check Casadi version to make sure it's up to date.
_MIN_CASADI_VERSION = "3.0"
_MAX_CASADI_VERSION = "4.0"
from distutils.version import LooseVersion as getVersion
import casadi
if getVersion(casadi.__version__) < getVersion(_MIN_CASADI_VERSION):
    raise ImportError("casadi version %s is too old (must be >=%s)"
        % (casadi.__version__, _MIN_CASADI_VERSION))
elif getVersion(casadi.__version__) > getVersion(_MAX_CASADI_VERSION):
    raise ImportError("casadi version %s is too new (must be <=%s)"
        % (casadi.__version__, _MAX_CASADI_VERSION))

# Add modules and some specific functions.
import tools
import plots
import util
import colloc
import solvers
from tools import nmpc, nmhe, sstarg, getCasadiFunc, DiscreteSimulator
from util import keyboard, mtimes, ekf
from solvers import callSolver
