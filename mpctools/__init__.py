"""
Import common functions here to make things a bit easier for the users.

Also check the version of Casadi and make sure it is new enough but not too
new (due to backward incompatibility in Casadi).
"""

# Version for mpctools.
__version__ = "2.3.1"

# Check Casadi version to make sure it's up to date.
_MIN_CASADI_VERSION = "3.0.0-rc3"
_MAX_CASADI_VERSION = "4.0"
def _getVersion(vstring, N=3):
    """Returns a tuple with version number."""
    parts = vstring.split(".")
    parts = parts[:-1] + parts[-1].split("-")
    if parts[-1].startswith("rc"):
        rc = parts.pop(-1)[2:]
    else:
        rc = "inf"
    parts = parts + ["0"]*(N - len(parts))
    if len(parts) > N:
        raise ValueError("Too many parts found (found %d, expected %d)!"
                         % (len(parts), N))
    parts.append(rc)
    return tuple([float(p) for p in parts])
import casadi
if _getVersion(casadi.__version__) < _getVersion(_MIN_CASADI_VERSION):
    raise ImportError("casadi version %s is too old (must be >=%s)"
        % (casadi.__version__, _MIN_CASADI_VERSION))
elif _getVersion(casadi.__version__) > _getVersion(_MAX_CASADI_VERSION):
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
vcat = tools.safevertcat
