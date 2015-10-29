"""
Import common functions here to make things a bit easier for the users.

Also check the version of Casadi and make sure it is new enough.
"""

MIN_CASADI_VERSION = "2.4"
from distutils.version import LooseVersion as getVersion
import casadi
if getVersion(casadi.__version__) < getVersion(MIN_CASADI_VERSION):
    raise ImportError("casadi version %s is too old (must be >=%s)"
        % (casadi.__version__, MIN_CASADI_VERSION))

import tools
import plots
import util
import colloc
from tools import nmpc, nmhe, sstarg, getCasadiFunc, DiscreteSimulator
from util import keyboard, mtimes, ekf
