# Runs all of the example files.
import sys, os, traceback
import matplotlib.pyplot as plt
import mpctools.solvers, mpctools.plots

mpctools.solvers.setMaxVerbosity(0)
mpctools.plots.SHOW_FIGURE_WINDOWS = False
mpctools.plots.SHOWANDSAVE_DEFAULT_CHOICE = "y" # Save figures without prompting.

stdout = sys.__stdout__
devnull = open(os.devnull,"w")
logfile = open("runall.log","w")

# List of files. We hard-code these so that we explicitly pick everything.
# Note: The commented-out files have yet to be updated viz. removal of legacy.
examplefiles = [
    "cstr.py",
    "cstr_startup.py",
    "cstr_nmpc_nmhe.py",
    "collocationexample.py",
    "comparison_casadi.py",
    "comparison_mtc.py",
    "example2-8.py",
    "mheexample.py",
    "mpcexampleclosedloop.py",
    "mpcmodelcomparison.py",
    "nmheexample.py",
    "nmpcexample.py",
    "periodicmpcexample.py",
    "vdposcillator.py",
    "ballmaze.py",
    "econmpc.py",
    "airplane.py",
]

for f in examplefiles:
    plt.ioff()
    print "%s ... " % (f,),
    sys.stdout = devnull
    try:
        execfile(f)
        status = "succeeded"
    except:
        err = sys.exc_info()
        logfile.write("*** Error running <%s>:\n" % (f,))    
        traceback.print_exc(file=logfile)
        status = "FAILED"
    sys.stdout = stdout
    print "%s." % (status,)
    plt.close('all')

plt.ion()
logfile.close()
        