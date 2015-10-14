# Runs all of the example files.
import sys, os, traceback
import matplotlib.pyplot as plt
import mpctools.solvers, mpctools.plots

# Turn off output
mpctools.solvers.setMaxVerbosity(0)
mpctools.plots.SHOW_FIGURE_WINDOWS = False
choice = "y" if len(sys.argv) < 2 else sys.argv[1]
mpctools.plots.SHOWANDSAVE_DEFAULT_CHOICE = choice

stdout = sys.__stdout__
devnull = open(os.devnull,"w")
logfile = open("runall.log","w")

# List of files. We hard-code these so that we explicitly pick everything.
examplefiles = [
    "airplane.py",
    "ballmaze.py",
    "cstr.py",
    "cstr_startup.py",
    "cstr_nmpc_nmhe.py",
    "collocationexample.py",
    "comparison_casadi.py",
    "comparison_mtc.py",
    "econmpc.py",    
    "example2-8.py",
    "mheexample.py",
    "mpcexampleclosedloop.py",
    "mpcmodelcomparison.py",
    "nmheexample.py",
    "nmpcexample.py",
    "periodicmpcexample.py",
    "vdposcillator.py",
]

plt.ioff()
for f in examplefiles:
    print "%s ... " % (f,),
    sys.stdout = devnull        
    try:
        execfile(f, {}) # Run files in dummy namespace.
        status = "succeeded"
    except:
        err = sys.exc_info()
        logfile.write("*** Error running <%s>:\n" % (f,))    
        traceback.print_exc(file=logfile)
        status = "FAILED"
    sys.stdout = stdout
    print "%s." % (status,)
    plt.close("all")

plt.ion()
logfile.close()
devnull.close()
        