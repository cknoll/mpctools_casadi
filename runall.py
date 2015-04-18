# Runs all of the example files.
import sys, os, traceback
import matplotlib.pyplot as plt
import mpctools.solvers, mpctools.legacy.solvers

mpctools.solvers.setMaxVerbosity(0)
mpctools.legacy.solvers.setMaxVerbosity(0)

stdout = sys.__stdout__
devnull = open(os.devnull,"w")
logfile = open("runall.log","w")

# List of files. We hard-code these so that we explicitly pick everything.
examplefiles = [
    "cstr.py",
    "cstr_startup.py",
    "cstr_nonlinear.py",
    "collocationexample.py",
    "comparison_casadi.py",
    "comparison_mtc.py",
    "mheexample.py",
    "mheexample_legacy.py",
    "mpcexampleclosedloop.py",
    "mpcmodelcomparison.py",
    "mpcoopexample.py",
    "nmheexample.py",
    "nmpcexample.py",
    "periodicmpcexample.py",
    "vdposcillator.py",
]

for f in examplefiles:
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

logfile.close()        