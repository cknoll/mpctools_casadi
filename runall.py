"""
Runs all the example files in mpc-tools-casadi showing only success or failure
By default, pdf plots are created, although this can be prevented by running
the script with an 'n' option, e.g.

    python runall.py -n

Note that you will be unable to see any plots in this case.
"""
import sys, traceback
import matplotlib.pyplot as plt
import mpctools.solvers, mpctools.plots
from mpctools.util import stdout_redirected, strcolor, dummy_context
import casadi

# Turn off output
mpctools.solvers.setMaxVerbosity(0)
mpctools.plots.SHOW_FIGURE_WINDOWS = False
choice = "y" if len(sys.argv) < 2 else sys.argv[1]
mpctools.plots.SHOWANDSAVE_DEFAULT_CHOICE = choice

logfile = open("runall.log","w")

# List of files. We hard-code these so that we explicitly pick everything.
examplefiles = []
if casadi.has_conic("qpoases"):
    examplefiles.append("mpcexampleclosedloop.py")
if casadi.has_nlpsol("bonmin"):
    examplefiles += ["fishing.py", "cargears.py"]
examplefiles += [
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
    "mpcmodelcomparison.py",
    "nmheexample.py",
    "nmpcexample.py",
    "periodicmpcexample.py",
    "vdposcillator.py",
    "predatorprey.py",
    "softconstraints.py",
    "sstargexample.py",
    "customconstraints.py",
]
showoutput = set(["fishing.py", "cargears.py"])

# Now loop through everybody.
plt.ioff()
print __doc__
abort = False
for f in examplefiles:
    print strcolor("%s ... " % (f,), "blue"),    
    try:
        context = dummy_context if f in showoutput else stdout_redirected
        with context():
            execfile(f, {}) # Run files in dummy namespace.
            status = strcolor("succeeded", "green", bold=True)
    except KeyboardInterrupt:
        status =  "\n\n%s\n" % (strcolor("*** USER INTERRUPT ***", "yellow"),) 
        abort = True
    except:
        err = sys.exc_info()
        logfile.write("*** Error running <%s>:\n" % (f,))    
        traceback.print_exc(file=logfile)
        status = strcolor("FAILED", "red", bold=True)
    finally:
        print "%s" % (status,)
        plt.close("all")
        if abort:
            break
    import pdb; pdb.set_trace()

plt.ion()
logfile.close()
        