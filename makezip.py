"""Adds all distribution files to a zip file for upload to Bitbucket."""

import argparse
import os
import sys
import zipfile

# Command-line arguments.
# TODO: add option to convert to windows newlines in .txt files.
parser = argparse.ArgumentParser(description=__doc__, add_help=False)
parser.add_argument("--help", help="print this help", action="help")
group = parser.add_mutually_exclusive_group()
group.add_argument("--root-folder", help="name for root folder in zip file")
group.add_argument("--no-root-folder", action="store_true",
                   help="don't include root folder in zip file")
parser.add_argument("--name", help="specify name for zip file",
                    default="mpc-tools-casadi.zip")
kwargs = vars(parser.parse_args(sys.argv[1:]))

# Specify files explicitly. Yes, wildcards would be faster and less prone to
# accidental omissions, but we want to be very explicit here.
files = [
    # Core mpctools files.
    "mpctools/__init__.py",
    "mpctools/colloc.py",
    "mpctools/plots.py",
    "mpctools/solvers.py",
    "mpctools/tools.py",
    "mpctools/util.py",
    
    # Example scripts.
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
    "runall.py",
    
    # Documentation.
    "doc/install.pdf",
    "doc/cheatsheet.pdf",
    "doc/introslides.pdf",
    "doc/octave-vs-python.pdf",
    
    # Casadi installer.
    "installer/casadiinstaller.py",
    "installer/casadisetup.py",
    "installer/README.txt",
    
    # Matlab/Octave files.
    "cstr.m",
    "cstr-matlab/main.m",
    "cstr-matlab/massenbal.m",
    "cstr-matlab/massenbalstst.m",
    "cstr-matlab/partial.m",
    
    # Miscellaneous files.
    "COPYING.txt", # Readme is handled specially.
    "mpctoolssetup.py",
]

# Get name of zip file and decide if there should be a root folder.
zipname = kwargs["name"]
if kwargs["no_root_folder"]:
    root = ""
else:
    root = kwargs["root_folder"]
    if root is None:
        root = os.path.splitext(os.path.split(zipname)[1])[0]

# Now add files.
with zipfile.ZipFile(zipname, "w", zipfile.ZIP_DEFLATED) as vizip:
    # Recurse through VI directories.
    for fl in files:
        readfile = fl
        writefile = os.path.join(root, fl)
        vizip.write(readfile, writefile)
    
    # Also add readme with txt extension to play nice with Windows.
    vizip.write("README.md", os.path.join(root, "README.txt"))
    print "Wrote zip file '%s'." % vizip.filename
