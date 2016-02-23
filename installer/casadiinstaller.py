# Full help message generated by argparse.
"""
With the default structure, Casadi splits its files between three different
directories (casadi, include, and lib). This makes it potentially difficult
to completely remove casadi because you have to either track down every
Casadi file inside include and lib, or delete those entire directories and risk
losing other important files.

This script puts a simple wrapper around Casadi so that all of its files can be
within a single, easily removable directory.
"""

import argparse
import os
import shutil
import sys

CASADI_SETUP_SCRIPT = "casadisetup.py"

# An argument checking functions.
def archivefile(f, extensions=(".zip",".tar.gz")):
    """Checks if f is some kind of archive file that exists."""
    f = str(f).strip()
    if not any([f.endswith(ext) for ext in extensions]):
        raise argparse.ArgumentError("Not an archive file!")
    elif not os.path.isfile(f):
        raise argparse.ArgumentError("File %s does not exist!" % (f,))
    return f
    
# Parser for command line arguments.
parser = argparse.ArgumentParser(add_help=False, description=
    "Creates file structure to install Casadi inside a single directory.",
    epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument("--help", help="print this help",action="help")
parser.add_argument("--noimport", help="Don't try to import the package",
                    action="store_true")
parser.add_argument("file",help="CasADi download file", type=archivefile)

# Parse command line arguments.
options = vars(parser.parse_args(sys.argv[1:]))
filename = options["file"]
tryimport = not options["noimport"]

# Remove any existing directory and setup script.
casadifolders = ["casadi", "lib", "include"]
for f in casadifolders:
    if os.path.isdir(f):
        shutil.rmtree(f)
if os.path.isfile(CASADI_SETUP_SCRIPT):
    os.remove(CASADI_SETUP_SCRIPT)

# Unzip archive.
print "*** Unzipping <%s> ***" % (filename,)
extrafiles = []
if filename.endswith(".zip"):
    import zipfile
    with zipfile.ZipFile(filename, "r") as z:
        z.extractall()
        casadifiles = z.infolist()
        for c in casadifiles:
            if not os.path.isdir(c.filename):            
                extrafiles.append(os.path.normpath(c.filename))
elif filename.endswith("tar.gz"):
    import tarfile
    with tarfile.open(filename, "r") as t:
        t.extractall()
        casadifiles = t.getmembers()
        for c in casadifiles:
            if c.isfile():
                extrafiles.append(os.path.normpath(c.name))
else:
    raise ValueError("Invalid archive file type!")

# Now try to import the new package.
if tryimport:
    print "*** Attempting to import casadi ***"
    try:
        import casadi
        version = casadi.__version__
    except ImportError as err:
        print err.message
        raise IOError("Unable to import casadi. Something went wrong.")
    print "*** Import successful ***"
else:
    version = ""

# Create setup file. Need a list of all Casadi's internals.
print "*** Creating setup.py ***"
front = "    r'.." + os.sep
back = "',\n"
files =  "\n" + front + (back + front).join(extrafiles) + back
setup = '''"""
Setup script for casadi.

This will install all Casadi files to a single directory, which is handy for
replacing old versions without disturbing any other Python packages.
"""
from distutils.core import setup

files = [{files}]

setup(name="CasADi",
    description="Symbolic framework for automatic differentiation and optimization.",
    author="Joel Andersson",    
    url="casadi.org",
    packages=["{packagename}"],
    package_data=dict({packagename}=files),
    version="{version}",
    license="LGPL",
)
'''.format(packagename="casadi", files=files, version=version)
with open(CASADI_SETUP_SCRIPT, "w") as setupfile:
    setupfile.write(setup)

# Finally, tell the user how to actually install things.
instructions = """
To install casadi for just the current user,

    python casadisetup.py install --user --quiet

To see full verbose output, omit the --quiet option (there may be a lot of
output).

Alternatively, to install systemwide, use

    python casadisetup.py build
    sudo python casadisetup.py install --quiet

For more flexibility, use

    python casadisetup.py install --help

to get a list of additional options.
"""
print instructions
