#!/usr/bin/env python3
"""Adds all distribution files to a zip file for upload to Bitbucket."""

import argparse
import os
import sys
import zipfile
import mpctools.hg

# Command-line arguments.
parser = argparse.ArgumentParser(description=__doc__, add_help=False)
parser.add_argument("--help", help="print this help", action="help")
group = parser.add_mutually_exclusive_group()
group.add_argument("--root-folder", help="name for root folder in zip file")
group.add_argument("--no-root-folder", action="store_true",
                   help="don't include root folder in zip file")
parser.add_argument("--name", help="specify name for zip file",
                    default="mpc-tools-casadi.zip")
parser.add_argument("--python2", help="make distribution for Python 2.7",
                    action="store_true")
parser.add_argument("--windows", help="use Windows newlines in text files",
                    action="store_true")
parser.add_argument("files", default=[], nargs="*",
                    help="Files to include")

# Constants.
CHANGESET_ID = mpctools.hg.get_changeset_id()
PYTHON_2_HEADER = "from __future__ import division, print_function"


# Helper functions.
def clean_py_file(file, python2=False):
    """Iterator for cleaned file."""
    if python2:
        yield PYTHON_2_HEADER
    with open(file, "r") as read:
        for line in read:
            line = read.rstrip()
            if "#CHANGESET_ID" in line:
                pad = line[:len(line) - len(line.rstrip())]
                line = "{}changeset_id = {}".format(pad, CHANGESET_ID)
            if python2 and "from .compat import" in line:
                continue
            yield line


def clean_txt_file(file):
    """Iterator for cleaned txt file."""
    with open(file, "r") as read:
        for line in read:
            yield line


def makefileolderthan(target, relto, delta=1, changeatime=False,
                      mustexist=False):
    """
    Sets modification time of target to be older than relto.
    
    Argument delta (default 1) gives the time difference to use. Argument
    chanteatime decides whether to also change the access time.
    
    If target does not exist and mustexist is False, nothing happens; if
    mustexist is True, then an error is raised.
    
    Returns mtime if set, otherwise None.
    """
    if os.path.isfile(target):
        mtime = os.path.getmtime(relto) - delta
        atime = mtime if changeatime else os.path.getatime(relto)
        os.utime(target, (atime, mtime))
    elif mustexist:
        raise IOError("File {} does not exist!".format(target))
    else:
        mtime = None
    return mtime


# Main function.
def main(files, zipname, root="", python2=False, newline="\n"):
    """Writes the zip file."""
    includereadme = ("README.md" in files)
    if includereadme:
        files.remove("README.md")
    files.discard("mpctools/hg.py")
    if python2:
        files.discard("mpctools/compat.py")
    if root is None:
        root = os.path.splitext(os.path.split(zipname)[1])[0]
    
    # Now add files.
    with zipfile.ZipFile(zipname, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            readfile = f
            writefile = os.path.join(root, f)
            if f.endswith(".py"):
                z.writestr(writefile,
                           newline.join(clean_py_file(writefile,
                                                      python2=python2)))
            elif f.endswith(".pdf"):
                z.write(readfile, writefile)
            else:
                z.writestr(writefile, newline.join(clean_txt_file(readfile)))
            
            z.write(readfile, writefile)
        
        # Also add readme with txt extension to play nice with Windows.
        if includereadme:
            z.write("README.md", os.path.join(root, "README.txt"))
        print("Wrote zip file '%s'." % z.filename)


# Script logic.
if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    if len(args.files) == 0:
        raise ValueError("Must provide at least 1 file!")
    if args.no_root_folder:
        root = ""
    else:
        root = args.root_folder
    try:
        main(args.files, args.name, root=root, python2=args.python2,
             newline="\r\n" if args.windows else "\n")
    except Exception as exc:
        makefileolderthan(args.name, args.files[0])
        raise RuntimeError("Error writing zip!") from exc
