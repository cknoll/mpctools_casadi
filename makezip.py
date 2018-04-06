#!/usr/bin/env python3
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
parser.add_argument("files", default=[], nargs="*",
                    help="Files to include")
kwargs = vars(parser.parse_args(sys.argv[1:]))

# Files are read from command line via make.
files = set(kwargs["files"])
includereadme = ("README.md" in files)
if includereadme:
    files.remove("README.md")

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
    if includereadme:
        vizip.write("README.md", os.path.join(root, "README.txt"))
    print("Wrote zip file '%s'." % vizip.filename)

