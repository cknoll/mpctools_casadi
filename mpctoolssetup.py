#!/usr/bin/env python
from distutils.core import setup
import mpctools

with open("README.md", "r") as f:
    description = "".join(f)

setup(name="mpc-tools-casadi",
    version=mpctools.__version__,
    description="Nonlinear MPC tools for use with CasADi",
    author="Michael Risbeck",
    author_email="risbeck@wisc.edu",
    url="https://hg.cae.wisc.edu/hg/mpc-tools-casadi",
    long_description=description,
    packages=["mpctools"],
    platforms=["N/A"],
    license="GNU GPLv3", 
)
