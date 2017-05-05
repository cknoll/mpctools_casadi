# Nonlinear Model Predictive Control Tools for Casadi (mpc-tools-casadi) #

Copyright (C) 2015

Michael J. Risbeck, Nishith R. Patel, and James B. Rawlings.

mpc-tools-casadi is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by the
Free Software Foundation; either version 3, or (at your option) any later
version.

mpc-tools-casadi is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the file
COPYING for more details.

## Availability ##

The latest development sources of mpc-tools-casadi are also available via
anonymous access to a read-only Mercurial archive. There is also a web
interface to the archive available at
<https://bitbucket.org/rawlings-group/mpc-tools-casadi>

## Installation ##

To use mpc-tools-casadi, you will need a recent versions of

* Python 2.7
* Numpy
* Scipy
* Matplotlib
* Tkinter (only needed for `*_mpcsim.py` examples)
* Casadi (Version >=3.0; download from <http://files.casadi.org>)

With these packages installed, mpc-tools-casadi can be downloaded from the
website above, and the mpctools folder can be manually placed in the user's
Python path, or the provided setup script mpctoolssetup.py can be used, e.g.,

    python mpctoolssetup.py install --user

to install for the current user only, or

    sudo python mpctoolssetup.py install

to install systemwide.

Code is used by importing mpctools within python scripts. See sample
files for  complete examples.

## Bugs ##

Questions, comments, bug reports, and contributions should be sent to
risbeck@wisc.edu.

## Documentation ##

Documentation for mpc-tools-casadi is included in each function. We also
provide a cheatsheet (doc/cheatsheet.pdf). See sample files for complete
examples.

Michael J. Risbeck  
<risbeck@wisc.edu>  
University of Wisconsin-Madison  
Department of Chemical Engineering
