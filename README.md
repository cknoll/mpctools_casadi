# MPCTools: Nonlinear Model Predictive Control Tools for Casadi (Python Interface) #

Copyright (C) 2017

Michael J. Risbeck and James B. Rawlings.

MPCTools is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by the
Free Software Foundation; either version 3, or (at your option) any later
version.

MPCTools is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the file
COPYING for more details.

## Availability ##

The latest development sources of MPCTools are also available via
anonymous access to a read-only Mercurial archive. There is also a web
interface to the archive available at
[Bitbucket](https://bitbucket.org/rawlings-group/mpc-tools-casadi)

## Installation ##

To use MPCTools, you will need a recent versions of

* Python 2.7
* Numpy
* Scipy
* Matplotlib
* Tkinter (only needed for `*_mpcsim.py` examples)
* Casadi (Version >=3.0; [download here](http://files.casadi.org>))

With these packages installed, MPCTools can be downloaded from the
website above, and the mpctools folder can be manually placed in the user's
Python path, or the provided setup script mpctoolssetup.py can be used, e.g.,

    python mpctoolssetup.py install --user

to install for the current user only, or

    sudo python mpctoolssetup.py install

to install systemwide.

Code is used by importing `mpctools` within python scripts. See sample
files for  complete examples.

## Documentation ##

Documentation for MPCTools is included in each function. We also
provide a cheatsheet (`doc/cheatsheet.pdf`). See sample files for complete
examples.

## Citing MPCTools ##

Because MPCTools is primarily an interface to CasADi, you should cite CasADi as
described on its [website](https://github.com/casadi/casadi/wiki/Publications).
In addition, you can cite MPCTools as

- Risbeck, M.J., Rawlings, J.B., 2015. MPCTools: Nonlinear Model Predictive
  Control Tools for Casadi (Python Interface).
  `https://bitbucket.org/rawlings-group/mpc-tools-casadi`.

## Bugs ##

Questions, comments, bug reports, and contributions should be sent to
risbeck@wisc.edu.

Michael J. Risbeck  
<risbeck@wisc.edu>  
University of Wisconsin-Madison  
Department of Chemical Engineering
