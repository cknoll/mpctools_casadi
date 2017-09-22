# MPCTools: Nonlinear Model Predictive Control Tools for CasADi (Python Interface) #

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

The most recent release of MPCTools is available for download from the
[Downloads][bbdownloads] section. Choose the appropriate version for Python 2
or 3. The development sources are hosted in a Mercurial repository on 
[Bitbucket][bitbucket].

## Installation ##

To use MPCTools, you will need a recent versions of

* Python 2.7 or 3.5+ (see below for Python 3 support)
* Numpy
* Scipy
* Matplotlib
* Tkinter (only needed for `*_mpcsim.py` examples)
* CasADi (Version >=3.0; [download here](http://files.casadi.org))

With these packages installed, MPCTools can be downloaded from the
[downloads][bbdownloads] section, and the `mpctools` folder can be manually 
placed in the user's Python path, or the provided setup script
`mpctoolssetup.py` can be used, e.g.,

    python mpctoolssetup.py install --user

to install for the current user only, or

    sudo python mpctoolssetup.py install

to install systemwide.

Code is used by importing `mpctools` within python scripts. See sample
files for complete examples.

### Python 3 Support ###

Support for Python 3.4+ has been added on an experimental basis. To use
MPCTools with Python 3, you will need to download the Python 3 zip from the
[Downloads][bbdownloads] section.

The Python 3 files are generated automatically from the Python 2 sources using
Python's `2to3` conversion utility. This translation seems to work, but there
may be subtle bugs. Please report any issues you discover.

## Documentation ##

Documentation for MPCTools is included in each function. We also
provide a cheatsheet (`doc/cheatsheet.pdf`). See sample files for complete
examples.

## Citing MPCTools ##

Because MPCTools is primarily an interface to CasADi, you should cite CasADi as
described on its [website][casadipubs]. In addition, you can cite MPCTools as

- Risbeck, M.J., Rawlings, J.B., 2015. MPCTools: Nonlinear model predictive
  control tools for CasADi (Python interface).
  `https://bitbucket.org/rawlings-group/mpc-tools-casadi`.

## Bugs ##

Questions, comments, bug reports can be posted on the
[issue tracker][bbissues] on Bitbucket.

Michael J. Risbeck  
<risbeck@wisc.edu>  
University of Wisconsin-Madison  
Department of Chemical Engineering

[bitbucket]: https://bitbucket.org/rawlings-group/mpc-tools-casadi
[bbissues]: https://bitbucket.org/rawlings-group/mpc-tools-casadi/issues
[bbdownloads]: https://bitbucket.org/rawlings-group/mpc-tools-casadi/downloads
[casadi]: https://casadi.org
[casadipubs]: https://github.com/casadi/casadi/wiki/Publications
[casadidownloads]: https://files.casadi.org
