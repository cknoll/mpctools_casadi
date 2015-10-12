The Python script casadiinstaller.py is written to help automate some of the
installation process. To use it, you should download the appropriate .zip/.tar
file from casadi.org (filename is something like casadi-py27-np1.9.1-v2.4.0.zip)
to this folder, and then invoke the installer with the command

    python casadiinstaller.py casadi-py27-np1.9.1-v2.4.0.zip

where you replace the second argument with the name of the actual file you
downloaded. This will unzip some files and then make a new Python script called
setup.py that can be invoked, e.g., as

    python setup.py install

to install Casadi to the appropriate custom packages folder on your machine.
