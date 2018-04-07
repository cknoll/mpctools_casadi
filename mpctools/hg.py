"""
Module to get mercurial changeset id.

Note that this is only used in development sources, and in distributions, the
changeset id is hard-coded.
"""
import os
import subprocess

def get_changeset_id():
    """Returns the mercurial changeset ID for the repo."""
    mpctoolsdir = os.path.dirname(__file__)
    hgdir = os.path.dirname(mpctoolsdir)
    changeset = subprocess.check_output(["hg", "id", "-i"],
                                        cwd=hgdir)
    return changeset.decode().strip()
