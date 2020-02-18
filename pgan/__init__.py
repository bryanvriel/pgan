#-*- coding: utf-8 -*-

# The submodules
from . import networks
from . import dynamics
from . import data
from . import math
from . import logging

# Tensorflow compatibility
from . import tensorflow

# Make action public
from .components import action

def main():
    """
    The main entrypoint to pgan using the plexus.
    """
    return pgan.run()

def boot():
    """
    Internal function to create the plexus and initialize the dashboard. Used the
    merlin package as a template.
    """
    # Access the plexus factory
    from .components import pgan
    # Build one
    plexus = pgan(name='pgan.plexus')

    # Get the dashboard
    from .components import dashboard
    # Attach the singletons
    import weakref
    dashboard.pgan = weakref.proxy(plexus)

    return plexus

# Call boot() to get a pgan plexus
pgan = boot()

# Meta information
from . import meta

# end of file
