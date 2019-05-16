#-*- coding: utf-8 -*-

import pyre
from .Dashboard import Dashboard

class Action(pyre.action, Dashboard, family='pgan.tasks'):
    """
    Protocol for pgan commands.
    """

# end of file
