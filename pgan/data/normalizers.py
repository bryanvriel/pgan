#-*- coding: utf-8 -*-

from collections import OrderedDict
import numpy as np

class Normalizer:
    """
    Simple convenience class that performs transformations to/from normalized values.
    Here, we use the norm range [-1, 1] for pos=False or [0, 1] for pos=True.
    """

    def __init__(self, xmin, xmax, pos=False, log=False):
        self.xmin = xmin
        self.xmax = xmax
        self.denom = xmax - xmin
        self.pos = pos
        self.log = log
        self.log_eps = 0.05

    def __call__(self, x):
        """
        Alias for Normalizer.forward()
        """
        return self.forward(x)

    def forward(self, x):
        """
        Normalize data.
        """
        if self.pos:
            return (x - self.xmin) / self.denom
        elif self.log:
            xn = (x - self.xmin + self.log_eps) / self.denom
            return np.log(xn)
        else:
            return 2.0 * (x - self.xmin) / self.denom - 1.0

    def inverse(self, xn):
        """
        Un-normalize data.
        """
        if self.pos:
            return self.denom * xn + self.xmin
        elif self.log:
            return self.denom * np.exp(xn) + self.xmin - self.log_eps
        else:
            return 0.5 * self.denom * (xn + 1.0) + self.xmin


# end of file
