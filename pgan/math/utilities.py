#!/usr/bin/env python3

def normalize(x, xmin, xmax):
    """
    Normalize data to be roughly in the range of [-1, 1].
    """
    return 2.0 * (x - xmin) / (xmax - xmin) - 1.0

def identity(x, *args):
    """
    Do nothing. Simply return input.
    """
    return x

# end of file
