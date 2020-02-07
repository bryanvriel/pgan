#!/usr/bin/env python3

import pgan.tensorflow as tf

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

def leaky_elu(x, slope=0.2):
    """
    Custom activation function that combines leaky ReLU + exponential LU.
    """
    cond = tf.greater_equal(x, tf.constant(0.0))
    pos_values = x
    neg_values = tf.exp(x) + slope * x - tf.exp(0.0)
    return tf.where(cond, pos_values, neg_values)

# end of file
