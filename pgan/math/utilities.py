#!/usr/bin/env python3

import numpy as np
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

def gaussian_smoothing_kernel(win_size=9):

    # Construct kernel
    if win_size == 5:
        kernel = np.array([0.06136, 0.24477, 0.38774, 0.24477, 0.06136])
    elif win_size == 9:
        kernel = np.array([0.029, 0.067, 0.124, 0.179, 0.202, 0.179, 0.124, 0.067, 0.029])
    else:
        raise ValueError('Unsupported window size for Gaussian kernel.')
    
    # Normalize
    kernel /= np.sum(kernel)

    # Return as tensorflow tensor with shape [filter_width, in_channels, out_channels]
    return tf.convert_to_tensor(kernel.reshape(-1, 1, 1).astype(np.float32))

def smoothe1d(x, filters):
    """
    Performs 1D convolution for smoothing.
    """
    # Reshape input to include channels dimension 
    xn = tf.expand_dims(x, axis=-1)
    
    # Perform 1D convolution (NOTE: result will have edge effects)
    out = tf.nn.conv1d(xn, filters=filters, stride=1, padding='SAME', data_format='NWC')

    # Squeeze dimensions and return
    return out[:, :, 0]

def grad1d(x, D):
    """
    Multiplies finite difference operator with tensor.
    """
    return tf.einsum('ij,kj->ki', D, x)


# end of file
