#-*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from pgan.networks.common import DenseNet, Model
from pgan import math as pmath

class Encoder(tf.keras.Model):
    """
    Feedforward network that encodes data points to latent vectors.
    """

    def __init__(self, layer_sizes, variational=False, name='encoder'):
        """
        Initialize and create layers.
        """
        # Initialize parent class
        super().__init__(name=name)

        # Create dense network
        self.dense = DenseNet(layer_sizes)

        # Flag for variational encoding tells us the latent dimensionality
        self.variational = variational
        if variational:
            self.latent_dim = layer_sizes[-1] // 2
        else:
            self.latent_dim = layer_sizes[-1]

        # Flatten layer
        self.flatten = tf.keras.layers.Flatten()

        return

    def call(self, w, t=None, training=False):
        """
        Pass inputs through network and generate an output.
        """
        # Flatten images
        wflat = self.flatten(w)

        # Concatenate (column stack) solution and time if necessary
        if t is not None:
            Xn = tf.concat(values=[wflat, t], axis=1)
        else:
            Xn = wflat

        # Dense inference network outputs latent distribution parameters
        if self.variational:
            # Unpack mean and std
            gaussian_params = self.dense(Xn, training=training)
            mean = gaussian_params[:,:self.latent_dim]
            std = tf.nn.softplus(gaussian_params[:,self.latent_dim:])
            # Feed mean and std into distributions object to allow differentiation
            q_z_given_x = tf.distributions.Normal(loc=mean, scale=std)
            return q_z_given_x, mean, std

        else:
            return self.dense(Xn, training=training, activate_outputs=False)


class Discriminator(tf.keras.Model):
    """
    Feedforward network that predicts whether a given data point is real or generated.
    """

    def __init__(self, layer_sizes, name='discriminator'):
        """
        Initialize and create layers.
        """
        # Initialize parent class
        super().__init__(name=name)

        # Create dense network
        self.dense = DenseNet(layer_sizes)

        return

    def call(self, x, y, t, w, training=False):
        """
        Pass inputs through network and generate an output.
        """
        # Concatenate (column stack) spatial coordinate, time, and solution
        Xn = tf.concat(values=[x, y, t, w], axis=1)

        # Compute dense network output (logits)
        p = self.dense(Xn, training=training)

        return p


class Generator(tf.keras.Model):
    """
    Feedforward network that generates solutions given a laten code.
    """

    def __init__(self, Ny, Nx, layer_sizes, name='generator'):
        """
        Initialize and create layers.
        """
        # Initialize parent class
        super().__init__(name=name)

        # Create dense network
        self.dense = DenseNet(layer_sizes)

        # Save image parameters
        self.Ny, self.Nx = Ny, Nx
        self.Npix = Ny * Nx

        return

    def call(self, z, t=None, training=False):
        """
        Pass inputs through network and generate an output.
        """
        # Concatenate (column stack) the latent and time variables if necessary
        if t is not None:
            Xn = tf.concat(values=[z, t], axis=1)
        else:
            Xn = z

        # Compute dense network output
        u = self.dense(Xn, training=training, activate_outputs=False)

        # Reshape and return
        conv_shape = [-1, self.Ny, self.Nx, 1]
        return tf.reshape(u, conv_shape)


class VariationalGenerator(tf.keras.Model):
    """
    Feedforward network that generates solutions given a laten code.
    """

    def __init__(self, layer_sizes, name='vaegenerator'):
        """
        Initialize and create layers.
        """
        # Initialize parent class
        super().__init__(name=name)

        # Create dense network
        self.dense = DenseNet(layer_sizes)

        # Cache number of outputs
        self.nout = layer_sizes[-1] // 2

        return

    def call(self, x, y, t, z, training=False):
        """
        Pass inputs through network and generate an output.
        """
        # Concatenate (column stack) the spatial, time, and latent input variables
        Xn = tf.concat(values=[x, y, t, z], axis=1)

        # Dense inference network outputs likelihood distribution parameters
        gaussian_params = self.dense(Xn, training=training)
        mean = gaussian_params[:,:self.nout]
        std = tf.nn.softplus(gaussian_params[:,self.nout:])

        # Feed mean and std into distributions object to allow differentiation
        # (reparameterization trick under the hood)
        q_x_given_z = tf.distributions.Normal(loc=mean, scale=std)

        return q_x_given_z, mean, std


class VariationalFeedforward(tf.keras.Model):
    """
    Feedforward network that generates solutions given a laten code.
    """

    def __init__(self, layer_sizes, name='feedforward'):
        """
        Initialize and create layers.
        """
        # Initialize parent class
        super().__init__(name=name)

        # Create dense network
        self.dense = DenseNet(layer_sizes)

        # Cache number of outputs
        self.nout = layer_sizes[-1] // 2

        return

    def call(self, x, y, t, training=False):
        """
        Pass inputs through network and generate an output.
        """
        # Concatenate (column stack) the spatial, time, and latent input variables
        Xn = tf.concat(values=[x, y, t], axis=1)

        # Dense inference network outputs likelihood distribution parameters
        gaussian_params = self.dense(Xn, training=training)
        mean = gaussian_params[:,:self.nout]
        std = tf.nn.softplus(gaussian_params[:,self.nout:])

        # Feed mean and std into distributions object to allow differentiation
        # (reparameterization trick under the hood)
        q_x_given_z = tf.distributions.Normal(loc=mean, scale=std)

        return q_x_given_z, mean, std


class PDENet(tf.keras.Model):
    """
    Feedforward network that takes in a solution tensor, computes gradients, and
    passes them through a neural network.
    """

    def __init__(self, layer_sizes, name='pde'):
        """
        Initialize and create layers.
        """
        # Initialize parent class
        super().__init__(name=name)

        # Create dense network
        self.dense = DenseNet(layer_sizes)

        return

    def call(self, w, x, y, t, training=False):
        """
        Compute gradients on inputs and generate an output.
        """
        # Compute gradients of vorticity
        w_t = tf.gradients(w, t)[0]
        w_x = tf.gradients(w, x)[0]
        w_y = tf.gradients(w, y)[0]
        w_xx = tf.gradients(w_x, x)[0]
        w_xy = tf.gradients(w_x, y)[0]
        w_yy = tf.gradients(w_y, y)[0]

        # Send to dense net
        inputs = tf.concat(values=[w, w_x, w_y, w_xx, w_xy, w_yy], axis=1)
        pde = self.dense(inputs, training=training, activate_outputs=False)

        # Residual output
        f = w_t - pde
        return f


class KnownPDENetFD:
    """
    Feedforward network that takes in a solution tensor, computes gradients, and
    passes them through a neural network.
    """

    def __init__(self, dy, dx, dt, name='pde'):
        """
        Save spacing information.
        """
        self.dy, self.dx, self.dt = dy, dx, dt
        return

    def __call__(self, W_bwd, W_pde, W_fwd, u, v, nu=0.001, **kwargs):
        """
        Compute gradients on inputs and generate an output.
        """
        # Compute horizontal gradients
        w_x = pmath.image_gradient(W_pde, self.dx, mode='horizontal')
        w_xx = pmath.image_gradient(w_x, self.dx, mode='horizontal')

        # Compute vertical gradients
        w_y = pmath.image_gradient(W_pde, self.dy, mode='vertical')
        w_yy = pmath.image_gradient(w_y, self.dy, mode='vertical')

        # Compute spatial activations
        pde = -u * w_x - v * w_y + nu * (w_xx + w_yy)

        # Compute temporal gradient using central difference
        w_t = (W_fwd - W_bwd) / (2.0 * self.dt)
        
        # Residual output
        f = w_t - pde
        return f


class SolutionNet(tf.keras.Model):
    """
    Feedforward network that takes in time variable.
    """

    def __init__(self, Ny, Nx, layer_sizes, name='generator'):
        """
        Initialize and create layers.
        """
        # Initialize parent class
        super().__init__(name=name)

        # Create dense network
        self.dense = DenseNet(layer_sizes)

        # Save image parameters
        self.Ny, self.Nx = Ny, Nx
        self.Npix = Ny * Nx

        return

    def call(self, z, t=None, training=False):
        """
        Pass inputs through network and generate an output.
        """
        # Concatenate (column stack) the latent and time variables if necessary
        if t is not None:
            Xn = tf.concat(values=[z, t], axis=1)
        else:
            Xn = z

        # Compute dense network output
        u = self.dense(Xn, training=training, activate_outputs=False)

        # Reshape and return
        conv_shape = [-1, self.Ny, self.Nx, 1]
        return tf.reshape(u, conv_shape)


# end of file
