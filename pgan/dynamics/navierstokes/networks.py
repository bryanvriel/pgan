#-*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from pgan.networks.common import DenseNet, Model


class Encoder(tf.keras.Model):
    """
    Feedforward network that encodes data points to latent vectors.
    """

    def __init__(self, layer_sizes, name='encoder'):
        """
        Initialize and create layers.
        """
        # Initialize parent class
        super().__init__(name=name)

        # Create dense network
        self.dense = DenseNet(layer_sizes)

        # The last layer size tells us the latent dimension
        self.latent_dim = layer_sizes[-1] // 2

        return

    def call(self, x, y, t, w, training=False):
        """
        Pass inputs through network and generate an output.
        """
        # Concatenate (column stack) spatial coordinate, time, and solution
        Xn = tf.concat(values=[x, y, t, w], axis=1)

        # Dense inference network outputs latent distribution parameters
        gaussian_params = self.dense(Xn, training=training)
        mean = gaussian_params[:,:self.latent_dim]
        std = tf.nn.softplus(gaussian_params[:,self.latent_dim:])

        # Feed mean and std into distributions object to allow differentiation
        # (reparameterization trick under the hood)
        q_z_given_x = tf.distributions.Normal(loc=mean, scale=std)

        return q_z_given_x, mean, std


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

    def __init__(self, layer_sizes, name='generator'):
        """
        Initialize and create layers.
        """
        # Initialize parent class
        super().__init__(name=name)

        # Create dense network
        self.dense = DenseNet(layer_sizes)

        return

    def call(self, x, y, t, z, training=False):
        """
        Pass inputs through network and generate an output.
        """
        # Concatenate (column stack) the spatial, time, and latent input variables
        Xn = tf.concat(values=[x, y, t, z], axis=1)

        # Compute dense network output
        u = self.dense(Xn, training=training)
        return u


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

    def call(self, w, x, y, u, v, t, training=False):
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
        inputs = tf.concat(values=[u, v, w, w_x, w_y, w_xx, w_xy, w_yy], axis=1)
        pde = self.dense(inputs, training=training, activate_outputs=False)

        # Residual output
        f = w_t - pde
        return f


class KnownPDENet:
    """
    Feedforward network that takes in a solution tensor, computes gradients, and
    passes them through a neural network.
    """

    def __init__(self, name='pde'):
        """
        Don't need to do anything here.
        """
        return

    def __call__(self, w, x, y, u, v, t, **kwargs):
        """
        Compute gradients on inputs and generate an output.
        """
        # Compute gradients of vorticity
        w_t = tf.gradients(w, t)[0]
        w_x = tf.gradients(w, x)[0]
        w_y = tf.gradients(w, y)[0]
        w_xx = tf.gradients(w_x, x)[0]
        w_yy = tf.gradients(w_y, y)[0]

        # Compute spatial activations
        pde = -u * w_x - v * w_y + 0.01 * (w_xx + w_yy)
        
        # Residual output
        f = w_t - pde
        return f


class SolutionNet(tf.keras.Model):
    """
    Feedforward network that takes in time and space variables.
    """

    def __init__(self, layer_sizes, name='solution'):
        """
        Initialize and create layers.
        """
        # Initialize parent class
        super().__init__(name=name)

        # Create dense network
        self.dense = DenseNet(layer_sizes)

        return

    def call(self, x, y, t, training=False):
        """
        Pass inputs through network and generate an output.
        """
        # Concatenate the spatial and temporal input variables
        Xn = tf.concat(values=[x, y, t], axis=1)

        # Compute dense network output
        w = self.dense(Xn, training=training, activate_outputs=False)
        return w


# end of file
