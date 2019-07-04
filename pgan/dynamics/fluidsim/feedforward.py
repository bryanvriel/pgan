#-*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import logging
from pgan.networks.common import DenseNet, Model

from .networks import VariationalFeedforward, SolutionNet


class Feedforward(Model):
    """
    Model for generating samples of observed data informed by physical dynamics.
    """

    def __init__(self, generator_layers, physical_model, pde_beta=1.0, variational_loss=False,
                 name='FF'):
        """
        Store metadata about network architectures and domain bounds.
        """
        # Initialize parent class
        super().__init__(name=name)

        # Create generators
        self.variational_loss = variational_loss
        if self.variational_loss:
            self.feedforward = VariationalFeedforward(generator_layers, name='feedforward')
        else:
            self.feedforward = SolutionNet(generator_layers, name='feedforward')

        # Cache pre-trained and pre-configured physics model
        self.physics = physical_model

        # Create dictionary of models
        self.submodels = {'feedforward': self.feedforward}

        # Store PDE loss regularization parameters
        self.pde_beta = pde_beta

        return

    def build(self, graph=None, inter_op_cores=1, intra_op_threads=1):
        """
        Construct all computation graphs, placeholders, loss functions, and optimizers.
        """
        # Placeholders for boundary points
        self.X = tf.placeholder(tf.float32, shape=[None, 1])
        self.Y = tf.placeholder(tf.float32, shape=[None, 1])
        self.T = tf.placeholder(tf.float32, shape=[None, 1])
        self.W = tf.placeholder(tf.float32, shape=[None, 1])

        # Placeholder for collocation points for PDE consistency evaluation
        self.Xpde = tf.placeholder(tf.float32, shape=[None, 1])
        self.Ypde = tf.placeholder(tf.float32, shape=[None, 1])
        self.Upde = tf.placeholder(tf.float32, shape=[None, 1])
        self.Vpde = tf.placeholder(tf.float32, shape=[None, 1])
        self.Tpde = tf.placeholder(tf.float32, shape=[None, 1])

        # Placeholder for learning rate
        self.learning_rate = tf.placeholder(tf.float32)

        # Feedforward generator outputs and losses
        if self.variational_loss:
            # Data fit
            p_likelihood, W_pred, p_std = self.feedforward(self.X, self.Y, self.T)
            self.likelihood = -1.0 * tf.reduce_mean(p_likelihood.log_prob(self.W))
            # PDE generation
            self.Wpde, self.Wpde_std = self.feedforward(self.Xpde, self.Ypde, self.Tpde)[1:]
    
        else:
            # Data fit
            W_pred = self.feedforward(self.X, self.Y, self.T)
            self.likelihood = tf.reduce_mean(tf.square(self.W - W_pred))
            # PDE generation
            self.Wpde = self.feedforward(self.Xpde, self.Ypde, self.Tpde)
            self.Wpde_std = self.Wpde
            
        # Append physics-based loss to likelihood
        F_pde = self.physics(self.Wpde, self.Xpde, self.Ypde, self.Upde, self.Vpde, self.Tpde)
        self.pde_loss = self.pde_beta * 1000.0 * tf.reduce_mean(tf.square(F_pde))
        self.total_loss = self.likelihood + self.pde_loss

        # Compute error (only for monitoring purposes)
        self.error = tf.reduce_mean(tf.square(self.W - W_pred))

        # List of losses to track during training
        self._losses = [self.likelihood, self.error, self.pde_loss]

        # Create optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.total_loss)

        # Finalize building via the super class
        super().build(graph=graph,
                      inter_op_cores=inter_op_cores,
                      intra_op_threads=intra_op_threads)

        return

    def constructFeedDict(self, batch, batch_pde, lr_val=None):
        """
        Construct feed dictionary for filling in tensor placeholders.
        """
        # Fill in batch data
        feed_dict = {self.X: batch['X'],
                     self.Y: batch['Y'],
                     self.T: batch['T'],
                     self.W: batch['W'],
                     self.Xpde: batch_pde['X'],
                     self.Ypde: batch_pde['Y'],
                     self.Upde: batch_pde['U'],
                     self.Vpde: batch_pde['V'],
                     self.Tpde: batch_pde['T']}

        # Optionally add learning rate data
        if lr_val is not None:
            feed_dict[self.learning_rate] = lr_val
       
        # Done 
        return feed_dict


# end of file
