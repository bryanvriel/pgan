#-*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from pgan.networks.common import Model
from tqdm import tqdm
import logging

from .networks import SolutionNet, PDENet

class PINN(Model):
    """
    Model for generating solutions to a PDE.
    """

    def __init__(self, solution_layers, physical_model, pde_beta=1.0, name='PINN'):
        """
        Store metadata about network architectures and domain bounds.
        """
        # Initialize parent class
        super().__init__(name=name)

        # Create solution network
        self.solution_net = SolutionNet(solution_layers, name='solution')

        # Cache pre-trained and pre-configured physics model
        self.physics = physical_model

        # Save the PDE loss penalty parameter
        self.pde_beta = pde_beta

        # Create dictionary of models
        self.submodels = {'solution': self.solution_net}

        return

    def build(self, graph=None, inter_op_cores=1, intra_op_threads=1):
        """
        Construct all computation graphs, placeholders, loss functions, and optimizers.
        """
        # Placeholders for likelihood points
        self.X = tf.placeholder(tf.float32, shape=[None, 1])
        self.Y = tf.placeholder(tf.float32, shape=[None, 1])
        self.T = tf.placeholder(tf.float32, shape=[None, 1])
        self.W = tf.placeholder(tf.float32, shape=[None, 1])

        # Placeholder for collocation (PDE) points
        self.Xpde = tf.placeholder(tf.float32, shape=[None, 1])
        self.Ypde = tf.placeholder(tf.float32, shape=[None, 1])
        self.Upde = tf.placeholder(tf.float32, shape=[None, 1])
        self.Vpde = tf.placeholder(tf.float32, shape=[None, 1])
        self.Tpde = tf.placeholder(tf.float32, shape=[None, 1])

        # Placeholder for learning rate
        self.learning_rate = tf.placeholder(tf.float32)

        # Compute graph for boundary and initial data
        self.W_pred = self.solution_net(self.X, self.Y, self.T)

        # Compute graph for collocation points (physics consistency)
        self.Wpde = self.solution_net(self.Xpde, self.Ypde, self.Tpde)
        F_pred = self.physics(self.Wpde, self.Xpde, self.Ypde,
                              self.Upde, self.Vpde, self.Tpde)

        # Scalar value for all loss functions to improve precision
        self.scale = 1000.0

        # Loss functions
        self.b_loss = self.scale * tf.reduce_mean(tf.square(self.W_pred - self.W))
        self.f_loss = self.pde_beta * self.scale * tf.reduce_mean(tf.square(F_pred))
        self.loss = self.b_loss + self.f_loss

        # List of losses to keep track of during training
        self._losses = [self.b_loss, self.f_loss]

        # Optimization steps (only for SOLUTION net)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(
            self.loss, var_list=self.solution_net.trainable_variables
        )

        # Finalize building via the super class
        super().build(graph=graph,
                      inter_op_cores=inter_op_cores,
                      intra_op_threads=intra_op_threads)

        return

    def predict(self, X, Y, T):
        """
        Generate predictions from PINN.
        """
        # Feed dictionary will be the same for all samples
        feed_dict = {self.Xpde: X.reshape(-1, 1),
                     self.Ypde: Y.reshape(-1, 1),
                     self.Tpde: T.reshape(-1, 1)}

        # Run graph for solution for collocation points
        W = self.sess.run(self.Wpde, feed_dict=feed_dict)

        return W

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



class DeepHPM(Model):
    """
    Model for learning hidden dynamics from data.
    """

    def __init__(self, solution_layers, pde_layers, name='deepHPM'):
        """
        Store metadata about network architectures and domain bounds.
        """
        # Initialize parent class
        super().__init__(name=name)

        # Create PDE (physics consistency) network
        self.pde_net = PDENet(pde_layers, name='pde')

        # Create solution network
        self.solution_net = SolutionNet(solution_layers, name='solution')

        # Create dictionary of models
        self.submodels = {'pde': self.pde_net, 'solution': self.solution_net}

        return

    def build(self, graph=None, inter_op_cores=1, intra_op_threads=1):
        """
        Construct all computation graphs, placeholders, loss functions, and optimizers.
        """
        # Placeholders for data
        self.T = tf.placeholder(tf.float32, shape=[None, 1])
        self.X = tf.placeholder(tf.float32, shape=[None, 1])
        self.Y = tf.placeholder(tf.float32, shape=[None, 1])
        self.W = tf.placeholder(tf.float32, shape=[None, 1])

        # Placeholder for learning rate
        self.learning_rate = tf.placeholder(tf.float32)

        # Compute graph for solution network
        self.W_pred = self.solution_net(self.X, self.Y, self.T)

        # Compute graph for residual network
        self.F_pred = self.pde_net(self.W_pred, self.X, self.Y, self.T)

        # Loss function
        self.solution_loss = 1000.0 * tf.reduce_mean(tf.square(self.W_pred - self.W))
        self.pde_loss = 1000.0 * tf.reduce_mean(tf.square(self.F_pred))
        self.loss = self.solution_loss + self.pde_loss
        self._losses = [self.solution_loss, self.pde_loss]

        # Optimization step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

        # Finalize building via the super class
        super().build(graph=graph,
                      inter_op_cores=inter_op_cores,
                      intra_op_threads=intra_op_threads)

        return

    def constructFeedDict(self, batch, dummy, lr_val=None):
        """
        Construct feed dictionary for filling in tensor placeholders.
        """
        # Fill in batch data
        feed_dict = {self.X: batch['X'],
                     self.Y: batch['Y'],
                     self.T: batch['T'],
                     self.W: batch['W']}

        # Optionally add learning rate data
        if lr_val is not None:
            feed_dict[self.learning_rate] = lr_val

        # Done 
        return feed_dict
    

# end of file
