#-*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from .common import DenseNet, Model
from tqdm import tqdm
import logging

class DeepHPM(Model):
    """
    Model for learning hidden dynamics from data.
    """

    def __init__(self, solution_layers, pde_layers, lower_bound, upper_bound, name='deepHPM'):
        """
        Store metadata about network architectures and domain bounds.
        """
        # Initialize parent class
        super().__init__(name=name)

        # Create PDE (physics consistency) network
        self.pde_net = PDENet(pde_layers, name='pde')

        # Create solution network
        self.solution_net = SolutionNet(
            solution_layers, np.array(upper_bound), np.array(lower_bound), name='solution'
        )

        # Create dictionary of models
        self.submodels = {'pde': self.pde_net, 'solution': self.solution_net}

        return

    def build(self, learning_rate=0.001, graph=None, inter_op_cores=1, intra_op_threads=1):
        """
        Construct all computation graphs, placeholders, loss functions, and optimizers.
        """
        # Placeholders for data
        self.T = tf.placeholder(tf.float32, shape=[None, 1])
        self.X = tf.placeholder(tf.float32, shape=[None, 1])
        self.U = tf.placeholder(tf.float32, shape=[None, 1])

        # Compute graph for solution network
        self.U_pred = self.solution_net(self.X, self.T)

        # Compute graph for residual network
        self.F_pred = self.pde_net(self.U_pred, self.X, self.T)

        # Loss function
        self.solution_loss = 1000.0 * tf.reduce_mean(tf.square(self.U_pred - self.U))
        self.pde_loss = 1000.0 * tf.reduce_mean(tf.square(self.F_pred))
        self.loss = self.solution_loss + self.pde_loss

        # Optimization step
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

        # Finalize building via the super class
        super().build(graph=graph,
                      inter_op_cores=inter_op_cores,
                      intra_op_threads=intra_op_threads)

        return

    def train(self, x_train, t_train, u_train, test=None,
              batch_size=128, n_epochs=1000, verbose=True):
        """
        Run training.
        """
        # Compute the number of batches
        n_train = t_train.shape[0]
        n_batches = int(np.ceil(n_train / batch_size))

        # Training iterations
        for epoch in tqdm(range(n_epochs)):

            # Loop over minibatches for training
            losses = np.zeros((n_batches, 2))
            start = 0
            for b in range(n_batches):

                # Construct feed dictionary
                Tmb = t_train[start:start+batch_size]
                Xmb = x_train[start:start+batch_size]
                Umb = u_train[start:start+batch_size]
                feed_dict = {self.T: Tmb, self.X: Xmb, self.U: Umb}

                # Run training operation
                _, uloss, floss = self.sess.run(
                    [self.train_op, self.solution_loss, self.pde_loss],
                    feed_dict=feed_dict
                )
                losses[b,:] = [uloss, floss]

                # Update starting batch index
                start += batch_size

            # Compute testing losses
            if test is not None:
                feed_dict = {self.X: test[0], self.T: test[1], self.U: test[2]}
                uloss_test, floss_test = self.sess.run(
                    [self.solution_loss, self.pde_loss],
                    feed_dict=feed_dict
                )

            # Log training performance
            if verbose:
                u_loss, f_loss = np.mean(losses, axis=0)
                msg = '%06d %f %f' % (epoch, u_loss, f_loss)
                if test is not None:
                    msg += ' %f %f' % (uloss_test, floss_test)
                logging.info(msg)

        return


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

    def call(self, u, x, t, training=False):
        """
        Compute gradients on inputs and generate an output.
        """
        # Compute gradients
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]

        # Send to dense net
        inputs = tf.concat(values=[u, u_x, u_xx], axis=1)
        pde = self.dense(inputs, training=training, activate_outputs=False)

        # Residual output
        f = u_t - pde
        return f


class SolutionNet(tf.keras.Model):
    """
    Feedforward network that takes in time and space variables.
    """

    def __init__(self, layer_sizes, upper_bound, lower_bound, name='solution'):
        """
        Initialize and create layers.
        """
        # Initialize parent class
        super().__init__(name=name)

        # Save domain bounds
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        # Create dense network
        self.dense = DenseNet(layer_sizes)

        return

    def call(self, x, t, training=False):
        """
        Pass inputs through network and generate an output.
        """
        # Concatenate the spatial and temporal input variables
        X = tf.concat(values=[x, t], axis=1)

        # Normalize by the domain boundaries
        Xn = 2.0 * (X - self.lower_bound) / (self.upper_bound - self.lower_bound) - 1.0

        # Compute dense network output
        u = self.dense(Xn, training=training, activate_outputs=False)
        return u


# end of file
