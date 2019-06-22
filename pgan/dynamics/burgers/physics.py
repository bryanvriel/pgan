#-*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from pgan.networks.common import DenseNet, Model
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
        # Placeholder for time
        self.T = tf.placeholder(tf.float32, shape=[None, 1])
        # Placeholder for spatial coordinates
        self.X = tf.placeholder(tf.float32, shape=[None, 1])
        # Placeholder for output solution
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

    def train(self, train, test=None, batch_size=128, n_epochs=1000, verbose=True):
        """
        Run training.
        """
        # Compute the number of batches
        n_train = train.t.shape[0]
        n_batches = int(np.ceil(n_train / batch_size))

        # Training iterations
        for epoch in tqdm(range(n_epochs)):

            # Loop over minibatches for training
            losses = np.zeros((n_batches, 2))
            start = 0
            for b in range(n_batches):

                # Construct feed dictionary
                Tmb = train.t[start:start+batch_size]
                Xmb = train.x[start:start+batch_size]
                Umb = train.u[start:start+batch_size]
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
                feed_dict = {self.X: test.x, self.T: test.t, self.U: test.u}
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

# end of file
