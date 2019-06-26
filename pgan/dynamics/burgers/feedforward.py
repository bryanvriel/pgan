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
        self.T = tf.placeholder(tf.float32, shape=[None, 1])
        self.U = tf.placeholder(tf.float32, shape=[None, 1])

        # Placeholder for collocation points for PDE consistency evaluation
        self.Xpde = tf.placeholder(tf.float32, shape=[None, 1])
        self.Tpde = tf.placeholder(tf.float32, shape=[None, 1])

        # Placeholder for learning rate
        self.learning_rate = tf.placeholder(tf.float32)

        # Feedforward generator outputs and losses
        if self.variational_loss:
            # Data fit
            p_likelihood, U_pred, p_std = self.feedforward(self.X, self.T)
            self.likelihood = -1.0 * tf.reduce_mean(p_likelihood.log_prob(self.U))
            # PDE generation
            self.Upde, self.Upde_std = self.feedforward(self.Xpde, self.Tpde)[1:]
    
        else:
            # Data fit
            U_pred = self.feedforward(self.X, self.T)
            self.likelihood = 1000.0 * tf.reduce_mean(tf.square(self.U - U_pred))
            # PDE generation
            self.Upde = self.feedforward(self.Xpde, self.Tpde)
            self.Upde_std = self.Upde
            
        # Append physics-based loss to likelihood
        F_pde = self.physics(self.Upde, self.Xpde, self.Tpde)
        self.pde_loss = self.pde_beta * 1000.0 * tf.reduce_mean(tf.square(F_pde))
        self.total_loss = self.likelihood + self.pde_loss

        # Compute error (only for monitoring purposes)
        self.error = tf.reduce_mean(tf.square(self.U - U_pred))

        # Create optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.total_loss)

        # Finalize building via the super class
        super().build(graph=graph,
                      inter_op_cores=inter_op_cores,
                      intra_op_threads=intra_op_threads)

        return

    def train(self,
              data,
              data_pde,
              n_iterations=100000,
              dskip=5,
              learning_rate=0.0001,
              verbose=True):

        """
        Run training over batches of collocation points.
        """
        # Compute time scale for exponential cooling of learning rate if tuple provided
        if isinstance(learning_rate, tuple):
            initial_learning_rate, final_learning_rate = learning_rate
            lr_tau = -n_iterations / np.log(final_learning_rate / initial_learning_rate)
            print('Learning rate tau:', lr_tau)
        else:
            print('Using constant learning rate:', learning_rate)
            lr_tau = None

        # Training iterations
        for iternum in tqdm(range(n_iterations)):

            # Compute learning rate
            if lr_tau is not None:
                lr_val = initial_learning_rate * np.exp(-iternum / lr_tau)
            else:
                lr_val = learning_rate

            # Get batch of training data
            batch = data.train_batch()
            batch_pde = data_pde.train_batch()

            # Construct feed dictionary
            feed_dict = {self.X: batch['X'],
                         self.T: batch['T'],
                         self.U: batch['U'],
                         self.Xpde: batch_pde['X'],
                         self.Tpde: batch_pde['T'],
                         self.learning_rate: lr_val}

            # Run updates
            values = self.sess.run(
                [self.train_op, self.likelihood, self.error, self.pde_loss],
                feed_dict=feed_dict
            )
            
            # Run losses periodically for test data
            if iternum % 100 == 0:
                batch = data.test
                batch_pde = data_pde.test
                feed_dict = {self.X: batch['X'],
                             self.T: batch['T'],
                             self.U: batch['U'],
                             self.Xpde: batch_pde['X'],
                             self.Tpde: batch_pde['T']}
                test = self.sess.run(
                    [self.likelihood, self.error, self.pde_loss],
                    feed_dict=feed_dict
                )

            # Log training performance
            if verbose:
                logging.info('%d %f %f %f %f %f %f' % 
                            (iternum,
                             values[1], values[2], values[3],
                             test[0], test[1], test[2]))

            # Temporarily save checkpoints
            if iternum % 10000 == 0 and iternum != 0:
                self.save(outdir='temp_checkpoints')

        return


# end of file
