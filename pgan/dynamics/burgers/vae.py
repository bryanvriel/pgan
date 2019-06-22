#-*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import logging
from pgan.networks.common import DenseNet, Model

from .networks import Encoder, VariationalGenerator


class VAE(Model):
    """
    Model for generating samples of observed data informed by physical dynamics.
    """

    def __init__(self, encoder_layers, generator_layers, physical_model,
                 encoder_beta=1.0, pde_beta=1.0, name='VAE'):
        """
        Store metadata about network architectures and domain bounds.
        """
        # Initialize parent class
        super().__init__(name=name)

        # Create variational encoder
        self.encoder = Encoder(encoder_layers, name='encoder')

        # Create variational generator
        self.vaegenerator = VariationalGenerator(generator_layers, name='vaegenerator')

        # Cache number of dimensions for latent space
        self.latent_dims = encoder_layers[-1] // 2

        # Cache pre-trained and pre-configured physics model
        self.physics = physical_model

        # Create dictionary of models
        self.submodels = {'encoder': self.encoder,
                          'vaegenerator': self.vaegenerator}

        # Store encoder and PDE loss regularization parameters
        self.encoder_beta = encoder_beta
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

        # Create prior p(z)
        latent_dims = [tf.shape(self.X)[0], self.latent_dims]
        prior = tf.distributions.Normal(
            loc=tf.zeros(latent_dims, dtype=tf.float32),
            scale=tf.ones(latent_dims, dtype=tf.float32)
        )

        # Get encoder (inference network) outputs (posterior distribution)
        q_z_given_u, q_mean, q_std = self.encoder(self.X, self.T, self.U)

        # Get variational generator outputs (likelihood)
        p_u_given_z, p_mean, p_std = self.vaegenerator(self.X, self.T, q_z_given_u.sample())

        # Compute KL divergence between variational posterior and prior
        kl_div = tf.distributions.kl_divergence(q_z_given_u, prior) # shape (?, latent_dim)
        KL = self.encoder_beta * tf.reduce_sum(kl_div, axis=1)

        # Compute evidence lower bound (ELBO)
        expected_log_likelihood = tf.reduce_sum(p_u_given_z.log_prob(self.U), axis=1)
        self.elbo = tf.reduce_mean(expected_log_likelihood - KL)
        self.KL_mean = tf.reduce_mean(KL)
        self.likelihood_mean = tf.reduce_mean(expected_log_likelihood)

        # Create prior for PDE points
        latent_dims = [tf.shape(self.Xpde)[0], self.latent_dims]
        prior_pde = tf.distributions.Normal(
            loc=tf.zeros(latent_dims, dtype=tf.float32),
            scale=tf.ones(latent_dims, dtype=tf.float32)
        )
        Zpde = prior_pde.sample()
        
        # Generate solutions for PDE points (keep predicted mean and std value only)
        self.Upde_mean, self.Upde_std = self.vaegenerator(self.Xpde, self.Tpde, Zpde)[1:]

        # Append physics-based loss
        F_pde = self.physics(self.Upde_mean, self.Xpde, self.Tpde)
        self.pde_loss = self.pde_beta * 1000.0 * tf.reduce_mean(tf.square(F_pde))
        self.total_loss = -1.0 * self.elbo + self.pde_loss

        # Compute error (only for monitoring purposes)
        self.error = tf.reduce_mean(tf.square(self.U - p_mean))

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

        Arguments:
            X_coll: ndarray
                Array of collocation points to loop over.
            batch_size: int
                Batch size.
            n_epochs: int
                Number of epochs (each epoch loops over all batches)
            dskip: int
                Skip factor for discriminator training ops.
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
                [self.train_op, self.elbo, self.likelihood_mean, self.KL_mean,
                 self.error, self.pde_loss], feed_dict=feed_dict
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
                    [self.elbo, self.likelihood_mean, self.KL_mean, self.error, self.pde_loss],
                    feed_dict=feed_dict
                )

            # Log training performance
            if verbose:
                logging.info('%d %f %f %f %f %f %f %f %f %f %f' % 
                            (iternum,
                             values[1], values[2], values[3], values[4], values[5],
                             test[0], test[1], test[2], test[3], test[4]))

            # Temporarily save checkpoints
            if iternum % 10000 == 0 and iternum != 0:
                self.save(outdir='temp_checkpoints')

        return


# end of file
