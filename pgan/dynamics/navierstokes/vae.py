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

        # Create prior p(z)
        latent_dims = [tf.shape(self.X)[0], self.latent_dims]
        prior = tf.distributions.Normal(
            loc=tf.zeros(latent_dims, dtype=tf.float32),
            scale=tf.ones(latent_dims, dtype=tf.float32)
        )

        # Get encoder (inference network) outputs (posterior distribution)
        q_posterior, q_mean, q_std = self.encoder(self.X, self.Y, self.T, self.W)

        # Sample posterior and get variational generator outputs (likelihood)
        self.Z = q_posterior.sample()
        p_likelihood, p_mean, p_std = self.vaegenerator(self.X, self.Y, self.T, self.Z)

        # Compute KL divergence between variational posterior and prior
        kl_div = tf.distributions.kl_divergence(q_posterior, prior) # shape (?, latent_dim)
        KL = self.encoder_beta * tf.reduce_sum(kl_div, axis=1)

        # Compute evidence lower bound (ELBO)
        expected_log_likelihood = tf.reduce_sum(p_likelihood.log_prob(self.W), axis=1)
        self.elbo = tf.reduce_mean(expected_log_likelihood - KL)
        self.KL_mean = tf.reduce_mean(KL)
        self.likelihood_mean = tf.reduce_mean(expected_log_likelihood)

        # Create prior for PDE points
        latent_dims = [tf.shape(self.Xpde)[0], self.latent_dims]
        prior_pde = tf.distributions.Normal(
            loc=tf.zeros(latent_dims, dtype=tf.float32),
            scale=tf.ones(latent_dims, dtype=tf.float32)
        )
        self.Zpde = prior_pde.sample()
        
        # Generate solutions for PDE points (keep predicted mean and std value only)
        self.Wpde_mean, self.Wpde_std = self.vaegenerator(
            self.Xpde, self.Ypde, self.Tpde, self.Zpde
        )[1:]

        # Append physics-based loss
        F_pde = self.physics(self.Wpde_mean, self.Xpde, self.Ypde,
                             self.Upde, self.Vpde, self.Tpde)
        self.pde_loss = self.pde_beta * 1000.0 * tf.reduce_mean(tf.square(F_pde))
        self.total_loss = -1.0 * self.elbo + self.pde_loss

        # Compute error (only for monitoring purposes)
        self.error = tf.reduce_mean(tf.square(self.W - p_mean))

        # List of losses to track during training
        self._losses = [self.elbo, self.likelihood_mean, self.KL_mean, self.error, self.pde_loss]

        # Create optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.total_loss)

        # Finalize building via the super class
        super().build(graph=graph,
                      inter_op_cores=inter_op_cores,
                      intra_op_threads=intra_op_threads)

        return
    

# end of file
