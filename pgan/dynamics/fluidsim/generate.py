#-*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import logging
from pgan.networks.common import DenseNet, Model

from .networks import Generator, Discriminator, Encoder


class GAN(Model):
    """
    Model for generating samples of observed data informed by physical dynamics.
    """

    def __init__(self, generator_layers, discriminator_layers, latent_dims,
                 physical_model, encoder_layers=None, entropy_reg=1.5, pde_beta=1.0,
                 name='GAN'):
        """
        Store metadata about network architectures and domain bounds.
        """
        # Initialize parent class
        super().__init__(name=name)

        # Create generator
        self.generator = Generator(generator_layers, name='generator')

        # Create discriminator
        self.discriminator = Discriminator(discriminator_layers, name='discriminator')

        # Cache number of dimensions for latent space
        self.latent_dims = latent_dims

        # Cache pre-trained and pre-configured physics model
        self.physics = physical_model

        # Create dictionary of models
        self.submodels = {'generator': self.generator,
                          'discriminator': self.discriminator}

        # Optionally add encoder
        self.encoder = None
        if encoder_layers is not None:
            # Check consistency of latent dimensions
            assert self.latent_dims == encoder_layers[-1] // 2, \
                'Encoder layers do not match latent dimensions'
            # Create encoder
            self.encoder = Encoder(encoder_layers, name='encoder')
            self.submodels['encoder'] = self.encoder
            self.entropy_reg = entropy_reg

        # Store PDE loss regularization parameter
        self.pde_beta = pde_beta

        return

    def build(self, graph=None, inter_op_cores=1, intra_op_threads=1):
        """
        Construct all computation graphs, placeholders, loss functions, and optimizers.
        """
        # Placeholders for points for GAN loss
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

        # Sample latent vectors from prior p(z)
        latent_dims = [tf.shape(self.X)[0], self.latent_dims]
        prior = tf.distributions.Normal(
            loc=tf.zeros(latent_dims, dtype=tf.float32),
            scale=tf.ones(latent_dims, dtype=tf.float32)
        )
        z_prior = prior.sample()

        # Generate solution at boundary points using sampled latent codes
        W_sol = self.generator(self.X, self.Y, self.T, z_prior)

        # Compute discriminator loss (Note: labels switched from standard GAN)
        disc_logits_real = self.discriminator(self.X, self.Y, self.T, self.W)
        disc_logits_fake = self.discriminator(self.X, self.Y, self.T, W_sol)
        disc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_logits_real,
            labels=tf.zeros_like(disc_logits_real)
        ))
        disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_logits_fake,
            labels=tf.ones_like(disc_logits_real)
        ))
        self.disc_loss = 0.5 * (disc_loss_real + disc_loss_fake)

        # Compute standard generator loss
        # The paper uses logits directly, but cross entropy works slightly better
        self.gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_logits_fake,
            labels=tf.zeros_like(disc_logits_real)
        ))

        # Sample latent vectors from prior p(z) for collocation points
        latent_dims = [tf.shape(self.Xpde)[0], self.latent_dims]
        prior_pde = tf.distributions.Normal(
            loc=tf.zeros(latent_dims, dtype=tf.float32),
            scale=tf.ones(latent_dims, dtype=tf.float32)
        )
        self.z_prior_pde = prior_pde.sample()

        # Compute PDE loss at collocation points
        self.Wpde = self.generator(self.Xpde, self.Ypde, self.Tpde, self.z_prior_pde)
        F_pred = self.physics(self.Wpde, self.Xpde, self.Ypde, self.Upde,
                              self.Vpde, self.Tpde)
        self.pde_loss = self.pde_beta * 1000.0 * tf.reduce_mean(tf.square(F_pred))

        # Keep track of error for monitoring purposes (we don't optimize this)
        self.error = tf.reduce_mean(tf.square(self.W - W_sol))

        # Total generator loss
        total_gen_loss = self.gen_loss + self.pde_loss
        self._losses = [self.disc_loss, self.gen_loss, self.error, self.pde_loss]

        # Optinally add variational inference entropy and cycle-consistency loss
        if self.encoder is not None:
            # First pass generated data through encoder
            q_z_posterior, q_mean, q_std = self.encoder(self.X, self.T, self.U_sol)
            # Then compute variational loss
            self.variational_loss = (1.0 - self.entropy_reg) * \
                tf.reduce_mean(q_z_posterior.log_prob(z_prior))
            # Add to losses
            total_gen_loss = total_gen_loss + self.variational_loss
            self._losses.append(self.variational_loss)
            gen_variables = self.generator.trainable_variables + self.encoder.trainable_variables
        else:
            gen_variables = self.generator.trainable_variables

        # Optimizers for discriminator and generator training objectives
        self.disc_opt = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1=0.9
        )
        self.gen_opt = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate,
            beta1=0.9
        )

        # Training steps
        self.disc_train_op = self.disc_opt.minimize(
            self.disc_loss,
            var_list=self.discriminator.trainable_variables
        )
        self.gen_train_op = self.gen_opt.minimize(
            total_gen_loss,
            var_list=gen_variables
        )

        # Finalize building via the super class
        super().build(graph=graph,
                      inter_op_cores=inter_op_cores,
                      intra_op_threads=intra_op_threads)

        return

    def predict(self, X, Y, T, n_samples=100):
        """
        Generate random predictions.
        """
        # Feed dictionary will be the same for all samples
        feed_dict = {self.Xpde: X.reshape(-1, 1),
                     self.Ypde: Y.reshape(-1, 1),
                     self.Tpde: T.reshape(-1, 1)}

        # Loop over samples and run graph for collocation (PDE) points
        W = np.zeros((n_samples, X.size), dtype=np.float32)
        for i in tqdm(range(n_samples)):
            Wi = self.sess.run(self.Wpde, feed_dict=feed_dict)
            W[i] = Wi.squeeze()

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


# end of file
