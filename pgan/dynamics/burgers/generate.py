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
        # Placeholders for boundary points
        self.X = tf.placeholder(tf.float32, shape=[None, 1])
        self.T = tf.placeholder(tf.float32, shape=[None, 1])
        self.U = tf.placeholder(tf.float32, shape=[None, 1])

        # Placeholder for collocation points for PDE consistency evaluation
        self.Xpde = tf.placeholder(tf.float32, shape=[None, 1])
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
        self.U_sol = self.generator(self.X, self.T, z_prior)

        # Compute discriminator loss (Note: labels switched from standard GAN)
        disc_logits_real = self.discriminator(self.X, self.T, self.U)
        disc_logits_fake = self.discriminator(self.X, self.T, self.U_sol)
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
        self.Upde = self.generator(self.Xpde, self.Tpde, self.z_prior_pde)
        F_pred = self.physics(self.Upde, self.Xpde, self.Tpde)
        self.pde_loss = self.pde_beta * 1000.0 * tf.reduce_mean(tf.square(F_pred))

        # Keep track of error for monitoring purposes (we don't optimize this)
        self.error = tf.reduce_mean(tf.square(self.U - self.U_sol))

        # Total generator loss and variables
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

    def train(self,
              data_gan,
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
            batch_gan = data_gan.train_batch()
            batch_pde = data_pde.train_batch()

            # Construct feed dictionary
            feed_dict = self.constructFeedDict(batch_gan, batch_pde, lr_val=lr_val)

            # Run weight updates and compute training loss
            values = self.sess.run([self.gen_train_op] + self._losses, feed_dict=feed_dict)
            # For some reason, tensorflow sticks update return value at the end
            train = values[:-1]

            # Periodically run update for discriminator
            if iternum % dskip == 0:
                self.sess.run(self.disc_train_op, feed_dict=feed_dict)

            # Run losses periodically for test data
            if iternum % 200 == 0:
                test_feed_dict = self.constructFeedDict(data_gan.test, data_pde.test)
                test = self.sess.run(self._losses, feed_dict=test_feed_dict)
            
            # Log training performance
            if verbose:
                out = '%d ' + '%f ' * 2 * len(self._losses)
                logging.info(out % tuple([iternum] + train + test))

            if iternum % 5000 == 0 and iternum != 0:
                self.save(outdir='temp_checkpoints')

        return


    def predict(self, X, T, n_samples=100):
        """
        Generate random predictions.
        """
        # Allocate memory for predictions
        U = np.zeros((n_samples, X.size), dtype=np.float32)

        # Feed dictionary will be the same for all samples
        feed_dict = {self.X: X.reshape(-1, 1),
                     self.T: T.reshape(-1, 1)}

        # Loop over samples
        for i in tqdm(range(n_samples)):
            # Run graph for solution for collocation points
            Ui = self.sess.run(self.U_sol, feed_dict=feed_dict)
            U[i] = Ui.squeeze()

        return U


    def constructFeedDict(self, batch, batch_pde, lr_val=None):
        """
        Construct feed dictionary for filling in tensor placeholders.
        """
        # Fill in batch data
        feed_dict = {self.X: batch['X'],
                     self.T: batch['T'],
                     self.U: batch['U'],
                     self.Xpde: batch_pde['X'],
                     self.Tpde: batch_pde['T']}

        # Optionally add learning rate data
        if lr_val is not None:
            feed_dict[self.learning_rate] = lr_val

        # Done 
        return feed_dict


# end of file
