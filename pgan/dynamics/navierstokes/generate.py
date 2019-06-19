#-*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import logging
from pgan.networks.common import DenseNet, Model


class GAN(Model):
    """
    Model for generating samples of observed data informed by physical dynamics.
    """

    def __init__(self, generator_layers, discriminator_layers, latent_dims,
                 physical_model, pde_beta=1.0, name='GAN'):
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

        # Total generator loss
        total_gen_loss = self.gen_loss + self.pde_loss

        # Keep track of error for monitoring purposes (we don't optimize this)
        self.error = tf.reduce_mean(tf.square(self.W - W_sol))

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
            var_list=self.generator.trainable_variables
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
            n_iternums: int
                Number of iternums (each iternum loops over all batches)
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
            feed_dict = {self.X: batch_gan['X'],
                         self.Y: batch_gan['Y'],
                         self.T: batch_gan['T'],
                         self.W: batch_gan['W'],
                         self.Xpde: batch_pde['X'],
                         self.Ypde: batch_pde['Y'],
                         self.Upde: batch_pde['U'],
                         self.Vpde: batch_pde['V'],
                         self.Tpde: batch_pde['T'],
                         self.learning_rate: lr_val}

            # Run update for generator
            _, gen_loss, error, pde_loss = self.sess.run(
                [self.gen_train_op, self.gen_loss, self.error, self.pde_loss],
                feed_dict=feed_dict
            )

           # Periodically run update for discriminator
            if iternum % dskip == 0:
                _, disc_loss = self.sess.run([self.disc_train_op, self.disc_loss],
                                             feed_dict=feed_dict)
            
            # Run losses periodically for test data
            if iternum % 100 == 0:
                batch_gan = data_gan.test
                batch_pde = data_pde.test
                feed_dict = {self.X: batch_gan['X'],
                             self.Y: batch_gan['Y'],
                             self.T: batch_gan['T'],
                             self.W: batch_gan['W'],
                             self.Xpde: batch_pde['X'],
                             self.Ypde: batch_pde['Y'],
                             self.Upde: batch_pde['U'],
                             self.Vpde: batch_pde['V'],
                             self.Tpde: batch_pde['T']}
                test_loss = self.sess.run(
                    [self.disc_loss, self.gen_loss, self.error, self.pde_loss],
                    feed_dict=feed_dict
                )

            # Log training performance
            if verbose:
                logging.info('%d %f %f %f %f %f %f %f %f' % 
                            (iternum,
                             disc_loss, gen_loss, error, pde_loss,
                             test_loss[0], test_loss[1], test_loss[2], test_loss[3]))

            if iternum % 5000 == 0 and iternum != 0:
                self.save(outdir='temp_checkpoints')

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

        return q_z_given_x, mean


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


# end of file
