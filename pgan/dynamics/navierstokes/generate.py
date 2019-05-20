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

    def __init__(self, generator_layers, discriminator_layers, encoder_layers,
                 physical_model, entropy_reg=1.5, pde_beta=1.0, name='GAN'):
        """
        Store metadata about network architectures and domain bounds.
        """
        # Initialize parent class
        super().__init__(name=name)

        # Create generator
        self.generator = Generator(generator_layers, name='generator')

        # Create discriminator
        self.discriminator = Discriminator(discriminator_layers, name='discriminator')

        # Create encoder
        self.encoder = Encoder(encoder_layers, name='encoder')

        # Cache pre-trained and pre-configured physics model
        self.physics = physical_model

        # Create dictionary of models
        self.submodels = {'generator': self.generator,
                          'discriminator': self.discriminator,
                          'encoder': self.encoder}

        # Store entropy regularization parameter
        # NOTE: λ is generally >= 1 and controls strength of mode collapse mitigation
        self.entropy_reg = entropy_reg

        # Store PDE loss regularization parameter
        self.pde_beta = pde_beta

        return

    def build(self, disc_learning_rate=0.001, gen_learning_rate=0.001,
              graph=None, inter_op_cores=1, intra_op_threads=1):
        """
        Construct all computation graphs, placeholders, loss functions, and optimizers.
        """
        # Placeholders for boundary points
        self.Xb = tf.placeholder(tf.float32, shape=[None, 1])
        self.Yb = tf.placeholder(tf.float32, shape=[None, 1])
        self.Tb = tf.placeholder(tf.float32, shape=[None, 1])
        self.Wb = tf.placeholder(tf.float32, shape=[None, 1])

        # Placeholder for collocation points
        self.Xcoll = tf.placeholder(tf.float32, shape=[None, 1])
        self.Ycoll = tf.placeholder(tf.float32, shape=[None, 1])
        self.Ucoll = tf.placeholder(tf.float32, shape=[None, 1])
        self.Vcoll = tf.placeholder(tf.float32, shape=[None, 1])
        self.Tcoll = tf.placeholder(tf.float32, shape=[None, 1])

        # Sample latent vectors from prior p(z)
        latent_dims = [tf.shape(self.Xb)[0], self.encoder.latent_dim]
        prior = tf.distributions.Normal(
            loc=tf.zeros(latent_dims, dtype=tf.float32),
            scale=tf.ones(latent_dims, dtype=tf.float32)
        )
        z_prior = prior.sample()

        # Generate solution at boundary points using sampled latent codes
        Wb_sol = self.generator(self.Xb, self.Yb, self.Tb, z_prior)

        # Pass generated data through encoder
        q_z_given_x_u, q_mean = self.encoder(self.Xb, self.Yb, self.Tb, Wb_sol)

        # Compute discriminator loss (Note: labels switched from standard GAN)
        disc_logits_real = self.discriminator(self.Xb, self.Yb, self.Tb, self.Wb)
        disc_logits_fake = self.discriminator(self.Xb, self.Yb, self.Tb, Wb_sol)
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
        gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=disc_logits_fake,
            labels=tf.zeros_like(disc_logits_real)
        ))

        # Compute variational inference entropy and cycle-consistency loss
        self.variational_loss = (1.0 - self.entropy_reg) * \
            tf.reduce_mean(q_z_given_x_u.log_prob(z_prior))

        # Sample latent vectors from prior p(z) for collocation points
        latent_dims = [tf.shape(self.Xcoll)[0], self.encoder.latent_dim]
        prior_coll = tf.distributions.Normal(
            loc=tf.zeros(latent_dims, dtype=tf.float32),
            scale=tf.ones(latent_dims, dtype=tf.float32)
        )
        self.z_prior_coll = prior_coll.sample()

        # Compute PDE loss at collocation points
        self.Wcoll = self.generator(self.Xcoll, self.Ycoll, self.Tcoll, self.z_prior_coll)
        self.pde_loss = self.pde_beta * tf.reduce_mean(
            tf.square(self.physics(self.Wcoll, self.Xcoll, self.Ycoll, self.Ucoll,
                                   self.Vcoll, self.Tcoll))
        )

        # Total generator loss
        self.gen_loss = gen_loss + self.variational_loss + self.pde_loss

        # Optimizers for discriminator and generator training objectives
        self.disc_opt = tf.train.AdamOptimizer(learning_rate=disc_learning_rate)
        self.gen_opt = tf.train.AdamOptimizer(learning_rate=gen_learning_rate)

        # Training steps
        self.disc_train_op = self.disc_opt.minimize(
            self.disc_loss,
            var_list=self.discriminator.trainable_variables
        )
        self.gen_train_op = self.gen_opt.minimize(
            self.gen_loss,
            var_list=self.generator.trainable_variables + self.encoder.trainable_variables
        )

        # Finalize building via the super class
        super().build(graph=graph,
                      inter_op_cores=inter_op_cores,
                      intra_op_threads=intra_op_threads)

        return

    def train(self, train, test=None, batch_size=128, n_epochs=1000, dskip=5, verbose=True):
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
        # Compute the number of batches
        n_train = train.xcoll.shape[0]
        n_batches = int(np.ceil(n_train / batch_size))
        print('Using %d batches of size %d' % (n_batches, batch_size))

        # Pre-construct feed dictionary for training
        feed_dict = {self.Xb: train.x,
                     self.Yb: train.y,
                     self.Tb: train.t,
                     self.Wb: train.w,
                     self.Xcoll: None,
                     self.Ycoll: None,
                     self.Ucoll: None,
                     self.Vcoll: None,
                     self.Tcoll: None}

        # Training iterations
        losses = np.zeros((n_epochs, 4))
        for epoch in tqdm(range(n_epochs)):

            # Get random indices to shuffle training examples
            ind = np.random.permutation(n_train)
            Xb = train.x[ind]
            Yb = train.y[ind]
            Tb = train.t[ind]
            Wb = train.w[ind]
            Xcoll = train.xcoll[ind]
            Ycoll = train.ycoll[ind]
            Ucoll = train.ucoll[ind]
            Vcoll = train.vcoll[ind]
            Tcoll = train.tcoll[ind]

            # Loop over minibatches
            gen_losses = np.zeros((n_batches, 3))
            disc_losses = np.zeros(n_batches)
            start = 0
            for b in range(n_batches):

                # Create feed dictionary for training points
                feed_dict = {
                    self.Xb: Xb[start:start+batch_size].reshape(-1, 1),
                    self.Yb: Yb[start:start+batch_size].reshape(-1, 1),
                    self.Tb: Tb[start:start+batch_size].reshape(-1, 1),
                    self.Wb: Wb[start:start+batch_size].reshape(-1, 1),
                    self.Xcoll: Xcoll[start:start+batch_size].reshape(-1, 1),
                    self.Ycoll: Ycoll[start:start+batch_size].reshape(-1, 1),
                    self.Ucoll: Ucoll[start:start+batch_size].reshape(-1, 1),
                    self.Vcoll: Vcoll[start:start+batch_size].reshape(-1, 1),
                    self.Tcoll: Tcoll[start:start+batch_size].reshape(-1, 1)
                }

                # Run training operation for generator and compute losses
                values = self.sess.run(
                    [self.gen_train_op, self.gen_loss, self.variational_loss, self.pde_loss],
                    feed_dict=feed_dict
                )
                gen_losses[b,:] = values[1:]

                # Optionally run training operations for disciminator
                if epoch % dskip == 0:
                    _, value = self.sess.run([self.disc_train_op, self.disc_loss],
                                             feed_dict=feed_dict)
                    disc_losses[b] = value

                # Update starting batch index
                start += batch_size

            # Average losses over all minibatches
            if epoch % dskip == 0:
                disc_loss = np.mean(disc_losses)
            gen_loss, var_loss, pde_loss = np.mean(gen_losses, axis=0)

            # Log training performance
            if verbose:
                logging.info('%d %f %f %f %f' % (epoch, disc_loss, gen_loss, var_loss, pde_loss))

            if epoch % 10000 == 0 and epoch != 0:
                self.save(outdir='checkpoints_%d' % epoch)

            # Save losses
            losses[epoch,:] = [disc_loss, gen_loss, var_loss, pde_loss]

        return losses

    def predict(self, X, Y, T, n_samples=100):
        """
        Generate random predictions.
        """
        # Allocate memory for predictions
        W = np.zeros((n_samples, X.size), dtype=np.float32)
        z = np.zeros((n_samples, X.size), dtype=np.float32)

        # Feed dictionary will be the same for all samples
        feed_dict = {self.Xcoll: X.reshape(-1, 1),
                     self.Ycoll: Y.reshape(-1, 1),
                     self.Tcoll: T.reshape(-1, 1)}

        # Loop over samples
        for i in tqdm(range(n_samples)):
            # Run graph for solution for collocation points
            Wi, zi = self.sess.run([self.Wcoll, self.z_prior_coll], feed_dict=feed_dict)
            W[i] = Wi.squeeze()
            z[i] = zi.squeeze()

        return W, z


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
