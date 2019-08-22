#-*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from pgan.networks.common import Model
from pgan import math as pmath

from .networks import Encoder, Generator

class Autoencoder(Model):
    """
    Model for generating samples of observed data informed by physical dynamics.
    """

    def __init__(self, encoder_layers, generator_layers, physical_model, image_shape,
                 pde_beta=1.0, name='VAE'):
        """
        Store metadata about network architectures and domain bounds.
        """
        # Initialize parent class
        super().__init__(name=name)

        # Cache image shapes
        if len(image_shape) == 3:
            self.Ny, self.Nx, self.Nc = image_shape
        else:
            self.Ny, self.Nx = image_shape
            self.Nc = 1

        # Create variational encoder
        self.encoder = Encoder(encoder_layers, name='encoder')

        # Create generator (decoder)
        self.generator = Generator(self.Ny, self.Nx, generator_layers, name='generator')

        # Cache pre-trained and pre-configured physics model
        self.physics = physical_model

        # Create dictionary of models
        self.submodels = {'encoder': self.encoder,
                          'generator': self.generator}

        # Store PDE loss regularization parameters
        self.pde_beta = pde_beta

        return

    def build(self, graph=None, inter_op_cores=1, intra_op_threads=1):
        """
        Construct all computation graphs, placeholders, loss functions, and optimizers.
        """
        # Placeholders for data examples
        self.W = tf.placeholder(tf.float32, shape=[None, self.Ny, self.Nx, self.Nc])

        # Placeholder for collocation/PDE examples
        self.Wpde = tf.placeholder(tf.float32, shape=[None, self.Ny, self.Nx, self.Nc])
        self.Upde = tf.placeholder(tf.float32, shape=[None, self.Ny, self.Nx, self.Nc])
        self.Vpde = tf.placeholder(tf.float32, shape=[None, self.Ny, self.Nx, self.Nc])

        # The time variables here represent time RELATIVE TO EMBEDDING
        self.T = tf.placeholder(tf.float32, shape=[None, 1])    # nominally set to 0
        self.Tpde = tf.placeholder(tf.float32, shape=[None, 1]) # nominally set to 0
        self.T_bwd = self.Tpde - self.physics.dt
        self.T_fwd = self.Tpde + self.physics.dt

        # Placeholder for learning rate
        self.learning_rate = tf.placeholder(tf.float32)

        # Get encoder outputs
        Z = self.encoder(self.W)
        # Decode
        W_recon = self.generator(Z, t=self.T)

        # Data loss
        self.error = 1000.0 * tf.reduce_mean(tf.square(W_recon - self.W))
        
        # Get encoder outputs for PDE examples
        self.Z_pde = self.encoder(self.Wpde)
        # Decode at three different times for central differencing
        W_recon_bwd = self.generator(self.Z_pde, t=self.T_bwd)
        W_recon_fwd = self.generator(self.Z_pde, t=self.T_fwd)
        self.W_recon_pde = self.generator(self.Z_pde, t=self.Tpde)

        # PDE loss
        self.F_pde = self.physics(W_recon_bwd, self.W_recon_pde, W_recon_fwd,
                                  self.Upde, self.Vpde)
        self.pde_loss = self.pde_beta * 1000.0 * tf.reduce_mean(tf.square(self.F_pde))

        # Boundary loss
        self.boundary_loss = pmath.compute_boundary_loss(self.W_recon_pde, scale=1000.0)

        # Vorticity sum loss
        ref_sum = 622.517 # without normalization
        W_sum = tf.reduce_sum(self.W_recon_pde, axis=[1, 2, 3])
        self.W_sum_misfit = 1.0e-2 * tf.reduce_mean(tf.square(W_sum - ref_sum))
        
        # Combined loss
        self.total_loss = self.error + self.pde_loss + self.boundary_loss

        # List of losses to track during training
        self._losses = [self.error, self.pde_loss, self.boundary_loss, self.W_sum_misfit]

        # Create optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.total_loss)

        # Finalize building via the super class
        super().build(graph=graph,
                      inter_op_cores=inter_op_cores,
                      intra_op_threads=intra_op_threads)

        return

    def predict(self, data):
        """
        Run predictions for data object.
        """
        # Allocate array for outputs
        Nt, Ny, Nx, Nc = data.train['W'].shape
        pred = np.zeros((Nt, Ny, Nx, Nc), dtype=np.float32)
        Z = np.zeros((Nt, self.encoder.latent_dim), dtype=np.float32)

        # Loop over batches
        n_batches = int(np.ceil(data.n_train / data.batch_size))
        for b in tqdm(range(n_batches)):

            # Get batch
            batch = data.train_batch()
            Np = batch['W'].shape[0]

            # Create feed dictionary
            feed_dict = {self.Tpde: np.zeros((Np, 1)),
                         self.Wpde: batch['W']}

            # Run prediction
            pvals, zvals = self.sess.run([self.W_recon_pde, self.Z_pde], feed_dict=feed_dict)

            # Store
            bslice = slice(b * data.batch_size, (b + 1) * data.batch_size)
            pred[bslice] = pvals
            Z[bslice] = zvals

        return pred, Z

    def constructFeedDict(self, batch, batch_pde, lr_val=None):
        """
        Construct feed dictionary for filling in tensor placeholders.
        """
        # Get batch sizes
        Nd = batch['W'].shape[0]
        Np = batch_pde['W'].shape[0]

        # Fill in batch data
        feed_dict = {self.T: batch['T'],
                     self.W: batch['W'],
                     self.Upde: batch_pde['U'],
                     self.Vpde: batch_pde['V'],
                     self.Tpde: batch_pde['T'],
                     self.Wpde: batch_pde['W']}

        # Optionally add learning rate data
        if lr_val is not None:
            feed_dict[self.learning_rate] = lr_val
       
        # Done 
        return feed_dict
    

# end of file
