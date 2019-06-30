#-*- coding: utf-8 -*-

import pgan
import pyre
import numpy as np
import tensorflow as tf
import logging
import h5py
import sys
import os

class TrainGAN(pgan.components.task, family='pgan.traingan'):
    """
    Train a generative physics model with adversarial learning.
    """
    
    data_file = pyre.properties.str()
    data_file.doc = 'Input HDF5 of data'

    dynamics = pyre.properties.str()
    dynamics.doc = 'Name of dynamics submodule'

    n_iterations = pyre.properties.int(default=100000)
    n_iterations.doc = 'Number of training iterationss (default: 100000)'

    generator_layers = pyre.properties.str()
    generator_layers.doc = 'Layer sizes for generator'

    discriminator_layers = pyre.properties.str()
    discriminator_layers.doc = 'Layer sizes for discriminator'

    encoder_layers = pyre.properties.str(default=None)
    encoder_layers.doc = 'Layer sizes for encoder'

    latent_dims = pyre.properties.int(default=1)
    latent_dims.doc = 'Number of dimensions for latent space (default: 1)'

    use_known_pde = pyre.properties.bool(default=False)
    use_known_pde.doc = 'Use known PDE (default: False)'

    pde_layers = pyre.properties.str()
    pde_layers.doc = 'Layer sizes for PDE net'

    pde_checkdir = pyre.properties.str()
    pde_checkdir.doc = 'Input checkpoint directory for PDE network'

    learning_rate = pyre.properties.float(default=0.0001)
    learning_rate.doc = 'Learning rate (default: 0.0001)'

    initial_learning_rate = pyre.properties.float(default=None)
    initial_learning_rate.doc = 'Initial learning rate'

    final_learning_rate = pyre.properties.float(default=None)
    final_learning_rate.doc = 'Final learning rate'

    gan_batch_size = pyre.properties.int(default=256)
    gan_batch_size.doc = 'Batch size for GAN data (default: 256)'

    pde_batch_size = pyre.properties.int(default=1024)
    pde_batch_size.doc = 'Batch size for PDE data (default: 1024)'

    train_fraction = pyre.properties.float(default=0.9)
    train_fraction.doc = 'Fraction of data to use for training (default: 0.9)'

    disc_skip = pyre.properties.int(default=5)
    disc_skip.doc = 'Number of training steps to skip for discriminator (default: 5)'

    entropy_reg = pyre.properties.float(default=1.5)
    entropy_reg.doc = 'Variational entropy loss penalty parameter (default: 1.5)'

    pde_beta = pyre.properties.float(default=100.0)
    pde_beta.doc = 'PDE loss penalty parameter (default: 100.0)'

    checkdir = pyre.properties.str(default='checkpoints')
    checkdir.doc = 'Output checkpoint directory'

    input_checkdir = pyre.properties.str(default=None)
    input_checkdir.doc = 'Optional input checkpoint directory'

    @pyre.export
    def main(self, plexus, argv):
        """
        Main entrypoint into this application.
        """
        # Initialize checkpoint directory
        if not os.path.isdir(self.checkdir):
            os.mkdir(self.checkdir)

        # Initialize logging data
        logfile = os.path.join(self.checkdir, 'train.log')
        logging.basicConfig(filename=logfile, filemode='w', level=logging.INFO)

        # Get dynamics submodule
        module = getattr(pgan.dynamics, self.dynamics)

        # Load data objects
        data_gan, data_pde = module.unpack(self.data_file,
                                           train_fraction=self.train_fraction,
                                           batch_likelihood=self.gan_batch_size,
                                           batch_pde=self.pde_batch_size)

        # Convert layers to lists
        generator_layers = [int(n) for n in self.generator_layers.split(',')]
        discriminator_layers = [int(n) for n in self.discriminator_layers.split(',')]
        if self.encoder_layers is not None:
            encoder_layers = [int(n) for n in self.encoder_layers.split(',')]
        else:
            encoder_layers = None
        
        # Separately create the PDE network
        if self.use_known_pde:
            pde_net = module.KnownPDENet()
        else:
            pde_layers = [int(n) for n in self.pde_layers.split(',')]
            pde_net = module.PDENet(pde_layers)

        # Create the GAN model
        model = module.GAN(generator_layers=generator_layers,
                           discriminator_layers=discriminator_layers,
                           latent_dims=self.latent_dims,
                           physical_model=pde_net,
                           encoder_layers=encoder_layers,
                           entropy_reg=self.entropy_reg,
                           pde_beta=self.pde_beta)

        # Construct graphs
        model.build(inter_op_cores=plexus.inter_op_cores,
                    intra_op_threads=plexus.intra_op_threads)
        model.print_variables()

        # Load PDE weights separately
        if not self.use_known_pde:
            saver = tf.train.Saver(var_list=pde_net.trainable_variables)
            saver.restore(model.sess, os.path.join(self.pde_checkdir, 'pde.ckpt'))

        # Load previous checkpoints
        if self.input_checkdir is not None:
            model.load(indir=self.input_checkdir)

        # Construct learning rate input
        if self.initial_learning_rate is not None and self.final_learning_rate is not None:
            learning_rate = (self.initial_learning_rate, self.final_learning_rate)
        else:
            learning_rate = self.learning_rate

        # Train the model
        model.train(data_gan,
                    data_pde,
                    n_iterations=self.n_iterations,
                    dskip=self.disc_skip,
                    learning_rate=learning_rate)

        # Save the weights
        model.save(outdir=self.checkdir)


# end of file
