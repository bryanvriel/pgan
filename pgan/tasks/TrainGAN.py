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

    n_epoch = pyre.properties.int(default=1000)
    n_epoch.doc = 'Number of training epochs (default: 1000)'

    generator_layers = pyre.properties.str()
    generator_layers.doc = 'Layer sizes for generator'

    discriminator_layers = pyre.properties.str()
    discriminator_layers.doc = 'Layer sizes for discriminator'

    encoder_layers = pyre.properties.str()
    encoder_layers.doc = 'Layer sizes for encoder'

    pde_layers = pyre.properties.str()
    pde_layers.doc = 'Layer sizes for PDE net'

    pde_checkdir = pyre.properties.str()
    pde_checkdir.doc = 'Input checkpoint directory for PDE network'

    disc_learning_rate = pyre.properties.float(default=0.001)
    disc_learning_rate.doc = 'Discriminator learning rate (default: 0.001)'

    gen_learning_rate = pyre.properties.float(default=0.001)
    gen_learning_rate.doc = 'Generator/encoder learning rate (default: 0.001)'

    entropy_reg = pyre.properties.float(default=1.5)
    entropy_reg.doc = 'Variational entropy penalty parameter (default: 1.5)'

    pde_beta = pyre.properties.float(default=100.0)
    pde_beta.doc = 'PDE loss penalty parameter (default: 100.0)'

    checkdir = pyre.properties.str(default='checkpoints')
    checkdir.doc = 'Output checkpoint directory'

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

        # Load data
        train, test, lower, upper = module.unpack(self.data_file)

        # Convert layers to lists
        generator_layers = [int(n) for n in self.generator_layers.split(',')]
        discriminator_layers = [int(n) for n in self.discriminator_layers.split(',')]
        encoder_layers = [int(n) for n in self.encoder_layers.split(',')]
        pde_layers = [int(n) for n in self.pde_layers.split(',')]

        # Separately create the PDE network
        pde_net = module.PDENet(pde_layers)

        # Create the GAN model
        model = module.GAN(
            generator_layers=generator_layers,
            discriminator_layers=discriminator_layers,
            encoder_layers=encoder_layers,
            physical_model=pde_net,
            entropy_reg=self.entropy_reg,
            pde_beta=self.pde_beta
        )

        # Construct graphs
        model.build(disc_learning_rate=self.disc_learning_rate,
                    gen_learning_rate=self.gen_learning_rate,
                    inter_op_cores=plexus.inter_op_cores,
                    intra_op_threads=plexus.intra_op_threads)
        model.print_variables()

        # Load PDE weights separately
        saver = tf.train.Saver(var_list=pde_net.trainable_variables)
        saver.restore(model.sess, os.path.join(self.pde_checkdir, 'pde.ckpt'))

        # Train the model
        model.train(train, test=test, n_epochs=self.n_epoch, batch_size=plexus.batch_size)

        # Save the weights
        model.save(outdir=self.checkdir)


# end of file
