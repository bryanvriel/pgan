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

    learning_rate = pyre.properties.float(default=0.001)
    learning_rate.doc = 'Learning rate (default: 0.001)'

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

        # Load data
        train, test, lower, upper = pgan.data.unpack(self.data_file)

        print(train)

# end of file
