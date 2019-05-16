#-*- coding: utf-8 -*-

import pgan
import pyre
import numpy as np
import tensorflow as tf
import logging
import h5py
import sys
import os

class TrainHPM(pgan.components.task, family='pgan.trainhpm'):
    """
    Train a hidden physics model from scattered data.
    """

    data_file = pyre.properties.str()
    data_file.doc = 'Input HDF5 of data'

    n_train = pyre.properties.int()
    n_train.doc = 'Number of training examples'

    n_epoch = pyre.properties.int(default=1000)
    n_epoch.doc = 'Number of training epochs (default: 1000)'

    solution_layers = pyre.properties.str()
    solution_layers.doc = 'Layer sizes for solution net'

    pde_layers = pyre.properties.str()
    pde_layers.doc = 'Layer sizes for PDE net'

    learning_rate = pyre.properties.float(default=0.001)
    learning_rate.doc = 'Learning rate (default: 0.001)'

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
        with h5py.File(self.data_file, 'r') as fid:

            # Use all training data 
            if self.n_train is None:
                x = fid['x_train'][()]
                t = fid['t_train'][()]
                u = fid['u_train'][()]
            # Or a subset
            else:
                x = fid['x_train'][:self.n_train]
                t = fid['t_train'][:self.n_train]
                u = fid['u_train'][:self.n_train]

            # Convert all data into column vectors
            x, t, u = [a.reshape((-1, 1)) for a in (x, t, u)]

            # Get testing data as well
            x_test = fid['x_test'][()].reshape((-1, 1))
            t_test = fid['t_test'][()].reshape((-1, 1))
            u_test = fid['u_test'][()].reshape((-1, 1))

            # Get space and time bounds
            lower = fid['lower'][()]
            upper = fid['upper'][()]

        # Convert layers to lists
        solution_layers = [int(n) for n in self.solution_layers.split(',')]
        pde_layers = [int(n) for n in self.pde_layers.split(',')]

        # Create the deep HPM model
        model = pgan.networks.DeepHPM(
            solution_layers=solution_layers,
            pde_layers=pde_layers,
            lower_bound=lower,
            upper_bound=upper
        )

        # Construct graphs
        model.build(learning_rate=self.learning_rate,
                    inter_op_cores=plexus.inter_op_cores,
                    intra_op_threads=plexus.intra_op_threads)
        model.print_variables()

        # Train the model
        model.train(x, t, u, test=(x_test, t_test, u_test), n_epochs=self.n_epoch)

        # Save the weights
        model.save(outdir=self.checkdir)


# end of file
