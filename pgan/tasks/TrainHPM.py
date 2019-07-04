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

    dynamics = pyre.properties.str()
    dynamics.doc = 'Name of dynamics submodule'

    n_iterations = pyre.properties.int(default=10000)
    n_iterations.doc = 'Number of training iterations (default: 10000)'

    solution_layers = pyre.properties.str()
    solution_layers.doc = 'Layer sizes for solution net'

    pde_layers = pyre.properties.str()
    pde_layers.doc = 'Layer sizes for PDE net'

    learning_rate = pyre.properties.float(default=0.0001)
    learning_rate.doc = 'Learning rate (default: 0.0001)'

    initial_learning_rate = pyre.properties.float(default=None)
    initial_learning_rate.doc = 'Initial learning rate'

    final_learning_rate = pyre.properties.float(default=None)
    final_learning_rate.doc = 'Final learning rate'

    batch_size = pyre.properties.int(default=256)
    batch_size.doc = 'Batch size for data (default: 256)'

    train_fraction = pyre.properties.float(default=0.9)
    train_fraction.doc = 'Fraction of data to use for training (default: 0.9)'

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
        data, data_pde = module.unpack(self.data_file,
                                       train_fraction=self.train_fraction,
                                       batch_likelihood=self.batch_size,
                                       batch_pde=self.batch_size)

        # Convert layers to lists
        solution_layers = [int(n) for n in self.solution_layers.split(',')]
        pde_layers = [int(n) for n in self.pde_layers.split(',')]

        # Create the deep HPM model
        model = module.DeepHPM(
            solution_layers=solution_layers,
            pde_layers=pde_layers
        )

        # Construct graphs
        model.build(inter_op_cores=plexus.inter_op_cores,
                    intra_op_threads=plexus.intra_op_threads)
        model.print_variables()

        # Load previous checkpoints
        if self.input_checkdir is not None:
            model.load(indir=self.input_checkdir)

        # Construct learning rate input
        if self.initial_learning_rate is not None and self.final_learning_rate is not None:
            learning_rate = (self.initial_learning_rate, self.final_learning_rate)
        else:
            learning_rate = self.learning_rate

        # Train the model
        model.train(data, n_iterations=self.n_iterations, learning_rate=learning_rate)

        # Save the weights
        model.save(outdir=self.checkdir)


# end of file
