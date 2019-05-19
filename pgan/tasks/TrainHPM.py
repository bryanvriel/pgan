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

    input_checkdir = pyre.properties.str()
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

        # Load data
        train, test, lower, upper = module.unpack(self.data_file)
        
        # Convert layers to lists
        solution_layers = [int(n) for n in self.solution_layers.split(',')]
        pde_layers = [int(n) for n in self.pde_layers.split(',')]

        # Create the deep HPM model
        model = module.DeepHPM(
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

        # Load previous checkpoints
        if self.input_checkdir is not None:
            model.load(indir=self.input_checkdir)

        # Train the model
        model.train(train, test=test, n_epochs=self.n_epoch, batch_size=plexus.batch_size)

        # Save the weights
        model.save(outdir=self.checkdir)


# end of file
