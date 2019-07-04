#-*- coding: utf-8 -*-

import h5py
from pgan.data import Data

def unpack(filename, train_fraction=0.9, batch_likelihood=256, batch_pde=1024):
    """
    Unpack HDF5 arrays and pack into DataTuple instances.
    """

    # Open the file
    with h5py.File(filename, 'r') as fid:

        # Get group for likelihood data
        likelihood = fid['likelihood']

        # Create data object for likelihood data
        data_likelihood = Data(train_fraction=train_fraction,
                               batch_size=batch_likelihood,
                               X=likelihood['X'][()],
                               Y=likelihood['Y'][()],
                               T=likelihood['T'][()],
                               W=likelihood['W'][()])

        # Get group for pde data
        pde = fid['pde']

        # Create data object for PDE data
        data_pde = Data(train_fraction=train_fraction,
                        batch_size=batch_pde,
                        X=pde['X'][()],
                        Y=pde['Y'][()],
                        T=pde['T'][()])

    # Done
    return data_likelihood, data_pde

# end of file
