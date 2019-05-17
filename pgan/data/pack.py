#-*- coding: utf-8 -*-

from .structures import DataTuple
import h5py

def unpack(filename):
    """
    Unpack HDF5 arrays and pack into DataTuple instances.
    """

    # Open the file
    with h5py.File(filename, 'r') as fid:

        # Pre-load collocation training points
        try:
            x_coll = fid['x_train_coll'][()].reshape(-1, 1)
            t_coll = fid['t_train_coll'][()].reshape(-1, 1)
        except KeyError:
            x_coll = t_coll = None

        # Make the training tuple
        train = DataTuple(x=fid['x_train'][()].reshape(-1, 1),
                          t=fid['t_train'][()].reshape(-1, 1),
                          u=fid['u_train'][()].reshape(-1, 1),
                          xcoll=x_coll,
                          tcoll=t_coll)

        # Pre-load collocation testing points
        try:
            x_coll = fid['x_test_coll'][()].reshape(-1, 1)
            t_coll = fid['t_test_coll'][()].reshape(-1, 1)
        except KeyError:
            x_coll = t_coll = None

        # Make the testing tuple
        test = DataTuple(x=fid['x_test'][()].reshape(-1, 1),
                         t=fid['t_test'][()].reshape(-1, 1),
                         u=fid['u_test'][()].reshape(-1, 1),
                         xcoll=x_coll,
                         tcoll=t_coll)

        # Load bounds
        try:
            lower = fid['lower'][()]
            upper = fid['upper'][()]
        except KeyError:
            lower = upper = None

        # Done
        return train, test, lower, upper


# end of file
