#-*- coding: utf-8 -*-

import collections
import h5py

# Standard data tuple
DataTuple = collections.namedtuple('DataTuple', 'x y u v w t xcoll ycoll ucoll vcoll tcoll')

def unpack(filename):
    """
    Unpack HDF5 arrays and pack into DataTuple instances.
    """

    # Open the file
    with h5py.File(filename, 'r') as fid:

        # Pre-load collocation training points
        try:
            x_coll = atleast_2d(fid['x_train_coll'][()])
            y_coll = atleast_2d(fid['y_train_coll'][()])
            u_coll = atleast_2d(fid['u_train_coll'][()])
            v_coll = atleast_2d(fid['v_train_coll'][()])
            t_coll = atleast_2d(fid['t_train_coll'][()])
        except KeyError:
            x_coll = y_coll = u_coll = v_coll = t_coll = None

        # Make the training tuple
        train = DataTuple(x=atleast_2d(fid['x_train'][()]),
                          y=atleast_2d(fid['y_train'][()]),
                          u=atleast_2d(fid['u_train'][()]),
                          v=atleast_2d(fid['v_train'][()]),
                          w=atleast_2d(fid['w_train'][()]),
                          t=atleast_2d(fid['t_train'][()]),
                          xcoll=x_coll,
                          ycoll=y_coll,
                          ucoll=u_coll,
                          vcoll=v_coll,
                          tcoll=t_coll)

        # Pre-load collocation testing points
        try:
            x_coll = atleast_2d(fid['x_test_coll'][()])
            y_coll = atleast_2d(fid['y_test_coll'][()])
            u_coll = atleast_2d(fid['u_test_coll'][()])
            v_coll = atleast_2d(fid['v_test_coll'][()])
            t_coll = atleast_2d(fid['t_test_coll'][()])
        except KeyError:
            x_coll = y_coll = u_coll = v_coll = t_coll = None

        # Make the testing tuple
        test = DataTuple(x=atleast_2d(fid['x_test'][()]),
                         y=atleast_2d(fid['y_test'][()]),
                         u=atleast_2d(fid['u_test'][()]),
                         v=atleast_2d(fid['v_test'][()]),
                         w=atleast_2d(fid['w_test'][()]),
                         t=atleast_2d(fid['t_test'][()]),
                         xcoll=x_coll,
                         ycoll=y_coll,
                         ucoll=u_coll,
                         vcoll=v_coll,
                         tcoll=t_coll)

        # Load bounds
        try:
            lower = fid['lower'][()]
            upper = fid['upper'][()]
        except KeyError:
            lower = upper = None

        # Done
        return train, test, lower, upper

def atleast_2d(x):
    """
    Convenience function to ensure arrays are column vectors.
    """
    if x.ndim == 1:
        return x.reshape(-1, 1)
    elif x.ndim == 2:
        return x
    else:
        raise NotImplementedError('Input array has greater than 2 dimensions')

# end of file
