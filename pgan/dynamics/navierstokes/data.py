#-*- coding: utf-8 -*-

import collections
import h5py

# Standard data tuple
DataTuple = collections.namedtuple('DataTuple', 'x y w t xcoll ycoll ucoll vcoll tcoll wcoll')

def unpack(filename):
    """
    Unpack HDF5 arrays and pack into DataTuple instances.
    """

    # Open the file
    with h5py.File(filename, 'r') as fid:

        keys = ['x_train', 'y_train', 'w_train', 't_train', 'x_train_coll', 'y_train_coll',
                'u_train_coll', 'v_train_coll', 't_train_coll', 'w_train_coll']
        data = {}
        for key in keys:
            try:
                data[key] = atleast_2d(fid[key][()])
            except KeyError:
                data[key] = None

        # Make the training tuple
        train = DataTuple(x=data['x_train'],
                          y=data['y_train'],
                          w=data['w_train'],
                          t=data['t_train'],
                          xcoll=data['x_train_coll'],
                          ycoll=data['y_train_coll'],
                          ucoll=data['u_train_coll'],
                          vcoll=data['v_train_coll'],
                          tcoll=data['t_train_coll'],
                          wcoll=data['w_train_coll'])

        keys = ['x_test', 'y_test', 'w_test', 't_test', 'x_test_coll', 'y_test_coll',
                'u_test_coll', 'v_test_coll', 't_test_coll', 'w_test_coll']
        data = {}
        for key in keys:
            try:
                data[key] = atleast_2d(fid[key][()])
            except KeyError:
                data[key] = None
        
        # Make the testing tuple
        test = DataTuple(x=data['x_test'],
                         y=data['y_test'],
                         w=data['w_test'],
                         t=data['t_test'],
                         xcoll=data['x_test_coll'],
                         ycoll=data['y_test_coll'],
                         ucoll=data['u_test_coll'],
                         vcoll=data['v_test_coll'],
                         tcoll=data['t_test_coll'],
                         wcoll=data['w_test_coll'])

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
